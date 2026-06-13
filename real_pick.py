#!/usr/bin/env python3
# =============================================================================
# real_pick.py — FULL real G1 pick in ONE process (run in the `tv` conda env).
# Drives BOTH the body (unitree_sdk2py, rt/lowcmd) and the inspire FTP hand
# (inspire_sdkpy, rt/inspire_hand/ctrl/r) in a single process, so there is NO
# ENTER and NO separate hand script.
#
#   PHASE1  ramp the upper body to the sim start (waist=0, left arm=0, right arm=traj start)
#   PHASE2  approach + grasp  (right arm -> the grasp pose; hand OPEN)
#   GRASP   close the hand: full thumb rotation, wait, then curl  (automatic)
#   PHASE3  lift (+0.15 m); hand stays closed
#   HOLD    hold the lifted glass until Ctrl-C
#
# Reads the trajectory planned by real_plan_approach.py (approach+grasp+lift). The
# body holds the grasp pose in a background thread while the hand closes, so the arm
# never goes limp during the grasp.
#
# PREREQ: Headless_driver_double.py running (bridges the hands to DDS).
# SAFETY: DRY-RUN by default. Robot HUNG/supported, zero-torque/low-level, no other
#         rt/lowcmd publisher. Ctrl-C freezes.
#
#   RUN:  python real_pick.py                 # dry run (prints the plan, sends nothing)
#         python real_pick.py --execute       # full pick: approach -> grasp -> lift
#         python real_pick.py --open-hand --execute   # just open the hand
# =============================================================================
import argparse, json, os, sys, time, threading
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from inspire_sdkpy import inspire_dds, inspire_hand_defaut

DEFAULT_TRAJ = "/home/dishant/Projects/GR00T-WholeBodyControl/scripts/approach_traj.json"
ARM_NAMES = ["right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
             "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"]
# ---- body (official G1 motor order) ----
WAIST_IDX = [12, 13, 14]; LARM_IDX = list(range(15, 22)); RARM_IDX = list(range(22, 29)); N = 29
MOTOR_KP = [150,150,150,200,40,40, 150,150,150,200,40,40, 250,250,250,
            100,100,40,40,20,20,20, 100,100,40,40,20,20,20]
MOTOR_KD = [2,2,2,4,2,2, 2,2,2,4,2,2, 5,5,5,
            5,5,2,2,2,2,2, 5,5,2,2,2,2,2]
# ---- hand (inspire FTP, angle order [pinky,ring,middle,index,thumb_curl,thumb_ROT]; 1000=open) ----
HAND_CTRL = "rt/inspire_hand/ctrl/r"; HAND_STATE = "rt/inspire_hand/state/r"
H_OPEN = 1000; H_THUMB_ROT = 5; H_CURL_IDX = [0, 1, 2, 3, 4]; H_FINGERS = [0, 1, 2, 3]
H_STATUS = {0:"open", 1:"closing", 2:"pos", 3:"FORCE", 5:"CURRENT", 6:"STALL", 7:"fault", 255:"err"}


class Body:
    """Right-arm body control over rt/lowcmd. ChannelFactory must be initialized first."""
    def __init__(self):
        self.crc = CRC()
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_); self.pub.Init()
        self._st = None
        self.sub = ChannelSubscriber("rt/lowstate", LowState_); self.sub.Init(self._cb, 10)
        self.cmd = unitree_hg_msg_dds__LowCmd_(); self.cmd.mode_pr = 0
        for i in range(N):
            self.cmd.motor_cmd[i].mode = 1
            self.cmd.motor_cmd[i].kp = float(MOTOR_KP[i]); self.cmd.motor_cmd[i].kd = float(MOTOR_KD[i])
            self.cmd.motor_cmd[i].dq = 0.0; self.cmd.motor_cmd[i].tau = 0.0
    def _cb(self, m): self._st = m
    def read(self, timeout=5.0):
        t0 = time.time()
        while self._st is None and time.time() - t0 < timeout: time.sleep(0.02)
        if self._st is None: return None, None
        q = np.array([self._st.motor_state[i].q for i in range(N)], dtype=np.float64)
        return q, int(self._st.mode_machine)
    def send(self, q29, dq29=None):
        for i in range(N):
            self.cmd.motor_cmd[i].q = float(q29[i])
            self.cmd.motor_cmd[i].dq = float(dq29[i]) if dq29 is not None else 0.0
        self.cmd.crc = self.crc.Crc(self.cmd); self.pub.Write(self.cmd)


class Hand:
    """Right inspire FTP hand over rt/inspire_hand/ctrl/r. ChannelFactory must be initialized first."""
    def __init__(self):
        self.pub = ChannelPublisher(HAND_CTRL, inspire_dds.inspire_hand_ctrl); self.pub.Init()
        self.sub = ChannelSubscriber(HAND_STATE, inspire_dds.inspire_hand_state); self.sub.Init()
    def send(self, angles, n=12, dt=0.02):
        c = inspire_hand_defaut.get_inspire_hand_ctrl()
        c.angle_set = [int(max(0, min(1000, a))) for a in angles]; c.mode = 0b0001
        for _ in range(n): self.pub.Write(c); time.sleep(dt)
    def read(self):
        s = self.sub.Read()
        return None if s is None else dict(angle=list(s.angle_act), force=list(s.force_act), status=list(s.status))
    def open(self): self.send([H_OPEN]*6, n=12)
    def close(self, thumb_rot, curl, rotate_wait):
        rot = [H_OPEN]*5 + [thumb_rot]                  # [1000,1000,1000,1000,1000, rot]
        cur = [curl]*5 + [thumb_rot]                    # [500,500,500,500,500, rot]
        print(f"[hand] (1) thumb rotation {rot}")
        self.send(rot, n=12); time.sleep(rotate_wait)
        print(f"[hand] (2) curl {cur}")
        self.send(cur, n=12); time.sleep(0.5)
        s = self.read()
        if s:
            contacts = [i for i in H_FINGERS if s['status'][i] in (3, 5, 6)]
            print(f"[hand] angle={s['angle']} status={[H_STATUS.get(s['status'][i], s['status'][i]) for i in H_FINGERS]} "
                  f"force={[s['force'][i] for i in H_FINGERS]} -> {'CONTACT '+str(contacts) if contacts else 'no contact'}")
        return s


def release_mode():
    try:
        msc = MotionSwitcherClient(); msc.SetTimeout(5.0); msc.Init()
        status, result = msc.CheckMode(); print(f"[mode] CheckMode -> {status} {result}")
        tries = 0
        while result and result.get("name"):
            msc.ReleaseMode(); status, result = msc.CheckMode(); time.sleep(1.0); tries += 1
            if tries > 5: break
        print("[mode] low-level enabled")
    except Exception as e:
        print(f"[mode] switcher unreachable ({e}); assuming already low-level/zero-torque")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default=DEFAULT_TRAJ, help="trajectory JSON from real_plan_approach.py")
    ap.add_argument("--interface", default="eno1", help="NIC on the 192.168.123.x robot subnet (body + hand DDS)")
    ap.add_argument("--domain-id", type=int, default=0)
    ap.add_argument("--to-start-sec", type=float, default=4.0)
    ap.add_argument("--start-hold", type=float, default=1.5)
    ap.add_argument("--hz", type=float, default=100.0)
    ap.add_argument("--speed", type=float, default=0.5, help="arm playback speed scale (<1 slower)")
    ap.add_argument("--grasp-settle", type=float, default=0.8, help="s to hold after the hand closes, before lifting")
    ap.add_argument("--thumb-rot", type=int, default=0, help="thumb rotation target (6th value; 0 = full opposition)")
    ap.add_argument("--curl", type=int, default=500, help="curl target for the first 5 (fingers + thumb curl)")
    ap.add_argument("--rotate-wait", type=float, default=2.5, help="s for the thumb to rotate before the curl")
    ap.add_argument("--no-lift", action="store_true", help="grasp but do not lift (stay at the grasp pose)")
    ap.add_argument("--force", action="store_true", help="run even if the to-start move is unusually large")
    ap.add_argument("--skip-mode-release", action="store_true")
    ap.add_argument("--open-hand", action="store_true", help="just open the hand and exit")
    ap.add_argument("--execute", action="store_true", help="ACTUALLY MOVE (else dry run)")
    args = ap.parse_args()

    ChannelFactoryInitialize(args.domain_id, args.interface)

    if args.open_hand:
        hand = Hand(); time.sleep(1.0)
        print("[hand] OPEN")
        if args.execute: hand.open()
        return

    if not os.path.exists(args.traj):
        print(f"[traj] ERROR {args.traj} not found — run real_plan_approach.py first."); sys.exit(2)
    T = json.load(open(args.traj))
    assert T["joint_names"] == ARM_NAMES, T["joint_names"]
    pts = T["points"]; lift_pts = T.get("lift_points")
    arm0 = np.array(pts[0]["positions"]); armT = np.array(pts[-1]["positions"])
    print(f"[traj] {args.traj}: {len(pts)} approach+grasp pts, {len(lift_pts) if lift_pts else 0} lift pts, "
          f"target_wrist={T.get('target_wrist')}")

    body = Body(); hand = Hand()
    print(f"[net] iface={args.interface} domain={args.domain_id}; reading rt/lowstate ...")
    q_cur, mode_machine = body.read()
    if q_cur is None:
        print("[net] ERROR no rt/lowstate — robot on? NIC right? other DDS app? Aborting."); sys.exit(2)
    body.cmd.mode_machine = mode_machine
    hs = hand.read()
    print(f"[hand] state: {('angle='+str(hs['angle'])) if hs else 'NONE (is Headless_driver_double.py running?)'}")

    start_pose = q_cur.copy(); start_pose[WAIST_IDX] = 0.0; start_pose[LARM_IDX] = 0.0; start_pose[RARM_IDX] = arm0
    ub = WAIST_IDX + LARM_IDX + RARM_IDX
    to_start_delta = np.abs(start_pose[ub] - q_cur[ub])
    print(f"[state] mode_machine={mode_machine}  right arm now={['%+.3f'%v for v in q_cur[RARM_IDX]]}")
    print(f"[state] PHASE1 to-sim-start max|delta|={to_start_delta.max():.3f} rad")
    if to_start_delta.max() > 3.0 and not args.force:
        print("[state] ABORT: to-start move >3 rad — check state; --force to override."); sys.exit(3)

    dt = 1.0 / args.hz
    q_hold = start_pose.copy()
    def full(rarm, rdq=None):
        q = q_hold.copy(); q[RARM_IDX] = rarm
        dq = np.zeros(N)
        if rdq is not None: dq[RARM_IDX] = rdq
        return q, dq
    def play(points, label):
        times = np.array([p["time"] for p in points]) / max(1e-3, args.speed)
        pos = np.array([p["positions"] for p in points])
        vel = np.array([p["velocities"] if p["velocities"] else [0]*7 for p in points]) * args.speed
        t_end = float(times[-1]); t0 = time.time()
        while True:
            tt = time.time() - t0
            if tt >= t_end: break
            j = int(np.searchsorted(times, tt)); j = min(max(j, 1), len(points) - 1)
            f = (tt - times[j-1]) / max(1e-6, times[j] - times[j-1])
            rarm = pos[j-1] + f * (pos[j] - pos[j-1]); rdq = vel[j-1] + f * (vel[j] - vel[j-1])
            q, dq = full(rarm, rdq); body.send(q, dq)
            if int(tt / dt) % max(1, int(0.5 / dt)) == 0:
                print(f"  [{label}] t={tt:4.2f}/{t_end:.2f}s arm={['%+.2f'%v for v in rarm]}")
            time.sleep(dt)

    if not args.execute:
        print("\n[dry-run] would: release mode -> PHASE1 to start -> PHASE2 approach+grasp -> "
              f"CLOSE hand (rot={args.thumb_rot}, curl={args.curl}) -> "
              f"{'NO lift' if args.no_lift else 'PHASE3 lift'} -> hold. Nothing sent. Add --execute.")
        return

    print("\n[execute] moving in 2s. Ctrl-C aborts (freezes).")
    time.sleep(2.0)
    if not args.skip_mode_release: release_mode()

    try:
        # PHASE1 to sim start
        n = max(1, int(args.to_start_sec * args.hz))
        print(f"[exec] PHASE1 to sim start over {args.to_start_sec:.1f}s ...")
        for k in range(1, n + 1):
            q = q_cur + (k / n) * (start_pose - q_cur); body.send(q); time.sleep(dt)
        for _ in range(int(args.start_hold * args.hz)): body.send(start_pose); time.sleep(dt)

        # PHASE2 approach + grasp
        print(f"[exec] PHASE2 approach+grasp ({len(pts)} pts, speed x{args.speed}) ...")
        play(pts, "approach")

        # hold the grasp pose in a background thread while the hand closes
        q_grasp, _ = full(armT)
        stop = threading.Event()
        def holder():
            while not stop.is_set(): body.send(q_grasp); time.sleep(dt)
        th = threading.Thread(target=holder, daemon=True); th.start()
        print("[exec] GRASP: closing the hand (arm holding) ...")
        hand.close(args.thumb_rot, args.curl, args.rotate_wait)
        time.sleep(args.grasp_settle)
        stop.set(); th.join(timeout=1.0)

        # PHASE3 lift
        if args.no_lift or not lift_pts:
            print("[exec] holding grasp pose (no lift) until Ctrl-C ...")
            while True: body.send(q_grasp); time.sleep(dt)
        print(f"[exec] PHASE3 LIFT ({len(lift_pts)} pts, +0.15m) ...")
        play(lift_pts, "lift")
        qL, _ = full(np.array(lift_pts[-1]["positions"]))
        print("[exec] ✅ PICKED — holding the lifted glass until Ctrl-C (then --open-hand to release).")
        while True: body.send(qL); time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[exec] Ctrl-C — freezing for 0.5s")
        qn, _ = body.read(timeout=1.0)
        for _ in range(int(0.5 * args.hz)): body.send(qn if qn is not None else q_hold); time.sleep(dt)


if __name__ == "__main__":
    main()
