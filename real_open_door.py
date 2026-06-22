#!/usr/bin/env python3
# =============================================================================
# real_open_door.py — REAL G1 LEFT-arm microwave-door OPEN, in ONE process (`tv` env).
#
#   PHASE0  ramp WHATEVER -> RELAXED, DIRECTLY (exactly like go_to_relaxed.py): waist 0,
#           both arms to the relaxed pose, legs held. Never via arms=0 (that hits the table).
#   PHASE1  from relaxed, replay the LEFT-arm door-open trajectory (reach -> slide -> swing)
#           planned by real_plan_door.py. RIGHT arm held FROZEN at relaxed, legs held, LEFT
#           hand OPEN (the open fingers hook the gap behind the handle and pull the door).
#   PHASE2  (if the plan has a grab) RIGHT arm grabs the glass: approach -> grasp -> CLOSE the right
#           hand -> bring it to the staging pose (BRING_BACK_XYZ). LEFT arm HOLDS the door open. No
#           insert into the microwave -- just like sim_grasp_viz.py phases 1-3.
#   HOLD    hold the final pose (glass at the staging pose) until Ctrl-C.
#
# Reads /tmp/door_open_traj.json (from real_plan_door.py). Its point 0 IS the relaxed left
# arm, so PHASE0 lands exactly where PHASE1 begins.
#
# PREREQ (for the LEFT hand): Headless_driver_double.py running. SAFETY: DRY-RUN by default.
# Robot HUNG/supported, zero-torque/low-level, no other rt/lowcmd publisher. Ctrl-C freezes.
#
#   RUN:  python real_open_door.py                 # dry run (prints the plan, sends nothing)
#         python real_open_door.py --execute       # relaxed -> open door -> hold
# =============================================================================
import argparse, json, os, sys, time, threading
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from real_pick import Body, Hand, release_mode, N, WAIST_IDX, LARM_IDX, RARM_IDX

DEFAULT_TRAJ = "/tmp/door_open_traj.json"
LARM_NAMES = ["left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
              "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint"]
RARM_NAMES = ["right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
              "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"]
# relaxed pose fallback (matches go_to_relaxed.py) if the traj omits it
ELBOW_BEND, SHOULDER_PITCH, SHOULDER_ROLL = 1.125, 0.0, 0.2
LARM_RELAXED = [SHOULDER_PITCH,  SHOULDER_ROLL, 0.0, ELBOW_BEND, 0.0, 0.0, 0.0]
RARM_RELAXED = [SHOULDER_PITCH, -SHOULDER_ROLL, 0.0, ELBOW_BEND, 0.0, 0.0, 0.0]
LHAND_CTRL = "rt/inspire_hand/ctrl/l"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default=DEFAULT_TRAJ, help="left-arm door trajectory JSON from real_plan_door.py")
    ap.add_argument("--interface", default="eno1", help="NIC on the 192.168.123.x robot subnet")
    ap.add_argument("--domain-id", type=int, default=0)
    ap.add_argument("--to-relaxed-sec", type=float, default=6.0, help="PHASE0 ramp current -> relaxed (slow)")
    ap.add_argument("--relaxed-hold", type=float, default=1.0, help="hold at relaxed before the door motion")
    ap.add_argument("--hz", type=float, default=100.0)
    ap.add_argument("--speed", type=float, default=0.2, help="door playback speed (<1 slower; 0.1 = very slow). Ctrl-C any time freezes & holds.")
    ap.add_argument("--no-pause", action="store_true", help="skip the ENTER confirmations between phases")
    ap.add_argument("--no-hand", action="store_true", help="do NOT open the left hand (assume already open)")
    ap.add_argument("--no-grab", action="store_true", help="open the door only (skip the right-arm glass grab even if planned)")
    ap.add_argument("--thumb-rot", type=int, default=0, help="RIGHT-hand thumb rotation target for the glass grasp (0 = full opposition)")
    ap.add_argument("--curl", type=int, default=300, help="RIGHT-hand curl target for the glass grasp (1000=open, smaller=tighter)")
    ap.add_argument("--rotate-wait", type=float, default=2.5, help="s for the thumb to rotate before the curl")
    ap.add_argument("--grasp-settle", type=float, default=0.8, help="s to hold the grasp after the hand closes")
    ap.add_argument("--release-settle", type=float, default=0.8, help="s to hold after the hand opens (glass released in the microwave)")
    ap.add_argument("--no-insert", action="store_true", help="stop after bring-back (do NOT put the glass in the microwave / close the door even if planned)")
    ap.add_argument("--force", action="store_true", help="run even if the to-relaxed move is >3 rad")
    ap.add_argument("--skip-mode-release", action="store_true")
    ap.add_argument("--execute", action="store_true", help="ACTUALLY MOVE (else dry run)")
    args = ap.parse_args()

    if not os.path.exists(args.traj):
        print(f"[traj] ERROR {args.traj} not found — run real_plan_door.py first."); sys.exit(2)
    T = json.load(open(args.traj))
    assert T["joint_names"] == LARM_NAMES, T["joint_names"]
    pts = T["points"]
    larm0 = np.array(T.get("relaxed_larm", pts[0]["positions"]))      # relaxed left arm = PHASE1 start
    rarm_hold = np.array(T.get("right_arm_hold", RARM_RELAXED))       # freeze the right arm here (relaxed)
    print(f"[traj] {args.traj}: {len(pts)} door pts, total {pts[-1]['time']:.2f}s, "
          f"door opens to {T.get('reached_deg','?')} deg, handle={T.get('handle_xyz')}")

    ChannelFactoryInitialize(args.domain_id, args.interface)
    body = Body()
    print(f"[net] iface={args.interface} domain={args.domain_id}; reading rt/lowstate ...")
    q_cur, mode_machine = body.read()
    if q_cur is None:
        print("[net] ERROR no rt/lowstate — robot on? NIC right? other DDS app? Aborting."); sys.exit(2)
    body.cmd.mode_machine = mode_machine

    # RELAXED full-body pose: legs HELD at current, waist 0, both arms relaxed (left = traj start)
    relaxed = q_cur.copy()
    relaxed[WAIST_IDX] = 0.0
    relaxed[LARM_IDX] = larm0
    relaxed[RARM_IDX] = rarm_hold
    ub = WAIST_IDX + LARM_IDX + RARM_IDX
    dmax = float(np.abs(relaxed[ub] - q_cur[ub]).max())
    print(f"[state] mode_machine={mode_machine}  PHASE0 to-relaxed max|delta|={dmax:.3f} rad (legs held; direct, no arms=0)")
    if dmax > 3.0 and not args.force:
        print("[state] ABORT: to-relaxed move >3 rad — check state/table; --force to override."); sys.exit(3)

    dt = 1.0 / args.hz
    has_grab = ("grab" in T) and not args.no_grab

    # ---- PERSISTENT command STREAMER -------------------------------------------------------------------
    # ONE thread always publishes rt/lowcmd at `hz`, sending the latest target in state["cmd"]. The robot
    # therefore NEVER loses command between phases -- not during the Hand() init/open, the ENTER pauses, the
    # finger close, or any phase switch -- so the arms never sag/jerk. Foreground code only UPDATES the target
    # (hold()/play()); it never calls body.send() itself, so there is exactly one sender (no gaps, no races).
    state = {"cmd": (q_cur.copy(), np.zeros(N))}                     # (q, dq); whole-tuple swap is atomic under the GIL
    stream_stop = threading.Event()
    def streamer():
        while not stream_stop.is_set():
            q, dq = state["cmd"]; body.send(q, dq); time.sleep(dt)
    def hold(q):                                                    # hold a static pose (zero feed-forward velocity)
        state["cmd"] = (np.array(q, dtype=float), np.zeros(N))
    def play(points, idx, q_base, label):                          # feed the streamer an interpolated arm trajectory
        times = np.array([p["time"] for p in points]) / max(1e-3, args.speed)
        pos = np.array([p["positions"] for p in points])
        vel = np.array([p["velocities"] if p["velocities"] else [0]*7 for p in points]) * args.speed
        t_end = float(times[-1]); t0 = time.time()
        while True:
            tt = time.time() - t0
            if tt >= t_end: break
            j = int(np.searchsorted(times, tt)); j = min(max(j, 1), len(points) - 1)
            f = (tt - times[j-1]) / max(1e-6, times[j] - times[j-1])
            arm = pos[j-1] + f * (pos[j] - pos[j-1]); adq = vel[j-1] + f * (vel[j] - vel[j-1])
            q = q_base.copy(); q[idx] = arm; dq = np.zeros(N); dq[idx] = adq
            state["cmd"] = (q, dq)                                  # hand the target to the streamer
            if int(tt / dt) % max(1, int(0.5 / dt)) == 0:
                print(f"  [{label}] t={tt:4.2f}/{t_end:.2f}s arm={['%+.2f'%v for v in arm]}")
            time.sleep(dt)
        qf = q_base.copy(); qf[idx] = pos[-1]; hold(qf)             # settle exactly on the final pose (no leftover vel)

    if not args.execute:
        grab_msg = (" -> PHASE2 RIGHT arm grab glass (approach->grasp->close hand->bring back to %s)"
                    % (T["grab"]["bring_back_xyz"])) if has_grab else ""
        ins_msg = (" -> PHASE3 put glass IN microwave -> release -> retract right to relaxed -> PHASE4 close door + relax left"
                   if has_grab and ("insert" in T["grab"]) and not args.no_insert else "")
        print("\n[dry-run] would: release mode -> PHASE0 ramp current->relaxed (%.1fs) -> %s%sPHASE1 open door "
              "(reach->slide->swing, ~%.0f deg)%s%s -> hold. Nothing sent. Add --execute."
              % (args.to_relaxed_sec, "" if args.no_hand else "open LEFT hand -> ",
                 "" if args.no_pause else "ENTER -> ", T.get("reached_deg", 0), grab_msg, ins_msg))
        return

    print("\n[execute] moving in 2s. Ctrl-C aborts (freezes).")
    time.sleep(2.0)
    if not args.skip_mode_release: release_mode()

    # START the streamer NOW (holding the current pose) and keep it running for the WHOLE sequence -- the body
    # is commanded continuously from here to the end, so there is no command gap at any phase switch.
    stream_th = threading.Thread(target=streamer, daemon=True); stream_th.start()

    # open the LEFT hand so the fingers can hook the handle gap (best-effort; the streamer holds the body meanwhile)
    if not args.no_hand:
        try:
            from inspire_sdkpy import inspire_dds, inspire_hand_defaut
            lpub = ChannelPublisher(LHAND_CTRL, inspire_dds.inspire_hand_ctrl); lpub.Init()
            c = inspire_hand_defaut.get_inspire_hand_ctrl(); c.angle_set = [1000]*6; c.mode = 0b0001
            for _ in range(12): lpub.Write(c); time.sleep(0.02)
            print("[hand] LEFT hand opened.")
        except Exception as e:
            print(f"[hand] could not open LEFT hand ({e}); continuing — ensure it is open.")

    try:
        # PHASE0: whatever -> relaxed, DIRECTLY (like go_to_relaxed.py) -- feed the streamer
        n = max(1, int(args.to_relaxed_sec * args.hz))
        print(f"[exec] PHASE0 ramp current -> relaxed over {args.to_relaxed_sec:.1f}s (direct) ...")
        for k in range(1, n + 1):
            state["cmd"] = (q_cur + (k / n) * (relaxed - q_cur), np.zeros(N)); time.sleep(dt)
        hold(relaxed)
        for _ in range(int(args.relaxed_hold * args.hz)): time.sleep(dt)

        # confirmation pause -- the streamer keeps holding `relaxed` while we wait (Ctrl-C aborts/freezes)
        if not args.no_pause:
            input("[exec] at RELAXED — inspect, then press ENTER to OPEN the door (Ctrl-C to abort) ...")

        # PHASE1: relaxed -> door open (LEFT arm replays; RIGHT arm + legs held at relaxed)
        print(f"[exec] PHASE1 open door ({len(pts)} pts, speed x{args.speed}) -- Ctrl-C any time to freeze ...")
        play(pts, LARM_IDX, relaxed, "door")
        qD = relaxed.copy(); qD[LARM_IDX] = np.array(pts[-1]["positions"]); hold(qD)   # streamer now holds the door-open pose
        print(f"[exec] ✅ DOOR OPEN (~{T.get('reached_deg','?')} deg).")
        if not has_grab:
            print("[exec] holding the door-open pose until Ctrl-C.")
            while True: time.sleep(0.2)

        # PHASE2: RIGHT arm grabs the glass + brings it to the staging pose. LEFT arm HOLDS the door open.
        # The streamer holds qD/q_grab CONTINUOUSLY through the Hand() init/open and the ENTER pause, so the
        # arms never lose command at this switch (this is the jerk we were seeing).
        grab = T["grab"]; assert grab["joint_names"] == RARM_NAMES, grab["joint_names"]
        left_hold = np.array(T["left_hold"])
        q_grab = relaxed.copy(); q_grab[LARM_IDX] = left_hold; hold(q_grab)            # hold the LEFT arm at door-open
        handR = None
        try:
            handR = Hand(); time.sleep(0.5); handR.open(); print("[hand] RIGHT hand opened.")
        except Exception as e:
            print(f"[hand] RIGHT hand unavailable ({e}); continuing — ensure it is open.")
        if not args.no_pause:
            input("[exec] door open, holding. Press ENTER to GRAB the glass (Ctrl-C to abort) ...")
        print(f"[exec] PHASE2 grab glass (right arm; left holds door) -- Ctrl-C any time to freeze ...")
        play(grab["approach"], RARM_IDX, q_grab, "approach")
        play(grab["grasp"],    RARM_IDX, q_grab, "grasp")
        # close the RIGHT hand -- the streamer holds the grasp pose, so the arm never goes limp
        q_grasp = q_grab.copy(); q_grasp[RARM_IDX] = np.array(grab["grasp"][-1]["positions"]); hold(q_grasp)
        print("[exec] GRASP: closing the RIGHT hand (arm held by the streamer) ...")
        if handR is not None: handR.close(args.thumb_rot, args.curl, args.rotate_wait)
        time.sleep(args.grasp_settle)
        # bring the grasped glass to the staging pose (hand stays closed)
        print(f"[exec] PHASE2 bring glass to staging {grab['bring_back_xyz']} ...")
        play(grab["bringback"], RARM_IDX, q_grab, "bringback")
        qF = q_grab.copy(); qF[RARM_IDX] = np.array(grab["bringback"][-1]["positions"]); hold(qF)

        # PHASE3: PUT the glass IN the microwave -> release -> retract the right arm to relaxed (LEFT still holds door)
        has_insert = ("insert" in grab) and ("retract" in grab) and not args.no_insert
        if not has_insert:
            print("[exec] ✅ glass at the staging pose — holding until Ctrl-C (no put-in planned / --no-insert).")
            while True: time.sleep(0.2)
        print(f"[exec] PHASE3 put glass IN microwave ({grab.get('place_angle','?')}deg x {grab.get('place_dist','?')}m) -- Ctrl-C to freeze ...")
        play(grab["insert"], RARM_IDX, q_grab, "insert")
        qIns = q_grab.copy(); qIns[RARM_IDX] = np.array(grab["insert"][-1]["positions"]); hold(qIns)   # streamer holds while the hand opens
        # RELEASE in the cavity: open ONLY the curl (fingers + thumb-curl -> 1000); KEEP the thumb rotation where
        # it is (thumb_ROT = args.thumb_rot, i.e. still rotated in) so the thumb does NOT sweep the glass while the
        # hand is still inside the microwave. Angle order: [pinky,ring,middle,index,thumb_curl,thumb_ROT].
        print("[exec] RELEASE: uncurl to 1000, KEEP thumb rotation (glass stays in the microwave) ...")
        if handR is not None: handR.send([1000, 1000, 1000, 1000, 1000, args.thumb_rot], n=12)
        time.sleep(args.release_settle)
        print("[exec] PHASE3 retract right arm out + to relaxed ...")
        play(grab["retract"], RARM_IDX, q_grab, "retract")          # inserted -> staging -> relaxed (hand uncurled, thumb still rotated in)
        # hand is now clear of the microwave at relaxed -> NOW rotate the thumb out (full open)
        print("[exec] thumb rotation -> 1000 (hand fully open, clear of the microwave) ...")
        if handR is not None: handR.send([1000, 1000, 1000, 1000, 1000, 1000], n=12)

        # PHASE4: CLOSE the door with the LEFT arm (reverse of the open) + retract the left arm to relaxed.
        # Right arm is now at relaxed, so q_base = relaxed; the play drives the LEFT arm from door-open back home.
        if "door_close" in T:
            dc = T["door_close"]
            print(f"[exec] PHASE4 close door + retract left arm ({len(dc)} pts) -- Ctrl-C to freeze ...")
            play(dc, LARM_IDX, relaxed, "door_close")
        hold(relaxed)
        print("[exec] ✅ glass placed, door closed, both arms relaxed — holding until Ctrl-C.")
        while True: time.sleep(0.2)
    except KeyboardInterrupt:
        # FREEZE and HOLD the last commanded pose -- the streamer keeps running (no gap, no jerk). Because the
        # robot is actively tracking state["cmd"], freezing on it (zero velocity) is where the arms already are.
        print("\n[exec] Ctrl-C — FREEZING and HOLDING the current pose (Ctrl-C again to release).")
        hold(state["cmd"][0])
        try:
            while True: time.sleep(0.2)
        except KeyboardInterrupt:
            stream_stop.set(); print("[exec] released.")


if __name__ == "__main__":
    main()
