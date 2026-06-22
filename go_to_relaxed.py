#!/usr/bin/env python3
# =============================================================================
# go_to_relaxed.py — move the real G1 UPPER BODY to the RELAXED pose and HOLD FOREVER.
#   waist -> 0; both arms -> RELAXED (shoulder_pitch 0, shoulder_roll +-0.2 abducted,
#   elbow 1.125 bent ~60 deg, wrists 0). Legs are HELD at their current position.
#
# This is the SAME relaxed start pose used by sim_grasp_viz.py / microwave_door_open_sim.py
# (the planning start for the door-open). It ramps DIRECTLY from the current pose to relaxed
# in ONE smooth interpolation — it never passes through the arms-straight (arms=0) "sim start"
# first, because the straight-arm pose hangs LOW and would sweep the hands through the table.
# Going straight to the bent-elbow relaxed pose keeps the arms up and clear of the table.
#
# Runs in the `tv` env, reusing real_pick.Body (unitree_sdk2py rt/lowcmd).
#
# SAFETY: DRY-RUN by default (reads state, prints current vs target, sends nothing). Robot
# HUNG/supported, zero-torque/low-level, NO other rt/lowcmd publisher. Ctrl-C freezes.
#
#   RUN:  python go_to_relaxed.py              # dry run (prints the move)
#         python go_to_relaxed.py --execute    # ramp DIRECTLY to relaxed, then hold forever
# =============================================================================
import argparse, time
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from real_pick import Body, release_mode, N, WAIST_IDX, LARM_IDX, RARM_IDX

# Relaxed pose (matches RELAXED_ARM in sim_grasp_viz.py / microwave_door_open_sim.py).
ELBOW_BEND, SHOULDER_PITCH, SHOULDER_ROLL = 1.125, 0.0, 0.2
# 7-joint arm order: [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw]
LARM_RELAXED = [SHOULDER_PITCH,  SHOULDER_ROLL, 0.0, ELBOW_BEND, 0.0, 0.0, 0.0]   # left  roll +0.2 (abduct off torso)
RARM_RELAXED = [SHOULDER_PITCH, -SHOULDER_ROLL, 0.0, ELBOW_BEND, 0.0, 0.0, 0.0]   # right roll -0.2 (abduct off torso)

JOINT_NAMES = [
    "left_hip_pitch","left_hip_roll","left_hip_yaw","left_knee","left_ankle_pitch","left_ankle_roll",
    "right_hip_pitch","right_hip_roll","right_hip_yaw","right_knee","right_ankle_pitch","right_ankle_roll",
    "waist_yaw","waist_roll","waist_pitch",
    "left_shoulder_pitch","left_shoulder_roll","left_shoulder_yaw","left_elbow","left_wrist_roll","left_wrist_pitch","left_wrist_yaw",
    "right_shoulder_pitch","right_shoulder_roll","right_shoulder_yaw","right_elbow","right_wrist_roll","right_wrist_pitch","right_wrist_yaw",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interface", default="eno1", help="NIC on the 192.168.123.x robot subnet")
    ap.add_argument("--domain-id", type=int, default=0)
    ap.add_argument("--ramp-sec", type=float, default=4.0, help="seconds to ramp current -> relaxed")
    ap.add_argument("--hz", type=float, default=100.0)
    ap.add_argument("--arms-only", action="store_true", help="leave the waist where it is")
    ap.add_argument("--force", action="store_true", help="run even if the move is unusually large (>3 rad)")
    ap.add_argument("--skip-mode-release", action="store_true")
    ap.add_argument("--execute", action="store_true", help="ACTUALLY MOVE (else dry run)")
    args = ap.parse_args()

    ChannelFactoryInitialize(args.domain_id, args.interface)
    body = Body()
    print(f"[net] iface={args.interface} domain={args.domain_id}; reading rt/lowstate ...")
    q_cur, mode_machine = body.read()
    if q_cur is None:
        print("[net] ERROR no rt/lowstate — robot on? NIC right? other DDS app? Aborting."); return
    body.cmd.mode_machine = mode_machine

    target = q_cur.copy()                                # legs (0-11) HELD at current
    target[LARM_IDX] = LARM_RELAXED
    target[RARM_IDX] = RARM_RELAXED
    if not args.arms_only:
        target[WAIST_IDX] = 0.0
    ub = WAIST_IDX + LARM_IDX + RARM_IDX

    print(f"[state] mode_machine={mode_machine}")
    print("\n  idx  joint                     current      target       delta")
    print("  " + "-" * 66)
    for i in ub:
        d = target[i] - q_cur[i]
        flag = "  <-- MOVE" if abs(d) > 1e-3 else ""
        print(f"  {i:>3}  {JOINT_NAMES[i]:<24} {q_cur[i]:>9.4f}   {target[i]:>9.4f}   {d:>9.4f}{flag}")
    dmax = float(np.abs(target[ub] - q_cur[ub]).max())
    print(f"  {'-'*66}\n  legs (0-11) HELD at current | max |delta| = {dmax:.3f} rad")
    print("  ramp goes DIRECTLY current -> relaxed (never via arms=0), so the arms stay clear of the table.\n")
    if dmax > 3.0 and not args.force:
        print("[ABORT] move >3 rad — check the state/table; --force to override."); return

    if not args.execute:
        print("[dry-run] would ramp current -> RELAXED over %.1fs, then HOLD FOREVER. Add --execute to move."
              % args.ramp_sec)
        return

    print("\n[execute] moving in 2s. Ctrl-C aborts (freezes).")
    time.sleep(2.0)
    if not args.skip_mode_release:
        release_mode()
    dt = 1.0 / args.hz
    n = max(1, int(args.ramp_sec * args.hz))
    try:
        print(f"[exec] ramping {args.ramp_sec:.1f}s DIRECTLY to the relaxed pose ...")
        for k in range(1, n + 1):
            q = q_cur + (k / n) * (target - q_cur)       # single smooth interpolation to relaxed
            body.send(q)
            if k % max(1, n // 5) == 0:
                print(f"  {100*k/n:4.0f}%  max|q-target|={np.abs(q[ub]-target[ub]).max():.3f}")
            time.sleep(dt)
        print("[exec] ✅ at the relaxed pose; HOLDING FOREVER (Ctrl-C to release) ...")
        while True:
            body.send(target); time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[exec] Ctrl-C — freezing for 0.5s")
        for _ in range(int(0.5 * args.hz)):
            body.send(target); time.sleep(dt)


if __name__ == "__main__":
    main()
