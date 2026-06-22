#!/usr/bin/env python3
# =============================================================================
# check_door_reach.py -- TEXT-ONLY feasibility check for the door-open motion (no browser).
# Reuses the constants from sim_grasp_viz.py so the two never drift, then asks the real
# cuRobo planner three things:
#   [1] is the RELAXED start pose collision-free against the (solid, moved-away) table?
#   [2] can the right hand REACH the handle from that start?  (sweeps the wrist yaw)
#   [3] is the door-open SWING arc reachable, step by step, from the grasp config?
#
# RUN (cumotion_venv):  /home/dishant/cumotion_venv/bin/python check_door_reach.py
#
# Reading the output:
#  - "Start ... in collision" on the reaches  -> the table is still too close: raise TABLE_BACK
#    in sim_grasp_viz.py (e.g. 0.34 -> 0.40) and re-run this.
#  - reaches FAIL with some other status      -> the handle is out of reach for that wrist yaw:
#    try a different DOOR_WRIST_YAW, or nudge MW_CX / MW_CY / DOOR_REACH.
#  - swing stops early                          -> that's the max the door opens before the
#    handle gets too close to the robot (a real fixed-base limit) -- lower DOOR_OPEN_DEG.
# =============================================================================
import numpy as np, torch
import sim_grasp_viz as sv                       # pulls TABLE, HANDLE_XYZ, HINGE_XYZ, RELAXED_ARM, DOOR_* etc.

from curobo.scene import Scene, Cuboid
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState
from curobo._src.state.state_joint_trajectory_ops import get_joint_state_at_horizon_index

print("building planner (~15 s) ...", flush=True)
cfg = MotionPlannerCfg.create(robot=sv.OUT, scene_model=None, max_goalset=8, collision_cache={"obb": 8, "mesh": 4})
pl = MotionPlanner(cfg)
pl.update_world(Scene(cuboid=[Cuboid(name="table", dims=sv.TABLE[0], pose=sv.TABLE[1])]))   # SAME solid table
pl.warmup(enable_graph=True, num_warmup_iterations=5)
kin = pl.kinematics

bx, dx = sv.TABLE[1][0], sv.TABLE[0][0]
print(f"tool frame(s): {pl.tool_frames}")
print(f"table: x-range=[{bx-dx/2:.3f},{bx+dx/2:.3f}] (back edge {bx-dx/2:.3f}), top_z={sv.TABLE_TOP_Z:.3f}")
print(f"handle={tuple(round(v,3) for v in sv.HANDLE_XYZ)}  hinge={tuple(round(v,3) for v in sv.HINGE_XYZ)}  "
      f"elbow_bend={sv.ELBOW_BEND:.3f}rad")

# ---- relaxed start config (mirrors sim_grasp_viz.relaxed_qstart) ----
jn = list(pl.default_joint_state.joint_names)
pos = pl.default_joint_state.position.clone().reshape(-1)
for n, v in sv.RELAXED_ARM.items():
    if n in jn: pos[jn.index(n)] = float(v)
qstart = kin.get_active_js(JointState.from_position(pos.unsqueeze(0), joint_names=jn))

def goal(p, q):
    return GoalToolPose(tool_frames=pl.tool_frames,
                        position=torch.tensor([[[[list(p)]]]], device="cuda", dtype=torch.float32),
                        quaternion=torch.tensor([[[[list(q)]]]], device="cuda", dtype=torch.float32))
def ok(r): return r is not None and r.success is not None and bool(r.success.any())

def rot(px, py, phi):
    c, s = np.cos(phi), np.sin(phi); rx, ry = px - sv.HINGE_XYZ[0], py - sv.HINGE_XYZ[1]
    return sv.HINGE_XYZ[0] + c*rx - s*ry, sv.HINGE_XYZ[1] + s*rx + c*ry
def wrist(yaw_deg, phi_deg):                     # identical geometry to sim_grasp_viz.door_wrist_pose
    th = np.deg2rad(yaw_deg); c, s = np.cos(th), np.sin(th); ox, oy = -sv.DOOR_REACH, -sv.DOOR_HGAP
    wx0 = sv.HANDLE_XYZ[0] + (c*ox - s*oy); wy0 = sv.HANDLE_XYZ[1] + (s*ox + c*oy)
    wx, wy = rot(wx0, wy0, np.deg2rad(phi_deg)); ph = np.deg2rad(phi_deg)
    quat = sv.qmul([np.cos(ph/2), 0, 0, np.sin(ph/2)], sv.euler_quat(yaw_deg, 0))
    return (wx, wy, sv.HANDLE_XYZ[2]), [float(v) for v in quat]

print("\n[1+2] reach the handle from the relaxed start (sweep wrist yaw):")
best = None
for yaw in [0, -30, -45, -60, -90]:
    (wx, wy, wz), q = wrist(yaw, 0)
    r = pl.plan_pose(goal((wx, wy, wz), q), qstart, max_attempts=6)
    tag = "OK  " if ok(r) else "FAIL"
    print(f"    yaw={yaw:+4d}  wrist=({wx:+.3f},{wy:+.3f},{wz:+.3f}) -> {tag} {'' if ok(r) else getattr(r,'status',None)}")
    if ok(r) and best is None: best = (yaw, r)

if best is None:
    print("\n  >> nothing reached. See the 'Reading the output' notes at the top of this file.")
else:
    yaw, r = best
    print(f"\n[3] swing the door open from yaw={yaw} (seed-chained, negative = toward the robot):")
    seed = kin.get_active_js(get_joint_state_at_horizon_index(r.js_solution, -1).squeeze(0))
    N = max(2, int(round(sv.DOOR_OPEN_DEG / 10.0)))
    for i in range(1, N + 1):
        phi = -sv.DOOR_OPEN_DEG * (i / N)
        (wx, wy, wz), q = wrist(yaw, phi)
        r = pl.plan_pose(goal((wx, wy, wz), q), seed, max_attempts=4)
        print(f"    open {abs(phi):5.1f} deg  handle->({wx:+.3f},{wy:+.3f}) -> {'OK' if ok(r) else 'FAIL'}")
        if not ok(r): break
        seed = kin.get_active_js(get_joint_state_at_horizon_index(r.js_solution, -1).squeeze(0))
print("\ndone.")
