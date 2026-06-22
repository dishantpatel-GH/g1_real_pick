#!/usr/bin/env python3
# =============================================================================
# diagnostic_collision.py -- pin down WHY the door reach reports "Start or End state in collision".
#
# The message comes from cuRobo's GRAPH planner and only fires AFTER IK finds a goal, so it is
# genuinely ambiguous about WHICH endpoint and WHAT it hits. This script answers both:
#   (A) plan_pose A/B matrix: {relaxed vs default start} x {table present vs moved away} -> isolates
#       whether the RELAXED START or the HANDLE GOAL is the blocker, and whether the TABLE matters.
#   (B) validate() the start configs directly (cuRobo's own collision check) with table present vs
#       removed -> SELF-collision (fails both) vs WORLD/table collision (fails only with table).
#   (C) sweep right-arm relaxed poses and list the collision-free ones -> a start pose that works.
#
# RUN (cumotion_venv):  /home/dishant/cumotion_venv/bin/python diagnostic_collision.py
# =============================================================================
import numpy as np, torch
import sim_grasp_viz as sv
from curobo.scene import Scene, Cuboid
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState

print("building planner (~15 s) ...", flush=True)
cfg = MotionPlannerCfg.create(robot=sv.OUT, scene_model=None, max_goalset=8, collision_cache={"obb": 8, "mesh": 4})
pl = MotionPlanner(cfg); kin = pl.kinematics; chk = pl.scene_collision_checker

TABLE = Scene(cuboid=[Cuboid(name="table", dims=sv.TABLE[0], pose=sv.TABLE[1])])
FAR   = Scene(cuboid=[Cuboid(name="table", dims=sv.TABLE[0], pose=[10.0, 10.0, -10.0, 1, 0, 0, 0])])  # table out of the way
pl.update_world(TABLE); pl.warmup(enable_graph=True, num_warmup_iterations=5)
jn = list(pl.default_joint_state.joint_names)
print("active dof:", len(pl.joint_names), "->", pl.joint_names)

def rarm_js(sp=0.0, sr=0.0, sy=0.0, eb=sv.ELBOW_BEND):
    """Active (right-arm) JointState for a chosen shoulder_pitch/roll/yaw + elbow; wrists 0."""
    pos = pl.default_joint_state.position.clone().reshape(-1)
    for n, v in {"right_shoulder_pitch_joint": sp, "right_shoulder_roll_joint": sr,
                 "right_shoulder_yaw_joint": sy, "right_elbow_joint": eb}.items():
        if n in jn: pos[jn.index(n)] = v
    return kin.get_active_js(JointState.from_position(pos.unsqueeze(0), joint_names=jn))

def arms0_js():
    return kin.get_active_js(pl.default_joint_state.clone().unsqueeze(0))

def is_free(qjs):
    """True if cuRobo considers this config collision-free (self + world + joint bounds)."""
    try:
        return bool(chk.validate(qjs.position.reshape(1, 1, -1)).all())
    except Exception as e:
        return f"validate-error: {e}"

def goal(p, q):
    return GoalToolPose(tool_frames=pl.tool_frames,
                        position=torch.tensor([[[[list(p)]]]], device="cuda", dtype=torch.float32),
                        quaternion=torch.tensor([[[[list(q)]]]], device="cuda", dtype=torch.float32))
def ok(r): return r is not None and r.success is not None and bool(r.success.any())
def wrist(yaw, phi=0.0):
    th = np.deg2rad(yaw); c, s = np.cos(th), np.sin(th); ox, oy = -sv.DOOR_REACH, -sv.DOOR_HGAP
    wx = sv.HANDLE_XYZ[0] + (c*ox - s*oy); wy = sv.HANDLE_XYZ[1] + (s*ox + c*oy); ph = np.deg2rad(phi)
    return (wx, wy, sv.HANDLE_XYZ[2]), [float(v) for v in sv.qmul([np.cos(ph/2), 0, 0, np.sin(ph/2)], sv.euler_quat(yaw, 0))]

relaxed = rarm_js(); arms0 = arms0_js()

print("\n========== (B) validate() the START configs directly ==========")
for scene, sname in ((TABLE, "table PRESENT"), (FAR, "table MOVED AWAY")):
    pl.update_world(scene)
    print(f"  [{sname}]  relaxed(elbow {sv.ELBOW_BEND:.3f})-> free={is_free(relaxed)}   arms-0 -> free={is_free(arms0)}")
print("  Reading: relaxed FALSE in BOTH scenes  => SELF-collision (folded arm hits the body).")
print("           relaxed FALSE only WITH table => the table is still too close (move it further).")

print("\n========== (A) plan_pose A/B: reach the handle ==========")
for yaw in [0, -45]:
    (wx, wy, wz), q = wrist(yaw)
    print(f"  -- yaw={yaw}, wrist=({wx:.3f},{wy:.3f},{wz:.3f}) --")
    for scene, sname in ((TABLE, "table"), (FAR, "no-table")):
        pl.update_world(scene)
        for start, stname in ((relaxed, "relaxed"), (arms0, "arms-0")):
            r = pl.plan_pose(goal((wx, wy, wz), q), start, max_attempts=4)
            print(f"       start={stname:8s} {sname:9s} -> {'OK' if ok(r) else 'FAIL ' + str(getattr(r,'status',None))}")

print("\n========== (C) collision-free relaxed right-arm poses (table present) ==========")
pl.update_world(TABLE)
found = []
for sp in [0.0, 0.15, 0.3]:
    for sr in [0.0, -0.15, -0.3, 0.15, 0.3]:
        for eb in [0.6, 0.9, sv.ELBOW_BEND]:
            if is_free(rarm_js(sp=sp, sr=sr, eb=eb)) is True:
                found.append((sp, sr, eb))
if found:
    print("  collision-free candidates (shoulder_pitch sp, shoulder_roll sr, elbow eb):")
    for sp, sr, eb in found:
        print(f"     sp={sp:+.2f}  sr={sr:+.2f}  eb={eb:.3f}")
    print("  -> set RELAXED_ARM in sim_grasp_viz.py to one of these (note: sr is right_shoulder_roll_joint).")
else:
    print("  none collision-free in the swept set -- widen the sweep or check the table position.")
print("\ndone.")
