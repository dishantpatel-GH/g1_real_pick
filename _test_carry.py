#!/usr/bin/env python3
# TEMP experiment: is the CARRY motion feasible with the MICROWAVE COLLISION ENABLED?
#   bring-back (microwave-aware) -> put-in (STRAIGHT line, yaw free to dodge) -> pull-out
# Mirrors exactly what would go into real_plan_door.py. Prints OK/FAIL per phase. Makes NO file changes.
#   RUN: /home/dishant/cumotion_venv/bin/python _test_carry.py --place-dist 0.28 --place-angle 52 \
#        --bring-back-z 0.15 --bring-back-x 0.29 --glass-grip-z 0.06
import argparse, json, numpy as np, torch
import microwave_door_open_sim as S
import sim_grasp_viz as SV
from curobo.scene import Scene, Cuboid, Cylinder, Mesh
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState
from curobo._src.state.state_joint_trajectory_ops import get_joint_state_at_horizon_index

ap = argparse.ArgumentParser()
ap.add_argument("--glass", default="/tmp/glass_pose.json")
ap.add_argument("--glass-grip-z", type=float, default=0.06)
ap.add_argument("--glass-x-nudge", type=float, default=0.015)
ap.add_argument("--glass-y-nudge", type=float, default=0.01)
ap.add_argument("--table-gap", type=float, default=S.GRASP_TABLE_GAP)
ap.add_argument("--bring-back-x", type=float, default=None)
ap.add_argument("--bring-back-y", type=float, default=None)
ap.add_argument("--bring-back-z", type=float, default=None)
ap.add_argument("--place-angle", type=float, default=None)
ap.add_argument("--place-dist", type=float, default=None)
ap.add_argument("--straight-steps", type=int, default=12)
ap.add_argument("--insert-yaw-span", type=float, default=60.0)
ap.add_argument("--mw-right-pad", type=float, default=0.01)
args = ap.parse_args()

G = json.load(open(args.glass)); gc = G["center"]
gx, gy = float(gc[0]), float(gc[1]); gR = float(G["radius_m"]); gH = float(G["height_m"])
gzg = float(gc[2]) + S.Z_LIFT
DX, GAP, AP, PITCH = SV.SG_DX, SV.SG_DYGAP, SV.FG_APPROACH, 0.0
XOFF, YOFF = args.glass_x_nudge, args.glass_y_nudge
grip_z = args.glass_grip_z; FROMTOP = (gzg + gH/2.0) - grip_z
BBYAW = SV.BRING_BACK_YAW
BBX = args.bring_back_x if args.bring_back_x is not None else SV.BRING_BACK_XYZ[0]
BBY = args.bring_back_y if args.bring_back_y is not None else SV.BRING_BACK_XYZ[1]
BBZ = args.bring_back_z if args.bring_back_z is not None else SV.BRING_BACK_XYZ[2] + S.Z_LIFT
PA = args.place_angle if args.place_angle is not None else SV.PLACE_IN_ANGLE
PD = args.place_dist  if args.place_dist  is not None else SV.PLACE_IN_DIST
print(f"[cfg] glass=({gx:.3f},{gy:.3f},{gzg:.3f}) grip_z={grip_z:.3f} | staging=({BBX:.3f},{BBY:.3f},{BBZ:.3f}) | place {PA:.0f}deg x {PD:.2f}m | right_pad {args.mw_right_pad*100:.0f}cm | steps {args.straight_steps} yaw_span +/-{args.insert_yaw_span:.0f}")

pR = MotionPlanner(MotionPlannerCfg.create(robot=S.OUT, scene_model=None, max_goalset=8, collision_cache={"obb": 12, "mesh": 4}))
tri = Cylinder(name="object", radius=gR, height=gH, pose=[gx, gy, gzg, 1,0,0,0]).get_trimesh_mesh()
glass_mesh = Mesh(name="object", vertices=tri.vertices.tolist(), faces=tri.faces.tolist(), pose=[gx, gy, gzg, 1,0,0,0])
gt_top = gzg - gH/2.0 - args.table_gap; gt_near = 0.32
gt_dx = max(0.70, max(gx+0.25, S.MW_CX+S.MW_DIMS[0]/2+0.10) - gt_near)
GTABLE = Cuboid(name="table", dims=[gt_dx, 1.2, 0.6], pose=[gt_near+gt_dx/2.0, -0.05, gt_top-0.005-0.3, 1,0,0,0])
# carry-collision: ONLY the microwave RIGHT side (button panel block) -- the surface the glass grazes -- padded -y.
# (The full closed box over-constrains the right arm and breaks the grasp; the right wall is what matters here.)
hxw, hyw, hzw = S.MW_DIMS[0]/2, S.MW_DIMS[1]/2, S.MW_DIMS[2]/2; W = S.MW_WALL; RSp = S.MW_WALL + S.MW_BTN_W + args.mw_right_pad
carry_walls = [Cuboid(name="mw_right", dims=[S.MW_DIMS[0], RSp, S.MW_DIMS[2]], pose=[S.MW_CX, S.MW_CY-hyw+RSp/2, S.MW_CZ, 1,0,0,0])]
pR.update_world(Scene(cuboid=[GTABLE], mesh=[glass_mesh])); pR.warmup(enable_graph=True, num_warmup_iterations=5)  # grasp in the clean world

def ok(r): return r is not None and r.success is not None and bool(r.success.any())
def trim_last(P):
    Hn = P.shape[0]
    if Hn <= 1: return Hn
    d = np.linalg.norm(np.diff(P, axis=0), axis=1); mov = np.where(d > 1e-5)[0]
    return min(int(mov[-1]+2) if len(mov) else Hn, Hn)
def relaxed_qstart_R():
    full = pR.default_joint_state.clone(); jn = list(full.joint_names); pos = full.position.clone().reshape(-1)
    for n, v in S.RELAXED_ARM.items():
        if n in jn: pos[jn.index(n)] = float(v)
    return pR.kinematics.get_active_js(JointState.from_position(pos.unsqueeze(0), joint_names=jn))
def goal_R(p, q):
    return GoalToolPose(tool_frames=pR.tool_frames,
        position=torch.tensor([[[[list(p)]]]], device="cuda", dtype=torch.float32),
        quaternion=torch.tensor([[[[list(q)]]]], device="cuda", dtype=torch.float32))
def rwrist(cx, cy, cz, yawd):
    cc, ss = np.cos(np.deg2rad(yawd)), np.sin(np.deg2rad(yawd)); ox, oy = -DX, -(gR+GAP)
    return (cx + (cc*ox - ss*oy) + XOFF, cy + (ss*ox + cc*oy) - YOFF, cz + (gH/2.0 - FROMTOP))
def yaw_candidates(y0):
    out_y = [y0]; d = 10.0
    while d <= args.insert_yaw_span + 1e-6:
        out_y += [y0-d, y0+d]; d += 10.0
    return out_y

# ---- grasp (to get a realistic seed at the glass) ----
def grasp_wrist(yaw):
    th = np.deg2rad(yaw); c, s = np.cos(th), np.sin(th); ox, oy = -DX, -(gR+GAP)
    return (gx + (c*ox - s*oy) + XOFF, gy + (s*ox + c*oy) - YOFF, grip_z)
res = None; gyaw = None
for yaw in [-90, -60, -30, 0]:
    try:
        r = pR.plan_grasp(goal_R(grasp_wrist(yaw), S.euler_quat(yaw, PITCH)), relaxed_qstart_R(),
                          grasp_approach_axis="y", grasp_approach_offset=AP, grasp_approach_in_tool_frame=True,
                          grasp_lift_axis="z", grasp_lift_offset=0.15, grasp_lift_in_tool_frame=False,
                          plan_approach_to_grasp=True, plan_grasp_to_lift=True, disable_collision_links=SV.HAND_LINKS)
    except Exception as e:
        print(f"[grasp] yaw={yaw} cuRobo error ({type(e).__name__}); skip"); continue
    if ok(r): res = r; gyaw = yaw; break
print(f"[grasp] -> {'OK yaw='+str(gyaw) if res else 'FAIL'}")
if res is None: raise SystemExit("grasp failed; cannot test carry")
gseed = pR.kinematics.get_active_js(get_joint_state_at_horizon_index(res.grasp_trajectory, -1).squeeze(0))

# ---- CARRY world: table + microwave walls (glass removed -> now in hand) ----
pR.update_world(Scene(cuboid=[GTABLE] + carry_walls))

# ---- bring-back (microwave-aware) ----
rp = pR.plan_pose(goal_R(rwrist(BBX, BBY, BBZ, BBYAW), S.euler_quat(BBYAW, 0)), gseed, max_attempts=10)
print(f"[bring-back] microwave-aware -> {'OK' if ok(rp) else 'FAIL'}")
if not ok(rp): raise SystemExit("bring-back failed with collision on (tune bring-back-x/y/z or mw-right-pad)")
sBB = pR.kinematics.get_active_js(get_joint_state_at_horizon_index(rp.js_solution, -1).squeeze(0))

# ---- put-in: STRAIGHT line, microwave-aware, yaw free per step ----
ca, sa = np.cos(np.deg2rad(PA)), np.sin(np.deg2rad(PA)); ig = (BBX + PD*ca, BBY + PD*sa, BBZ)
M = max(2, args.straight_steps); cseed = sBB; cur_yaw = BBYAW; ins_ok = True; yaws = []
for k in range(1, M+1):
    fk = k/M; cx = BBX + fk*(ig[0]-BBX); cy = BBY + fk*(ig[1]-BBY); cz = BBZ + fk*(ig[2]-BBZ)
    step = None
    for yk in yaw_candidates(cur_yaw):
        rk = pR.plan_pose(goal_R(rwrist(cx, cy, cz, yk), S.euler_quat(yk, 0)), cseed, max_attempts=4)
        if ok(rk): step = (rk, yk); break
    if step is None: print(f"[put-in] STUCK at step {k}/{M} ({fk*PD:.2f}m in, target glass=({cx:.3f},{cy:.3f},{cz:.3f}))"); ins_ok = False; break
    rk, cur_yaw = step; yaws.append(int(round(cur_yaw)))
    cseed = pR.kinematics.get_active_js(get_joint_state_at_horizon_index(rk.js_solution, -1).squeeze(0))
print(f"[put-in] STRAIGHT {PA:.0f}deg x {PD:.2f}m -> ({ig[0]:.3f},{ig[1]:.3f},{ig[2]:.3f}) -> {'OK' if ins_ok else 'FAIL'}  yaws={yaws}")

# ---- pull-out: cspace staging -> relaxed (microwave-aware) ----
rr = pR.plan_cspace(relaxed_qstart_R(), sBB, max_attempts=10)
print(f"[pull-out] staging->relaxed (cspace, microwave-aware) -> {'OK' if ok(rr) else 'FAIL'}")
print("\n[RESULT] carry feasible with collision ON:" , bool(ok(rp) and ins_ok and ok(rr)))
