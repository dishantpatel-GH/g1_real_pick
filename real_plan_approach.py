#!/usr/bin/env python3
# =============================================================================
# real_plan_approach.py  —  PLAN the approach for the REAL G1 pick (cuMotion).
# Runs in cumotion_venv (cuRobo + torch + GPU). NO ROS, NO robot comms.
#
#   reads   : the detected glass pose (base/pelvis frame) from object_detection --emit
#   plans   : cuRobo plan_grasp (== Viser SideGrasp) -> approach(-y) then grasp, from the arm
#             START pose (arms=0). Exports approach+grasp so the hand ends AT the grasp pose
#             (the "perfect position"). NO lift, and FINGERS ARE NOT CLOSED (positioning only).
#   writes  : the right-arm joint trajectory to a JSON the docker executor streams.
#
# Two modes:
#   --probe (RECOMMENDED when a single pose fails): sweep a grid of approach poses
#           (wrist-behind x  ×  -y standoff  ×  height), report which are REACHABLE,
#           auto-pick the best reachable one, and plan to it.
#   default: plan the single pose set by --sg-dx/--standoff/--from-top.
#
# RUN (cumotion_venv):
#   /home/dishant/cumotion_venv/bin/python real_plan_approach.py --pose /tmp/glass_pose.json --probe
# =============================================================================
import argparse, json
import numpy as np

CONFIG = "/home/dishant/g1_ws/cumotion/config/g1_inspire_right.yml"
ARM_JOINTS = ["right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
              "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"]
SG_DYGAP = 0.01
SLOW = 2.0
# probe grid (right wrist, identity/side-grasp quat). x=cx-dx, y=cy-(R+gap)-standoff, z=topz-from_top.
# The inspire hand reaches FORWARD ~0.215 m from the wrist, so the WRIST must sit ~SG_DX behind the glass
# for the fingers to land ON it (Viser-confirmed). Probe AROUND the sim geometry and pick the pose closest
# to it. (Earlier bug: tiny dx + smallest-standoff selection left the hand short/beside the glass.)
SIM_DX, SIM_STANDOFF, SIM_FROMTOP = 0.215, 0.05, 0.035
DX_GRID       = [0.18, 0.215, 0.25]     # wrist behind glass center in x (LARGER dx -> fingers reach forward onto it)
STANDOFF_GRID = [0.03, 0.05, 0.08]      # -y pre-grasp clearance from the grasp pose
FROMTOP_GRID  = [0.035, 0.0, -0.03]     # wrist z = topz - from_top  (0.035 = sim, below top; negative = above)

# right-hand links whose collisions are DISABLED during plan_grasp, so the OPEN hand can sit at the grasp
# pose touching the glass (exactly like the Viser SideGrasp). plan_pose would reject that pose as a collision.
HAND_LINKS = ["right_base_link","right_palm_force_sensor",
              "right_thumb_1","right_thumb_2","right_thumb_3","right_thumb_4",
              "right_thumb_force_sensor_1","right_thumb_force_sensor_2","right_thumb_force_sensor_3","right_thumb_force_sensor_4",
              "right_index_1","right_index_2","right_index_force_sensor_1","right_index_force_sensor_2","right_index_force_sensor_3",
              "right_middle_1","right_middle_2","right_middle_force_sensor_1","right_middle_force_sensor_2","right_middle_force_sensor_3",
              "right_ring_1","right_ring_2","right_ring_force_sensor_1","right_ring_force_sensor_2","right_ring_force_sensor_3",
              "right_little_1","right_little_2","right_little_force_sensor_1","right_little_force_sensor_2","right_little_force_sensor_3"]


def _trim_last(P):
    Hn = P.shape[0]
    if Hn <= 1: return Hn
    d = np.linalg.norm(np.diff(P, axis=0), axis=1); mov = np.where(d > 1e-6)[0]
    return min(int(mov[-1] + 2) if len(mov) else Hn, Hn)


def _seg_points(seg, interp_dt):
    t = seg.squeeze(0); sjn = list(t.joint_names)
    P = t.position[0].detach().cpu().numpy()
    V = t.velocity[0].detach().cpu().numpy() if t.velocity is not None else None
    L = _trim_last(P); perm = [sjn.index(a) for a in ARM_JOINTS]; Pa = P[:L][:, perm]
    out = []
    for i in range(L):
        v = (V[:L][i, perm] * (1.0 / SLOW)).tolist() if V is not None else None
        if v is not None and i == L - 1: v = [0.0] * 7
        out.append((Pa[i].tolist(), i * interp_dt * SLOW, v))
    return out


def _qmul(a, b):  # [w,x,y,z]
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return [aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
            aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw]
def euler_quat(yaw_deg, pitch_deg):  # qz(yaw) * qy(pitch) — yaw about z, then pitch about y
    y = np.deg2rad(yaw_deg); p = np.deg2rad(pitch_deg)
    return [float(v) for v in _qmul([np.cos(y/2),0,0,np.sin(y/2)], [np.cos(p/2),0,np.sin(p/2),0])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default="/tmp/glass_pose.json")
    ap.add_argument("--out", default="/home/dishant/Projects/GR00T-WholeBodyControl/scripts/approach_traj.json",
                    help="output trajectory JSON (default = GR00T repo scripts/, bind-mounted into the container)")
    ap.add_argument("--probe", action="store_true", help="sweep the grid, report reachability, auto-pick the best")
    ap.add_argument("--no-table", action="store_true", help="diagnostic: drop the table collision box")
    # BestAngle grasp params (tuned in sim_grasp_viz; rotate the grasp by yaw about the cylinder axis)
    ap.add_argument("--standoff", type=float, default=0.05, help="pre-grasp clearance (tool-frame approach offset)")
    ap.add_argument("--sg-dx", type=float, default=0.215, help="reach: wrist behind the glass along the finger axis")
    ap.add_argument("--dygap", type=float, default=0.01, help="palm-to-glass gap")
    ap.add_argument("--from-top", type=float, default=0.055, help="wrist z below glass top (BIGGER value = LOWER hand)")
    ap.add_argument("--xoff", type=float, default=0.01, help="wrist X nudge (+ = grasp more ahead)")
    ap.add_argument("--yoff", type=float, default=0.02, help="wrist Y nudge (+ = grasp more right / -y)")
    ap.add_argument("--pitch-deg", type=float, default=0.0, help="tilt the hand about the lateral axis to level the pinky")
    ap.add_argument("--yaws", type=int, nargs="+", default=[0, -30, -60, -90],
                    help="BestAngle yaw set (deg): 0=side, -90=front. Sweeps MOST-FRONT first, takes first reachable")
    ap.add_argument("--yaw", type=int, default=None, help="force a single yaw (skip the BestAngle sweep)")
    ap.add_argument("--start-q", type=float, nargs=7, default=[0.0]*7)
    args = ap.parse_args()

    with open(args.pose) as f:
        P = json.load(f)
    cx, cy, topz = P["center"][0], P["center"][1], P["top_z"]
    R, H = P["radius_m"], P["height_m"]
    print(f"[plan] glass(base): center=({cx:+.3f},{cy:+.3f}) top_z={topz:+.3f} r={R:.3f} h={H:.3f}")

    import torch
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.scene import Scene, Cuboid, Cylinder, Mesh
    from curobo.types import GoalToolPose, JointState

    print("[plan] building cuRobo planner (~15s) ...", flush=True)
    cfg = MotionPlannerCfg.create(robot=CONFIG, scene_model=None, max_goalset=8, collision_cache={"obb": 8, "mesh": 4})
    pl = MotionPlanner(cfg)
    assert list(pl.joint_names) == ARM_JOINTS, pl.joint_names
    pl.warmup(enable_graph=True, num_warmup_iterations=5)
    interp_dt = pl.trajopt_solver.config.interpolation_dt

    table_top = (topz - H) - 0.005
    cz = topz - H / 2.0
    # table box: back edge fixed at x=0.25 (center 0.60) so it CLEARS the fixed-base robot. Centering it
    # on the glass put the back edge at x≈0.01 under the robot -> the arms=0 start was inside the slab ->
    # every plan_pose returned None. (Sim used the same fixed (0.6,*) table.)
    TABLE_CX = 0.60
    tri = Cylinder(name="object", radius=R, height=H, pose=[cx, cy, cz, 1, 0, 0, 0]).get_trimesh_mesh()
    cuboids = [] if args.no_table else [Cuboid(name="table", dims=[0.7, 0.9, 0.04],
                                               pose=[TABLE_CX, cy, table_top - 0.02, 1, 0, 0, 0])]
    scene = Scene(cuboid=cuboids,
                  mesh=[Mesh(name="object", vertices=tri.vertices.tolist(), faces=tri.faces.tolist(),
                             pose=[cx, cy, cz, 1, 0, 0, 0])])
    pl.update_world(scene)
    print(f"[plan] scene: table_top={table_top:+.3f} table_cx={'NONE' if args.no_table else TABLE_CX} "
          f"obstacles={pl.scene_collision_checker.get_obstacle_names()}")

    qs = JointState.from_position(torch.tensor([args.start_q], device="cuda", dtype=torch.float32), joint_names=ARM_JOINTS)

    # ---- BestAngle (== sim_grasp_viz): rotate the grasp by `yaw` about the cylinder axis (0=side, -90=front);
    #      the wrist offset rotates WITH the palm so the fingers always reach the glass, and the approach is in
    #      the TOOL frame so the hand comes in from the rotated direction. Sweep MOST-FRONT first, keep the first
    #      reachable. Hand-link collisions disabled so the open hand can sit at the glass. ----
    def plan_grasp_yaw(yaw):
        th = np.deg2rad(yaw); c, s = np.cos(th), np.sin(th); ox, oy = -args.sg_dx, -(R + args.dygap)
        wx = cx + (c*ox - s*oy) + args.xoff          # +xoff = grasp more ahead
        wy = cy + (s*ox + c*oy) - args.yoff          # +yoff = grasp more right (-y)
        wz = topz - args.from_top
        q = euler_quat(yaw, args.pitch_deg)
        goal = GoalToolPose(tool_frames=pl.tool_frames,
                            position=torch.tensor([[[[[wx, wy, wz]]]]], device="cuda", dtype=torch.float32),
                            quaternion=torch.tensor([[[[[q[0], q[1], q[2], q[3]]]]]], device="cuda", dtype=torch.float32))
        res = pl.plan_grasp(goal, qs, grasp_approach_axis="y", grasp_approach_offset=-args.standoff, grasp_approach_in_tool_frame=True,
                            grasp_lift_axis="z", grasp_lift_offset=0.15, grasp_lift_in_tool_frame=False,
                            plan_approach_to_grasp=True, plan_grasp_to_lift=True, disable_collision_links=HAND_LINKS)
        ok = res is not None and res.success is not None and bool(res.success.any())
        return ok, res, (wx, wy, wz)

    yaws = [args.yaw] if args.yaw is not None else sorted(args.yaws, key=lambda v: abs(v), reverse=True)  # most-front first
    print(f"[plan] BestAngle sweeping yaw {yaws}  (sg_dx={args.sg_dx} dygap={args.dygap} from_top={args.from_top} "
          f"xoff={args.xoff} yoff={args.yoff} pitch={args.pitch_deg} standoff={args.standoff}) ...")
    chosen = None
    for yaw in yaws:
        okk, res, w = plan_grasp_yaw(yaw)
        print(f"  [{'OK ' if okk else 'fail'}] yaw={yaw:+d}deg wrist=({w[0]:+.3f},{w[1]:+.3f},{w[2]:+.3f})"
              + ("" if okk else f"  {getattr(res, 'status', None)}"))
        if okk: chosen = (yaw, res, w); break
    if chosen is None:
        print("[plan] ❌ no reachable yaw — adjust --from-top/--xoff/--yoff/--sg-dx or re-emit a fresh pose.")
        return 1
    yaw, res, (ggx, ggy, ggz) = chosen
    print(f"[plan] ✅ BEST yaw={yaw:+d}deg  GRASP wrist=({ggx:+.3f},{ggy:+.3f},{ggz:+.3f})  glass=({cx:+.3f},{cy:+.3f})")

    ap_pts = _seg_points(res.approach_interpolated_trajectory, interp_dt)
    gr_pts = _seg_points(res.grasp_interpolated_trajectory, interp_dt)
    lf_pts = _seg_points(res.lift_interpolated_trajectory, interp_dt)   # grasp pose -> +z lifted
    t_off = ap_pts[-1][1] + interp_dt * SLOW
    pts = ap_pts + [(p, t + t_off, v) for (p, t, v) in gr_pts[1:]]
    print(f"[plan] ✅ plan_grasp OK: approach {len(ap_pts)} + grasp {len(gr_pts)-1} = {len(pts)} waypoints, T={pts[-1][1]:.2f}s")
    print(f"[plan]    + lift {len(lf_pts)} waypoints (grasp pose -> +0.15m up), T_lift={lf_pts[-1][1]:.2f}s")
    print(f"[plan] start_q ={['%+.3f'%v for v in args.start_q]}")
    print(f"[plan] end_q (GRASP pose) ={['%+.3f'%v for v in pts[-1][0]]}")
    print(f"[plan] lifted_q          ={['%+.3f'%v for v in lf_pts[-1][0]]}")
    out = {"joint_names": ARM_JOINTS, "interp_dt": float(interp_dt), "slow": SLOW,
           "target_wrist": [ggx, ggy, ggz], "yaw": yaw, "glass": P,
           "points": [{"positions": p, "time": t, "velocities": v} for (p, t, v) in pts],
           "lift_points": [{"positions": p, "time": t, "velocities": v} for (p, t, v) in lf_pts]}
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"[plan] wrote trajectory -> {args.out} ({len(pts)} pts). Hand ends at the GRASP position; fingers NOT closed. "
          f"Then run real_exec_approach.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
