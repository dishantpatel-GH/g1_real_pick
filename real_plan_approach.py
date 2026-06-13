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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default="/tmp/glass_pose.json")
    ap.add_argument("--out", default="/home/dishant/Projects/GR00T-WholeBodyControl/scripts/approach_traj.json",
                    help="output trajectory JSON (default = GR00T repo scripts/, bind-mounted into the container)")
    ap.add_argument("--probe", action="store_true", help="sweep the grid, report reachability, auto-pick the best")
    ap.add_argument("--no-table", action="store_true", help="diagnostic: drop the table collision box")
    ap.add_argument("--standoff", type=float, default=0.05, help="(non-probe) -y pre-grasp clearance from grasp pose")
    ap.add_argument("--sg-dx", type=float, default=0.220, help="wrist x BEHIND glass (bigger=more behind; hand reaches fwd ~0.215)")
    ap.add_argument("--dygap", type=float, default=0.01, help="palm-to-glass -y gap (bigger=hand more to the RIGHT)")
    ap.add_argument("--from-top", type=float, default=0.035, help="wrist z below glass top (BIGGER value = LOWER hand)")
    ap.add_argument("--pitch-deg", type=float, default=0.0,
                    help="tilt the grasp hand about the lateral axis to level the pinky (+ = pitch fingers up)")
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

    def wrist_of(dx, so, ft):
        return cx - dx, cy - (R + args.dygap) - so, topz - ft

    def try_plan(x, y, z, attempts):
        goal = GoalToolPose(tool_frames=pl.tool_frames,
                            position=torch.tensor([[[[[x, y, z]]]]], device="cuda", dtype=torch.float32),
                            quaternion=torch.tensor([[[[[1., 0, 0, 0]]]]], device="cuda", dtype=torch.float32))
        res = pl.plan_pose(goal, qs, max_attempts=attempts, use_implicit_goal=True)
        ok = res is not None and res.success is not None and bool(res.success.any())
        return ok, res

    # ---- choose the geometry (sg_dx, standoff=-y approach offset, from_top) ----
    if args.probe:
        print("[probe] sweeping pre-grasp poses (PASS = reachable) ...")
        reach = []
        for dx in DX_GRID:
            for so in STANDOFF_GRID:
                for ft in FROMTOP_GRID:
                    x, y, z = wrist_of(dx, so, ft)
                    ok, _ = try_plan(x, y, z, attempts=4)
                    print(f"  [{'PASS' if ok else 'fail'}] dx={dx:.2f} standoff={so:.2f} from_top={ft:+.2f} "
                          f"-> pre-grasp wrist=({x:+.3f},{y:+.3f},{z:+.3f})")
                    if ok:
                        reach.append((dx, so, ft))
        if not reach:
            print("[probe] ❌ NOTHING reachable near the glass. Glass too low/close for the right arm OR the "
                  "camera->base extrinsic has a z/x error. Verify the base pose / extrinsic next.")
            return 1
        dx, so, ft = min(reach, key=lambda r: (abs(r[0]-SIM_DX), abs(r[1]-SIM_STANDOFF), abs(r[2]-SIM_FROMTOP)))
        print(f"[probe] {len(reach)} reachable. PICK (closest to sim grasp): dx={dx:.3f} standoff={so:.2f} from_top={ft:+.3f}")
    else:
        dx, so, ft = args.sg_dx, args.standoff, args.from_top

    # GRASP pose = wrist sg_dx behind the glass, at its -y surface+gap, from_top below top (== Viser SideGrasp pose)
    ggx, ggy, ggz = cx - dx, cy - (R + args.dygap), topz - ft
    fwd = cx - ggx
    print(f"[plan] GRASP wrist=({ggx:+.3f},{ggy:+.3f},{ggz:+.3f})  [sg_dx={dx} dygap={args.dygap} from_top={ft}; "
          f"approach standoff={so:.2f} on -y]  -> fingers reach ~{fwd:.2f}m fwd to the glass. glass=({cx:+.3f},{cy:+.3f})")

    # ---- plan_grasp: approach(-y) -> grasp -> lift, hand-link collisions disabled so the OPEN hand can sit at
    #      the glass (plan_pose would reject it). We export approach+grasp (NOT lift, NOT fingers) = SideGrasp's path.
    p = np.deg2rad(args.pitch_deg); qw, qy = float(np.cos(p/2)), float(np.sin(p/2))   # tilt about lateral axis
    print(f"[plan] grasp pitch={args.pitch_deg:+.1f}deg -> quat=[{qw:.3f},0,{qy:.3f},0]")
    goal = GoalToolPose(tool_frames=pl.tool_frames,
                        position=torch.tensor([[[[[ggx, ggy, ggz]]]]], device="cuda", dtype=torch.float32),
                        quaternion=torch.tensor([[[[[qw, 0., qy, 0.]]]]], device="cuda", dtype=torch.float32))
    res = pl.plan_grasp(goal, qs, grasp_approach_axis="y", grasp_approach_offset=-so, grasp_approach_in_tool_frame=False,
                        grasp_lift_axis="z", grasp_lift_offset=0.15, grasp_lift_in_tool_frame=False,
                        plan_approach_to_grasp=True, plan_grasp_to_lift=True, disable_collision_links=HAND_LINKS)
    ok = res is not None and res.success is not None and bool(res.success.any())
    if not ok:
        print("[plan] ❌ plan_grasp FAILED status=", getattr(res, "status", None),
              "\n   -> adjust --sg-dx/--dygap/--from-top/--standoff or re-emit a fresh pose.")
        return 1

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
           "target_wrist": [ggx, ggy, ggz], "glass": P,
           "points": [{"positions": p, "time": t, "velocities": v} for (p, t, v) in pts],
           "lift_points": [{"positions": p, "time": t, "velocities": v} for (p, t, v) in lf_pts]}
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"[plan] wrote trajectory -> {args.out} ({len(pts)} pts). Hand ends at the GRASP position; fingers NOT closed. "
          f"Then run real_exec_approach.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
