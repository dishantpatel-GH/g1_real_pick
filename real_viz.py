#!/usr/bin/env python3
# =============================================================================
# real_viz.py — tune the REAL-glass grasp in the browser (cuMotion, no robot).
# Same viewer as phase0_viz.py, but the glass + table come from the DETECTED pose
# (object_detection --emit -> /tmp/glass_pose.json) instead of the hardcoded sim
# cylinder. Open http://localhost:8080, drag sliders, click SidePose / SideGrasp,
# and WATCH where the inspire hand lands relative to the glass.
#
#   SidePose  -> plan_pose to the side-grasp WRIST pose (cheap reach test)
#   SideGrasp -> plan_grasp (approach -y, grasp, lift +z); renders the hand mesh closing
#
# TELL ME: the SG_DX / SG_DYGAP / SG_FROMTOP / APPROACH_OFF values where the hand
# visibly reaches/grasps the glass — I'll bake them into real_plan_approach.py.
# (This uses the SAME planner/config as the real run, so reachable-here == reachable-there.
#  Caveat: the glass POSE is the unverified detection, so a good viz grasp still needs the
#  extrinsic sanity-checked before the real grasp — but this nails the geometry + reach.)
#
# RUN (cumotion_venv):  /home/dishant/cumotion_venv/bin/python real_viz.py [--pose /tmp/glass_pose.json]
# =============================================================================
import sys, time, threading, traceback, json, argparse

OUT  = "/home/dishant/g1_ws/cumotion/config/g1_inspire_right.yml"
PORT = 8080

ap = argparse.ArgumentParser()
ap.add_argument("--pose", default="/tmp/glass_pose.json")
args = ap.parse_args()
P = json.load(open(args.pose))
CX, CY = float(P["center"][0]), float(P["center"][1])
R, H = float(P["radius_m"]), float(P["height_m"])
TOP = float(P["top_z"])
CZ = TOP - H / 2.0                       # glass center (cylinder sits on the table)
TABLE_TOP = TOP - H
# table x fixed at 0.60 so its back edge (0.25) clears the fixed-base robot (centering on the
# glass buries the robot in the slab -> arms=0 start in collision -> all plans fail).
TABLE = [0.7, 0.9, 0.04], [0.60, CY, TABLE_TOP - 0.025, 1, 0, 0, 0]
print(f"[viz] REAL glass: center=({CX:+.3f},{CY:+.3f},{CZ:+.3f}) top={TOP:+.3f} r={R:.3f} h={H:.3f} | table_top={TABLE_TOP:+.3f}")

# side-grasp geometry (sim-validated start values; TUNE with the sliders while watching)
SG_DX      = 0.215
SG_DYGAP   = 0.01
SG_FROMTOP = 0.035
APPROACH_OFF = -0.05
LIFT_OFF     =  0.15

HAND_JOINTS = ["right_thumb_1_joint","right_thumb_2_joint","right_index_1_joint",
               "right_middle_1_joint","right_ring_1_joint","right_little_1_joint"]
HAND_OPEN   = dict(zip(HAND_JOINTS, [0,0,0,0,0,0]))
HAND_CLOSED = dict(zip(HAND_JOINTS, [1.0,0.5,1.4,1.4,1.4,1.4]))

HAND_LINKS = [
    "right_base_link","right_palm_force_sensor",
    "right_thumb_1","right_thumb_2","right_thumb_3","right_thumb_4",
    "right_thumb_force_sensor_1","right_thumb_force_sensor_2","right_thumb_force_sensor_3","right_thumb_force_sensor_4",
    "right_index_1","right_index_2","right_index_force_sensor_1","right_index_force_sensor_2","right_index_force_sensor_3",
    "right_middle_1","right_middle_2","right_middle_force_sensor_1","right_middle_force_sensor_2","right_middle_force_sensor_3",
    "right_ring_1","right_ring_2","right_ring_force_sensor_1","right_ring_force_sensor_2","right_ring_force_sensor_3",
    "right_little_1","right_little_2","right_little_force_sensor_1","right_little_force_sensor_2","right_little_force_sensor_3",
]

def main():
    import torch, numpy as np
    from curobo.viewer import ViserVisualizer
    from curobo.scene import Scene, Cuboid, Cylinder, Mesh
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import ContentPath, GoalToolPose, JointState, Pose

    def build_scene(cx, cy, cz):
        cyl = Cylinder(name="object", radius=R, height=H, pose=[cx, cy, cz, 1, 0, 0, 0])
        tri = cyl.get_trimesh_mesh()
        obj = Mesh(name="object", vertices=tri.vertices.tolist(), faces=tri.faces.tolist(), pose=[cx, cy, cz, 1, 0, 0, 0])
        return Scene(cuboid=[Cuboid(name="table", dims=TABLE[0], pose=TABLE[1])], mesh=[obj])

    print("building viewer + planner ...", flush=True)
    viz = ViserVisualizer(content_path=ContentPath(robot_config_absolute_path=OUT),
                          connect_ip="0.0.0.0", connect_port=PORT,
                          add_control_frames=True, visualize_robot_spheres=False)
    cfg = MotionPlannerCfg.create(robot=OUT, scene_model=None, max_goalset=8, collision_cache={"obb": 8, "mesh": 4})
    planner = MotionPlanner(cfg)
    planner.update_world(build_scene(CX, CY, CZ))
    print("obstacles:", planner.scene_collision_checker.get_obstacle_names())
    obstacle_frames = viz.add_scene(build_scene(CX, CY, CZ), add_control_frames=True)
    old_poses = {k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz) for k in obstacle_frames}

    current_state = planner.default_joint_state.clone().unsqueeze(0)
    print("warming up planner ...", flush=True)
    planner.warmup(enable_graph=True, num_warmup_iterations=5)
    viz.set_joint_state(planner.default_joint_state)

    VJ = list(viz.joint_names)
    ARM_SET = {"right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
               "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"}
    def render_full(arm_map, hand_map):
        d = {n: 0.0 for n in VJ}; d.update(arm_map); d.update(hand_map)
        viz.set_joint_state(JointState.from_position(
            torch.tensor([[d.get(n, 0.0) for n in VJ]], device="cuda", dtype=torch.float32), joint_names=list(VJ)))
    def arm_map_of(seg):
        t = seg.squeeze(0); jn = list(t.joint_names); last = t.position.squeeze(0)[-1].detach().cpu().numpy()
        return {a: float(last[jn.index(a)]) for a in jn if a in ARM_SET}

    is_moving = False
    server = viz._server
    sl_dx = server.gui.add_slider("SG_DX (wrist behind glass +x)", min=0.10, max=0.30, step=0.005, initial_value=SG_DX)
    sl_dy = server.gui.add_slider("SG_DYGAP (-y gap)",             min=-0.03, max=0.08, step=0.005, initial_value=SG_DYGAP)
    sl_dz = server.gui.add_slider("SG_FROMTOP (below top)",        min=-0.06, max=0.10, step=0.005, initial_value=SG_FROMTOP)
    sl_ap = server.gui.add_slider("APPROACH_OFF (-y standoff)",    min=-0.20, max=-0.01, step=0.01, initial_value=APPROACH_OFF)
    sl_grip = server.gui.add_slider("GRIP (finger closure frac)",  min=0.2, max=1.0, step=0.05, initial_value=0.55)
    sl_pitch = server.gui.add_slider("PITCH deg (level the pinky)", min=-40.0, max=40.0, step=1.0, initial_value=0.0)

    def cyl_center():
        f = obstacle_frames.get("object")
        if f is None: return (CX, CY, CZ)
        p = f.position; return (float(p[0]), float(p[1]), float(p[2]))
    def side_grasp_wrist():
        cx, cy, cz = cyl_center(); top = cz + H / 2.0
        return (cx - sl_dx.value, cy - (R + sl_dy.value), top - sl_dz.value)
    def goal_at(pos3):
        p = np.deg2rad(sl_pitch.value); qw, qy = float(np.cos(p/2)), float(np.sin(p/2))   # tilt about lateral axis
        return GoalToolPose(tool_frames=planner.tool_frames,
                            position=torch.tensor([[[[list(pos3)]]]], device="cuda", dtype=torch.float32),
                            quaternion=torch.tensor([[[[[qw, 0.0, qy, 0.0]]]]], device="cuda", dtype=torch.float32))
    def update_obstacles():
        for k in obstacle_frames:
            np_ = Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
            if np_ != old_poses[k]:
                planner.scene_collision_checker.update_obstacle_pose(k, np_); old_poses[k] = np_.clone()
    def trim_last(Pn):
        Hn = Pn.shape[0]
        if Hn <= 1: return Hn
        d = np.linalg.norm(np.diff(Pn, axis=0), axis=1); mov = np.where(d > 1e-5)[0]
        return min(int(mov[-1] + 2) if len(mov) else Hn, Hn)
    def _arm_row(jn, row): return {a: float(row[jn.index(a)]) for a in jn if a in ARM_SET}
    def eef(arm_map):
        order = list(planner.joint_names)
        q = torch.tensor([[arm_map.get(a, 0.0) for a in order]], device="cuda", dtype=torch.float32)
        st = planner.compute_kinematics(JointState.from_position(q, joint_names=order))
        return st.tool_poses.get_link_pose(planner.tool_frames[0]).position.detach().cpu().numpy().ravel()
    def execute(traj_in, hand=None):
        nonlocal current_state, is_moving
        traj = traj_in.squeeze(0); jn = list(traj.joint_names)
        Pn = traj.position[0].detach().cpu().numpy(); last = trim_last(Pn)
        idxs = np.unique(np.linspace(0, last - 1, min(last, 70)).astype(int))
        for i in idxs:
            if not is_moving: return
            if hand is None:
                viz.set_joint_state(JointState.from_position(
                    torch.tensor([Pn[i]], device="cuda", dtype=torch.float32), joint_names=jn).squeeze(0))
            else:
                render_full(_arm_row(jn, Pn[i]), hand)
            time.sleep(0.025)
        current_state = JointState.from_position(
            torch.tensor([Pn[last-1]], device="cuda", dtype=torch.float32), joint_names=jn)
    def grip_pose():
        g = sl_grip.value; return {k: HAND_CLOSED[k] * g for k in HAND_JOINTS}
    def close_fingers(arm_map):
        tgt = grip_pose()
        for f in np.linspace(0, 1, 12):
            if not is_moving: return
            render_full(arm_map, {k: (f*tgt[k] if 'thumb' in k else 0.0) for k in HAND_JOINTS}); time.sleep(0.03)
        for f in np.linspace(0, 1, 14):
            if not is_moving: return
            render_full(arm_map, {k: (tgt[k] if 'thumb' in k else f*tgt[k]) for k in HAND_JOINTS}); time.sleep(0.03)
    def run_async(fn):
        nonlocal is_moving
        if is_moving: return
        def work():
            nonlocal is_moving
            is_moving = True
            try: fn()
            except Exception: traceback.print_exc()
            is_moving = False
        threading.Thread(target=work, daemon=True).start()

    def on_sidepose(_):
        def f():
            update_obstacles()
            gx, gy, gz = side_grasp_wrist(); w = (gx, gy + sl_ap.value, gz)
            print(f"\nSidePose -> standoff {tuple(round(v,3) for v in w)}")
            active = planner.kinematics.get_active_js(current_state.clone())
            res = planner.plan_pose(goal_at(w), active, use_implicit_goal=True, max_attempts=4)
            if res is not None and res.success.any(): print("  SidePose: OK"); execute(res.get_interpolated_plan())
            else: print("  SidePose: FAILED status=", getattr(res, "status", None))
        run_async(f)
    def on_sidegrasp(_):
        def f():
            nonlocal current_state
            current_state = planner.default_joint_state.clone().unsqueeze(0); render_full({}, HAND_OPEN); update_obstacles()
            w = side_grasp_wrist()
            print(f"\nSideGrasp -> wrist {tuple(round(v,3) for v in w)} [DX={sl_dx.value:.3f} DYGAP={sl_dy.value:.3f} "
                  f"FROMTOP={sl_dz.value:.3f} APPROACH={sl_ap.value:.2f}]")
            active = planner.kinematics.get_active_js(current_state.clone())
            res = planner.plan_grasp(goal_at(w), active,
                grasp_approach_axis="y", grasp_approach_offset=sl_ap.value, grasp_approach_in_tool_frame=False,
                grasp_lift_axis="z", grasp_lift_offset=LIFT_OFF, grasp_lift_in_tool_frame=False,
                plan_approach_to_grasp=True, plan_grasp_to_lift=True, disable_collision_links=HAND_LINKS)
            ok = res is not None and res.success is not None and res.success.any()
            print(f"  SideGrasp: {'OK' if ok else 'FAILED'} status={getattr(res,'status',None)}")
            if ok:
                ap_, gr, lf = res.approach_interpolated_trajectory, res.grasp_interpolated_trajectory, res.lift_interpolated_trajectory
                try:
                    print("  [1/3] approach ..."); execute(ap_)
                    print("  [2/3] grasp ...");    execute(gr)
                    print("  [3/3] close fingers (thumb first, then curl) ..."); close_fingers(arm_map_of(gr))
                    print("  done — see if the fingers actually wrap the glass.")
                except Exception: traceback.print_exc()
        run_async(f)
    def on_move(_):
        def f():
            update_obstacles()
            active = planner.kinematics.get_active_js(current_state.clone())
            res = planner.plan_pose(GoalToolPose.from_poses(viz.get_control_frame_pose(), num_goalset=1),
                                    active, use_implicit_goal=True, max_attempts=3)
            if res is not None and res.success.any(): print("  Move: OK"); execute(res.get_interpolated_plan())
            else: print("  Move: FAILED status=", getattr(res, "status", None))
        run_async(f)

    server.gui.add_button("SidePose",  color="teal").on_click(on_sidepose)
    server.gui.add_button("SideGrasp", color="green").on_click(on_sidegrasp)
    server.gui.add_button("Move (target)", color="gray").on_click(on_move)
    print(f"side-grasp wrist for glass@({CX},{CY},{CZ}): {tuple(round(v,3) for v in side_grasp_wrist())}")
    print(f"\n================  OPEN  http://localhost:{PORT}  ================\n", flush=True)
    while True:
        time.sleep(1)

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\nbye")
    except Exception: print("\n❌ EXCEPTION:"); traceback.print_exc(); sys.exit(3)
