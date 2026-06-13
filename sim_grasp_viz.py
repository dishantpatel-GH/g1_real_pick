#!/usr/bin/env python3
# =============================================================================
# sim_grasp_viz.py — compare TWO right-hand grasp methods in the browser (cuRobo, no robot):
#   SIDE  (current): approach from the robot's RIGHT (-y), palm faces +y, identity wrist quat.
#   FRONT (new):     approach head-on from the FRONT (-x), palm faces +x. Tunable yaw/pitch.
# Plus a COVERAGE sweep that tests both methods over a grid of glass positions on the table and
# reports which method reaches more (right-side placements). Glass = r 3.6cm, h 10.5cm (fixed).
#
# Open http://localhost:8080. Drag the cylinder, tune the FG_* sliders, click SideGrasp / FrontGrasp
# (renders the inspire hand closing) and Coverage (prints a side-vs-front map to the console).
#
# RUN (cumotion_venv): /home/dishant/cumotion_venv/bin/python sim_grasp_viz.py
# =============================================================================
import sys, time, threading, traceback
import numpy as np

OUT  = "/home/dishant/g1_ws/cumotion/config/g1_inspire_right.yml"
PORT = 8080
R, H = 0.036, 0.105                                   # the glass under test (3.6 cm radius, 10.5 cm tall)
CX, CY = 0.40, -0.24                                  # starting position (robot's right)
CZ = -0.006 + H / 2.0                                 # center; table top ~ -0.006 (glass sits on it)
TABLE = [0.7, 0.9, 0.04], [0.60, -0.10, (CZ - H/2) - 0.025, 1, 0, 0, 0]   # back edge 0.25 clears the robot

# SIDE grasp (current, frozen for reference)
SG_DX, SG_DYGAP, SG_FROMTOP, SG_APPROACH = 0.215, 0.01, 0.035, -0.05
# FRONT grasp (new) defaults — the wrist position is DERIVED by rotating the side grasp by FG_YAW about the
# cylinder axis; FG_DX/FG_DY are small fine-tune nudges on top. FG_YAW=-90 => palm +x, approach from the front.
FG_DX, FG_DY, FG_FROMTOP, FG_APPROACH = 0.0, 0.0, 0.035, -0.05
FG_YAW0, FG_PITCH0 = -90.0, 0.0                       # palm +y -> +x is yaw -90 about z

HAND_JOINTS = ["right_thumb_1_joint","right_thumb_2_joint","right_index_1_joint",
               "right_middle_1_joint","right_ring_1_joint","right_little_1_joint"]
HAND_OPEN   = dict(zip(HAND_JOINTS, [0,0,0,0,0,0]))
HAND_CLOSED = dict(zip(HAND_JOINTS, [1.0,0.5,1.4,1.4,1.4,1.4]))
HAND_LINKS = ["right_base_link","right_palm_force_sensor",
    "right_thumb_1","right_thumb_2","right_thumb_3","right_thumb_4",
    "right_thumb_force_sensor_1","right_thumb_force_sensor_2","right_thumb_force_sensor_3","right_thumb_force_sensor_4",
    "right_index_1","right_index_2","right_index_force_sensor_1","right_index_force_sensor_2","right_index_force_sensor_3",
    "right_middle_1","right_middle_2","right_middle_force_sensor_1","right_middle_force_sensor_2","right_middle_force_sensor_3",
    "right_ring_1","right_ring_2","right_ring_force_sensor_1","right_ring_force_sensor_2","right_ring_force_sensor_3",
    "right_little_1","right_little_2","right_little_force_sensor_1","right_little_force_sensor_2","right_little_force_sensor_3"]


def qmul(a, b):  # [w,x,y,z]
    aw,ax,ay,az = a; bw,bx,by,bz = b
    return [aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
            aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw]
def euler_quat(yaw_deg, pitch_deg):  # qz(yaw) * qy(pitch), about pelvis z then y
    y = np.deg2rad(yaw_deg); p = np.deg2rad(pitch_deg)
    qz = [np.cos(y/2), 0, 0, np.sin(y/2)]; qy = [np.cos(p/2), 0, np.sin(p/2), 0]
    return [float(v) for v in qmul(qz, qy)]


def main():
    import torch
    from curobo.viewer import ViserVisualizer
    from curobo.scene import Scene, Cuboid, Cylinder, Mesh
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import ContentPath, GoalToolPose, JointState, Pose

    def build_scene(cx, cy, cz):
        tri = Cylinder(name="object", radius=R, height=H, pose=[cx, cy, cz, 1, 0, 0, 0]).get_trimesh_mesh()
        return Scene(cuboid=[Cuboid(name="table", dims=TABLE[0], pose=TABLE[1])],
                     mesh=[Mesh(name="object", vertices=tri.vertices.tolist(), faces=tri.faces.tolist(),
                                pose=[cx, cy, cz, 1, 0, 0, 0])])

    print("building viewer + planner ...", flush=True)
    viz = ViserVisualizer(content_path=ContentPath(robot_config_absolute_path=OUT),
                          connect_ip="0.0.0.0", connect_port=PORT,
                          add_control_frames=True, visualize_robot_spheres=False)
    cfg = MotionPlannerCfg.create(robot=OUT, scene_model=None, max_goalset=8, collision_cache={"obb": 8, "mesh": 4})
    planner = MotionPlanner(cfg)
    planner.update_world(build_scene(CX, CY, CZ))
    obstacle_frames = viz.add_scene(build_scene(CX, CY, CZ), add_control_frames=True)
    old_poses = {k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz) for k in obstacle_frames}
    print("warming up planner ...", flush=True)
    planner.warmup(enable_graph=True, num_warmup_iterations=5)
    viz.set_joint_state(planner.default_joint_state)

    VJ = list(viz.joint_names)
    ARM_SET = set(["right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
                   "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"])
    def render_full(arm_map, hand_map):
        d = {n: 0.0 for n in VJ}; d.update(arm_map); d.update(hand_map)
        viz.set_joint_state(JointState.from_position(
            torch.tensor([[d.get(n, 0.0) for n in VJ]], device="cuda", dtype=torch.float32), joint_names=list(VJ)))
    def arm_map_of(seg):
        t = seg.squeeze(0); jn = list(t.joint_names); last = t.position.squeeze(0)[-1].detach().cpu().numpy()
        return {a: float(last[jn.index(a)]) for a in jn if a in ARM_SET}

    is_moving = False
    server = viz._server
    sl_fdx   = server.gui.add_slider("FG_DX fine-tune (x nudge)",      min=-0.10, max=0.10, step=0.005, initial_value=FG_DX)
    sl_fdy   = server.gui.add_slider("FG_DY fine-tune (y nudge)",      min=-0.10, max=0.10, step=0.005, initial_value=FG_DY)
    sl_fdz   = server.gui.add_slider("FG_FROMTOP (below top)",         min=-0.10, max=0.15, step=0.005, initial_value=FG_FROMTOP)
    sl_fap   = server.gui.add_slider("FG_APPROACH (-x standoff)",      min=-0.20, max=-0.01, step=0.01, initial_value=FG_APPROACH)
    sl_yaw   = server.gui.add_slider("FG_YAW deg (palm dir)",          min=-180.0, max=180.0, step=5.0, initial_value=FG_YAW0)
    sl_pitch = server.gui.add_slider("FG_PITCH deg",                   min=-90.0, max=90.0, step=2.0, initial_value=FG_PITCH0)
    sl_grip  = server.gui.add_slider("GRIP (finger closure frac)",     min=0.2, max=1.0, step=0.05, initial_value=0.55)

    def cyl_center():
        f = obstacle_frames.get("object")
        if f is None: return (CX, CY, CZ)
        p = f.position; return (float(p[0]), float(p[1]), float(p[2]))
    def update_obstacles():
        for k in obstacle_frames:
            np_ = Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
            if np_ != old_poses[k]:
                planner.scene_collision_checker.update_obstacle_pose(k, np_); old_poses[k] = np_.clone()
    def goal(pos3, quat):
        return GoalToolPose(tool_frames=planner.tool_frames,
                            position=torch.tensor([[[[list(pos3)]]]], device="cuda", dtype=torch.float32),
                            quaternion=torch.tensor([[[[list(quat)]]]], device="cuda", dtype=torch.float32))
    def trim_last(P):
        Hn = P.shape[0]
        if Hn <= 1: return Hn
        d = np.linalg.norm(np.diff(P, axis=0), axis=1); mov = np.where(d > 1e-5)[0]
        return min(int(mov[-1] + 2) if len(mov) else Hn, Hn)

    # --- the two grasp methods (return the plan_grasp result; None on failure) ---
    def plan_side(cx, cy, cz, qstart):
        top = cz + H/2.0
        w = (cx - SG_DX, cy - (R + SG_DYGAP), top - SG_FROMTOP)
        return planner.plan_grasp(goal(w, [1.,0,0,0]), qstart,
            grasp_approach_axis="y", grasp_approach_offset=SG_APPROACH, grasp_approach_in_tool_frame=False,
            grasp_lift_axis="z", grasp_lift_offset=0.15, grasp_lift_in_tool_frame=False,
            plan_approach_to_grasp=True, plan_grasp_to_lift=True, disable_collision_links=HAND_LINKS), w
    def plan_front(cx, cy, cz, qstart):
        # FRONT = the side grasp ROTATED by FG_YAW about the cylinder's vertical axis, so the wrist offset
        # rotates WITH the palm and the fingers still reach the glass. Approach is in the TOOL frame, so the
        # hand comes in along its rotated approach direction (FG_YAW=-90 -> palm +x, approach from -x = front).
        th = np.deg2rad(sl_yaw.value); c, s = np.cos(th), np.sin(th)
        ox, oy = -SG_DX, -(R + SG_DYGAP)                       # side-grasp wrist offset (glass frame, xy)
        wx = cx + (c*ox - s*oy) + sl_fdx.value                 # rotated offset + small x fine-tune
        wy = cy + (s*ox + c*oy) + sl_fdy.value                 # rotated offset + small y fine-tune
        wz = (cz + H/2.0) - sl_fdz.value
        q = euler_quat(sl_yaw.value, sl_pitch.value)
        return planner.plan_grasp(goal((wx, wy, wz), q), qstart,
            grasp_approach_axis="y", grasp_approach_offset=sl_fap.value, grasp_approach_in_tool_frame=True,
            grasp_lift_axis="z", grasp_lift_offset=0.15, grasp_lift_in_tool_frame=False,
            plan_approach_to_grasp=True, plan_grasp_to_lift=True, disable_collision_links=HAND_LINKS), (wx, wy, wz)
    def ok(res): return res is not None and res.success is not None and bool(res.success.any())

    def execute(seg, hand=None):
        nonlocal is_moving
        t = seg.squeeze(0); jn = list(t.joint_names); P = t.position[0].detach().cpu().numpy(); last = trim_last(P)
        for i in np.unique(np.linspace(0, last-1, min(last, 60)).astype(int)):
            if not is_moving: return
            if hand is None:
                viz.set_joint_state(JointState.from_position(
                    torch.tensor([P[i]], device="cuda", dtype=torch.float32), joint_names=jn).squeeze(0))
            else:
                render_full({a: float(P[i][jn.index(a)]) for a in jn if a in ARM_SET}, hand)
            time.sleep(0.025)
    def grip(): g = sl_grip.value; return {k: HAND_CLOSED[k]*g for k in HAND_JOINTS}
    def run_async(fn):
        nonlocal is_moving
        if is_moving: return
        def work():
            nonlocal is_moving; is_moving = True
            try: fn()
            except Exception: traceback.print_exc()
            is_moving = False
        threading.Thread(target=work, daemon=True).start()

    def demo(method_name, planfn):
        def f():
            nonlocal is_moving
            update_obstacles(); render_full({}, HAND_OPEN)
            cx, cy, cz = cyl_center()
            qstart = planner.kinematics.get_active_js(planner.default_joint_state.clone().unsqueeze(0))
            res, w = planfn(cx, cy, cz, qstart)
            print(f"\n{method_name} @glass({cx:.3f},{cy:.3f}) wrist=({w[0]:.3f},{w[1]:.3f},{w[2]:.3f}) "
                  f"-> {'OK' if ok(res) else 'FAILED'} {getattr(res,'status',None)}")
            if not ok(res): return
            execute(res.approach_interpolated_trajectory)
            execute(res.grasp_interpolated_trajectory)
            arm = arm_map_of(res.grasp_interpolated_trajectory)
            tgt = grip()
            for fr in np.linspace(0,1,10): render_full(arm, {k:(fr*tgt[k] if 'thumb' in k else 0.0) for k in HAND_JOINTS}); time.sleep(0.03)
            for fr in np.linspace(0,1,12): render_full(arm, {k:(tgt[k] if 'thumb' in k else fr*tgt[k]) for k in HAND_JOINTS}); time.sleep(0.03)
            print(f"  {method_name}: see if the fingers wrap the glass.")
        run_async(f)

    def coverage():
        def f():
            qstart = planner.kinematics.get_active_js(planner.default_joint_state.clone().unsqueeze(0))
            XS = np.round(np.arange(0.30, 0.49, 0.03), 3)
            YS = np.round(np.arange(-0.32, -0.02, 0.03), 3)
            print("\n================ COVERAGE SWEEP (right-side placements) ================")
            print(f"glass r={R} h={H}; rows=y (-0.02..-0.32), cols=x (0.30..0.48). "
                  f"B=both S=side-only F=front-only .=neither")
            ns = nf = nb = 0; grid = []
            for y in YS:
                row = []
                for x in XS:
                    cz = (-0.006) + H/2.0
                    planner.update_world(build_scene(x, y, cz))
                    s = ok(plan_side(x, y, cz, qstart)[0])
                    fr = ok(plan_front(x, y, cz, qstart)[0])
                    c = "B" if (s and fr) else "S" if s else "F" if fr else "."
                    ns += s; nf += fr; nb += (s and fr); row.append(c)
                grid.append((y, row))
            hdr = "  y\\x  " + " ".join(f"{x:+.2f}" for x in XS); print(hdr)
            for y, row in grid: print(f"  {y:+.2f} " + "    ".join(row))
            tot = len(XS)*len(YS)
            print(f"  side reachable: {ns}/{tot} | front reachable: {nf}/{tot} | both: {nb}/{tot}")
            print("=======================================================================\n")
            planner.update_world(build_scene(*cyl_center()))   # restore
        run_async(f)

    server.gui.add_button("SideGrasp (current)", color="teal").on_click(lambda _: demo("SIDE", plan_side))
    server.gui.add_button("FrontGrasp (new)", color="green").on_click(lambda _: demo("FRONT", plan_front))
    server.gui.add_button("Coverage sweep (side vs front)", color="orange").on_click(lambda _: coverage())
    print(f"\n================  OPEN  http://localhost:{PORT}  ================\n", flush=True)
    while True: time.sleep(1)


if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\nbye")
    except Exception: print("\n❌ EXCEPTION:"); traceback.print_exc(); sys.exit(3)
