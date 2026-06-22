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
CX, CY = 0.38, -0.30                                  # starting position (robot's right)
CZ = -0.006 + H / 2.0                                 # center; table top ~ -0.006 (glass sits on it)
# SOLID wooden-box table (was a 4 cm sheet). TABLE_BACK = the edge nearest the robot; raise it to move the
# table further AWAY from the body if the relaxed start pose reports "Start ... in collision". Top stays at -0.011.
TABLE_THICK, TABLE_BACK = 0.6, 0.34                                       # was a sheet at back edge 0.25
# y-dim widened to 1.0 (center -0.05 -> +y edge 0.45) so the LEFT-shifted microwave stays seated on the table.
TABLE = [0.7, 1.0, TABLE_THICK], [TABLE_BACK + 0.35, -0.05, (CZ - H/2) - 0.005 - TABLE_THICK/2, 1, 0, 0, 0]

# SIDE grasp (current, frozen for reference)
SG_DX, SG_DYGAP, SG_FROMTOP, SG_APPROACH = 0.215, 0.01, 0.035, -0.05
# FRONT grasp (new) defaults — the wrist position is DERIVED by rotating the side grasp by FG_YAW about the
# cylinder axis; FG_DX/FG_DY are small fine-tune nudges on top. FG_YAW=-90 => palm +x, approach from the front.
FG_DX, FG_DY, FG_FROMTOP, FG_APPROACH = 0.0, 0.0, 0.055, -0.05
FG_YAW0, FG_PITCH0 = -90.0, 0.0                       # palm +y -> +x is yaw -90 about z

# ---- RELAXED start pose: arms lowered, elbows bent ~60 deg, shoulders ABDUCTED slightly --------------
# The elbow flexes positive only (URDF limit [-1.0472, +2.0944]). IMPORTANT: with shoulder_roll=0 a bent
# elbow makes cuRobo's self-collision spheres reject the pose (the forearm tucks into the torso), even
# though it looks clear. A small abduction (SHOULDER_ROLL, arms rotated off the torso) clears it, so we
# keep self-collision ON and the relaxed look. This pose is the planning start for the door grasp.
ELBOW_BEND     = 1.125                              # elbow flexion (positive direction only)
SHOULDER_PITCH = 0.0                                # upper-arm forward hang (0 = straight down)
SHOULDER_ROLL  = 0.2                                # ABDUCT the upper arms off the torso -> avoids the self-collision reject
RELAXED_ARM = {"left_shoulder_pitch_joint":  SHOULDER_PITCH, "left_shoulder_roll_joint":   SHOULDER_ROLL,
               "left_elbow_joint":  ELBOW_BEND,
               "right_shoulder_pitch_joint": SHOULDER_PITCH, "right_shoulder_roll_joint": -SHOULDER_ROLL,
               "right_elbow_joint": ELBOW_BEND}

# ---- MICROWAVE (VISUAL ONLY -- not added to the planner collision world yet) ------------------------
# Pelvis frame: x fwd, y left (robot's right = -y), z up. The unit rests ON the table; its DOOR is on the
# FRONT (-x) face (nearest the robot). Opened with the LEFT arm, so the microwave is shifted to the robot's
# LEFT (+y) where the left arm can reach. HANDLE is on the unit's -y front edge, HINGE on its +y edge.
# Real Samsung unit measured: length(depth x)=31, width(y)=49, height(z)=27.5 cm.
MW_DIMS      = (0.31, 0.49, 0.275)                   # body (depth/length x, width y, height z)
MW_CX, MW_CY = 0.602, -0.015                           # body center x,y on the table
MW_WALL      = 0.04                                  # 4 cm insulation layer (cavity left/top/bottom/back + the wall between cavity and buttons)
MW_BTN_W     = 0.10                                  # right (-y) 10 cm = SOLID control/button panel (no cavity there)
MW_HANDLE_FROM_LEFT = 0.39                           # handle centre is 39 cm from the LEFT (+y) edge (= 10 cm from the right)
DOOR_THICK   = 0.02                                  # door slab thickness along x
TABLE_TOP_Z  = TABLE[1][2] + TABLE[0][2] / 2.0       # table pose z + half thickness (~ -0.011)
MW_CZ        = TABLE_TOP_Z + MW_DIMS[2] / 2.0        # body center z so the unit sits on the table
MW_FRONT_X   = MW_CX - MW_DIMS[0] / 2.0              # front face, toward the robot
MW_HINGE_Y   = MW_CY + MW_DIMS[1] / 2.0              # +y front edge ("left") -> hinge
MW_HANDLE_Y  = MW_HINGE_Y - MW_HANDLE_FROM_LEFT      # handle 39 cm from the left edge -> y = MW_CY - 0.145 (robot's right side)
HINGE_XYZ    = (MW_FRONT_X, MW_HINGE_Y, MW_CZ)       # hinge ROTATION AXIS is vertical (+z) through this point
HANDLE_XYZ   = (MW_FRONT_X - DOOR_THICK, MW_HANDLE_Y, MW_CZ)

# ---- DOOR-OPEN motion (LEFT ARM) --------------------------------------------------------------------
# The LEFT arm opens the door; the RIGHT arm stays FROZEN at its relaxed pose. The left hand reaches the
# handle from the robot side (palm toward -x), fingers hooking through the gap behind the handle, then
# grasps. Planned with a dedicated LEFT-arm planner (collision ON). Close = thumb FULL ROTATION first,
# then fingers + thumb curl together (same two-step procedure as the real Inspire hand).
DOOR_WRIST_YAW  = -60.0          # LEFT wrist yaw that lands the hand ON the handle (verified); slider-tunable
DOOR_PITCH      = 0.0            # wrist pitch (deg)
DOOR_YAW_TRACK  = 0.5            # how much the wrist yaw follows the door swing (1.0 = rigid 1:1; <1 = gentler roll, curled grip rolls in the fingers)
DOOR_STANDOFF   = 0.16           # (superseded by fingertip-targeting below)
FINGER_OFFSET   = (0.289, -0.008, 0.006)  # MEASURED: left_wrist_yaw_link -> MIDDLE fingertip (open hand), in the wrist frame
DOOR_GRIP_OFFSET= (0.16, -0.008, 0.006)   # CURLED-grip contact (closer to the wrist) -> wrist stays central -> door opens FURTHER
DOOR_GAP_DX     = 0.012          # push the FINGERTIP this far past the handle front, into the gap behind it
DOOR_INSERT_DY  = 0.05           # phase-1 pre-pose: fingertip this far to the robot's LEFT (+y) of the handle
DOOR_OPEN_DEG   = 80.0           # door-open target (deg); swing stops early at the reachable limit
# ---- PLACE the grasped glass INSIDE the microwave (RIGHT arm) ---------------------------------------
PLACE_X, PLACE_Z = 0.46, 0.10    # glass-center target inside the cavity (y = MW_CY); verified reachable by the right arm
PLACE_YAW        = 0.0           # wrist orientation for the place (the glass is a symmetric cylinder -> free to re-orient)
PLACE_INSERT_ANGLE = 40.0        # insert-LINE tilt from the front normal (deg): the glass slides in at an ANGLE (not
                                 # perpendicular) so it slips past the diagonal half-open door. Tune visually (+15..+60 reach).
PLACE_INSERT_BACK  = 0.18        # how far the glass is pulled back along the angled line before sliding in
# After grasping, the right arm brings the glass BACK to this staging coordinate, then pushes it in.
BRING_BACK_XYZ = (0.28, -0.08, 0.12)   # glass-center staging target after grasp -- TUNE to the pose you find optimal
BRING_BACK_YAW = 20.0                   # wrist yaw at the staging pose (glass is symmetric, so free to set)
# ---- straight-from-staging INSERT into the microwave (RIGHT arm) ------------------------------------
# From the staging pose the glass slides into the cavity along a line tilted PLACE_IN_ANGLE in the xy
# plane (toward +y / cavity-centre), so it walks OFF the right wall as it goes deeper. Planned with the
# microwave NOT a hard obstacle (handle has swung away, so its collision is ignored per request); you
# visually confirmed this line clears the body. 30deg@0.26 lands the glass dead-centre (~0.53,-0.02).
PLACE_IN_ANGLE = 42.0            # insert-line tilt in xy from +x toward +y (deg). 30 reaches deepest; 15 is shallower.
PLACE_IN_DIST  = 0.23            # distance to slide along that line from the staging pose (m)
DOOR_REACH      = SG_DX          # kept for the diagnostic scripts (check_door_reach.py / diagnostic_collision.py)
DOOR_HGAP       = 0.012          # kept for the diagnostic scripts
DOOR_THUMB_ROT  = 2.2            # *_thumb_1_joint -- full thumb rotation/opposition (close phase 1)
DOOR_THUMB_CURL = 1.0            # *_thumb_2_joint -- thumb curl, closes WITH the fingers (phase 2)
DOOR_FINGER     = 1.4            # index/middle/ring/little curl (phase 2)
DOOR_DO_SWING   = True           # after the fingers are in the gap, swing the door open
# LEFT-arm planner setup: active chain = left arm; the right arm is LOCKED (frozen) at its relaxed pose.
LARM_JOINTS = ["left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
               "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint"]
RIGHT_FROZEN = {"right_shoulder_pitch_joint": 0.0, "right_shoulder_roll_joint": -SHOULDER_ROLL,
                "right_shoulder_yaw_joint": 0.0, "right_elbow_joint": ELBOW_BEND,
                "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0, "right_wrist_yaw_joint": 0.0}
LHAND_JOINTS = ["left_thumb_1_joint","left_thumb_2_joint","left_index_1_joint",
                "left_middle_1_joint","left_ring_1_joint","left_little_1_joint"]
LHAND_OPEN = dict(zip(LHAND_JOINTS, [0, 0, 0, 0, 0, 0]))

HAND_JOINTS = ["right_thumb_1_joint","right_thumb_2_joint","right_index_1_joint",
               "right_middle_1_joint","right_ring_1_joint","right_little_1_joint"]
HAND_OPEN   = dict(zip(HAND_JOINTS, [0,0,0,0,0,0]))
HAND_CLOSED = dict(zip(HAND_JOINTS, [2.2,0.0,1.4,1.4,1.4,1.4]))
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
def quat_rotate(q, v):  # rotate vector v=[x,y,z] by quaternion q=[w,x,y,z]
    w, x, y, z = q
    return qmul(qmul(q, [0.0, v[0], v[1], v[2]]), [w, -x, -y, -z])[1:]


def main():
    import torch
    from curobo.viewer import ViserVisualizer
    from curobo.scene import Scene, Cuboid, Cylinder, Mesh
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import ContentPath, GoalToolPose, JointState, Pose
    from curobo._src.state.state_joint_trajectory_ops import get_joint_state_at_horizon_index

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

    # ---- LEFT-arm planner for the door: same robot, active chain = LEFT arm, RIGHT arm LOCKED at relaxed ----
    import yaml as _yaml
    _lc = _yaml.safe_load(open(OUT)); _lk = _lc["kinematics"]; _lk["tool_frames"] = ["left_wrist_yaw_link"]
    for _j in LARM_JOINTS: _lk["lock_joints"].pop(_j, None)        # unlock the left arm (now the active chain)
    _lk["lock_joints"].update(RIGHT_FROZEN)                        # freeze the right arm at its relaxed pose
    _yaml.safe_dump(_lc, open("/tmp/g1_inspire_left.yml", "w"))
    print("building LEFT-arm planner ...", flush=True)
    planner_L = MotionPlanner(MotionPlannerCfg.create(robot="/tmp/g1_inspire_left.yml", scene_model=None,
                                                      max_goalset=8, collision_cache={"obb": 12, "mesh": 4}))
    TABLE_C = Cuboid(name="table", dims=TABLE[0], pose=TABLE[1])
    def mw_body_walls():   # HOLLOW microwave: 4cm insulation walls (back, left/+y, top, bottom) + a SOLID right
                           #   block (4cm insulation + 10cm button panel = 14cm) on the -y side. FRONT (-x) open (door).
        hxw, hyw, hzw = MW_DIMS[0]/2, MW_DIMS[1]/2, MW_DIMS[2]/2
        W = MW_WALL; RS = MW_WALL + MW_BTN_W           # right solid = insulation(4) + buttons(10) = 14 cm
        return [Cuboid(name="mw_back",   dims=[W, MW_DIMS[1], MW_DIMS[2]], pose=[MW_CX+hxw-W/2, MW_CY, MW_CZ, 1,0,0,0]),
                Cuboid(name="mw_left",   dims=[MW_DIMS[0], W, MW_DIMS[2]], pose=[MW_CX, MW_CY+hyw-W/2, MW_CZ, 1,0,0,0]),
                Cuboid(name="mw_right",  dims=[MW_DIMS[0], RS, MW_DIMS[2]], pose=[MW_CX, MW_CY-hyw+RS/2, MW_CZ, 1,0,0,0]),
                Cuboid(name="mw_top",    dims=[MW_DIMS[0], MW_DIMS[1], W], pose=[MW_CX, MW_CY, MW_CZ+hzw-W/2, 1,0,0,0]),
                Cuboid(name="mw_bottom", dims=[MW_DIMS[0], MW_DIMS[1], W], pose=[MW_CX, MW_CY, MW_CZ-hzw+W/2, 1,0,0,0])]
    def mw_handle_obs():   # the handle bar as a collision object (for the LEFT pre-reach so fingers don't clip it)
        return [Cuboid(name="mw_handle", dims=[0.03, 0.03, MW_DIMS[2]*0.5], pose=[MW_FRONT_X-DOOR_THICK-0.015, MW_HANDLE_Y+0.03, MW_CZ, 1,0,0,0])]
    # show every COLLISION surface in translucent RED so it's visible (vs the solid grey viz boxes)
    for _wc in mw_body_walls() + mw_handle_obs():
        viz._server.scene.add_box("/mw_collision/" + _wc.name, color=(235, 25, 25), dimensions=tuple(_wc.dims),
                                  position=tuple(_wc.pose[:3]), wxyz=(1, 0, 0, 0), opacity=0.55)
    def mw_scene_L(with_mw): return Scene(cuboid=[TABLE_C] + (mw_body_walls() + mw_handle_obs() if with_mw else []))
    def mw_scene_R(with_mw): return Scene(cuboid=[TABLE_C] + (mw_body_walls() if with_mw else []))   # right place: body walls only
    planner_L.update_world(mw_scene_L(True))   # microwave (walls + handle) is a collision object for the LEFT arm (toggled per phase)
    planner_L.warmup(enable_graph=True, num_warmup_iterations=5)

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

    # ---- relaxed start pose: lower the arms, bend both elbows ~60 deg (overrides the arms-straight default) ----
    render_full(RELAXED_ARM, HAND_OPEN)

    # ---- microwave: STATIC boxes in the browser only (planner collision world is left untouched, so the
    #      SideGrasp/FrontGrasp/BestAngle/Coverage buttons behave exactly as before) -------------------------
    server.scene.add_box("/microwave/body", color=(120, 122, 130), dimensions=MW_DIMS,
                         position=(MW_CX, MW_CY, MW_CZ), wxyz=(1, 0, 0, 0), opacity=0.15)  # faint shell so the hollow shows
    # SOLID control/button panel: the right (-y) 10 cm of the front face
    server.scene.add_box("/microwave/buttons", color=(30, 32, 38), dimensions=(MW_DIMS[0], MW_BTN_W, MW_DIMS[2]),
                         position=(MW_CX, MW_CY - MW_DIMS[1] / 2.0 + MW_BTN_W / 2.0, MW_CZ), wxyz=(1, 0, 0, 0), opacity=0.92)
    DOOR_W          = MW_HINGE_Y - MW_HANDLE_Y                                       # door spans hinge(+y) -> handle (= 39 cm)
    DOOR_CY         = (MW_HINGE_Y + MW_HANDLE_Y) / 2.0                               # door slab center y (closed)
    DOOR_REST_POS   = (MW_FRONT_X - DOOR_THICK / 2.0, DOOR_CY, MW_CZ)                # door slab center, closed
    HANDLE_REST_POS = (MW_FRONT_X - DOOR_THICK - 0.015, MW_HANDLE_Y + 0.03, MW_CZ)   # handle bar center, closed
    door_box   = server.scene.add_box("/microwave/door", color=(55, 58, 68),
                         dimensions=(DOOR_THICK, DOOR_W * 0.98, MW_DIMS[2] * 0.88),
                         position=DOOR_REST_POS, wxyz=(1, 0, 0, 0), opacity=0.9)
    handle_box = server.scene.add_box("/microwave/handle", color=(15, 15, 18),
                         dimensions=(0.03, 0.03, MW_DIMS[2] * 0.5),
                         position=HANDLE_REST_POS, wxyz=(1, 0, 0, 0))
    # a frame whose +z (blue) axis IS the door's vertical hinge rotation axis
    server.scene.add_frame("/microwave/hinge", axes_length=MW_DIMS[2], axes_radius=0.006,
                           position=HINGE_XYZ, wxyz=(1, 0, 0, 0))
    print(f"[microwave] body center=({MW_CX:.3f},{MW_CY:.3f},{MW_CZ:.3f}) dims(L,W,H)={tuple(round(d*100,1) for d in MW_DIMS)}cm  sits on table top z={TABLE_TOP_Z:.3f}", flush=True)
    print(f"[microwave] cavity (hollow) = {(MW_DIMS[1]-MW_BTN_W-2*MW_WALL)*100:.1f}w x {(MW_DIMS[2]-2*MW_WALL)*100:.1f}h x {(MW_DIMS[0]-MW_WALL)*100:.1f}deep cm  | {MW_WALL*100:.0f}cm insulation (L/top/bottom/back), right {MW_BTN_W*100:.0f}cm = solid buttons", flush=True)
    print(f"[microwave] HINGE rotation axis = VERTICAL (+z) through pelvis xyz=({HINGE_XYZ[0]:.3f}, {HINGE_XYZ[1]:.3f}, {HINGE_XYZ[2]:.3f})", flush=True)
    print(f"[microwave] handle = pelvis xyz=({HANDLE_XYZ[0]:.3f}, {HANDLE_XYZ[1]:.3f}, {HANDLE_XYZ[2]:.3f})  (39cm from left edge / 10cm from right)", flush=True)

    # ALL buttons (Side/Front/Best) + Coverage read these live:
    sl_dx    = server.gui.add_slider("SG_DX (reach / wrist behind)",   min=0.10, max=0.30, step=0.005, initial_value=SG_DX)
    sl_dygap = server.gui.add_slider("SG_DYGAP (palm gap)",            min=-0.03, max=0.08, step=0.005, initial_value=SG_DYGAP)
    sl_xoff  = server.gui.add_slider("WRIST X nudge (+ = ahead)",      min=-0.12, max=0.12, step=0.005, initial_value=0.01)
    sl_yoff  = server.gui.add_slider("WRIST Y nudge (+ = right/-y)",   min=-0.12, max=0.12, step=0.005, initial_value=0.02)
    sl_fdz   = server.gui.add_slider("FROMTOP (below top; bigger=lower)", min=-0.10, max=0.15, step=0.005, initial_value=FG_FROMTOP)
    sl_fap   = server.gui.add_slider("APPROACH (tool standoff)",       min=-0.20, max=-0.01, step=0.01, initial_value=FG_APPROACH)
    sl_yaw   = server.gui.add_slider("FG_YAW deg (front angle, 0=side)", min=-180.0, max=180.0, step=5.0, initial_value=FG_YAW0)
    sl_pitch = server.gui.add_slider("PITCH deg (level pinky)",        min=-90.0, max=90.0, step=2.0, initial_value=FG_PITCH0)
    sl_grip  = server.gui.add_slider("GRIP (finger closure frac)",     min=0.2, max=1.0, step=0.05, initial_value=0.55)
    sl_dyaw  = server.gui.add_slider("DOOR wrist yaw deg (grasp config)", min=-180.0, max=180.0, step=5.0, initial_value=DOOR_WRIST_YAW)
    sl_dopen = server.gui.add_slider("DOOR open deg (swing target)",      min=0.0, max=120.0, step=5.0, initial_value=DOOR_OPEN_DEG)

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

    # --- unified grasp: rotated by `yaw` about the cylinder axis; reads ALL sliders live, so every button
    #     (Side/Front/Best) AND the Coverage sweep respond to SG_DX/SG_DYGAP/FROMTOP/APPROACH/PITCH. ---
    def plan_at_yaw(cx, cy, cz, yaw, qstart):
        dx, gap, fromtop, ap, pitch = sl_dx.value, sl_dygap.value, sl_fdz.value, sl_fap.value, sl_pitch.value
        th = np.deg2rad(yaw); c, s = np.cos(th), np.sin(th); ox, oy = -dx, -(R + gap)
        wx = cx + (c*ox - s*oy) + sl_xoff.value          # +x nudge = grasp more ahead
        wy = cy + (s*ox + c*oy) - sl_yoff.value          # +nudge = grasp more right (-y)
        wz = (cz + H/2.0) - fromtop
        return planner.plan_grasp(goal((wx, wy, wz), euler_quat(yaw, pitch)), qstart,
            grasp_approach_axis="y", grasp_approach_offset=ap, grasp_approach_in_tool_frame=True,
            grasp_lift_axis="z", grasp_lift_offset=0.15, grasp_lift_in_tool_frame=False,
            plan_approach_to_grasp=True, plan_grasp_to_lift=True, disable_collision_links=HAND_LINKS), (wx, wy, wz)
    def plan_side(cx, cy, cz, qstart):  return plan_at_yaw(cx, cy, cz, 0, qstart)             # yaw 0 = current side
    def plan_front(cx, cy, cz, qstart): return plan_at_yaw(cx, cy, cz, sl_yaw.value, qstart)  # the FG_YAW slider
    def ok(res): return res is not None and res.success is not None and bool(res.success.any())

    YAW_SET = [0, -30, -60, -90]          # side -> front; "best angle" = most-front reachable
    def plan_best(cx, cy, cz, qstart):    # sweep yaw from most-front toward side; take the first reachable
        for yaw in YAW_SET[::-1]:
            res, w = plan_at_yaw(cx, cy, cz, yaw, qstart)
            if ok(res): print(f"  BEST reachable yaw = {yaw}deg"); return res, w
        return None, (cx, cy, cz)

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
            XS = np.round(np.arange(0.30, 0.49, 0.04), 3)
            YS = np.round(np.arange(-0.32, -0.02, 0.04), 3)
            tot = len(XS)*len(YS)
            print(f"\n========= COVERAGE: side (yaw 0) vs BEST reachable angle, {tot} cells x {len(YAW_SET)} yaws =========")
            print(f"  glass r={R} h={H}. cell = most-front reachable yaw: 0=side 1=-30 2=-60 3=-90, '.'=none. "
                  f"(~takes a couple minutes)", flush=True)
            ns = nany = nnew = 0; grid = []
            for y in YS:
                row = []
                for x in XS:
                    cz = (-0.006) + H/2.0
                    planner.update_world(build_scene(x, y, cz))
                    reach = [yaw for yaw in YAW_SET if ok(plan_at_yaw(x, y, cz, yaw, qstart)[0])]
                    side = (0 in reach); anyok = bool(reach); mostfront = min(reach) if reach else None
                    sym = "." if not anyok else str(abs(mostfront)//30)
                    ns += side; nany += anyok; nnew += (anyok and not side); row.append(sym)
                grid.append((y, row)); print(f"  y={y:+.2f} " + " ".join(row), flush=True)
            print("  cols x = " + " ".join(f"{x:.2f}" for x in XS))
            print(f"  side(yaw 0): {ns}/{tot} | ANY-angle: {nany}/{tot} | NEW (front works, side fails): {nnew}/{tot}")
            print("  => allowing front-ish angles adds %d cells over side-only." % nnew)
            print("=================================================================================\n", flush=True)
            planner.update_world(build_scene(*cyl_center()))   # restore
        run_async(f)

    # ===== DOOR OPEN (LEFT ARM): right arm FROZEN at relaxed; left arm reaches the handle, grasps =====
    def rot_about_hinge(px, py, phi):                       # rotate a point about the vertical hinge by phi (rad)
        c, s = np.cos(phi), np.sin(phi); rx, ry = px - HINGE_XYZ[0], py - HINGE_XYZ[1]
        return HINGE_XYZ[0] + c*rx - s*ry, HINGE_XYZ[1] + s*rx + c*ry
    def set_door_angle(phi):                                # swing the door + handle viser boxes by phi (rad)
        qz = (float(np.cos(phi/2)), 0.0, 0.0, float(np.sin(phi/2)))
        for box, rest in ((door_box, DOOR_REST_POS), (handle_box, HANDLE_REST_POS)):
            x, y = rot_about_hinge(rest[0], rest[1], phi)
            box.position = (x, y, rest[2]); box.wxyz = qz

    LARM_SET = set(LARM_JOINTS)
    def relaxed_qstart_L():                                 # LEFT-arm relaxed start (active chain = left arm)
        full = planner_L.default_joint_state.clone(); jn = list(full.joint_names)
        pos = full.position.clone().reshape(-1)
        for name, val in RELAXED_ARM.items():
            if name in jn: pos[jn.index(name)] = float(val)
        return planner_L.kinematics.get_active_js(JointState.from_position(pos.unsqueeze(0), joint_names=jn))
    def goal_L(pos3, quat):
        return GoalToolPose(tool_frames=planner_L.tool_frames,
            position=torch.tensor([[[[list(pos3)]]]], device="cuda", dtype=torch.float32),
            quaternion=torch.tensor([[[[list(quat)]]]], device="cuda", dtype=torch.float32))
    def door_wrist_L(ft_xyz, yaw_deg, pitch_deg, offset=FINGER_OFFSET):  # place the CONTACT point at ft_xyz; wrist sits behind it
        Q = euler_quat(yaw_deg, pitch_deg); off = quat_rotate(Q, offset)
        return (ft_xyz[0] - off[0], ft_xyz[1] - off[1], ft_xyz[2] - off[2]), Q

    def open_door_left():
        def f():
            nonlocal is_moving
            def render_L(larm, hand):             # HOLD the right arm FROZEN at relaxed; move the left arm + hand
                render_full({**RIGHT_FROZEN, **larm}, hand)
            def render_traj(res):                 # play a plan_pose trajectory, left hand OPEN throughout
                t = res.interpolated_trajectory.squeeze(0); jn = list(t.joint_names)
                P = t.position[0].detach().cpu().numpy(); last = trim_last(P)
                for i in np.unique(np.linspace(0, last - 1, min(last, 60)).astype(int)):
                    if not is_moving: return False
                    render_L({a: float(P[i][jn.index(a)]) for a in jn if a in LARM_SET}, LHAND_OPEN); time.sleep(0.025)
                return True
            update_obstacles(); set_door_angle(0.0); render_full(RELAXED_ARM, LHAND_OPEN)
            yaw = sl_dyaw.value
            ftx, ftz = HANDLE_XYZ[0] + DOOR_GAP_DX, HANDLE_XYZ[2]
            FT1 = (ftx, HANDLE_XYZ[1] + DOOR_INSERT_DY, ftz)    # phase 1: fingertip to the LEFT of the handle (matched x,z)
            FT2 = (ftx, HANDLE_XYZ[1],                  ftz)    # phase 2: slide straight -y so the fingers enter the gap
            # phase 1: approach to the left of the handle -- microwave collision ON (don't go through the body)
            planner_L.update_world(mw_scene_L(True))
            w1, q1 = door_wrist_L(FT1, yaw, DOOR_PITCH)
            r1 = planner_L.plan_pose(goal_L(w1, q1), relaxed_qstart_L(), max_attempts=6)
            print(f"\nOPEN DOOR (LEFT) phase1 (left of handle) fingertip=({FT1[0]:.3f},{FT1[1]:.3f},{FT1[2]:.3f}) "
                  f"-> {'OK' if ok(r1) else 'FAILED'} {getattr(r1,'status',None)}")
            if not ok(r1) or not render_traj(r1): return
            seed = planner_L.kinematics.get_active_js(get_joint_state_at_horizon_index(r1.js_solution, -1).squeeze(0))
            # phase 2: slide straight -y into the gap -- microwave collision OFF (fingers must enter the gap;
            #          the swing also runs with it off so the hand can stay at the door front)
            planner_L.update_world(mw_scene_L(False))
            w2, q2 = door_wrist_L(FT2, yaw, DOOR_PITCH)
            r2 = planner_L.plan_pose(goal_L(w2, q2), seed, max_attempts=6)
            print(f"OPEN DOOR (LEFT) phase2 (slide -y into gap) fingertip=({FT2[0]:.3f},{FT2[1]:.3f},{FT2[2]:.3f}) "
                  f"-> {'OK' if ok(r2) else 'FAILED'} {getattr(r2,'status',None)}")
            if not ok(r2) or not render_traj(r2): return
            print("  fingers inserted into the gap behind the handle (open hand).")
            if not DOOR_DO_SWING: return
            # phase 3: OPEN the door -- OPEN hand (NO curl). The fingertip follows the handle's ARC about the
            #          hinge (same FINGER_OFFSET as the reach, so the swing continues smoothly from phase 2);
            #          the wrist yaw rolls PARTIALLY with the door (DOOR_YAW_TRACK) to keep contact.
            seed = planner_L.kinematics.get_active_js(get_joint_state_at_horizon_index(r2.js_solution, -1).squeeze(0))
            open_deg = sl_dopen.value; N = max(1, int(round(open_deg / 8.0))); reached = 0.0
            for i in range(1, N + 1):
                phid = -open_deg * (i / N); phi = np.deg2rad(phid)            # negative = swing toward the robot
                fx, fy = rot_about_hinge(ftx, HANDLE_XYZ[1], phi)
                wk, qk = door_wrist_L((fx, fy, ftz), yaw + DOOR_YAW_TRACK * phid, DOOR_PITCH)   # FINGER_OFFSET (default)
                rk = planner_L.plan_pose(goal_L(wk, qk), seed, max_attempts=4)
                if not ok(rk):
                    print(f"  door stuck at {abs(reached):.0f} deg (next {abs(phid):.0f} deg unreachable)"); break
                t = rk.interpolated_trajectory.squeeze(0); jn = list(t.joint_names)
                P = t.position[0].detach().cpu().numpy(); last = trim_last(P)
                idxs = np.unique(np.linspace(0, last - 1, min(last, 40)).astype(int))
                for k, fi in enumerate(idxs):
                    if not is_moving: return
                    render_L({a: float(P[fi][jn.index(a)]) for a in jn if a in LARM_SET}, LHAND_OPEN)
                    set_door_angle(np.deg2rad(reached + (phid - reached) * (k + 1) / len(idxs))); time.sleep(0.03)
                reached = phid
                seed = planner_L.kinematics.get_active_js(get_joint_state_at_horizon_index(rk.js_solution, -1).squeeze(0))
            print(f"  DOOR OPENED to {abs(reached):.0f} deg (open hand; left wrist yaw rolled {yaw:.0f} -> {yaw + DOOR_YAW_TRACK*reached:.0f}).")
        run_async(f)

    # ===== FULL bimanual sequence: LEFT opens door -> RIGHT grabs glass (from relaxed) -> RIGHT places inside =====
    def relaxed_qstart_R():                                 # RIGHT-arm relaxed start (active chain = right arm)
        full = planner.default_joint_state.clone(); jnn = list(full.joint_names)
        pr = full.position.clone().reshape(-1)
        for name, val in RELAXED_ARM.items():
            if name in jnn: pr[jnn.index(name)] = float(val)
        return planner.kinematics.get_active_js(JointState.from_position(pr.unsqueeze(0), joint_names=jnn))

    def full_sequence():
        def f():
            nonlocal is_moving
            update_obstacles(); set_door_angle(0.0)
            render_full(RELAXED_ARM, {**LHAND_OPEN, **HAND_OPEN})
            yaw = sl_dyaw.value; ftx, ftz = HANDLE_XYZ[0] + DOOR_GAP_DX, HANDLE_XYZ[2]
            # ---- PHASE 1: LEFT arm opens the door (right arm frozen at relaxed) ----
            def renL(lmap, lhand): render_full({**lmap, **RIGHT_FROZEN}, {**lhand, **HAND_OPEN})
            def playL(res):
                t = res.interpolated_trajectory.squeeze(0); jt = list(t.joint_names)
                P = t.position[0].detach().cpu().numpy(); lz = trim_last(P)
                for i in np.unique(np.linspace(0, lz - 1, min(lz, 60)).astype(int)):
                    if not is_moving: return False
                    renL({a: float(P[i][jt.index(a)]) for a in jt if a in LARM_SET}, LHAND_OPEN); time.sleep(0.025)
                return True
            def playL_rev(res):                          # play a left-arm trajectory BACKWARDS (door close / retract)
                t = res.interpolated_trajectory.squeeze(0); jt = list(t.joint_names)
                P = t.position[0].detach().cpu().numpy(); lz = trim_last(P)
                for i in reversed(np.unique(np.linspace(0, lz - 1, min(lz, 60)).astype(int))):
                    if not is_moving: return False
                    renL({a: float(P[i][jt.index(a)]) for a in jt if a in LARM_SET}, LHAND_OPEN); time.sleep(0.025)
                return True
            planner_L.update_world(mw_scene_L(True))     # microwave collision ON for the reach
            w, q = door_wrist_L((ftx, HANDLE_XYZ[1] + DOOR_INSERT_DY, ftz), yaw, DOOR_PITCH)
            r = planner_L.plan_pose(goal_L(w, q), relaxed_qstart_L(), max_attempts=6)
            print(f"\n[FULL] 1) door reach-left -> {'OK' if ok(r) else 'FAIL'}")
            if not ok(r) or not playL(r): return
            sL = planner_L.kinematics.get_active_js(get_joint_state_at_horizon_index(r.js_solution, -1).squeeze(0))
            reach_res = r                                # saved: reverse-played in PHASE 7 (FT1 -> relaxed)
            planner_L.update_world(mw_scene_L(False))    # microwave collision OFF for the -y insertion + swing
            w, q = door_wrist_L((ftx, HANDLE_XYZ[1], ftz), yaw, DOOR_PITCH)
            r = planner_L.plan_pose(goal_L(w, q), sL, max_attempts=6)
            print(f"[FULL] 1) door slide-in -> {'OK' if ok(r) else 'FAIL'}")
            if not ok(r) or not playL(r): return
            sL = planner_L.kinematics.get_active_js(get_joint_state_at_horizon_index(r.js_solution, -1).squeeze(0))
            slide_res = r                                # saved: reverse-played in PHASE 7 (FT2 -> FT1, fingers out of gap)
            od = sl_dopen.value; N = max(1, int(round(od / 8.0))); reached = 0.0
            LQ = {k: v for k, v in RELAXED_ARM.items() if k.startswith("left_")}
            for i in range(1, N + 1):
                phid = -od * (i / N); fx, fy = rot_about_hinge(ftx, HANDLE_XYZ[1], np.deg2rad(phid))
                w, q = door_wrist_L((fx, fy, ftz), yaw + DOOR_YAW_TRACK * phid, DOOR_PITCH)
                rk = planner_L.plan_pose(goal_L(w, q), sL, max_attempts=4)
                if not ok(rk): break
                t = rk.interpolated_trajectory.squeeze(0); jt = list(t.joint_names)
                P = t.position[0].detach().cpu().numpy(); lz = trim_last(P)
                ix = np.unique(np.linspace(0, lz - 1, min(lz, 40)).astype(int))
                for k, fi in enumerate(ix):
                    if not is_moving: return
                    renL({a: float(P[fi][jt.index(a)]) for a in jt if a in LARM_SET}, LHAND_OPEN)
                    set_door_angle(np.deg2rad(reached + (phid - reached) * (k + 1) / len(ix))); time.sleep(0.03)
                reached = phid; LQ = {a: float(P[lz - 1][jt.index(a)]) for a in jt if a in LARM_SET}
                sL = planner_L.kinematics.get_active_js(get_joint_state_at_horizon_index(rk.js_solution, -1).squeeze(0))
            print(f"[FULL] 1) door OPENED to {abs(reached):.0f} deg; left arm holds it.")
            # ---- PHASE 2: RIGHT arm grabs the glass (BestAngle from relaxed); left held at LQ ----
            def renR(rmap, rhand): render_full({**LQ, **rmap}, {**LHAND_OPEN, **rhand})
            def playR(seg, rhand):
                t = seg.squeeze(0); jt = list(t.joint_names)
                P = t.position[0].detach().cpu().numpy(); lz = trim_last(P)
                for i in np.unique(np.linspace(0, lz - 1, min(lz, 60)).astype(int)):
                    if not is_moving: return False
                    renR({a: float(P[i][jt.index(a)]) for a in jt if a in ARM_SET}, rhand); time.sleep(0.025)
                return True
            def playR_rev(seg, rhand):                   # play a right-arm trajectory BACKWARDS (retract out of cavity)
                t = seg.squeeze(0); jt = list(t.joint_names)
                P = t.position[0].detach().cpu().numpy(); lz = trim_last(P)
                for i in reversed(np.unique(np.linspace(0, lz - 1, min(lz, 60)).astype(int))):
                    if not is_moving: return False
                    renR({a: float(P[i][jt.index(a)]) for a in jt if a in ARM_SET}, rhand); time.sleep(0.025)
                return True
            cx, cy, cz = cyl_center(); res = None
            for gy in [-90, -60, -30, 0]:
                rr, ww = plan_at_yaw(cx, cy, cz, gy, relaxed_qstart_R())
                if ok(rr): res = rr; print(f"[FULL] 2) glass BestAngle yaw={gy} -> OK"); break
            if res is None: print("[FULL] 2) glass grab FAILED"); return
            if not playR(res.approach_interpolated_trajectory, HAND_OPEN): return
            if not playR(res.grasp_interpolated_trajectory, HAND_OPEN): return
            arm = arm_map_of(res.grasp_interpolated_trajectory); tgt = grip()
            for fr in np.linspace(0, 1, 10): renR(arm, {k: (fr*tgt[k] if 'thumb' in k else 0.0) for k in HAND_JOINTS}); time.sleep(0.03)
            for fr in np.linspace(0, 1, 12): renR(arm, {k: (tgt[k] if 'thumb' in k else fr*tgt[k]) for k in HAND_JOINTS}); time.sleep(0.03)
            closedR = {k: tgt[k] for k in HAND_JOINTS}
            gseed = planner.kinematics.get_active_js(get_joint_state_at_horizon_index(res.grasp_trajectory, -1).squeeze(0))
            print("[FULL] 2) glass grasped.")
            # ---- PHASE 3: bring the grasped glass BACK to a staging coordinate (NO place-inside) ----
            #      Just lift/carry the glass to BRING_BACK_XYZ (a tunable target). Set BRING_BACK_XYZ to the
            #      pose you find optimal; no insertion is attempted, so the hand never enters the microwave.
            planner.update_world(Scene(cuboid=[Cuboid(name="table", dims=TABLE[0], pose=TABLE[1])]))  # glass in-hand
            c, s = np.cos(np.deg2rad(BRING_BACK_YAW)), np.sin(np.deg2rad(BRING_BACK_YAW)); ox, oy = -sl_dx.value, -(R + sl_dygap.value)
            bw = (BRING_BACK_XYZ[0] + (c*ox - s*oy) + sl_xoff.value, BRING_BACK_XYZ[1] + (s*ox + c*oy) - sl_yoff.value, BRING_BACK_XYZ[2] + (H/2 - sl_fdz.value))
            rp = planner.plan_pose(goal(bw, euler_quat(BRING_BACK_YAW, 0)), gseed, max_attempts=8)
            print(f"[FULL] 3) bring glass back to {tuple(round(v,3) for v in BRING_BACK_XYZ)} wrist=({bw[0]:.3f},{bw[1]:.3f},{bw[2]:.3f}) -> {'OK' if ok(rp) else 'FAIL'}")
            if not ok(rp): return
            t = rp.interpolated_trajectory.squeeze(0); jt = list(t.joint_names); P = t.position[0].detach().cpu().numpy(); lz = trim_last(P)
            gf = obstacle_frames.get("object"); ix = np.unique(np.linspace(0, lz - 1, min(lz, 60)).astype(int))
            for k, i in enumerate(ix):
                if not is_moving: return
                renR({a: float(P[i][jt.index(a)]) for a in jt if a in ARM_SET}, closedR)
                if gf is not None:
                    fr = (k + 1) / len(ix)
                    try: gf.position = ((cx) + (BRING_BACK_XYZ[0]-cx)*fr, (cy) + (BRING_BACK_XYZ[1]-cy)*fr, (cz) + (BRING_BACK_XYZ[2]-cz)*fr)
                    except Exception: pass
                time.sleep(0.03)
            print("[FULL] 3) glass brought back to the staging pose (held).")
            # ---- PHASE 4: slide the glass straight INTO the microwave along an ANGLED line ----
            #      From the staging pose, move along PLACE_IN_ANGLE in the xy plane (toward +y/centre) so
            #      the glass walks off the right wall as it goes in. Microwave stays out of the collision
            #      world (handle has swung away -> its collision is ignored); table only.
            sBB = planner.kinematics.get_active_js(get_joint_state_at_horizon_index(rp.js_solution, -1).squeeze(0))
            ca, sa = np.cos(np.deg2rad(PLACE_IN_ANGLE)), np.sin(np.deg2rad(PLACE_IN_ANGLE))
            ig = (BRING_BACK_XYZ[0] + PLACE_IN_DIST*ca, BRING_BACK_XYZ[1] + PLACE_IN_DIST*sa, BRING_BACK_XYZ[2])  # glass-centre target
            iw = (ig[0] + (c*ox - s*oy) + sl_xoff.value, ig[1] + (s*ox + c*oy) - sl_yoff.value, ig[2] + (H/2 - sl_fdz.value))
            ri = planner.plan_pose(goal(iw, euler_quat(BRING_BACK_YAW, 0)), sBB, max_attempts=8)
            print(f"[FULL] 4) insert glass at {PLACE_IN_ANGLE:.0f}deg line -> {tuple(round(v,3) for v in ig)} wrist=({iw[0]:.3f},{iw[1]:.3f},{iw[2]:.3f}) -> {'OK' if ok(ri) else 'FAIL'}")
            if not ok(ri):
                print("[FULL] 4) insert FAILED (tune PLACE_IN_ANGLE / PLACE_IN_DIST / BRING_BACK_XYZ). Glass held at staging.")
                return
            t = ri.interpolated_trajectory.squeeze(0); jt = list(t.joint_names); P = t.position[0].detach().cpu().numpy(); lz = trim_last(P)
            ix = np.unique(np.linspace(0, lz - 1, min(lz, 60)).astype(int))
            for k, i in enumerate(ix):
                if not is_moving: return
                renR({a: float(P[i][jt.index(a)]) for a in jt if a in ARM_SET}, closedR)
                if gf is not None:
                    fr = (k + 1) / len(ix)
                    try: gf.position = (BRING_BACK_XYZ[0] + (ig[0]-BRING_BACK_XYZ[0])*fr, BRING_BACK_XYZ[1] + (ig[1]-BRING_BACK_XYZ[1])*fr, BRING_BACK_XYZ[2] + (ig[2]-BRING_BACK_XYZ[2])*fr)
                    except Exception: pass
                time.sleep(0.03)
            print("[FULL] 4) glass inserted into the microwave.")
            # ---- PHASE 5: open the fingers (release), then RETRACT the right hand out to relaxed ----
            #      Hard wall-collision planning into/out of the cavity is infeasible here (fixed base, tight
            #      cavity past the half-open door; also the RIGHT planner's locked LEFT arm sits inside the
            #      wall boxes, so any walls-on plan reports start/end in collision). So the hand RETRACES the
            #      insert line back OUT (guaranteed clear -- it just came in that way) and only then returns
            #      to relaxed through the open space in front of the microwave.
            insArm = arm_map_of(ri.interpolated_trajectory)
            for fr in np.linspace(1, 0, 12):                            # open fingers; the glass stays in the cavity
                if not is_moving: return
                renR(insArm, {k: closedR[k] * fr for k in HAND_JOINTS}); time.sleep(0.03)
            print("[FULL] 5) fingers opened; glass released in the microwave.")
            if not playR_rev(ri.interpolated_trajectory, HAND_OPEN): return   # retrace the angled insert OUT of the cavity
            rr = planner.plan_cspace(relaxed_qstart_R(), sBB, max_attempts=10)  # staging -> relaxed (table only; open space)
            print(f"[FULL] 5) retract right arm to relaxed -> {'OK' if ok(rr) else 'FAIL'}")
            if not ok(rr): return
            if not playR(rr.interpolated_trajectory, HAND_OPEN): return
            print("[FULL] 5) right arm back at relaxed (hand open).")
            # ---- PHASE 6: CLOSE the door with the LEFT arm -- reverse of the open swing (collision OFF) ----
            cur = reached; Nc = max(1, int(round(abs(reached) / 8.0)))
            for i in range(1, Nc + 1):
                phid = reached * (1.0 - i / Nc)                         # reached (open) -> 0 (closed)
                fx, fy = rot_about_hinge(ftx, HANDLE_XYZ[1], np.deg2rad(phid))
                w, q = door_wrist_L((fx, fy, ftz), yaw + DOOR_YAW_TRACK * phid, DOOR_PITCH)
                rk = planner_L.plan_pose(goal_L(w, q), sL, max_attempts=4)
                if not ok(rk):
                    print(f"[FULL] 6) door-close stuck at {abs(cur):.0f} deg"); break
                t = rk.interpolated_trajectory.squeeze(0); jt = list(t.joint_names)
                P = t.position[0].detach().cpu().numpy(); lz = trim_last(P)
                ix = np.unique(np.linspace(0, lz - 1, min(lz, 40)).astype(int))
                for k, fi in enumerate(ix):
                    if not is_moving: return
                    renL({a: float(P[fi][jt.index(a)]) for a in jt if a in LARM_SET}, LHAND_OPEN)
                    set_door_angle(np.deg2rad(cur + (phid - cur) * (k + 1) / len(ix))); time.sleep(0.03)
                cur = phid; sL = planner_L.kinematics.get_active_js(get_joint_state_at_horizon_index(rk.js_solution, -1).squeeze(0))
            set_door_angle(0.0)
            print(f"[FULL] 6) door closed (left arm tracked it from {abs(reached):.0f} deg back to 0).")
            # ---- PHASE 7: bring the LEFT hand back to relaxed -- reverse the slide, then the reach ----
            if not playL_rev(slide_res): return                         # FT2 -> FT1: pull the fingers +y out of the gap
            if not playL_rev(reach_res): return                         # FT1 -> relaxed (reverse the collision-ON reach)
            render_full(RELAXED_ARM, {**LHAND_OPEN, **HAND_OPEN})       # settle both arms at the relaxed pose
            print("[FULL] 7) left arm back at relaxed. Full bimanual sequence complete.")
        run_async(f)

    server.gui.add_button("FULL: open door (L) -> grab+insert glass (R) -> retract (R) -> close door + relax (L)", color="red").on_click(lambda _: full_sequence())
    server.gui.add_button("SideGrasp (yaw 0)", color="teal").on_click(lambda _: demo("SIDE", plan_side))
    server.gui.add_button("FrontGrasp (FG_YAW)", color="green").on_click(lambda _: demo("FRONT", plan_front))
    server.gui.add_button("BestAngle (most-front reachable)", color="green").on_click(lambda _: demo("BEST", plan_best))
    server.gui.add_button("Coverage: side vs best-angle", color="orange").on_click(lambda _: coverage())
    server.gui.add_button("GraspDoorHandle LEFT (right arm frozen; reach -> grasp)", color="violet").on_click(lambda _: open_door_left())
    print(f"\n================  OPEN  http://localhost:{PORT}  ================\n", flush=True)
    while True: time.sleep(1)


if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\nbye")
    except Exception: print("\n❌ EXCEPTION:"); traceback.print_exc(); sys.exit(3)
