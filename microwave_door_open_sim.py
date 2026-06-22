#!/usr/bin/env python3
# =============================================================================
# microwave_door_open_sim.py — LEFT-arm microwave-door APPROACH + OPEN, in viser/cuRobo,
# at the microwave pose we LOCALISED from the real head camera.
#
# This is a focused slice of sim_grasp_viz.py: the SAME left-arm door-open method
# (reach to the pre-pose -> slide -y into the gap behind the handle -> swing the door
# about the hinge, wrist yaw partially tracking it), and NOTHING ELSE (no glass grab,
# no insert, no right arm). The microwave is placed at the coordinates emitted by
# microwave_handle_real.py (/tmp/microwave_pose.json); its DIMENSIONS come from
# sim_grasp_viz.py; and the handle sits 3.8 cm in FRONT of the door front and 3.8 cm
# BELOW the door top (HANDLE_PROTRUDE / HANDLE_TOP_GAP) — matching the real unit.
#
# Sliders tune the APPROACH live: wrist yaw, grasp height, pre-pose +y offset, gap push,
# yaw-tracking, and the open angle.
#
# RUN (cumotion_venv):  /home/dishant/cumotion_venv/bin/python microwave_door_open_sim.py
#   -> open http://localhost:8081 ; click "Approach + Open door (LEFT)".
# =============================================================================
import sys, time, json, threading, traceback
import numpy as np

OUT  = "/home/dishant/g1_ws/cumotion/config/g1_inspire_right.yml"
PORT = 8081
POSE_JSON = "/tmp/microwave_pose.json"     # written by microwave_handle_real.py
TRAJ_JSON = "/tmp/door_open_traj.json"     # written by real_plan_door.py (the plan to replay / run on the robot)

# ---- Z LIFT: on the REAL robot we drive the handle AND the glass +5 cm above the z that depth gives us
# (the detectors read the FRONT/visible surface, which sits a few cm low; +5 cm is the offset that grabs in
# real). The detectors are NOT changed -- we apply the lift HERE, as a sim/plan setting, so the motion plan
# is computed for the heights we actually operate at. Lifting the grasp also clears the table (less clipping).
Z_LIFT = 0.05

# ---- GRASP TABLE GAP: the glass-grasp planner needs the table to sit clearly BELOW the glass bottom. If the
# table touches/overlaps the glass (the grasp TARGET), cuRobo's grasp-goal generation returns None ("the glass
# is in collision with the table") and EVERY yaw fails -- verified: table top >= glass bottom => FAIL, a clear
# gap => OK, regardless of grip height. So drop the grasp table this far below the glass bottom. The fingers
# still grip near the glass base; the +Z_LIFT is what keeps them clear of the REAL table. = plan --table-gap.
GRASP_TABLE_GAP = 0.03

# ---- microwave POSE: from the real perception (fallback = last localised values) -------------------
def _load_mw_center():
    try:
        d = json.load(open(POSE_JSON)); c = d["microwave_center"]; h = d.get("handle_center", c)
        # body-centre z is set to the HANDLE z (the reliable yellow-tape detection): the handle sits on
        # the unit's vertical mid-line, so MW_CZ == handle z. x,y still come from the body centroid.
        # +Z_LIFT: plan the handle at the height we actually contact it on the real robot.
        return float(c[0]), float(c[1]), float(h[2]) + Z_LIFT, f"perception (body z = handle z +{Z_LIFT*100:.0f}cm)"
    except Exception:
        return 0.754, 0.048, 0.045 + Z_LIFT, f"fallback (no /tmp/microwave_pose.json) +{Z_LIFT*100:.0f}cm"
MW_CX, MW_CY, MW_CZ, MW_SRC = _load_mw_center()

# ---- microwave DIMENSIONS + layout: identical to sim_grasp_viz.py ----------------------------------
MW_DIMS      = (0.31, 0.49, 0.275)          # body (depth/length x, width y, height z)
MW_WALL      = 0.04                         # 4 cm insulation walls (cavity left/top/bottom/back)
MW_BTN_W     = 0.10                         # right (-y) 10 cm = SOLID button panel
MW_HANDLE_FROM_LEFT = 0.39                  # handle centre is 39 cm from the +y (left) edge
DOOR_THICK   = 0.02                         # door slab thickness along x
HANDLE_PROTRUDE = 0.038                     # handle FRONT is 3.8 cm ahead of the door front
HANDLE_LEN      = 0.18                       # handle bar height (visual + collision)

MW_FRONT_X   = MW_CX - MW_DIMS[0] / 2.0     # front face (= door front), toward the robot
MW_TOP_Z     = MW_CZ + MW_DIMS[2] / 2.0     # microwave / door top
MW_HINGE_Y   = MW_CY + MW_DIMS[1] / 2.0     # +y front edge -> hinge
MW_HANDLE_Y  = MW_HINGE_Y - MW_HANDLE_FROM_LEFT     # handle y (robot's right side)
HANDLE_FRONT_X = MW_FRONT_X - HANDLE_PROTRUDE       # handle bar front (3.8 cm ahead of the door)
HANDLE_Z       = MW_CZ                               # handle CENTRE z = body-centre z (= perceived handle z)
HINGE_XYZ    = (MW_FRONT_X, MW_HINGE_Y, MW_CZ)      # vertical (+z) door-hinge axis

# ---- TABLE the unit rests on (top = microwave bottom), for left-arm collision ----------------------
# NEAR edge fixed at 0.34 (like sim_grasp_viz). A table reaching closer to the robot CLIPS the left
# forearm on the reach -- that (not the grasp height) is what made the earlier approach fail.
TABLE_TOP_Z = MW_CZ - MW_DIMS[2] / 2.0
TABLE_NEAR  = 0.32
_TFAR = MW_CX + MW_DIMS[0] / 2.0 + 0.20             # cover the microwave back + margin
_TDX  = max(0.70, _TFAR - TABLE_NEAR)
# y-extent: the RIGHT edge stops at the microwave's right edge so the (upper) table does NOT reach under the
# glass / the right hand's pre-grasp -- the old full 1.0 m width clipped the right hand. Extend LEFT for support.
TABLE_RIGHT_Y = MW_CY - MW_DIMS[1] / 2.0           # = microwave right edge (glass sits to the -y of this)
TABLE_LEFT_Y  = MW_CY + MW_DIMS[1] / 2.0 + 0.25    # extend left (counter/support; clear of the right arm)
_TWY = TABLE_LEFT_Y - TABLE_RIGHT_Y
TABLE = [_TDX, _TWY, 0.6], [TABLE_NEAR + _TDX / 2.0, (TABLE_LEFT_Y + TABLE_RIGHT_Y) / 2.0, TABLE_TOP_Z - 0.3, 1, 0, 0, 0]

# ---- RELAXED start pose (same as sim_grasp_viz.py) -------------------------------------------------
ELBOW_BEND, SHOULDER_PITCH, SHOULDER_ROLL = 1.125, 0.0, 0.2
RELAXED_ARM = {"left_shoulder_pitch_joint": SHOULDER_PITCH, "left_shoulder_roll_joint": SHOULDER_ROLL,
               "left_elbow_joint": ELBOW_BEND,
               "right_shoulder_pitch_joint": SHOULDER_PITCH, "right_shoulder_roll_joint": -SHOULDER_ROLL,
               "right_elbow_joint": ELBOW_BEND}

# ---- DOOR-OPEN defaults (same meaning/values as sim_grasp_viz.py) ----------------------------------
DOOR_WRIST_YAW = -60.0          # LEFT wrist yaw that lands the hand on the handle
DOOR_PITCH     = 0.0
DOOR_YAW_TRACK = 0.5            # how much the wrist yaw follows the door swing (1.0 = rigid)
FINGER_OFFSET  = (0.289, -0.008, 0.006)     # left_wrist_yaw_link -> MIDDLE fingertip (open hand), wrist frame
DOOR_GAP_DX    = 0.020          # push the fingertip this far past the handle front, into the gap behind it
DOOR_INSERT_DY = 0.05           # phase-1 pre-pose: fingertip this far to the +y (left) of the handle
DOOR_OPEN_DEG  = 80.0           # swing target (stops early at the reachable limit)

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
HAND_OPEN   = dict(zip(HAND_JOINTS, [0, 0, 0, 0, 0, 0]))
HAND_CLOSED = dict(zip(HAND_JOINTS, [2.2, 0.0, 1.4, 1.4, 1.4, 1.4]))   # inspire curl: [thumb_rot, thumb_curl, 4 fingers]
GRIP_FRAC   = 0.55                                                       # finger-closure fraction at the grasp (cosmetic)

# ---- GLASS the RIGHT arm grabs (perception emit from object_detection.py) -- the grab target ---------
GLASS_JSON = "/tmp/glass_pose.json"
def _load_glass():
    try:
        d = json.load(open(GLASS_JSON)); c = d["center"]
        # +Z_LIFT: grab the glass at the height we actually use on the real robot (depth reads a touch low).
        return float(c[0]), float(c[1]), float(c[2]) + Z_LIFT, float(d["radius_m"]), float(d["height_m"]), f"perception (/tmp/glass_pose.json) +{Z_LIFT*100:.0f}cm"
    except Exception:
        return 0.426, -0.281, 0.064 + Z_LIFT, 0.05, 0.109, f"fallback (no /tmp/glass_pose.json) +{Z_LIFT*100:.0f}cm"
GX, GY, GZ, GR, GH, GLASS_SRC = _load_glass()


def qmul(a, b):
    aw,ax,ay,az = a; bw,bx,by,bz = b
    return [aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
            aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw]
def euler_quat(yaw_deg, pitch_deg):
    y = np.deg2rad(yaw_deg); p = np.deg2rad(pitch_deg)
    qz = [np.cos(y/2), 0, 0, np.sin(y/2)]; qy = [np.cos(p/2), 0, np.sin(p/2), 0]
    return [float(v) for v in qmul(qz, qy)]
def quat_rotate(q, v):
    w, x, y, z = q
    return qmul(qmul(q, [0.0, v[0], v[1], v[2]]), [w, -x, -y, -z])[1:]


def main():
    import torch, trimesh
    import sim_grasp_viz as SV                        # grasp constants (SG_DX/SG_DYGAP/FG_*, BRING_BACK_*, HAND_LINKS)
    from curobo.viewer import ViserVisualizer
    from curobo.scene import Scene, Cuboid, Cylinder, Mesh
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import ContentPath, GoalToolPose, JointState, Pose
    from curobo._src.state.state_joint_trajectory_ops import get_joint_state_at_horizon_index

    print(f"[microwave] pose source: {MW_SRC}", flush=True)
    print(f"[microwave] body center=({MW_CX:.3f},{MW_CY:.3f},{MW_CZ:.3f}) dims(L,W,H)={tuple(round(d*100,1) for d in MW_DIMS)}cm  table top z={TABLE_TOP_Z:.3f}", flush=True)
    print(f"[microwave] door front x={MW_FRONT_X:.3f}  hinge=({HINGE_XYZ[0]:.3f},{HINGE_XYZ[1]:.3f})  handle front x={HANDLE_FRONT_X:.3f} y={MW_HANDLE_Y:.3f}  handle centre z={HANDLE_Z:.3f} (= body centre)", flush=True)

    print("building viewer + LEFT-arm planner ...", flush=True)
    viz = ViserVisualizer(content_path=ContentPath(robot_config_absolute_path=OUT),
                          connect_ip="0.0.0.0", connect_port=PORT,
                          add_control_frames=True, visualize_robot_spheres=False)

    # LEFT-arm planner: active chain = LEFT arm, RIGHT arm LOCKED at relaxed (same as sim_grasp_viz.py)
    import yaml as _yaml
    _lc = _yaml.safe_load(open(OUT)); _lk = _lc["kinematics"]; _lk["tool_frames"] = ["left_wrist_yaw_link"]
    for _j in LARM_JOINTS: _lk["lock_joints"].pop(_j, None)
    _lk["lock_joints"].update(RIGHT_FROZEN)
    _yaml.safe_dump(_lc, open("/tmp/g1_door_left.yml", "w"))
    planner_L = MotionPlanner(MotionPlannerCfg.create(robot="/tmp/g1_door_left.yml", scene_model=None,
                                                      max_goalset=8, collision_cache={"obb": 12, "mesh": 4}))
    TABLE_C = Cuboid(name="table", dims=TABLE[0], pose=TABLE[1])

    def mw_body_walls():    # HOLLOW microwave (same model as sim_grasp_viz.py): insulation walls + solid button block
        hxw, hyw, hzw = MW_DIMS[0]/2, MW_DIMS[1]/2, MW_DIMS[2]/2
        W = MW_WALL; RS = MW_WALL + MW_BTN_W
        return [Cuboid(name="mw_back",   dims=[W, MW_DIMS[1], MW_DIMS[2]], pose=[MW_CX+hxw-W/2, MW_CY, MW_CZ, 1,0,0,0]),
                Cuboid(name="mw_left",   dims=[MW_DIMS[0], W, MW_DIMS[2]], pose=[MW_CX, MW_CY+hyw-W/2, MW_CZ, 1,0,0,0]),
                Cuboid(name="mw_right",  dims=[MW_DIMS[0], RS, MW_DIMS[2]], pose=[MW_CX, MW_CY-hyw+RS/2, MW_CZ, 1,0,0,0]),
                Cuboid(name="mw_top",    dims=[MW_DIMS[0], MW_DIMS[1], W], pose=[MW_CX, MW_CY, MW_CZ+hzw-W/2, 1,0,0,0]),
                Cuboid(name="mw_bottom", dims=[MW_DIMS[0], MW_DIMS[1], W], pose=[MW_CX, MW_CY, MW_CZ-hzw+W/2, 1,0,0,0])]
    def mw_handle_obs():    # handle bar (protruding 3.8cm) as collision, so the LEFT fingers don't clip it on the reach
        return [Cuboid(name="mw_handle", dims=[HANDLE_PROTRUDE, 0.03, HANDLE_LEN],
                       pose=[HANDLE_FRONT_X + HANDLE_PROTRUDE/2, MW_HANDLE_Y, HANDLE_Z, 1,0,0,0])]
    def mw_scene_L(with_mw):
        return Scene(cuboid=[TABLE_C] + (mw_body_walls() + mw_handle_obs() if with_mw else []))

    for _wc in mw_body_walls() + mw_handle_obs():     # collision surfaces in translucent RED
        viz._server.scene.add_box("/mw_collision/" + _wc.name, color=(235, 25, 25), dimensions=tuple(_wc.dims),
                                  position=tuple(_wc.pose[:3]), wxyz=(1, 0, 0, 0), opacity=0.55)
    planner_L.update_world(mw_scene_L(True))
    print("warming up LEFT planner ...", flush=True)
    planner_L.warmup(enable_graph=True, num_warmup_iterations=5)

    # ---- RIGHT-arm planner + the GLASS to grab (identical setup to real_plan_door.py's grab block) ----
    print(f"[glass] pose source: {GLASS_SRC}", flush=True)
    print(f"[glass] center=({GX:.3f},{GY:.3f},{GZ:.3f}) r={GR*100:.1f}cm h={GH*100:.1f}cm", flush=True)
    # GLASS-level table: the glass rests on the real surface (its bottom = table top). The microwave-derived
    # table reads too high (perceived handle z = MW_CZ sits above the body centre), so use the glass's own
    # level for the right arm -- otherwise the table would bury the glass and every grasp returns None.
    _gt_top  = GZ - GH / 2.0 - GRASP_TABLE_GAP; _gt_near = 0.32   # table sits GAP below the glass bottom (no glass-table collision)
    _gt_dx   = max(0.70, max(GX + 0.25, MW_CX + MW_DIMS[0] / 2 + 0.10) - _gt_near)
    GTABLE   = Cuboid(name="table", dims=[_gt_dx, 1.2, 0.6], pose=[_gt_near + _gt_dx / 2.0, -0.05, _gt_top - 0.005 - 0.3, 1, 0, 0, 0])
    planner_R = MotionPlanner(MotionPlannerCfg.create(robot=OUT, scene_model=None,
                                                      max_goalset=8, collision_cache={"obb": 8, "mesh": 4}))
    _gtri = Cylinder(name="object", radius=GR, height=GH, pose=[GX, GY, GZ, 1, 0, 0, 0]).get_trimesh_mesh()
    glass_mesh = Mesh(name="object", vertices=_gtri.vertices.tolist(), faces=_gtri.faces.tolist(), pose=[GX, GY, GZ, 1, 0, 0, 0])
    def grasp_world(with_glass=True):                 # grasp: table (+glass). put-in/pull-out: table only (may graze)
        return Scene(cuboid=[GTABLE], mesh=[glass_mesh]) if with_glass else Scene(cuboid=[GTABLE])
    # BRING-BACK is collision-aware against the microwave RIGHT wall (button-panel block), padded -y, so the
    # in-hand glass does NOT touch the unit on the way to staging (same method as _test_carry.py). The full box
    # over-constrains the right arm, so only the right wall is used; put-in/pull-out stay table-only (may graze).
    MW_RIGHT_PAD = 0.01
    _hyw = MW_DIMS[1] / 2.0; _RSp = MW_WALL + MW_BTN_W + MW_RIGHT_PAD
    MW_RIGHT_R = Cuboid(name="mw_right", dims=[MW_DIMS[0], _RSp, MW_DIMS[2]], pose=[MW_CX, MW_CY - _hyw + _RSp/2, MW_CZ, 1, 0, 0, 0])
    def bringback_world():                            # table + padded microwave right wall (the surface the glass would graze)
        return Scene(cuboid=[GTABLE, MW_RIGHT_R])
    planner_R.update_world(grasp_world(True))
    print("warming up RIGHT planner ...", flush=True)
    planner_R.warmup(enable_graph=True, num_warmup_iterations=5)

    VJ = list(viz.joint_names)
    LARM_SET = set(LARM_JOINTS)
    def render_full(arm_map, hand_map):
        d = {n: 0.0 for n in VJ}; d.update(arm_map); d.update(hand_map)
        viz.set_joint_state(JointState.from_position(
            torch.tensor([[d.get(n, 0.0) for n in VJ]], device="cuda", dtype=torch.float32), joint_names=list(VJ)))
    is_moving = False
    server = viz._server
    render_full(RELAXED_ARM, {**LHAND_OPEN, **HAND_OPEN})

    # ---- microwave VIZ boxes (body shell, button panel, door, handle) -------------------------------
    server.scene.add_box("/microwave/body", color=(120,122,130), dimensions=MW_DIMS,
                         position=(MW_CX, MW_CY, MW_CZ), wxyz=(1,0,0,0), opacity=0.15)
    server.scene.add_box("/microwave/buttons", color=(30,32,38), dimensions=(MW_DIMS[0], MW_BTN_W, MW_DIMS[2]),
                         position=(MW_CX, MW_CY - MW_DIMS[1]/2 + MW_BTN_W/2, MW_CZ), wxyz=(1,0,0,0), opacity=0.92)
    DOOR_W  = MW_HINGE_Y - MW_HANDLE_Y
    DOOR_CY = (MW_HINGE_Y + MW_HANDLE_Y) / 2.0
    DOOR_REST_POS   = (MW_FRONT_X - DOOR_THICK/2.0, DOOR_CY, MW_CZ)
    HANDLE_REST_POS = (HANDLE_FRONT_X + HANDLE_PROTRUDE/2.0, MW_HANDLE_Y, HANDLE_Z)   # bar centre (= body centre z; matches collision)
    door_box   = server.scene.add_box("/microwave/door", color=(55,58,68),
                         dimensions=(DOOR_THICK, DOOR_W*0.98, MW_DIMS[2]*0.88), position=DOOR_REST_POS, wxyz=(1,0,0,0), opacity=0.9)
    handle_box = server.scene.add_box("/microwave/handle", color=(210,210,215),
                         dimensions=(HANDLE_PROTRUDE, 0.03, HANDLE_LEN), position=HANDLE_REST_POS, wxyz=(1,0,0,0))
    server.scene.add_frame("/microwave/hinge", axes_length=MW_DIMS[2], axes_radius=0.006, position=HINGE_XYZ, wxyz=(1,0,0,0))

    # ---- TABLE surfaces (rendered so you can SEE the +Z_LIFT) -----------------------------------------
    # The planner uses two tables: a LEFT/door table (microwave-derived) and a RIGHT/glass table
    # (glass-derived, a bit lower because the perceived handle z sits above the true body centre). Both
    # ride +Z_LIFT, so the handle/glass rest ON these tops. Shown as thin surfaces at the true top z
    # (the real collision boxes are 0.6 m thick slabs going down out of view).
    _dt_dims, _dt_pose = TABLE
    server.scene.add_box("/tables/door_left (microwave-derived)", color=(140,110,70),
                         dimensions=(_dt_dims[0], _dt_dims[1], 0.02), position=(_dt_pose[0], _dt_pose[1], TABLE_TOP_Z - 0.01),
                         wxyz=(1,0,0,0), opacity=0.30)
    server.scene.add_box("/tables/glass_right (glass-derived)", color=(180,150,95),
                         dimensions=(GTABLE.dims[0], GTABLE.dims[1], 0.02), position=(GTABLE.pose[0], GTABLE.pose[1], _gt_top - 0.01),
                         wxyz=(1,0,0,0), opacity=0.45)
    print(f"[table] door/left top z={TABLE_TOP_Z:.3f} | glass/right top z={_gt_top:.3f} | "
          f"handle z={HANDLE_Z:.3f} glass z={GZ:.3f}  (all include +{Z_LIFT*100:.0f}cm lift)", flush=True)

    # ---- GREEN glass viz (cylinder) -- a moveable node so it can ride along on the bring-back ----------
    _gmesh_viz = trimesh.creation.cylinder(radius=GR, height=GH, sections=28)
    _gmesh_viz.visual.vertex_colors = np.array([40, 170, 60, 255], np.uint8)
    glass_node = server.scene.add_mesh_trimesh("/glass", _gmesh_viz, position=(GX, GY, GZ), wxyz=(1, 0, 0, 0))

    # ---- APPROACH-TUNING sliders --------------------------------------------------------------------
    sl_yaw    = server.gui.add_slider("DOOR wrist yaw (deg)",          min=-180.0, max=180.0, step=5.0,  initial_value=DOOR_WRIST_YAW)
    sl_gz     = server.gui.add_slider("HANDLE grasp Z (m)",            min=MW_CZ-0.10, max=MW_CZ+0.10, step=0.005, initial_value=HANDLE_Z)
    sl_dy     = server.gui.add_slider("APPROACH pre-pose +y offset (FT1)", min=0.0, max=0.15, step=0.005, initial_value=DOOR_INSERT_DY)
    sl_slide  = server.gui.add_slider("SLIDE y past handle (- = deeper -y)", min=-0.08, max=0.04, step=0.005, initial_value=0.0)
    sl_gap    = server.gui.add_slider("APPROACH gap push (+x past handle front)", min=-0.02, max=0.06, step=0.004, initial_value=DOOR_GAP_DX)
    sl_track  = server.gui.add_slider("DOOR yaw-track (wrist roll w/ swing)", min=0.0, max=1.0, step=0.05, initial_value=DOOR_YAW_TRACK)
    sl_open   = server.gui.add_slider("DOOR open target (deg)",        min=0.0,  max=120.0, step=5.0,  initial_value=DOOR_OPEN_DEG)
    # ---- GLASS-grasp tuning (x/y/z of where the RIGHT hand grabs the glass) -- live, like the door knobs ----
    sl_ggx    = server.gui.add_slider("GLASS grasp X nudge (+ = more ahead/+x)", min=-0.06, max=0.06, step=0.005, initial_value=0.015)
    sl_ggy    = server.gui.add_slider("GLASS grasp Y nudge (+ = more right/-y)", min=-0.06, max=0.10, step=0.005, initial_value=0.01)
    _ggz_min, _ggz_max = round(GZ - GH/2, 3), round(GZ + GH/2 + 0.04, 3)   # range tracks THIS glass's height
    sl_ggz    = server.gui.add_slider("GLASS grasp Z (m, grip height)", min=_ggz_min, max=_ggz_max, step=0.005,
                                      initial_value=round(min(max(0.08, _ggz_min), _ggz_max), 3))  # default 0.08, clamped to this glass's range
    # ---- BRING-BACK staging pose (glass centre after the grasp) -- live, = real_plan_door --bring-back-x/y/z ----
    sl_bbx    = server.gui.add_slider("BRINGBACK X (m)", min=0.15, max=0.50, step=0.005, initial_value=round(SV.BRING_BACK_XYZ[0], 3))
    sl_bby    = server.gui.add_slider("BRINGBACK Y (m)", min=-0.30, max=0.15, step=0.005, initial_value=round(SV.BRING_BACK_XYZ[1], 3))
    sl_bbz    = server.gui.add_slider("BRINGBACK Z (m)", min=0.05, max=0.30, step=0.005, initial_value=round(SV.BRING_BACK_XYZ[2] + Z_LIFT, 3))
    # ---- PUT-IN: glass slides from staging into the cavity along a line tilted PLACE angle, by PLACE dist ----
    sl_pa     = server.gui.add_slider("PUT-IN angle (deg, +x->+y)", min=0.0, max=90.0, step=2.0, initial_value=round(SV.PLACE_IN_ANGLE, 1))
    sl_pd     = server.gui.add_slider("PUT-IN distance (m)", min=0.0, max=0.40, step=0.01, initial_value=round(SV.PLACE_IN_DIST, 2))

    def trim_last(P):
        Hn = P.shape[0]
        if Hn <= 1: return Hn
        d = np.linalg.norm(np.diff(P, axis=0), axis=1); mov = np.where(d > 1e-5)[0]
        return min(int(mov[-1] + 2) if len(mov) else Hn, Hn)
    def ok(res): return res is not None and res.success is not None and bool(res.success.any())
    def run_async(fn):
        nonlocal is_moving
        if is_moving: return
        def work():
            nonlocal is_moving; is_moving = True
            try: fn()
            except Exception: traceback.print_exc()
            is_moving = False
        threading.Thread(target=work, daemon=True).start()

    def rot_about_hinge(px, py, phi):
        c, s = np.cos(phi), np.sin(phi); rx, ry = px - HINGE_XYZ[0], py - HINGE_XYZ[1]
        return HINGE_XYZ[0] + c*rx - s*ry, HINGE_XYZ[1] + s*rx + c*ry
    def set_door_angle(phi):
        qz = (float(np.cos(phi/2)), 0.0, 0.0, float(np.sin(phi/2)))
        for box, rest in ((door_box, DOOR_REST_POS), (handle_box, HANDLE_REST_POS)):
            x, y = rot_about_hinge(rest[0], rest[1], phi)
            box.position = (x, y, rest[2]); box.wxyz = qz
    def relaxed_qstart_L():
        full = planner_L.default_joint_state.clone(); jn = list(full.joint_names)
        pos = full.position.clone().reshape(-1)
        for name, val in RELAXED_ARM.items():
            if name in jn: pos[jn.index(name)] = float(val)
        return planner_L.kinematics.get_active_js(JointState.from_position(pos.unsqueeze(0), joint_names=jn))
    def goal_L(pos3, quat):
        return GoalToolPose(tool_frames=planner_L.tool_frames,
            position=torch.tensor([[[[list(pos3)]]]], device="cuda", dtype=torch.float32),
            quaternion=torch.tensor([[[[list(quat)]]]], device="cuda", dtype=torch.float32))
    def door_wrist_L(ft_xyz, yaw_deg, pitch_deg, offset=FINGER_OFFSET):
        Q = euler_quat(yaw_deg, pitch_deg); off = quat_rotate(Q, offset)
        return (ft_xyz[0] - off[0], ft_xyz[1] - off[1], ft_xyz[2] - off[2]), Q

    # ---- RIGHT-arm grasp helpers (mirror real_plan_door.py's grab block / sim_grasp_viz full_sequence) ----
    RARM_SET = set(["right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
                    "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"])
    GDX, GGAP, GAP_AP, GPITCH = SV.SG_DX, SV.SG_DYGAP, SV.FG_APPROACH, 0.0   # x/y/z now come from the sl_gg* sliders
    def relaxed_qstart_R():
        full = planner_R.default_joint_state.clone(); jn = list(full.joint_names); pos = full.position.clone().reshape(-1)
        for name, val in RELAXED_ARM.items():
            if name in jn: pos[jn.index(name)] = float(val)
        return planner_R.kinematics.get_active_js(JointState.from_position(pos.unsqueeze(0), joint_names=jn))
    def goal_R(pos3, quat):
        return GoalToolPose(tool_frames=planner_R.tool_frames,
            position=torch.tensor([[[[list(pos3)]]]], device="cuda", dtype=torch.float32),
            quaternion=torch.tensor([[[[list(quat)]]]], device="cuda", dtype=torch.float32))
    def grasp_wrist(yaw):                              # wrist pose = side grasp rotated `yaw` about the glass axis
        th = np.deg2rad(yaw); c, s = np.cos(th), np.sin(th); ox, oy = -GDX, -(GR + GGAP)
        return (GX + (c*ox - s*oy) + sl_ggx.value, GY + (s*ox + c*oy) - sl_ggy.value, sl_ggz.value)  # x/y nudge + abs grip Z

    # ===== APPROACH + OPEN (LEFT arm) — identical method to sim_grasp_viz.py's open_door_left =====
    # Returns (LQ, reached_deg): the final LEFT-arm joint map (door held open) + signed swing angle, or None on fail.
    def _door_open_core():
        frames = []                                            # (larm_map, door_deg) per step -> replayed REVERSED to close the door
        def render_L(larm, hand): render_full({**RIGHT_FROZEN, **larm}, hand)
        def render_traj(res, dd=0.0):
            t = res.interpolated_trajectory.squeeze(0); jn = list(t.joint_names)
            P = t.position[0].detach().cpu().numpy(); last = trim_last(P)
            for i in np.unique(np.linspace(0, last - 1, min(last, 60)).astype(int)):
                if not is_moving: return False
                lm = {a: float(P[i][jn.index(a)]) for a in jn if a in LARM_SET}
                render_L(lm, LHAND_OPEN); frames.append((lm, dd)); time.sleep(0.025)
            return True
        set_door_angle(0.0); render_full(RELAXED_ARM, {**LHAND_OPEN, **HAND_OPEN})
        LQ = {a: float(RELAXED_ARM.get(a, 0.0)) for a in LARM_SET}    # left-arm hold (starts relaxed)
        yaw = sl_yaw.value                                       # live-tunable approach knobs
        gz = sl_gz.value; gap = sl_gap.value; insert_dy = sl_dy.value; track = sl_track.value; slide_y = sl_slide.value
        ftx, ftz = HANDLE_FRONT_X + gap, gz
        fty = MW_HANDLE_Y + slide_y           # SLIDE target + swing contact (slide_y<0 = deeper -y, past the handle)
        FT1 = (ftx, MW_HANDLE_Y + insert_dy, ftz)   # phase 1: pre-pose +y of the HANDLE (independent of the slide)
        FT2 = (ftx, fty, ftz)                 # phase 2: slide -y to the (possibly shifted) contact, fingers into the gap
        # phase 1: approach to the LEFT of the handle -- microwave collision ON (don't go through the body)
        planner_L.update_world(mw_scene_L(True))
        w1, q1 = door_wrist_L(FT1, yaw, DOOR_PITCH)
        r1 = planner_L.plan_pose(goal_L(w1, q1), relaxed_qstart_L(), max_attempts=6)
        print(f"\n[door] APPROACH phase1 (left of handle) ft=({FT1[0]:.3f},{FT1[1]:.3f},{FT1[2]:.3f}) -> {'OK' if ok(r1) else 'FAILED'} {getattr(r1,'status',None)}")
        if not ok(r1) or not render_traj(r1): return None
        seed = planner_L.kinematics.get_active_js(get_joint_state_at_horizon_index(r1.js_solution, -1).squeeze(0))
        # phase 2: slide straight -y into the gap -- microwave collision OFF (fingers enter the gap; swing keeps it off)
        planner_L.update_world(mw_scene_L(False))
        w2, q2 = door_wrist_L(FT2, yaw, DOOR_PITCH)
        r2 = planner_L.plan_pose(goal_L(w2, q2), seed, max_attempts=6)
        print(f"[door] APPROACH phase2 (slide -y into gap) ft=({FT2[0]:.3f},{FT2[1]:.3f},{FT2[2]:.3f}) -> {'OK' if ok(r2) else 'FAILED'} {getattr(r2,'status',None)}")
        if not ok(r2) or not render_traj(r2): return None
        print("  fingers inserted into the gap behind the handle (open hand).")
        # phase 3: OPEN the door -- fingertip follows the handle ARC about the hinge; wrist yaw rolls PARTIALLY
        seed = planner_L.kinematics.get_active_js(get_joint_state_at_horizon_index(r2.js_solution, -1).squeeze(0))
        open_deg = sl_open.value; N = max(1, int(round(open_deg / 8.0))); reached = 0.0
        for i in range(1, N + 1):
            phid = -open_deg * (i / N); phi = np.deg2rad(phid)
            fx, fy = rot_about_hinge(ftx, fty, phi)
            wk, qk = door_wrist_L((fx, fy, ftz), yaw + track * phid, DOOR_PITCH)
            rk = planner_L.plan_pose(goal_L(wk, qk), seed, max_attempts=4)
            if not ok(rk):
                print(f"  door stuck at {abs(reached):.0f} deg (next {abs(phid):.0f} deg unreachable)"); break
            t = rk.interpolated_trajectory.squeeze(0); jn = list(t.joint_names)
            P = t.position[0].detach().cpu().numpy(); last = trim_last(P)
            idxs = np.unique(np.linspace(0, last - 1, min(last, 40)).astype(int))
            for k, fi in enumerate(idxs):
                if not is_moving: return None
                lm = {a: float(P[fi][jn.index(a)]) for a in jn if a in LARM_SET}
                render_L(lm, LHAND_OPEN)
                dd = reached + (phid - reached) * (k + 1) / len(idxs)
                set_door_angle(np.deg2rad(dd)); frames.append((lm, dd)); time.sleep(0.03)
            reached = phid
            LQ = {a: float(P[last - 1][jn.index(a)]) for a in jn if a in LARM_SET}    # hold the left arm here
            seed = planner_L.kinematics.get_active_js(get_joint_state_at_horizon_index(rk.js_solution, -1).squeeze(0))
        print(f"  DOOR OPENED to {abs(reached):.0f} deg (wrist yaw rolled {yaw:.0f} -> {yaw + track*reached:.0f}).")
        return LQ, reached, frames

    def open_door_left():
        run_async(_door_open_core)

    # ===== GRAB the glass with the RIGHT arm, then BRING it back to BRING_BACK_XYZ — NO insert =====
    # Mirrors real_plan_door.py's grab block: BestAngle yaw sweep -> approach -> grasp -> close -> bring back.
    # `hold_larm` keeps the LEFT arm where the door-open left it; `door_phi` keeps the door at that angle.
    def _grasp_core(hold_larm, door_phi, do_insert=False):
        set_door_angle(np.deg2rad(door_phi))
        glass_node.position = (GX, GY, GZ)                  # reset the glass to the table for this run
        def render_R(rarm, rhand): render_full({**hold_larm, **rarm}, {**LHAND_OPEN, **rhand})
        def ride_glass(p0, p1, fr):                         # interpolate the glass node from centre p0 to p1
            try: glass_node.position = (p0[0]+(p1[0]-p0[0])*fr, p0[1]+(p1[1]-p0[1])*fr, p0[2]+(p1[2]-p0[2])*fr)
            except Exception: pass
        def play_seg(seg, rhand):
            t = seg.squeeze(0); jn = list(t.joint_names)
            P = t.position[0].detach().cpu().numpy(); last = trim_last(P)
            for i in np.unique(np.linspace(0, last - 1, min(last, 60)).astype(int)):
                if not is_moving: return False
                render_R({a: float(P[i][jn.index(a)]) for a in jn if a in RARM_SET}, rhand); time.sleep(0.025)
            return True
        def play_pose(res, rhand, gfrom=None, gto=None):    # animate a plan_pose/cspace result; optionally ride the glass along
            t = res.interpolated_trajectory.squeeze(0); jn = list(t.joint_names)
            P = t.position[0].detach().cpu().numpy(); last = trim_last(P)
            ix = np.unique(np.linspace(0, last - 1, min(last, 60)).astype(int))
            for k, i in enumerate(ix):
                if not is_moving: return False
                render_R({a: float(P[i][jn.index(a)]) for a in jn if a in RARM_SET}, rhand)
                if gfrom is not None: ride_glass(gfrom, gto, (k + 1) / len(ix))
                time.sleep(0.03)
            return True
        def play_pose_rev(res, rhand):                      # animate a plan_pose result BACKWARDS (retrace out)
            t = res.interpolated_trajectory.squeeze(0); jn = list(t.joint_names)
            P = t.position[0].detach().cpu().numpy(); last = trim_last(P)
            for i in reversed(np.unique(np.linspace(0, last - 1, min(last, 60)).astype(int))):
                if not is_moving: return False
                render_R({a: float(P[i][jn.index(a)]) for a in jn if a in RARM_SET}, rhand); time.sleep(0.03)
            return True
        def arm_map_of(seg):
            t = seg.squeeze(0); jn = list(t.joint_names); P = t.position[0].detach().cpu().numpy(); last = trim_last(P)
            return {a: float(P[last - 1][jn.index(a)]) for a in jn if a in RARM_SET}
        render_R({}, HAND_OPEN)
        # BestAngle grasp: most-front reachable yaw (glass is on the robot's right, approach along +tool-y)
        planner_R.update_world(grasp_world(True))
        _gw = grasp_wrist(0)
        print(f"[grab] glass=({GX:.3f},{GY:.3f},{GZ:.3f}) tuned wrist xyz=({_gw[0]:.3f},{_gw[1]:.3f},{_gw[2]:.3f}) "
              f"[Xnudge={sl_ggx.value:+.3f} Ynudge={sl_ggy.value:+.3f} gripZ={sl_ggz.value:.3f}]")
        res = None; gyaw = None
        for yaw in [-90, -60, -30, 0]:
            try:
                r = planner_R.plan_grasp(goal_R(grasp_wrist(yaw), euler_quat(yaw, GPITCH)), relaxed_qstart_R(),
                                         grasp_approach_axis="y", grasp_approach_offset=GAP_AP, grasp_approach_in_tool_frame=True,
                                         grasp_lift_axis="z", grasp_lift_offset=0.15, grasp_lift_in_tool_frame=False,
                                         plan_approach_to_grasp=True, plan_grasp_to_lift=True, disable_collision_links=SV.HAND_LINKS)
            except Exception as e:                       # cuRobo plan_grasp can raise on some poses (goalset-index edge case); skip this yaw
                print(f"[grab] grasp yaw={yaw} -> cuRobo error ({type(e).__name__}); skipping"); continue
            if ok(r): res = r; gyaw = yaw; print(f"[grab] BestAngle yaw={yaw} -> OK"); break
        if res is None:
            print("[grab] grasp FAILED at all yaws (tune the glass pose / grasp constants)."); return False
        if not play_seg(res.approach_interpolated_trajectory, HAND_OPEN): return False   # relaxed -> pre-grasp
        if not play_seg(res.grasp_interpolated_trajectory, HAND_OPEN): return False      # pre-grasp -> grasp
        # close the inspire hand: full thumb rotation/opposition first, THEN curl the fingers (two-step, like the real hand)
        arm = arm_map_of(res.grasp_interpolated_trajectory); tgt = {k: HAND_CLOSED[k] * GRIP_FRAC for k in HAND_JOINTS}
        for fr in np.linspace(0, 1, 10):
            if not is_moving: return False
            render_R(arm, {k: (fr * tgt[k] if 'thumb' in k else 0.0) for k in HAND_JOINTS}); time.sleep(0.03)
        for fr in np.linspace(0, 1, 12):
            if not is_moving: return False
            render_R(arm, {k: (tgt[k] if 'thumb' in k else fr * tgt[k]) for k in HAND_JOINTS}); time.sleep(0.03)
        closedR = {k: tgt[k] for k in HAND_JOINTS}
        gseed = planner_R.kinematics.get_active_js(get_joint_state_at_horizon_index(res.grasp_trajectory, -1).squeeze(0))
        print(f"[grab] glass grasped (yaw={gyaw}).")
        # bring the grasped glass to the staging pose -- COLLISION-AWARE vs the microwave right wall (must not touch)
        planner_R.update_world(bringback_world())
        BBYAW = SV.BRING_BACK_YAW
        BBX, BBY, BBZ = sl_bbx.value, sl_bby.value, sl_bbz.value     # staging target (live sliders)
        fromtop = (GZ + GH/2.0) - sl_ggz.value                      # grip-height-below-top implied by the Z slider
        c, s = np.cos(np.deg2rad(BBYAW)), np.sin(np.deg2rad(BBYAW)); ox, oy = -GDX, -(GR + GGAP)
        def rwrist(cx, cy, cz):                                     # wrist pose for a glass-CENTRE target (same yaw/offsets as the grasp)
            return (cx + (c*ox - s*oy) + sl_ggx.value, cy + (s*ox + c*oy) - sl_ggy.value, cz + (GH/2.0 - fromtop))
        rp = planner_R.plan_pose(goal_R(rwrist(BBX, BBY, BBZ), euler_quat(BBYAW, 0)), gseed, max_attempts=8)
        print(f"[grab] bring-back to ({BBX:.3f},{BBY:.3f},{BBZ:.3f}) -> {'OK' if ok(rp) else 'FAILED'}")
        if not ok(rp):
            planner_R.update_world(grasp_world(True)); return False
        if not play_pose(rp, closedR, (GX, GY, GZ), (BBX, BBY, BBZ)): return False
        print("[grab] glass brought back to the staging pose (held).")
        if not do_insert:
            planner_R.update_world(grasp_world(True)); return True
        # ===== PUT-IN: slide the glass into the cavity along the PLACE line, release, retract to relaxed =====
        planner_R.update_world(grasp_world(False))                  # put-in/pull-out: table only (may graze -- that's OK)
        sBB = planner_R.kinematics.get_active_js(get_joint_state_at_horizon_index(rp.js_solution, -1).squeeze(0))
        PA, PD = sl_pa.value, sl_pd.value
        ca, sa = np.cos(np.deg2rad(PA)), np.sin(np.deg2rad(PA))
        ig = (BBX + PD*ca, BBY + PD*sa, BBZ)                        # inserted glass-centre target
        ri = planner_R.plan_pose(goal_R(rwrist(*ig), euler_quat(BBYAW, 0)), sBB, max_attempts=8)
        print(f"[grab] PUT-IN {PA:.0f}deg x {PD:.2f}m -> ({ig[0]:.3f},{ig[1]:.3f},{ig[2]:.3f}) -> {'OK' if ok(ri) else 'FAILED'}")
        if not ok(ri):
            print("[grab] insert FAILED (tune PUT-IN angle/distance / BRINGBACK x/y/z)."); planner_R.update_world(grasp_world(True)); return False
        if not play_pose(ri, closedR, (BBX, BBY, BBZ), ig): return False
        for fr in np.linspace(1, 0, 12):                            # release: open the fingers (glass stays in the cavity)
            if not is_moving: return False
            render_R(arm_map_of(ri.interpolated_trajectory), {k: closedR[k] * fr for k in HAND_JOINTS}); time.sleep(0.03)
        print("[grab] released; glass left in the microwave.")
        if not play_pose_rev(ri, HAND_OPEN): return False           # retrace the insert OUT (inserted -> staging)
        rr = planner_R.plan_cspace(relaxed_qstart_R(), sBB, max_attempts=10)   # staging -> relaxed (open space)
        print(f"[grab] retract right arm to relaxed -> {'OK' if ok(rr) else 'FAILED'}")
        if not ok(rr): return False
        if not play_pose(rr, HAND_OPEN): return False
        print("[grab] right arm back at relaxed (hand open).")
        planner_R.update_world(grasp_world(True))        # restore the glass in the world for the next run
        return True

    def grasp_only():     # grab the glass with the door CLOSED + left arm relaxed (right-arm motion on its own; no put-in)
        run_async(lambda: _grasp_core({a: float(RELAXED_ARM.get(a, 0.0)) for a in LARM_SET}, 0.0, do_insert=False))

    def full_sequence():  # open door (L) -> grab + PUT-IN glass + retract (R) -> close door + relax (L) -- the whole motion
        def f():
            r = _door_open_core()
            if r is None: print("[FULL] door open failed -- skipping the rest."); return
            LQ, reached, frames = r
            print("[FULL] door open; LEFT holds it -> RIGHT grabs the glass and puts it in the microwave ...")
            if not _grasp_core(LQ, reached, do_insert=True):
                print("[FULL] grab/put-in failed -- leaving as is (left arm still holding the door)."); return
            # CLOSE the door: replay the door-open frames in REVERSE -- the left arm tracks the handle back,
            # the door swings shut, and the arm returns to relaxed. (Right arm already retracted to relaxed.)
            print("[FULL] closing the door (reverse of the open) + relaxing the left arm ...")
            for lm, dd in reversed(frames):
                if not is_moving: return
                render_full({**RIGHT_FROZEN, **lm}, {**LHAND_OPEN, **HAND_OPEN}); set_door_angle(np.deg2rad(dd)); time.sleep(0.02)
            set_door_angle(0.0); render_full(RELAXED_ARM, {**LHAND_OPEN, **HAND_OPEN})
            print("[FULL] DONE: glass in the microwave, door closed, both arms relaxed.")
        run_async(f)

    # ===== REPLAY the exported trajectory (real_plan_door.py JSON) — exactly what real_open_door.py runs ====
    # Plays BOTH halves: LEFT arm opens the door, then (if a "grab" block exists) the RIGHT arm grabs the
    # glass + brings it back -- the same two phases real_open_door.py executes on the robot.
    def replay_traj():
        def f():
            nonlocal is_moving
            try:
                T = json.load(open(TRAJ_JSON))
            except Exception as e:
                print(f"[replay] cannot load {TRAJ_JSON}: {e}  -- run real_plan_door.py first"); return
            pts = T["points"]; jn = T["joint_names"]; grab = T.get("grab")
            ngrab = (len(grab["approach"]) + len(grab["grasp"]) + len(grab["bringback"])) if grab else 0
            print(f"\n[replay] {TRAJ_JSON}: door {len(pts)} pts (opens to {T.get('reached_deg','?')} deg)"
                  f"{f' + grab {ngrab} pts' if grab else ' (door only -- planned with --no-grab or grasp failed)'}")
            gc = (grab.get("glass", {}).get("center") if grab else None) or [GX, GY, GZ]
            bb = (grab.get("bring_back_xyz") if grab else None) or list(SV.BRING_BACK_XYZ)
            set_door_angle(0.0); render_full(RELAXED_ARM, {**LHAND_OPEN, **HAND_OPEN}); glass_node.position = tuple(gc)
            # ---- phase A: LEFT arm opens the door (RIGHT arm frozen at relaxed) ----
            for p in pts:
                if not is_moving: return
                larm = {jn[k]: float(p["positions"][k]) for k in range(len(jn))}
                render_full({**RIGHT_FROZEN, **larm}, {**LHAND_OPEN, **HAND_OPEN})
                set_door_angle(np.deg2rad(float(p.get("door_deg", 0.0))))   # signed: negative swings toward the robot
                time.sleep(0.03)
            door_deg = float(pts[-1].get("door_deg", 0.0)) if pts else 0.0
            print(f"[replay] door at {abs(door_deg):.0f} deg.")
            if not grab: return
            # ---- phase B: RIGHT arm grabs the glass (LEFT held at left_hold, door stays open) ----
            rjn = grab["joint_names"]; hold = {jn[k]: float(T["left_hold"][k]) for k in range(len(jn))}
            def render_R(rarm, rhand): render_full({**hold, **rarm}, {**LHAND_OPEN, **rhand})
            def play(seg, rhand):
                for p in seg:
                    if not is_moving: return False
                    render_R({rjn[k]: float(p["positions"][k]) for k in range(len(rjn))}, rhand); time.sleep(0.03)
                return True
            if not play(grab["approach"], HAND_OPEN): return        # relaxed -> pre-grasp
            if not play(grab["grasp"], HAND_OPEN): return           # pre-grasp -> grasp
            arm = {rjn[k]: float(grab["grasp"][-1]["positions"][k]) for k in range(len(rjn))}   # hold at the grasp pose
            tgt = {k: HAND_CLOSED[k] * GRIP_FRAC for k in HAND_JOINTS}                          # close: thumb rotate, then curl
            for fr in np.linspace(0, 1, 10):
                if not is_moving: return
                render_R(arm, {k: (fr * tgt[k] if 'thumb' in k else 0.0) for k in HAND_JOINTS}); time.sleep(0.03)
            for fr in np.linspace(0, 1, 12):
                if not is_moving: return
                render_R(arm, {k: (tgt[k] if 'thumb' in k else fr * tgt[k]) for k in HAND_JOINTS}); time.sleep(0.03)
            closedR = {k: tgt[k] for k in HAND_JOINTS}
            print(f"[replay] glass grasped (yaw={grab.get('grasp_yaw','?')}); bringing it back.")
            bbk = grab["bringback"]
            for i, p in enumerate(bbk):                             # bring-back; the glass rides along in the hand
                if not is_moving: return
                render_R({rjn[k]: float(p["positions"][k]) for k in range(len(rjn))}, closedR)
                fr = (i + 1) / max(1, len(bbk))
                try: glass_node.position = (gc[0] + (bb[0]-gc[0])*fr, gc[1] + (bb[1]-gc[1])*fr, gc[2] + (bb[2]-gc[2])*fr)
                except Exception: pass
                time.sleep(0.03)
            print(f"[replay] glass carried to staging {tuple(round(v,3) for v in bb)}.")
            if "insert" not in grab:
                print(f"[replay] done — door {abs(door_deg):.0f} deg (no put-in in this JSON)."); return
            # ---- phase C: PUT-IN the glass -> release -> retract the right arm to relaxed ----
            ins = grab["insert"]; ig = grab.get("insert_xyz", bb)
            for i, p in enumerate(ins):
                if not is_moving: return
                render_R({rjn[k]: float(p["positions"][k]) for k in range(len(rjn))}, closedR)
                fr = (i + 1) / max(1, len(ins))
                try: glass_node.position = (bb[0] + (ig[0]-bb[0])*fr, bb[1] + (ig[1]-bb[1])*fr, bb[2] + (ig[2]-bb[2])*fr)
                except Exception: pass
                time.sleep(0.03)
            insarm = {rjn[k]: float(ins[-1]["positions"][k]) for k in range(len(rjn))}
            for fr in np.linspace(1, 0, 12):                        # release: open the fingers (glass stays in the cavity)
                if not is_moving: return
                render_R(insarm, {k: closedR[k] * fr for k in HAND_JOINTS}); time.sleep(0.03)
            print("[replay] glass released in the microwave; retracting the right arm.")
            if not play(grab["retract"], HAND_OPEN): return         # inserted -> staging -> relaxed
            # ---- phase D: CLOSE the door with the LEFT arm (door_close = reverse of the open swing) ----
            dc = T.get("door_close")
            if dc:
                for p in dc:
                    if not is_moving: return
                    larm = {jn[k]: float(p["positions"][k]) for k in range(len(jn))}
                    render_full({**RIGHT_FROZEN, **larm}, {**LHAND_OPEN, **HAND_OPEN})
                    set_door_angle(np.deg2rad(float(p.get("door_deg", 0.0)))); time.sleep(0.03)
            set_door_angle(0.0); render_full(RELAXED_ARM, {**LHAND_OPEN, **HAND_OPEN})
            print("[replay] done — glass in microwave, door closed, both arms relaxed.")
        run_async(f)

    server.gui.add_button("Approach + Open door (LEFT)", color="violet").on_click(lambda _: open_door_left())
    server.gui.add_button("Grab glass (RIGHT only)", color="green").on_click(lambda _: grasp_only())
    server.gui.add_button("FULL: open door (L) -> grab+put-in glass (R) -> close door + relax", color="blue").on_click(lambda _: full_sequence())
    server.gui.add_button("Replay exported traj (door+grab+put-in+close, real_plan_door.py JSON)", color="teal").on_click(lambda _: replay_traj())
    print(f"\n================  OPEN  http://localhost:{PORT}  ================\n", flush=True)
    while True: time.sleep(1)


if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\nbye")
    except Exception: print("\n❌ EXCEPTION:"); traceback.print_exc(); sys.exit(3)
