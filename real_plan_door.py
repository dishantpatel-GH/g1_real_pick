#!/usr/bin/env python3
# =============================================================================
# real_plan_door.py — PLAN the LEFT-arm microwave-door OPEN (relaxed -> reach -> slide ->
# swing) and export the left-arm joint trajectory to JSON for real_open_door.py.
#
# This is the door-open counterpart of real_plan_approach.py. It runs the IDENTICAL
# planning as microwave_door_open_sim.py (same geometry, same reach/slide/swing, imported
# from it), but headless and with NO viser — it just writes the trajectory.
#
# The trajectory STARTS at the relaxed pose (the planning start), so the executor can ramp
# straight to relaxed and replay from there (no arms=0 detour through the table).
#
# ENV: cumotion_venv (torch + cuRobo). Reads the microwave pose from /tmp/microwave_pose.json
# (via microwave_door_open_sim). NOT run on the robot.
#
#   RUN:  /home/dishant/cumotion_venv/bin/python real_plan_door.py            # -> /tmp/door_open_traj.json
#         /home/dishant/cumotion_venv/bin/python real_plan_door.py --open-deg 40 --out /tmp/door_open_traj.json
# =============================================================================
import argparse, json
import numpy as np

import microwave_door_open_sim as S      # geometry + DOOR_* params + euler_quat/quat_rotate (loads the MW pose)

VMAX = 0.6      # cap per-joint speed (rad/s) when assigning waypoint times -> slow, safe replay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/door_open_traj.json")
    ap.add_argument("--open-deg", type=float, default=S.DOOR_OPEN_DEG, help="door-open swing target (deg)")
    ap.add_argument("--yaw", type=float, default=S.DOOR_WRIST_YAW, help="LEFT wrist yaw at the handle (deg)")
    ap.add_argument("--gap", type=float, default=S.DOOR_GAP_DX, help="fingertip push past the handle front (m)")
    ap.add_argument("--insert-dy", type=float, default=S.DOOR_INSERT_DY, help="pre-pose (FT1) +y offset from the handle (m)")
    ap.add_argument("--slide-dy", type=float, default=0.0, help="slide target shift (m): negative = slide deeper -y, past the handle")
    ap.add_argument("--grasp-z", type=float, default=S.HANDLE_Z, help="handle grasp height (m); default = body centre")
    ap.add_argument("--glass", default="/tmp/glass_pose.json", help="glass pose JSON from object_detection.py")
    ap.add_argument("--no-grab", action="store_true", help="plan only the door open (skip the right-arm glass grab)")
    # glass-grasp x/y/z tuning -- the same knobs as microwave_door_open_sim.py's sl_ggx/sl_ggy/sl_ggz sliders
    ap.add_argument("--glass-x-nudge", type=float, default=0.02, help="glass grasp +x nudge (more ahead); = sim sl_ggx")
    ap.add_argument("--glass-y-nudge", type=float, default=0.0, help="glass grasp y nudge (subtracted -> + = more right/-y); = sim sl_ggy")
    ap.add_argument("--glass-grip-z", type=float, default=0.07, help="glass grasp absolute grip height (m); default = glass top - FG_FROMTOP; = sim sl_ggz")
    ap.add_argument("--table-gap", type=float, default=S.GRASP_TABLE_GAP, help="drop the glass table this far BELOW the glass bottom (m) so it doesn't collide with the glass grasp target; = sim GRASP_TABLE_GAP")
    # bring-back staging pose (glass CENTRE target after the grasp) -- defaults = sim BRING_BACK_XYZ (z +Z_LIFT)
    ap.add_argument("--bring-back-x", type=float, default=None, help="staging x for the grasped glass (m); default = sim BRING_BACK_XYZ[0]")
    ap.add_argument("--bring-back-y", type=float, default=None, help="staging y (m); default = sim BRING_BACK_XYZ[1]")
    ap.add_argument("--bring-back-z", type=float, default=None, help="staging z (m); default = sim BRING_BACK_XYZ[2] + Z_LIFT")
    # put-in-microwave: from the staging pose the glass slides into the cavity along a line tilted PLACE_ANGLE
    ap.add_argument("--no-insert", action="store_true", help="stop after bring-back (do NOT put the glass in the microwave / close the door)")
    ap.add_argument("--place-angle", type=float, default=None, help="insert-line tilt in xy from +x toward +y (deg); default = sim PLACE_IN_ANGLE")
    ap.add_argument("--place-dist", type=float, default=None, help="distance to slide the glass into the cavity along that line (m); default = sim PLACE_IN_DIST")
    ap.add_argument("--straight-steps", type=int, default=12, help="Cartesian steps for the STRAIGHT-LINE put-in/pull-out (more = straighter; the hand follows a straight line, not a curve)")
    ap.add_argument("--insert-yaw-span", type=float, default=60.0, help="max wrist-yaw the planner may rotate (deg, +/-) along the put-in/pull-out to avoid the microwave -- the yaw is NOT fixed; it changes only if a step would collide")
    ap.add_argument("--mw-right-pad", type=float, default=0.01, help="extend the microwave RIGHT wall -y by this (m) in the carry/insert collision world -> margin so the (unmodelled in-hand) glass doesn't graze the unit")
    args = ap.parse_args()

    import torch, yaml
    from curobo.scene import Scene, Cuboid
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import GoalToolPose, JointState
    from curobo._src.state.state_joint_trajectory_ops import get_joint_state_at_horizon_index

    print(f"[plan] microwave: {S.MW_SRC}  body=({S.MW_CX:.3f},{S.MW_CY:.3f},{S.MW_CZ:.3f})  "
          f"handle front x={S.HANDLE_FRONT_X:.3f} y={S.MW_HANDLE_Y:.3f} z={args.grasp_z:.3f}")

    # LEFT-arm planner: active chain = LEFT arm, RIGHT arm LOCKED at relaxed (same as the sim)
    lc = yaml.safe_load(open(S.OUT)); lk = lc["kinematics"]; lk["tool_frames"] = ["left_wrist_yaw_link"]
    for j in S.LARM_JOINTS: lk["lock_joints"].pop(j, None)
    lk["lock_joints"].update(S.RIGHT_FROZEN)
    yaml.safe_dump(lc, open("/tmp/g1_door_left_plan.yml", "w"))
    pL = MotionPlanner(MotionPlannerCfg.create(robot="/tmp/g1_door_left_plan.yml", scene_model=None,
                                               max_goalset=8, collision_cache={"obb": 12, "mesh": 4}))
    TABLE_C = Cuboid(name="table", dims=S.TABLE[0], pose=S.TABLE[1])
    hxw, hyw, hzw = S.MW_DIMS[0]/2, S.MW_DIMS[1]/2, S.MW_DIMS[2]/2; W = S.MW_WALL; RS = S.MW_WALL + S.MW_BTN_W
    def walls():
        return [Cuboid(name="mw_back",   dims=[W, S.MW_DIMS[1], S.MW_DIMS[2]], pose=[S.MW_CX+hxw-W/2, S.MW_CY, S.MW_CZ, 1,0,0,0]),
                Cuboid(name="mw_left",   dims=[S.MW_DIMS[0], W, S.MW_DIMS[2]], pose=[S.MW_CX, S.MW_CY+hyw-W/2, S.MW_CZ, 1,0,0,0]),
                Cuboid(name="mw_right",  dims=[S.MW_DIMS[0], RS, S.MW_DIMS[2]], pose=[S.MW_CX, S.MW_CY-hyw+RS/2, S.MW_CZ, 1,0,0,0]),
                Cuboid(name="mw_top",    dims=[S.MW_DIMS[0], S.MW_DIMS[1], W], pose=[S.MW_CX, S.MW_CY, S.MW_CZ+hzw-W/2, 1,0,0,0]),
                Cuboid(name="mw_bottom", dims=[S.MW_DIMS[0], S.MW_DIMS[1], W], pose=[S.MW_CX, S.MW_CY, S.MW_CZ-hzw+W/2, 1,0,0,0])]
    def handle_obs():
        return [Cuboid(name="mw_handle", dims=[S.HANDLE_PROTRUDE, 0.03, S.HANDLE_LEN],
                       pose=[S.HANDLE_FRONT_X + S.HANDLE_PROTRUDE/2, S.MW_HANDLE_Y, S.HANDLE_Z, 1,0,0,0])]
    def scene_L(with_mw): return Scene(cuboid=[TABLE_C] + (walls() + handle_obs() if with_mw else []))
    pL.update_world(scene_L(True)); pL.warmup(enable_graph=True, num_warmup_iterations=5)

    def ok(r): return r is not None and r.success is not None and bool(r.success.any())
    def trim_last(P):
        Hn = P.shape[0]
        if Hn <= 1: return Hn
        d = np.linalg.norm(np.diff(P, axis=0), axis=1); mov = np.where(d > 1e-5)[0]
        return min(int(mov[-1] + 2) if len(mov) else Hn, Hn)
    def relaxed_qstart_L():
        full = pL.default_joint_state.clone(); jn = list(full.joint_names); pos = full.position.clone().reshape(-1)
        for name, val in S.RELAXED_ARM.items():
            if name in jn: pos[jn.index(name)] = float(val)
        return pL.kinematics.get_active_js(JointState.from_position(pos.unsqueeze(0), joint_names=jn))
    def goal_L(p, q):
        return GoalToolPose(tool_frames=pL.tool_frames,
                            position=torch.tensor([[[[list(p)]]]], device="cuda", dtype=torch.float32),
                            quaternion=torch.tensor([[[[list(q)]]]], device="cuda", dtype=torch.float32))
    def door_wrist_L(ft, yaw_deg, pitch_deg=S.DOOR_PITCH):
        Q = S.euler_quat(yaw_deg, pitch_deg); o = S.quat_rotate(Q, S.FINGER_OFFSET)
        return (ft[0]-o[0], ft[1]-o[1], ft[2]-o[2]), Q
    def rot_about_hinge(px, py, phi):
        c, s = np.cos(phi), np.sin(phi); rx, ry = px - S.HINGE_XYZ[0], py - S.HINGE_XYZ[1]
        return S.HINGE_XYZ[0] + c*rx - s*ry, S.HINGE_XYZ[1] + s*rx + c*ry

    LARM = S.LARM_JOINTS
    RARM_NAMES = ["right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
                  "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"]
    def sample(traj, order):                            # interpolated_trajectory -> list of joint waypoints (in `order`)
        t = traj.squeeze(0); jn = list(t.joint_names)
        P = t.position[0].detach().cpu().numpy(); last = trim_last(P)
        idx = np.unique(np.linspace(0, last - 1, min(last, 50)).astype(int))
        return [[float(P[i][jn.index(a)]) for a in order] for i in idx]
    def mk_points(ws, dgs=None):                        # de-dup, distance-based times (cap VMAX), finite-diff velocities
        out = []; tt = 0.0; pv = None
        for i, w in enumerate(ws):
            if pv is not None and max(abs(np.array(w) - np.array(pv))) < 1e-4: continue
            if pv is not None: tt += max(0.02, float(max(abs(np.array(w) - np.array(pv)))) / VMAX)
            vel = list((np.array(w) - np.array(pv)) / max(1e-3, tt - out[-1]["time"])) if pv is not None else [0.0]*7
            pt = {"positions": [round(x, 6) for x in w], "time": round(tt, 4), "velocities": [round(x, 4) for x in vel]}
            if dgs is not None: pt["door_deg"] = round(dgs[i], 2)
            out.append(pt); pv = w
        if out: out[0]["velocities"] = [0.0]*7
        return out
    wps = []; degs = []                                 # collected LEFT-arm waypoints (7-vectors) + door angle (deg, signed)
    def collect(res, door_deg):
        for w in sample(res.interpolated_trajectory, LARM):
            wps.append(w); degs.append(float(door_deg))

    yaw, gz, gap, dy, slide_y = args.yaw, args.grasp_z, args.gap, args.insert_dy, args.slide_dy
    ftx, ftz = S.HANDLE_FRONT_X + gap, gz
    fty = S.MW_HANDLE_Y + slide_y                        # slide target + swing contact (slide_y<0 = deeper -y)
    # phase 1: reach to the +y of the HANDLE (microwave collision ON) -- pre-pose independent of the slide
    pL.update_world(scene_L(True))
    w1, q1 = door_wrist_L((ftx, S.MW_HANDLE_Y + dy, ftz), yaw)
    r1 = pL.plan_pose(goal_L(w1, q1), relaxed_qstart_L(), max_attempts=6)
    print(f"[plan] phase1 reach -> {'OK' if ok(r1) else 'FAIL'}")
    if not ok(r1): raise SystemExit("phase1 reach failed")
    collect(r1, 0.0); seed = pL.kinematics.get_active_js(get_joint_state_at_horizon_index(r1.js_solution, -1).squeeze(0))
    # phase 2: slide -y into the gap (microwave collision OFF)
    pL.update_world(scene_L(False))
    w2, q2 = door_wrist_L((ftx, fty, ftz), yaw)
    r2 = pL.plan_pose(goal_L(w2, q2), seed, max_attempts=6)
    print(f"[plan] phase2 slide -> {'OK' if ok(r2) else 'FAIL'}")
    if not ok(r2): raise SystemExit("phase2 slide failed")
    collect(r2, 0.0); seed = pL.kinematics.get_active_js(get_joint_state_at_horizon_index(r2.js_solution, -1).squeeze(0))
    # phase 3: swing the door open (fingertip arcs about the hinge; wrist yaw partially tracks)
    Nn = max(1, int(round(args.open_deg / 8.0))); reached = 0.0
    for i in range(1, Nn + 1):
        phid = -args.open_deg * (i / Nn); phi = np.deg2rad(phid)
        fx, fy = rot_about_hinge(ftx, fty, phi)
        wk, qk = door_wrist_L((fx, fy, ftz), yaw + S.DOOR_YAW_TRACK * phid)
        rk = pL.plan_pose(goal_L(wk, qk), seed, max_attempts=4)
        if not ok(rk):
            print(f"[plan] swing stuck at {abs(reached):.0f} deg"); break
        collect(rk, phid); reached = phid          # phid is signed (negative = swing toward the robot)
        seed = pL.kinematics.get_active_js(get_joint_state_at_horizon_index(rk.js_solution, -1).squeeze(0))
    print(f"[plan] door swings to {abs(reached):.0f} deg over {len(wps)} waypoints")

    door_pts = mk_points(wps, degs)                     # LEFT-arm door trajectory (top-level, unchanged)
    rarm_relaxed = [S.RIGHT_FROZEN[n] for n in RARM_NAMES]
    out = {"joint_names": LARM, "points": door_pts,
           "relaxed_larm": door_pts[0]["positions"],            # left-arm relaxed (the start)
           "right_arm_names": RARM_NAMES, "right_arm_hold": rarm_relaxed,   # freeze the right arm here during the door open
           "reached_deg": float(abs(reached)),
           "handle_xyz": [round(S.HANDLE_FRONT_X, 4), round(S.MW_HANDLE_Y, 4), round(gz, 4)],
           "microwave_center": [round(S.MW_CX, 4), round(S.MW_CY, 4), round(S.MW_CZ, 4)],
           "left_hold": door_pts[-1]["positions"]}              # HOLD the left arm here (door open) during the grab

    # ===== RIGHT ARM: grab the glass, then bring it to the staging pose (NO insert) -- like sim_grasp_viz =====
    if not args.no_grab:
        try:
            G = json.load(open(args.glass)); gc = G["center"]
            gx, gy = float(gc[0]), float(gc[1]); gR = float(G["radius_m"]); gH = float(G["height_m"])
            gzg = float(gc[2]) + S.Z_LIFT                          # +5cm: grab at the height we actually use in real (= the sim)
            print(f"[plan] glass: center=({gx:.3f},{gy:.3f},{gzg:.3f}) r={gR*100:.1f}cm h={gH*100:.1f}cm  (z +{S.Z_LIFT*100:.0f}cm above depth)")
        except Exception as e:
            print(f"[plan] no glass pose at {args.glass} ({e}); skipping the grab."); G = None
        if G is not None:
            import sim_grasp_viz as SV
            from curobo.scene import Cylinder, Mesh
            # GLASS-level table: the glass rests on the real surface (its bottom = table top). This differs
            # from the microwave-derived table -- the perceived handle z (= MW_CZ) sits above the body centre,
            # so MW's table reads too high and would bury the glass. Use the glass's own level here.
            gt_top = gzg - gH / 2.0 - args.table_gap; gt_near = 0.32        # table sits GAP below the glass bottom (no glass-table collision)
            gt_dx = max(0.70, max(gx + 0.25, S.MW_CX + S.MW_DIMS[0]/2 + 0.10) - gt_near)
            GTABLE = Cuboid(name="table", dims=[gt_dx, 1.2, 0.6], pose=[gt_near + gt_dx/2.0, -0.05, gt_top - 0.005 - 0.3, 1,0,0,0])
            print(f"[plan] glass table top z={gt_top:.3f} (glass bottom {gzg-gH/2.0:.3f} - gap {args.table_gap:.3f}; vs microwave table {S.TABLE_TOP_Z:.3f})")
            pR = MotionPlanner(MotionPlannerCfg.create(robot=S.OUT, scene_model=None, max_goalset=8, collision_cache={"obb": 8, "mesh": 4}))
            tri = Cylinder(name="object", radius=gR, height=gH, pose=[gx, gy, gzg, 1,0,0,0]).get_trimesh_mesh()
            glass_mesh = Mesh(name="object", vertices=tri.vertices.tolist(), faces=tri.faces.tolist(), pose=[gx, gy, gzg, 1,0,0,0])
            pR.update_world(Scene(cuboid=[GTABLE], mesh=[glass_mesh])); pR.warmup(enable_graph=True, num_warmup_iterations=5)
            def relaxed_qstart_R():
                full = pR.default_joint_state.clone(); jn = list(full.joint_names); pos = full.position.clone().reshape(-1)
                for name, val in S.RELAXED_ARM.items():
                    if name in jn: pos[jn.index(name)] = float(val)
                return pR.kinematics.get_active_js(JointState.from_position(pos.unsqueeze(0), joint_names=jn))
            def goal_R(p, q):
                return GoalToolPose(tool_frames=pR.tool_frames,
                                    position=torch.tensor([[[[list(p)]]]], device="cuda", dtype=torch.float32),
                                    quaternion=torch.tensor([[[[list(q)]]]], device="cuda", dtype=torch.float32))
            DX, GAP, AP, PITCH = SV.SG_DX, SV.SG_DYGAP, SV.FG_APPROACH, 0.0
            XOFF, YOFF = args.glass_x_nudge, args.glass_y_nudge                 # x/y tuning (= sim sl_ggx/sl_ggy)
            grip_z = args.glass_grip_z if args.glass_grip_z is not None else (gzg + gH/2.0) - SV.FG_FROMTOP   # abs grip Z (= sim sl_ggz)
            FROMTOP = (gzg + gH/2.0) - grip_z                                   # below-top implied by grip_z; reused on bring-back
            print(f"[plan] glass grasp tune: Xnudge={XOFF:+.3f} Ynudge={YOFF:+.3f} gripZ={grip_z:.3f} (top={gzg+gH/2.0:.3f})")
            def grasp_wrist(yaw):                            # wrist pose = rotate the side grasp by `yaw` about the glass axis
                th = np.deg2rad(yaw); c, s = np.cos(th), np.sin(th); ox, oy = -DX, -(gR + GAP)
                return (gx + (c*ox - s*oy) + XOFF, gy + (s*ox + c*oy) - YOFF, grip_z)
            res = None; gyaw = None
            for yaw in [-90, -60, -30, 0]:                  # BestAngle: most-front reachable
                try:
                    r = pR.plan_grasp(goal_R(grasp_wrist(yaw), S.euler_quat(yaw, PITCH)), relaxed_qstart_R(),
                                      grasp_approach_axis="y", grasp_approach_offset=AP, grasp_approach_in_tool_frame=True,
                                      grasp_lift_axis="z", grasp_lift_offset=0.15, grasp_lift_in_tool_frame=False,
                                      plan_approach_to_grasp=True, plan_grasp_to_lift=True, disable_collision_links=SV.HAND_LINKS)
                except Exception as e:                       # cuRobo plan_grasp can raise on some poses (goalset-index edge case); skip this yaw
                    print(f"[plan] grasp yaw={yaw} -> cuRobo error ({type(e).__name__}); skipping"); continue
                if ok(r): res = r; gyaw = yaw; print(f"[plan] grasp BestAngle yaw={yaw} -> OK"); break
            if res is None:
                print("[plan] grasp FAILED at all yaws -- writing door-only JSON.")
            else:
                app_pts = mk_points(sample(res.approach_interpolated_trajectory, RARM_NAMES))   # relaxed -> pre-grasp
                grasp_pts = mk_points(sample(res.grasp_interpolated_trajectory, RARM_NAMES))     # pre-grasp -> grasp
                gseed = pR.kinematics.get_active_js(get_joint_state_at_horizon_index(res.grasp_trajectory, -1).squeeze(0))
                # ---- BRING-BACK is COLLISION-AWARE vs the microwave RIGHT wall (button block, grown -y by
                #      --mw-right-pad) so the in-hand glass does NOT graze the unit. The full box over-constrains
                #      the right arm, so only the right wall is modelled. Put-in/pull-out stay table-only (may graze). ----
                hyw = S.MW_DIMS[1] / 2.0; RSp = S.MW_WALL + S.MW_BTN_W + args.mw_right_pad
                mw_right = Cuboid(name="mw_right", dims=[S.MW_DIMS[0], RSp, S.MW_DIMS[2]], pose=[S.MW_CX, S.MW_CY - hyw + RSp/2, S.MW_CZ, 1,0,0,0])
                BBYAW = SV.BRING_BACK_YAW
                BBX = args.bring_back_x if args.bring_back_x is not None else SV.BRING_BACK_XYZ[0]
                BBY = args.bring_back_y if args.bring_back_y is not None else SV.BRING_BACK_XYZ[1]
                BBZ = args.bring_back_z if args.bring_back_z is not None else SV.BRING_BACK_XYZ[2] + S.Z_LIFT
                ox, oy = -DX, -(gR + GAP)
                def rwrist(cx, cy, cz, yawd):                 # wrist for a glass-CENTRE target at wrist yaw `yawd`
                    cc, ss = np.cos(np.deg2rad(yawd)), np.sin(np.deg2rad(yawd))
                    return (cx + (cc*ox - ss*oy) + XOFF, cy + (ss*ox + cc*oy) - YOFF, cz + (gH/2.0 - FROMTOP))
                def yaw_candidates(y0):                       # nominal first, then rotate +/- up to the span (only used if a step collides)
                    out_y = [y0]; d = 10.0
                    while d <= args.insert_yaw_span + 1e-6:
                        out_y += [y0 - d, y0 + d]; d += 10.0
                    return out_y
                pR.update_world(Scene(cuboid=[GTABLE, mw_right]))     # BRING-BACK world: table + padded right wall
                rp = pR.plan_pose(goal_R(rwrist(BBX, BBY, BBZ, BBYAW), S.euler_quat(BBYAW, 0)), gseed, max_attempts=10)
                print(f"[plan] bring-back to ({BBX:.3f},{BBY:.3f},{BBZ:.3f}) microwave-aware (right pad {args.mw_right_pad*100:.0f}cm) -> {'OK' if ok(rp) else 'FAIL'}")
                if not ok(rp):
                    print("[plan] bring-back FAILED -- writing door-only JSON (tune --bring-back-x/y/z or --mw-right-pad).")
                else:
                    grab = {"joint_names": RARM_NAMES, "approach": app_pts, "grasp": grasp_pts,
                            "bringback": mk_points(sample(rp.interpolated_trajectory, RARM_NAMES)),
                            "glass": {"center": [round(gx,4), round(gy,4), round(gzg,4)], "radius_m": round(gR,4), "height_m": round(gH,4)},
                            "bring_back_xyz": [round(BBX,4), round(BBY,4), round(BBZ,4)], "grasp_yaw": gyaw}
                    print(f"[plan] grab: approach {len(app_pts)} + grasp {len(grasp_pts)} + bringback {len(grab['bringback'])} pts")
                    # ===== PUT-IN (straight line, table-only/graze, yaw free) -> PULL-OUT -> CLOSE door (LEFT) =====
                    if not args.no_insert:
                        pR.update_world(Scene(cuboid=[GTABLE]))   # put-in/pull-out: table only (graze tolerated)
                        sBB = pR.kinematics.get_active_js(get_joint_state_at_horizon_index(rp.js_solution, -1).squeeze(0))
                        PA = args.place_angle if args.place_angle is not None else SV.PLACE_IN_ANGLE
                        PD = args.place_dist  if args.place_dist  is not None else SV.PLACE_IN_DIST
                        ca, sa = np.cos(np.deg2rad(PA)), np.sin(np.deg2rad(PA))
                        ig = (BBX + PD*ca, BBY + PD*sa, BBZ)                      # inserted glass-centre target
                        # STRAIGHT Cartesian line staging -> inserted. Yaw is NOT fixed: keep nominal if a step is
                        # reachable, else rotate (+/- insert_yaw_span) so the hand can thread without contorting.
                        M = max(2, args.straight_steps); insert_ws = []; cseed = sBB; cur_yaw = BBYAW; ins_ok = True; yaws_used = []
                        for k in range(1, M + 1):
                            fk = k / M; cx = BBX + fk*(ig[0]-BBX); cy = BBY + fk*(ig[1]-BBY); cz = BBZ + fk*(ig[2]-BBZ)
                            step = None
                            for yk in yaw_candidates(cur_yaw):
                                rk = pR.plan_pose(goal_R(rwrist(cx, cy, cz, yk), S.euler_quat(yk, 0)), cseed, max_attempts=4)
                                if ok(rk): step = (rk, yk); break
                            if step is None: print(f"[plan] put-in stuck at step {k}/{M} ({fk*PD:.2f}m in)"); ins_ok = False; break
                            rk, cur_yaw = step; yaws_used.append(int(round(cur_yaw)))
                            insert_ws += sample(rk.interpolated_trajectory, RARM_NAMES)
                            cseed = pR.kinematics.get_active_js(get_joint_state_at_horizon_index(rk.js_solution, -1).squeeze(0))
                        print(f"[plan] put-in straight {PA:.0f}deg x {PD:.2f}m yaw {BBYAW:.0f}->{cur_yaw:.0f} -> {'OK' if ins_ok else 'FAIL'}")
                        if not ins_ok or not insert_ws:
                            print("[plan] put-in FAILED -- grab kept through bring-back only (tune --place-angle/--place-dist/--bring-back-*/--insert-yaw-span).")
                        else:
                            rr = pR.plan_cspace(relaxed_qstart_R(), sBB, max_attempts=10)   # staging -> relaxed
                            if not ok(rr):
                                print("[plan] pull-out (staging->relaxed) FAILED -- grab kept through bring-back only.")
                            else:
                                retract_ws = list(reversed(insert_ws)) + sample(rr.interpolated_trajectory, RARM_NAMES)
                                close_ws = [p["positions"] for p in reversed(door_pts)]
                                close_dg = [p.get("door_deg", 0.0) for p in reversed(door_pts)]
                                grab["insert"]  = mk_points(insert_ws)
                                grab["retract"] = mk_points(retract_ws)
                                grab["insert_xyz"] = [round(v,4) for v in ig]; grab["place_angle"] = PA; grab["place_dist"] = PD
                                out["door_close"] = mk_points(close_ws, close_dg)
                                print(f"[plan] put-in {len(grab['insert'])} (yaws {yaws_used}) + pull-out {len(grab['retract'])} pts; door_close {len(out['door_close'])} pts")
                    out["grab"] = grab

    json.dump(out, open(args.out, "w"), indent=2)
    has_ins = ("grab" in out) and ("insert" in out["grab"])
    tag = (" + grab + put-in/close" if has_ins else (" + grab" if "grab" in out else " (door only)"))
    print(f"[plan] wrote door {len(door_pts)} pts{tag} -> {args.out}")


if __name__ == "__main__":
    main()
