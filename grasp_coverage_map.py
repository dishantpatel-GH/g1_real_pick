#!/usr/bin/env python3
# =============================================================================
# grasp_coverage_map.py — render a top-down PICTURE of where the right hand can grasp the glass:
#   SIDE (yaw 0) vs BEST reachable angle (sweep yaw [0,-30,-60,-90], take the most-front that works).
# Colors each table cell by the most-front reachable yaw; marks side-reachable cells; draws the robot
# base + right shoulder + reach arc and the current glass pose. Saves a PNG. Glass r3.6 h10.5.
#
# RUN (cumotion_venv):
#   /home/dishant/cumotion_venv/bin/python grasp_coverage_map.py --out /tmp/grasp_coverage.png
# =============================================================================
import argparse, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Circle, Wedge, Rectangle

CONFIG = "/home/dishant/g1_ws/cumotion/config/g1_inspire_right.yml"
ARM_JOINTS = ["right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
              "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"]
R, H = 0.036, 0.105
SG_DX, SG_DYGAP, SG_FROMTOP = 0.215, 0.01, 0.035
YAW_SET = [0, -30, -60, -90]
SHOULDER = np.array([0.0, -0.18])                      # approx right shoulder (pelvis xy)
HAND_LINKS = ["right_base_link","right_palm_force_sensor",
    "right_thumb_1","right_thumb_2","right_thumb_3","right_thumb_4",
    "right_thumb_force_sensor_1","right_thumb_force_sensor_2","right_thumb_force_sensor_3","right_thumb_force_sensor_4",
    "right_index_1","right_index_2","right_index_force_sensor_1","right_index_force_sensor_2","right_index_force_sensor_3",
    "right_middle_1","right_middle_2","right_middle_force_sensor_1","right_middle_force_sensor_2","right_middle_force_sensor_3",
    "right_ring_1","right_ring_2","right_ring_force_sensor_1","right_ring_force_sensor_2","right_ring_force_sensor_3",
    "right_little_1","right_little_2","right_little_force_sensor_1","right_little_force_sensor_2","right_little_force_sensor_3"]


def euler_quat_z(yaw_deg):
    y = np.deg2rad(yaw_deg)
    return [float(np.cos(y/2)), 0.0, 0.0, float(np.sin(y/2))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/grasp_coverage.png")
    ap.add_argument("--pose", default="/tmp/glass_pose.json", help="glass pose JSON to mark (optional)")
    ap.add_argument("--step", type=float, default=0.025, help="grid spacing (m); smaller = finer + slower")
    ap.add_argument("--xrange", type=float, nargs=2, default=[0.28, 0.50])
    ap.add_argument("--yrange", type=float, nargs=2, default=[-0.34, -0.02])
    args = ap.parse_args()

    import torch
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.scene import Scene, Cuboid, Cylinder, Mesh
    from curobo.types import GoalToolPose, JointState

    print("[cov] building cuRobo planner (~15s) ...", flush=True)
    cfg = MotionPlannerCfg.create(robot=CONFIG, scene_model=None, max_goalset=8, collision_cache={"obb": 8, "mesh": 4})
    pl = MotionPlanner(cfg); pl.warmup(enable_graph=True, num_warmup_iterations=5)
    qstart = pl.kinematics.get_active_js(pl.default_joint_state.clone().unsqueeze(0))
    table_top = -0.006

    def build_scene(cx, cy, cz):
        tri = Cylinder(name="object", radius=R, height=H, pose=[cx, cy, cz, 1, 0, 0, 0]).get_trimesh_mesh()
        return Scene(cuboid=[Cuboid(name="table", dims=[0.7, 0.9, 0.04], pose=[0.60, -0.10, (cz-H/2)-0.025, 1, 0, 0, 0])],
                     mesh=[Mesh(name="object", vertices=tri.vertices.tolist(), faces=tri.faces.tolist(), pose=[cx, cy, cz, 1, 0, 0, 0])])
    def goal(w, q):
        return GoalToolPose(tool_frames=pl.tool_frames,
                            position=torch.tensor([[[[list(w)]]]], device="cuda", dtype=torch.float32),
                            quaternion=torch.tensor([[[[list(q)]]]], device="cuda", dtype=torch.float32))
    def ok(res): return res is not None and res.success is not None and bool(res.success.any())
    def plan_at_yaw(cx, cy, cz, yaw):
        th = np.deg2rad(yaw); c, s = np.cos(th), np.sin(th); ox, oy = -SG_DX, -(R+SG_DYGAP)
        w = (cx + (c*ox - s*oy), cy + (s*ox + c*oy), (cz + H/2.0) - SG_FROMTOP)
        return pl.plan_grasp(goal(w, euler_quat_z(yaw)), qstart,
            grasp_approach_axis="y", grasp_approach_offset=-0.05, grasp_approach_in_tool_frame=True,
            grasp_lift_axis="z", grasp_lift_offset=0.15, grasp_lift_in_tool_frame=False,
            plan_approach_to_grasp=True, plan_grasp_to_lift=True, disable_collision_links=HAND_LINKS)

    XS = np.round(np.arange(args.xrange[0], args.xrange[1] + 1e-9, args.step), 3)
    YS = np.round(np.arange(args.yrange[0], args.yrange[1] + 1e-9, args.step), 3)
    best = np.full((len(YS), len(XS)), -1)             # -1 none; else index into YAW_SET (most-front)
    side = np.zeros((len(YS), len(XS)), bool)
    print(f"[cov] sweeping {len(XS)}x{len(YS)}={len(XS)*len(YS)} cells x {len(YAW_SET)} yaws ...", flush=True)
    for iy, y in enumerate(YS):
        for ix, x in enumerate(XS):
            cz = table_top + H/2.0
            pl.update_world(build_scene(x, y, cz))
            reach = [k for k, yaw in enumerate(YAW_SET) if ok(plan_at_yaw(x, y, cz, yaw))]
            if reach: best[iy, ix] = max(reach)
            side[iy, ix] = (0 in reach)
        print(f"  y={y:+.2f} done", flush=True)
    ns = int(side.sum()); nany = int((best >= 0).sum()); nnew = int(((best >= 0) & ~side).sum())
    print(f"[cov] side(yaw0): {ns} | any-angle: {nany} | NEW(front-only): {nnew} of {best.size}")

    # ---- plot (top-down: forward x up, lateral y horizontal; robot's right = -y on the right) ----
    colors = ["#dddddd", "#1f77b4", "#17becf", "#2ca02c", "#ffd11a"]   # none, side(0), -30, -60, -90
    cmap = ListedColormap(colors); norm = BoundaryNorm([-1.5,-0.5,0.5,1.5,2.5,3.5], cmap.N)
    fig, axp = plt.subplots(figsize=(8, 8))
    axp.pcolormesh(YS, XS, best.T, cmap=cmap, norm=norm, shading="nearest", edgecolors="white", linewidth=0.4)
    # mark side-reachable cells with a dot
    YY, XX = np.meshgrid(YS, XS)
    axp.scatter(YY[side.T], XX[side.T], c="black", s=8, marker="o", label="side (yaw 0) reachable")
    # robot schematic
    axp.add_patch(Rectangle((-0.20, -0.10), 0.40, 0.10, color="0.5", alpha=0.5))   # pelvis/torso slab
    axp.plot(SHOULDER[1], SHOULDER[0], "ks", ms=9); axp.annotate("right shoulder", (SHOULDER[1], SHOULDER[0]),
             textcoords="offset points", xytext=(6, 6), fontsize=8)
    axp.add_patch(Wedge((SHOULDER[1], SHOULDER[0]), 0.43, -10, 110, width=0.001, color="k", ls="--", fill=False, lw=1))
    axp.annotate("~reach 0.43m", (SHOULDER[1]+0.30, SHOULDER[0]+0.30), fontsize=8, color="k")
    # current glass
    if os.path.exists(args.pose):
        try:
            P = json.load(open(args.pose)); gx, gy = P["center"][0], P["center"][1]
            axp.add_patch(Circle((gy, gx), R, color="red", fill=False, lw=2))
            axp.plot(gy, gx, "r*", ms=14, label=f"glass ({gx:.2f},{gy:.2f})")
        except Exception: pass
    axp.set_xlabel("pelvis y  (robot's RIGHT →)"); axp.set_ylabel("pelvis x  (forward ↑)")
    axp.invert_xaxis()                                  # -y (robot right) on the right
    axp.set_aspect("equal"); axp.set_title(
        f"Right-hand grasp coverage  |  side {ns}  ·  any-angle {nany}  ·  +{nnew} from front angles  (of {best.size})")
    # legend for the colors
    from matplotlib.patches import Patch
    leg = [Patch(facecolor=colors[i+1], edgecolor="w", label=lab) for i, lab in
           enumerate(["best=side (0°)", "best=-30°", "best=-60°", "best=-90°"])] + [Patch(facecolor=colors[0], label="none")]
    axp.legend(handles=leg + [plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="k", label="side reachable")],
               loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print(f"[cov] saved {args.out}")


if __name__ == "__main__":
    main()
