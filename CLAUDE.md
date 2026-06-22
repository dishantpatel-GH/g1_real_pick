# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A real-robot **pick pipeline** for a Unitree G1 humanoid + Inspire FTP right hand: head RealSense
depth → detect a coloured cup → cuRobo `plan_grasp` → drive the right arm (Unitree SDK) and close
the Inspire hand → lift. There is **no AprilTag** and **no ROS** anywhere in the path.

This folder began as an aggregation layer of **symlinks** to live sources, but has since been
**vendored**: every entry is now a real committed copy (remote `dishantpatel-GH/g1_real_pick`).
First-party code is `real_pick.py`, `go_to_start.py`, and the planner/viz/detector/server scripts
(`real_plan_approach.py`, `real_viz.py`, `sim_grasp_viz.py`, `grasp_coverage_map.py`,
`object_detection.py`, `image_server.py`), originally authored under `/home/dishant/g1_ws/...`.
`inspire_hand_sdk/` is the Inspire SDK. `config/` holds a *snapshot* of the robot config
(`g1_inspire_right.yml`, `g1_inspire_cumotion.urdf`) — but the scripts do **NOT** read it: they hardcode
`/home/dishant/g1_ws/cumotion/config/g1_inspire_right.yml` (as `CONFIG`/`OUT`), so editing the repo's
`config/` changes nothing until you repoint those constants.

`curobo/` is a vendored copy of **NVlabs/curobo — a post-0.8.0, unreleased dev snapshot** (`.git`
stripped, so there's no exact commit hash; the CHANGELOG's top section is `## Latest Commit`, sitting
*above* `## Version 0.8.0`). It uses the **redesigned API, NOT v0.8.0's `MotionGen`**: the package lives
flat at `curobo/curobo/` (not `src/curobo/`), and the planner entry points are
`from curobo import MotionPlanner, MotionPlannerCfg`; `Scene`/`Cuboid`/`Cylinder`/`Mesh` from
`curobo.scene`; `GoalToolPose`/`JointState` from `curobo.types`; and `MotionPlanner.plan_grasp(...)`
(implemented in `curobo/curobo/_src/motion/motion_planner.py`). Don't go looking for
`wrap/reacher/motion_gen.py` — it does not exist in this tree.

**The copies have diverged from the symlink era:** editing a file here no longer changes the live source
under `/home/dishant/g1_ws/...` or `/home/dishant/Projects/...` (or vice-versa). The repo is a snapshot,
**not yet portable** — absolute paths are still baked into the scripts (`CONFIG`, `DEFAULT_TRAJ`, the
`cumotion_venv` interpreter, the GR00T `--out` dir); fix those before cloning to another machine.

## The three-environment split (the central architectural constraint)

cuRobo and the Unitree/Inspire SDKs **cannot co-install** in one environment, so the pipeline is split
across three Python environments and two machines. This is why planning and execution are separate
processes that hand off through a **file on disk**, not function calls.

| Stage | Where / env | Owns |
|---|---|---|
| `image_server.py` | **on the robot** (its python + `pyrealsense2`) | the head camera; publishes color+depth+intrinsics over ZMQ (port 5556) |
| `object_detection.py`, `real_plan_approach.py`, `real_viz.py` | host **`cumotion_venv`** (`/home/dishant/cumotion_venv/bin/python`) | torch + cuRobo + GPU planning |
| `real_pick.py`, `go_to_start.py` | host **`tv`** conda env | the ONLY env with BOTH `unitree_sdk2py` and `inspire_sdkpy` |

`object_detection.py` needs only zmq+cv2 (runs in `tv` or `cumotion_venv`). The host NIC `eno1`
(192.168.123.222) is on the robot subnet, so it reaches the camera (.164) and DDS-commands the
robot + hands. The Inspire hands also require **`Headless_driver_double.py`** running (bridges the
hands ↔ DDS) before `real_pick.py` can talk to them.

## Data flow / handoff contract

```
image_server (robot, ZMQ :5556)  --color+depth_raw+intrinsics-->  object_detection.py
object_detection --emit  -->  /tmp/glass_pose.json   {frame:"pelvis", center, top_z, radius_m, height_m}
real_plan_approach.py --pose /tmp/glass_pose.json  -->  approach_traj.json
real_pick.py --traj approach_traj.json  -->  drives arm + hand
```

- **`/tmp/glass_pose.json`** — object pose in the **base/pelvis frame**. Produced by
  `object_detection.py` applying `T_BASE_CAM` (a CAD/nominal `camera_optical → pelvis` extrinsic
  hardcoded near the top of that file) to the camera-frame centroid. If a grasp is geometrically
  wrong despite a good viz, suspect this extrinsic — it assumes waist=0 and is unverified against a
  known glass position.
- **`approach_traj.json`** — the executor's input. Schema: `joint_names` (must equal the 7
  `right_*_joint` names), `points` (approach+grasp, each `{positions[7], time, velocities[7]}`),
  and `lift_points` (grasp→+0.15 m up). Default path is the **GR00T repo `scripts/` dir**, which is
  bind-mounted into a container; `real_pick.py`'s `DEFAULT_TRAJ` and the planner's `--out` must agree.
  The planner exports **approach+grasp only — fingers are NOT closed and there is no lift in `points`**;
  `real_pick.py` owns finger closing and replays `lift_points` separately.

## Conventions that span files

- **G1 motor order is fixed at 29 joints**, indexed identically in `real_pick.py` and `go_to_start.py`:
  legs `0–11`, waist `12–14` (`WAIST_IDX`), left arm `15–21` (`LARM_IDX`), right arm `22–28` (`RARM_IDX`).
  `go_to_start.py` imports `Body`, `release_mode`, and these index constants directly from `real_pick`.
  Per-joint `MOTOR_KP`/`MOTOR_KD` gains live in `real_pick.py` in this same order.
- **"Sim start" pose** = waist 0, both arms 0; legs held where they are. Both scripts ramp to this; the
  planned trajectory begins from right-arm = 0, so the arm must be parked there first (`go_to_start.py`).
- **Inspire hand angle vector is 6 values** `[pinky, ring, middle, index, thumb_curl, thumb_ROT]`,
  `1000 = fully open`, `0 = closed`. Close is two-step: full thumb rotation/opposition first, *wait*,
  then curl — see `Hand.close()`.
- **Grasp geometry knobs** (`sg_dx`/`dygap`/`from_top`/`standoff`/`pitch_deg`) mean the same thing in
  `real_plan_approach.py` and `real_viz.py`. The wrist sits ~`sg_dx` (≈0.215 m) **behind** the glass in
  x because the hand reaches forward ~0.215 m. Tune them in `real_viz.py` (browser, localhost:8080)
  watching where the hand lands, then bake the values into `real_plan_approach.py`.
- **The executed grasp is a SIDE grasp** — approach from the robot's right (`grasp_approach_axis="y"`),
  palm facing +y. The **FRONT grasp** (approach head-on from −x) only exists in the offline exploration
  tools `sim_grasp_viz.py` (browser: side-vs-front comparison + coverage) and `grasp_coverage_map.py`
  (top-down reachability PNG, sweeping yaws `[0,-30,-60,-90]`); it is **not** wired into
  `real_plan_approach.py`/`real_pick.py`. Use these two tools to decide whether a target glass position is
  reachable at all before planning.
- The planner disables collisions on `HAND_LINKS` during `plan_grasp` so the open hand may sit touching
  the glass; and fixes the table box at `x=0.60` (back edge 0.25) so it clears the fixed-base robot —
  centering the table on the glass buries the arms-0 start inside the slab and **every plan returns None**.

## Safety model (applies to every executable here)

- **DRY-RUN by default.** `real_pick.py`, `go_to_start.py`, and `real_plan_approach.py` print the plan
  and send nothing unless `--execute` (`--probe` for the planner). Never add `--execute` to an example
  or default.
- Preconditions: robot **HUNG/supported**, zero-torque/low-level, and **no other `rt/lowcmd` publisher**
  (stop teleop first). The arm has a **fixed base** with a limited reachable band — keep the glass in the
  zone the planner/viz confirm.
- During the grasp, the arm holds the grasp pose in a **background thread** while the hand closes (so it
  never goes limp). **Ctrl-C freezes** (re-sends the current/held pose briefly), it does not relax.
- Large moves are guarded: `real_pick.py` aborts if the to-start delta > 3 rad without `--force`.

## Commands

There is no build/lint/test suite — these are operational scripts run by hand in their own envs. See
the README's "Run the full pick" for the canonical sequence. Key invocations (each in its env):

```bash
# robot:   python image_server.py
# laptop:  python Headless_driver_double.py            # hand↔DDS bridge (needed for real_pick)
# detect (tv or cumotion_venv):
python object_detection.py --host 192.168.123.164 --once --emit /tmp/glass_pose.json
python object_detection.py --calibrate                # click the cup to read its HSV band
# plan (cumotion_venv) — --probe sweeps a reachability grid and auto-picks:
/home/dishant/cumotion_venv/bin/python real_plan_approach.py --pose /tmp/glass_pose.json --probe
/home/dishant/cumotion_venv/bin/python real_viz.py            # browser tuner, http://localhost:8080
# reachability exploration (cumotion_venv, offline — not part of the pick path):
/home/dishant/cumotion_venv/bin/python sim_grasp_viz.py                                  # browser: SIDE vs FRONT + coverage
/home/dishant/cumotion_venv/bin/python grasp_coverage_map.py --out /tmp/grasp_coverage.png  # top-down reachability PNG
# execute (tv) — dry-run first, then --execute:
python go_to_start.py            # park arm at sim start (add --execute)
python real_pick.py              # full pick: approach→grasp→lift (add --execute)
python real_pick.py --open-hand --execute            # release
```

Useful `real_pick.py` knobs: `--speed 0.5 --curl 500 --thumb-rot 0 --rotate-wait 2.5 --no-lift`.
