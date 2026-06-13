# g1_real_pick — depth-driven cuMotion pick for the real Unitree G1 + Inspire FTP hand

One folder gathering everything for the real pick: **image server**, **cuRobo** (planner + config +
viewer), the **approach/grasp/lift** scripts, and the **inspire hand SDK**. Most entries are symlinks to the
live sources on this machine (so there's no drift) — the only first-party file here is **`real_pick.py`**.

Pipeline: head **RealSense depth** (no AprilTag) → **cuRobo `plan_grasp`** → drive the G1 right arm
(Unitree SDK) + close the Inspire FTP hand → lift.

## The three environments (this can't be one env — cuRobo and the Unitree/Inspire SDKs don't co-install cleanly)
| Stage | Env | Why |
|---|---|---|
| image server | on the **robot** (its python + pyrealsense2) | owns the head camera |
| detect + **plan** (cuRobo) | **`cumotion_venv`** (host) | torch + curobo + GPU |
| **execute** (arm) + **hand** | **`tv`** conda env (host) | has BOTH `unitree_sdk2py` and `inspire_sdkpy` |

`object_detection.py` needs zmq+cv2 (run it in `tv`). The host NIC `eno1` (192.168.123.222) is on the robot
subnet, so it reaches the camera (.164) and DDS-commands the robot + hands.

## Files
- **`real_pick.py`** — THE full pick in ONE process (`tv` env): approach + grasp(fingers) + lift, **no ENTER,
  no separate hand script**. Body = `unitree_sdk2py` (rt/lowcmd); hand = `inspire_sdkpy` (rt/inspire_hand/ctrl/r).
- **`go_to_start.py`** — move the upper body to the sim START pose (waist=0, both arms=0; legs held) and HOLD
  FOREVER (`tv` env). Park the arm here before `real_pick.py`, or to validate the start pose. `--execute`.
- `real_plan_approach.py` → cuMotion planner (`cumotion_venv`); writes the approach+grasp+lift trajectory.
- `real_viz.py` → browser tuner (`cumotion_venv`, http://localhost:8080) — dial SG_DX/DYGAP/FROMTOP/PITCH.
- `object_detection.py` → depth glass detector (`tv` env); `--emit` writes the pelvis-frame pose.
- `image_server.py` → runs on the robot; publishes color+depth+**intrinsics** over ZMQ.
- `Headless_driver_double.py` → run on the laptop first; bridges the Inspire hands ↔ DDS.
- `curobo/`, `config/`, `inspire_hand_sdk/` → the cuRobo repo, the robot config, and the Inspire SDK.

## Run the full pick
```bash
# 0) laptop: hand bridge
python Headless_driver_double.py
# 0) robot: image server (publishes intrinsics; any resolution)
python image_server.py            # on the G1 (.164)

# 1) tv env: detect the glass -> pelvis-frame pose
python object_detection.py --host 192.168.123.164 --once --emit /tmp/glass_pose.json

# 2) cumotion_venv: plan approach+grasp+lift  (tune --from-top / --pitch-deg / --dygap if needed)
/home/dishant/cumotion_venv/bin/python real_plan_approach.py --pose /tmp/glass_pose.json

# 3) tv env: ONE command does approach -> grasp(fingers) -> lift
python real_pick.py                 # DRY RUN (prints the plan)
python real_pick.py --execute       # full pick
python real_pick.py --open-hand --execute    # release afterwards
```
Knobs: `real_pick.py --speed 0.5 --curl 500 --thumb-rot 0 --rotate-wait 2.5 --no-lift`.

## Safety / prereqs
Robot HUNG/supported, zero-torque/low-level, **no other rt/lowcmd publisher** (stop teleop). `real_pick.py`
dry-runs by default; the arm holds the grasp pose in a background thread while the hand closes; Ctrl-C freezes.
The arm can only reach a limited band (fixed base) — keep the glass in the reachable zone the planner/viz confirm.

## Notes
- Entries are **symlinks** to the live sources (machine-local). To make this a portable/cloneable repo, replace
  the symlinks with copies and fix the absolute paths (cuRobo `CONFIG`, the traj `--traj`/`--out` defaults).
- cuRobo planning stays a separate step because its env can't be merged with the Unitree/Inspire SDKs without a
  heavy rebuild. `real_pick.py` consumes the planned trajectory file.
