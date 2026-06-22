#!/usr/bin/env python3
# =============================================================================
# microwave_handle_real.py  —  REAL-robot door-handle + microwave-centre localiser.
#
# Real-robot sibling of microwave_handle_sim.py. It runs the IDENTICAL perception
# (so what we validated in sim can't drift from what runs on the robot):
#   1. YOLOv11 INSTANCE SEGMENTATION -> appliance region seed (microwave/tv/...;
#      the COCO label does not matter, we only need the region), refined to the dark
#      body silhouette (white TABLE dropped) and closed to keep the handle area.
#   2. Find the HANDLE by the YELLOW TAPE stuck on it (HSV colour, constrained to the
#      microwave bbox). Colour is FAR more robust than the chrome strip's brightness,
#      which is view/lighting dependent (from an oblique angle the chrome goes dark).
#   3. Deproject the tape pixels with the aligned DEPTH -> handle 3D centroid in the
#      camera optical frame -> pelvis/base frame (T_BASE_CAM).
#   4. handle front sits HANDLE_PROTRUDE (3.8 cm) ahead of the body; the handle is
#      39 cm from the +y (left) edge of the 49x31x27.5 cm body -> infer microwave x,y;
#      the body HEIGHT (z) comes from the front-face vertical extent (region_center_z).
#
# DIFFERENCE from the sim file: the RGB + DEPTH come from the G1 head RealSense via
# the image_server ZMQ stream (run image_server.py on the robot at 1280x720), NOT a
# MuJoCo render. The reusable perception fns are imported from microwave_handle_sim;
# the ZMQ plumbing (recv/split/intrinsics) is imported from microwave_detection.
#
# ENV: laptop cumotion_venv (torch+CUDA for YOLO; pyzmq/opencv). Point --host at the
# robot. The robot only runs image_server.py (pyrealsense2); it does not need this.
#
# RUN:
#   # robot:  python image_server.py            # 1280x720 head RGB-D over ZMQ :5556 (default)
#   # laptop (cumotion_venv):
#   /home/dishant/cumotion_venv/bin/python microwave_handle_real.py --host 192.168.123.164
#   # headless one-shot: save the annotated view + emit the pose JSON:
#   /home/dishant/cumotion_venv/bin/python microwave_handle_real.py --host 192.168.123.164 \
#       --once --no-display --save /tmp/mw_handle.png --emit /tmp/microwave_pose.json
#   # tune the white-strip threshold for your lighting / a bigger model:
#   ... --white-thr 140 --model yolo11m-seg.pt
# =============================================================================
import argparse
import sys
import time

import numpy as np
import cv2
import zmq

# ZMQ plumbing for THIS image_server (handles the [color|depth_viz] hconcat + cx/cy intrinsics).
from microwave_detection import recv_latest, split_color, get_intr
# The SAME perception + real microwave/camera geometry that sim validated.
from microwave_handle_sim import (appliance_region, find_handle_yellow, deproject, region_center_z,
                                  microwave_center_from_handle, T_BASE_CAM, HALF, HANDLE_PROTRUDE)


def draw(bgr, region, hmask, hd, mw):
    """Annotated view: refined region (green), handle strip (red), 3D readout."""
    vis = bgr.copy()
    if region is not None:
        m = region > 0
        vis[m] = (0.55 * vis[m] + 0.45 * np.array([60, 200, 60])).astype(np.uint8)
    if hmask is not None:
        vis[hmask > 0] = (0, 0, 255)
    if hd is not None:
        cv2.circle(vis, hd["px"], 7, (0, 255, 255), 2)
        lines = [f"handle base ({hd['base'][0]:+.3f},{hd['base'][1]:+.3f},{hd['base'][2]:+.3f}) m  n={hd['n']}"]
        if mw is not None:
            lines.append(f"microwave  ({mw[0]:+.3f},{mw[1]:+.3f},{mw[2]:+.3f}) m")
        for i, t in enumerate(lines):
            org = (12, 30 + 26 * i)
            cv2.putText(vis, t, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, t, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def process(model, color, depth_m, intr, conf, yellow_lo, yellow_hi):
    """Run the full localiser on one RGB-D frame. Returns (region, hmask, hd, mw, status, hinfo)."""
    region, rinfo = appliance_region(model, color, conf)
    if region is None:
        return None, None, None, None, f"region: {rinfo}", {}
    hmask, hinfo = find_handle_yellow(color, region, hsv_lo=yellow_lo, hsv_hi=yellow_hi)
    if hmask is None:
        return region, None, None, None, f"handle (yellow tape): {hinfo.get('reason')}", hinfo
    if depth_m is None:
        return region, hmask, None, None, "no depth_raw (start image_server WITHOUT --no-depth)", hinfo
    hd, dstat = deproject(hmask, depth_m, intr, T_BASE_CAM)
    if hd is None:
        return region, hmask, None, None, f"deproject: {dstat}", hinfo
    cz = region_center_z(region, depth_m, intr, T_BASE_CAM)          # body height from the front-face extent
    mw = microwave_center_from_handle(hd["base"], cz)
    return region, hmask, hd, mw, "ok", hinfo


def main():
    ap = argparse.ArgumentParser(description="REAL G1 microwave door-handle + centre localiser (image_server ZMQ)")
    ap.add_argument("--host", default="127.0.0.1", help="image_server host (G1 IP from the laptop)")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--model", default="yolo11n-seg.pt", help="Ultralytics SEG weights (auto-downloads)")
    ap.add_argument("--conf", type=float, default=0.10, help="YOLO confidence (low; class label irrelevant)")
    ap.add_argument("--yellow-lo", type=int, nargs=3, default=[15, 60, 45], metavar=("H", "S", "V"),
                    help="lower HSV bound for the yellow handle tape (OpenCV H 0-179)")
    ap.add_argument("--yellow-hi", type=int, nargs=3, default=[45, 255, 255], metavar=("H", "S", "V"),
                    help="upper HSV bound for the yellow handle tape")
    ap.add_argument("--depth-range", type=float, nargs=2, default=[0.1, 5.0], metavar=("MIN", "MAX"))
    ap.add_argument("--once", action="store_true", help="one detection then exit")
    ap.add_argument("--emit", default=None, help="write the microwave + handle pose (base frame) to this JSON")
    ap.add_argument("--save", default=None, help="write the annotated view to this PNG")
    ap.add_argument("--no-display", action="store_true", help="headless (print only)")
    args = ap.parse_args()

    from ultralytics import YOLO
    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[yolo] loading {args.model} (seg) on device={device} ...")
    model = YOLO(args.model)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.CONFLATE, 1)              # keep only the latest frame
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    addr = f"tcp://{args.host}:{args.port}"
    sock.connect(addr)
    poller = zmq.Poller(); poller.register(sock, zmq.POLLIN)
    print(f"[net] subscribed {addr}; waiting for frames ...")

    printed_intr = False
    try:
        while True:
            payload = recv_latest(sock, poller)
            if payload is None:
                print("[net] no frame (is image_server.py running with depth enabled?)")
                if args.once:
                    sys.exit(2)
                continue
            color = split_color(payload)
            depth_raw = payload.get("depth_raw")
            if color is None:
                print("[err] payload missing/undecodable color image")
                if args.once:
                    sys.exit(2)
                continue
            intr = get_intr(payload, frame_shape=color.shape)
            if not printed_intr:
                print(f"[intr] fx={intr['fx']:.1f} fy={intr['fy']:.1f} ppx={intr['ppx']:.1f} "
                      f"ppy={intr['ppy']:.1f} depth_scale={intr['depth_scale']} frame={color.shape[1]}x{color.shape[0]}")
                printed_intr = True
            depth_m = depth_raw.astype(np.float32) * float(intr["depth_scale"]) if depth_raw is not None else None

            region, hmask, hd, mw, status, hinfo = process(model, color, depth_m, intr,
                                                            args.conf, tuple(args.yellow_lo), tuple(args.yellow_hi))
            if hinfo and "bbox" in hinfo:
                print(f"[yellow] tape bbox={hinfo.get('bbox')} area={hinfo.get('area')}")

            if hd is not None:
                b = hd["base"]
                print(f"[detect] handle base=({b[0]:+.3f},{b[1]:+.3f},{b[2]:+.3f})m "
                      f"cam=({hd['cam'][0]:+.3f},{hd['cam'][1]:+.3f},{hd['cam'][2]:+.3f}) n={hd['n']} "
                      f"| microwave center=({mw[0]:+.3f},{mw[1]:+.3f},{mw[2]:+.3f})m")
            else:
                print(f"[detect] FAIL — {status}")

            if args.emit and hd is not None:
                import json
                with open(args.emit, "w") as f:
                    json.dump({
                        "frame": "pelvis",
                        "handle_center": [float(v) for v in hd["base"]],
                        "microwave_center": [float(v) for v in mw],
                        "microwave_dims_m": [float(2 * h) for h in HALF],     # (depth x, width y, height z)
                        "handle_from_left_m": 0.39,                            # handle centre, from the +y (left) edge
                        "handle_protrude_m": float(HANDLE_PROTRUDE),
                        "cam_centroid": [float(v) for v in hd["cam"]],
                    }, f, indent=2)
                print(f"[emit] wrote microwave + handle pose -> {args.emit}")

            vis = draw(color, region, hmask, hd, mw)
            if args.save:
                cv2.imwrite(args.save, vis); print(f"[save] {args.save}")
                # debug siblings so a failed run is diagnosable (raw frame + region + bright mask)
                stem = args.save.rsplit(".", 1)[0]
                cv2.imwrite(f"{stem}_raw.png", color)
                if region is not None:
                    rv = color.copy(); rv[region > 0] = (0.5 * rv[region > 0] + 0.5 * np.array([60, 200, 60])).astype(np.uint8)
                    cv2.imwrite(f"{stem}_region.png", rv)
                if hinfo.get("bright") is not None:
                    cv2.imwrite(f"{stem}_bright.png", hinfo["bright"])
            if not args.no_display:
                cv2.imshow("microwave_handle_real (q quits)", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.once:
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        sock.close(); ctx.term()


if __name__ == "__main__":
    main()
