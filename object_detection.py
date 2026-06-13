#!/usr/bin/env python3
# =============================================================================
# object_detection.py  —  REAL-robot depth object detection for the G1 head camera.
#
# Mirrors the sim depth detector (g1_perception/detect_object_server.py): it
# deprojects the aligned RealSense depth to 3D and localises the target object.
# Difference: the real target is a vividly COLOURED cup, so we IDENTIFY it by
# colour (HSV) — far more robust than pure table-plane geometry for one object —
# then use the aligned depth at those pixels for the 3D POSE.
#
# It does NOT open the camera (the image_server owns it). It SUBSCRIBES to the
# image_server ZMQ stream and reads color + depth_raw + intrinsics from the payload.
# Run it on the laptop (where cuRobo lives) pointing at the G1, or on the G1 itself.
#
# RUN:
#   # on the G1 (same box as the server):
#   python object_detection.py --once
#   # from the laptop, against the robot's server:
#   python object_detection.py --host 192.168.123.222 --save /tmp/glass_det.png
#   # tune the colour to YOUR object (click on it; prints HSV):
#   python object_detection.py --calibrate
#
# Output per detection: object centroid in the CAMERA optical frame (x right,
# y down, z forward, metres) + approx width/height + pixel bbox. To get a pose in
# the robot BASE/pelvis frame (what cuRobo needs), fill T_BASE_CAM below after a
# camera->base calibration (next step).
# =============================================================================
import argparse
import pickle
import sys
import time

import numpy as np
import cv2
import zmq

# --- camera_optical -> robot base (pelvis) extrinsic, 4x4 (metres). -----------
# Derived from the OFFICIAL G1 head d435 pose in our sim (torso_link camera, waist=0
# start pose) — valid because the real head cam IS that official camera. This is a
# CAD/nominal estimate: the camera-frame pose + radius are independent of it, but the
# BASE-frame numbers below should be sanity-checked against one known glass position
# (and re-derived if the waist isn't at 0). Set to None to report camera-frame only.
T_BASE_CAM = np.array([[ 0.0,      -0.737333,  0.67553,   0.05366 ],
                       [-1.0,       0.0,       0.0,       0.01753 ],
                       [ 0.0,      -0.67553,  -0.737333,  0.47387 ],
                       [ 0.0,       0.0,       0.0,       1.0     ]])

# D435 640x480 factory-ish fallback if the server doesn't publish intrinsics.
_FALLBACK_INTR = {"fx": 605.0, "fy": 605.0, "ppx": 320.0, "ppy": 240.0,
                  "width": 640, "height": 480, "depth_scale": 0.001}

# Default green band (OpenCV HSV: H 0-179). Tune with --calibrate for your cup.
GREEN_LO = (35, 60, 40)
GREEN_HI = (90, 255, 255)


def recv_latest(sock, poller, timeout_ms=2000):
    """Return the most recent payload (drains the queue so we never act on a stale frame)."""
    if not poller.poll(timeout_ms):
        return None
    msg = sock.recv()
    while poller.poll(0):          # drain backlog -> keep only newest
        msg = sock.recv()
    return pickle.loads(msg)


def split_color(payload):
    """Decode the JPEG and return the LEFT (color) half (server hconcats [color | depth_viz])."""
    arr = cv2.imdecode(np.frombuffer(payload["image"], np.uint8), cv2.IMREAD_COLOR)  # BGR
    if arr is None:
        return None
    depth = payload.get("depth_raw")
    w = depth.shape[1] if depth is not None else arr.shape[1] // 2
    return arr[:, :w].copy()       # BGR color, aligned to depth_raw


def get_intr(payload):
    intr = payload.get("intrinsics")
    if not intr:
        print("[warn] server published no intrinsics — using D435 fallback (run the patched image_server.py).")
        intr = dict(_FALLBACK_INTR)
    intr.setdefault("depth_scale", 0.001)
    return intr


def detect(color_bgr, depth_m, intr, hsv_lo, hsv_hi, min_area):
    """Identify the coloured object and localise it in the camera optical frame.
    Returns a dict (or None). depth_m is float metres, same HxW as color."""
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lo), np.array(hsv_hi))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask, "no colour blob"
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area:
        return None, mask, f"largest blob {int(area)}px < min_area {min_area}"

    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [c], -1, 255, -1)        # filled object mask
    x, y, w, h = cv2.boundingRect(c)

    ys, xs = np.where(clean > 0)
    Z = depth_m[ys, xs]
    valid = np.isfinite(Z) & (Z > 0.1) & (Z < 2.5)
    if valid.sum() < 30:
        return None, mask, f"object has too few valid-depth px ({int(valid.sum())}) — transparent/holes?"
    xs, ys, Z = xs[valid], ys[valid], Z[valid]

    # reject depth outliers (background bleeding through the mask edges)
    zmed = np.median(Z)
    keep = np.abs(Z - zmed) < 0.08
    xs, ys, Z = xs[keep], ys[keep], Z[keep]

    fx, fy, ppx, ppy = intr["fx"], intr["fy"], intr["ppx"], intr["ppy"]
    X = (xs - ppx) * Z / fx
    Y = (ys - ppy) * Z / fy
    pts = np.stack([X, Y, Z], axis=1)               # camera optical frame
    centroid = np.median(pts, axis=0)

    # approximate metric size from the pixel bbox at the object's median depth
    width_m = w * zmed / fx
    height_m = h * zmed / fy

    res = dict(
        centroid_cam=centroid.tolist(),             # [x_right, y_down, z_fwd] m
        distance=float(zmed),
        width_m=float(width_m), height_m=float(height_m),
        radius_m=float(width_m / 2.0),
        bbox=(int(x), int(y), int(w), int(h)),
        centroid_px=(int(np.median(xs)), int(np.median(ys))),
        n_pts=int(pts.shape[0]),
        hsv_mean=[int(v) for v in hsv[ys, xs].mean(axis=0)] if len(ys) else None,
    )
    if T_BASE_CAM is not None:
        Pb = (T_BASE_CAM[:3, :3] @ pts.T + T_BASE_CAM[:3, 3:4]).T
        cb = np.median(Pb, axis=0)
        res["centroid_base"] = cb.tolist()
        res["top_base_z"] = float(Pb[:, 2].max())   # highest point = object top in base frame
    return res, clean, "ok"


def draw(color_bgr, mask, res):
    vis = color_bgr.copy()
    overlay = vis.copy()
    overlay[mask > 0] = (0, 0, 255)
    vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
    if res:
        x, y, w, h = res["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cx, cy = res["centroid_px"]
        cv2.circle(vis, (cx, cy), 4, (0, 255, 255), -1)
        cm = res["centroid_cam"]
        lines = [f"R={res['radius_m']*100:.1f}cm (D={res['width_m']*100:.1f}) H={res['height_m']*100:.1f}cm",
                 f"cam xyz=({cm[0]:+.3f},{cm[1]:+.3f},{cm[2]:+.3f})m dist={res['distance']:.3f}",
                 f"pts={res['n_pts']}"]
        if "centroid_base" in res:
            b = res["centroid_base"]
            lines.append(f"base xyz=({b[0]:+.3f},{b[1]:+.3f},{b[2]:+.3f}) top_z={res['top_base_z']:+.3f}")
        for i, t in enumerate(lines):
            cv2.putText(vis, t, (x, max(18, y - 8 - 16 * (len(lines) - 1 - i))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def calibrate(color_bgr):
    """Click on the object; prints the HSV there so you can set GREEN_LO/HI."""
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    samples = []

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            h, s, v = hsv[y, x]
            samples.append((h, s, v))
            print(f"[calib] px=({x},{y}) HSV=({h},{s},{v})")
            if len(samples) >= 2:
                a = np.array(samples)
                print(f"[calib] suggest  GREEN_LO=({a[:,0].min()-10},{max(0,a[:,1].min()-50)},{max(0,a[:,2].min()-50)})"
                      f"  GREEN_HI=({a[:,0].max()+10},255,255)")
    cv2.namedWindow("calibrate (click object, q to quit)")
    cv2.setMouseCallback("calibrate (click object, q to quit)", on_mouse)
    while True:
        cv2.imshow("calibrate (click object, q to quit)", color_bgr)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="image_server host (G1 IP from the laptop)")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--hsv-lo", type=int, nargs=3, default=list(GREEN_LO), metavar=("H", "S", "V"))
    ap.add_argument("--hsv-hi", type=int, nargs=3, default=list(GREEN_HI), metavar=("H", "S", "V"))
    ap.add_argument("--min-area", type=int, default=400, help="min object blob area (px)")
    ap.add_argument("--once", action="store_true", help="one detection then exit")
    ap.add_argument("--emit", default=None, help="write the detected object pose (base frame) to this JSON")
    ap.add_argument("--save", default=None, help="write the annotated view to this PNG")
    ap.add_argument("--no-display", action="store_true", help="headless (print only)")
    ap.add_argument("--calibrate", action="store_true", help="click the object to read its HSV")
    args = ap.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.CONFLATE, 1)            # keep only the latest message
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    addr = f"tcp://{args.host}:{args.port}"
    sock.connect(addr)
    poller = zmq.Poller(); poller.register(sock, zmq.POLLIN)
    print(f"[net] subscribed {addr}; waiting for frames ...")

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
            if color is None or depth_raw is None:
                print("[err] payload missing color/depth_raw — start the server WITHOUT --no-depth")
                if args.once:
                    sys.exit(2)
                continue
            intr = get_intr(payload)

            if args.calibrate:
                calibrate(color)
                return

            depth_m = depth_raw.astype(np.float32) * float(intr["depth_scale"])
            res, mask, status = detect(color, depth_m, intr, tuple(args.hsv_lo), tuple(args.hsv_hi), args.min_area)

            if res:
                cm = res["centroid_cam"]
                print(f"[detect] OK  RADIUS={res['radius_m']*100:5.1f}cm (diam={res['width_m']*100:.1f}cm) "
                      f"height={res['height_m']*100:.1f}cm | cam=({cm[0]:+.3f},{cm[1]:+.3f},{cm[2]:+.3f})m "
                      f"dist={res['distance']:.3f}m | px_bbox={res['bbox']} n={res['n_pts']} hsv_mean={res['hsv_mean']}"
                      + (f" | base=({res['centroid_base'][0]:+.3f},{res['centroid_base'][1]:+.3f},"
                         f"{res['centroid_base'][2]:+.3f}) top_z={res['top_base_z']:+.3f}" if "centroid_base" in res else ""))
            else:
                print(f"[detect] FAIL — {status}")

            if args.emit and res and "centroid_base" in res:
                import json
                with open(args.emit, "w") as f:
                    json.dump({"frame": "pelvis",
                               "center": res["centroid_base"], "top_z": res["top_base_z"],
                               "radius_m": res["radius_m"], "height_m": res["height_m"],
                               "cam_centroid": res["centroid_cam"], "distance": res["distance"]}, f, indent=2)
                print(f"[emit] wrote object pose -> {args.emit}")
            elif args.emit and res and "centroid_base" not in res:
                print("[emit] SKIPPED: T_BASE_CAM is None (no base-frame pose to emit)")

            vis = draw(color, mask, res)
            if args.save:
                cv2.imwrite(args.save, vis)
                print(f"[save] {args.save}")
            if not args.no_display:
                cv2.imshow("object_detection (q quits)", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.once:
                break
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        sock.close(); ctx.term()


if __name__ == "__main__":
    main()
