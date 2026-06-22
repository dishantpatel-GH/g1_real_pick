#!/usr/bin/env python3
# =============================================================================
# microwave_detection.py  —  YOLOv11 microwave detector for the G1 head camera.
#
# Sibling of object_detection.py (the coloured-cup HSV detector). Same plumbing,
# different identifier: instead of an HSV colour band it runs an Ultralytics
# YOLO11 model pretrained on COCO and keeps only the "microwave" class (COCO id
# 68). When the server publishes depth + intrinsics, it ALSO deprojects the box
# to a 3D pose in the camera optical frame and the robot base/pelvis frame
# (same T_BASE_CAM extrinsic as object_detection.py), so a downstream stage can
# reason about where the microwave is.
#
# It does NOT open the camera (image_server.py owns it). It SUBSCRIBES to the
# image_server ZMQ stream (port 5556) and reads color + depth_raw + intrinsics
# from the payload — exactly like object_detection.py.
#
# ENV: run in cumotion_venv (it has torch+CUDA; ultralytics/torchvision/pyzmq
#      were installed there). torch runs YOLO11 on the GPU.
#
# RUN:
#   # on the G1 (same box as the server):
#   /home/dishant/cumotion_venv/bin/python microwave_detection.py --once
#   # from the laptop, against the robot's server:
#   /home/dishant/cumotion_venv/bin/python microwave_detection.py --host 192.168.123.164
#   # headless, save an annotated frame + emit the pose:
#   /home/dishant/cumotion_venv/bin/python microwave_detection.py --once --no-display \
#       --save /tmp/microwave_det.png --emit /tmp/microwave_pose.json
#   # bigger/more-accurate model:
#   /home/dishant/cumotion_venv/bin/python microwave_detection.py --model yolo11m.pt
#
# Per-detection output: class, confidence, pixel bbox, and (with depth) the
# microwave centroid in the camera optical frame (x right, y down, z forward,
# metres) and the robot base/pelvis frame.
# =============================================================================
import argparse
import pickle
import sys
import time

import numpy as np
import cv2
import zmq

# --- camera_optical -> robot base (pelvis) extrinsic, 4x4 (metres). -----------
# Identical to object_detection.py — the real head cam IS the official G1 head
# d435 (torso_link camera, waist=0). CAD/nominal estimate; sanity-check against a
# known position if the waist isn't at 0. Set to None to report camera-frame only.
T_BASE_CAM = np.array([[ 0.0,      -0.737333,  0.67553,   0.05366 ],
                       [-1.0,       0.0,       0.0,       0.01753 ],
                       [ 0.0,      -0.67553,  -0.737333,  0.47387 ],
                       [ 0.0,       0.0,       0.0,       1.0     ]])

# D435 640x480 factory-ish fallback if the server doesn't publish intrinsics.
_FALLBACK_INTR = {"fx": 605.0, "fy": 605.0, "ppx": 320.0, "ppy": 240.0,
                  "width": 640, "height": 480, "depth_scale": 0.001}

DEFAULT_TARGETS = ["microwave"]     # COCO class names to keep (microwave = id 68)


def recv_latest(sock, poller, timeout_ms=2000):
    """Return the most recent payload (drains the queue so we never act on a stale frame)."""
    if not poller.poll(timeout_ms):
        return None
    msg = sock.recv()
    while poller.poll(0):          # drain backlog -> keep only newest
        msg = sock.recv()
    return pickle.loads(msg)


def split_color(payload):
    """Decode the JPEG and return the color image (BGR), aligned to depth_raw.

    The server hconcats [color | depth_viz] only when depth is enabled, so the
    color half is the left depth.shape[1] columns. With depth disabled the head
    frame is the color image alone — use it whole (do NOT halve)."""
    arr = cv2.imdecode(np.frombuffer(payload["image"], np.uint8), cv2.IMREAD_COLOR)  # BGR
    if arr is None:
        return None
    depth = payload.get("depth_raw")
    if depth is not None:
        return arr[:, :depth.shape[1]].copy()
    return arr


def _pick(d, *keys):
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None


def get_intr(payload, frame_shape=None):
    """Return a normalised intrinsics dict with fx/fy/ppx/ppy/depth_scale always present.

    Different image_server builds name things differently (some publish the principal
    point as cx/cy, some omit it entirely). We alias the common names and, if the
    principal point is still missing, fall back to the image centre — a safe estimate
    for a RealSense (principal point sits within a few px of centre)."""
    intr = payload.get("intrinsics")
    if not intr:
        print("[warn] server published no intrinsics — using D435 fallback (run the patched image_server.py).")
        return dict(_FALLBACK_INTR)

    fx = _pick(intr, "fx", "focal_x")
    fy = _pick(intr, "fy", "focal_y")
    ppx = _pick(intr, "ppx", "cx", "principal_x", "ppx_px")
    ppy = _pick(intr, "ppy", "cy", "principal_y", "ppy_px")
    width = _pick(intr, "width", "w")
    height = _pick(intr, "height", "h")

    if fx is None or fy is None:
        print(f"[warn] intrinsics missing focal length (keys: {sorted(intr)}) — using D435 fallback focal.")
        fx = fx if fx is not None else _FALLBACK_INTR["fx"]
        fy = fy if fy is not None else _FALLBACK_INTR["fy"]
    if ppx is None or ppy is None:
        if frame_shape is not None:
            H, W = frame_shape[0], frame_shape[1]
        else:
            H, W = (height or _FALLBACK_INTR["height"]), (width or _FALLBACK_INTR["width"])
        ppx = ppx if ppx is not None else W / 2.0
        ppy = ppy if ppy is not None else H / 2.0
        print(f"[warn] intrinsics published no principal point (keys: {sorted(intr)}); "
              f"using image centre ({ppx:.1f},{ppy:.1f}).")

    out = dict(intr)
    out.update(fx=float(fx), fy=float(fy), ppx=float(ppx), ppy=float(ppy))
    out.setdefault("depth_scale", 0.001)
    return out


def resolve_target_ids(model, names):
    """Map requested COCO class names -> model class ids. Exits if any is unknown."""
    name_to_id = {n: i for i, n in model.names.items()}
    ids = []
    for n in names:
        if n not in name_to_id:
            print(f"[err] '{n}' is not a class of this model. Available include: "
                  f"{sorted(name_to_id)[:12]}... ({len(name_to_id)} total)")
            sys.exit(2)
        ids.append(name_to_id[n])
    return ids


def localize(bbox, depth_m, intr, depth_range):
    """Deproject the box to a 3D pose. bbox=(x,y,w,h) px; depth_m is float metres.

    Returns a dict (cam + base centroid, distance) or None if too few valid-depth
    pixels (microwave interiors are dark/holey, so this can legitimately fail)."""
    if depth_m is None:
        return None
    x, y, w, h = bbox
    H, W = depth_m.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return None

    sub = depth_m[y0:y1, x0:x1]
    ys, xs = np.mgrid[y0:y1, x0:x1]
    Z = sub.reshape(-1)
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)
    lo, hi = depth_range
    valid = np.isfinite(Z) & (Z > lo) & (Z < hi)
    if valid.sum() < 50:
        return None
    xs, ys, Z = xs[valid], ys[valid], Z[valid]

    # reject background bleeding through the box edges: keep the nearest cluster
    zmed = float(np.median(Z))
    keep = np.abs(Z - zmed) < 0.20
    xs, ys, Z = xs[keep], ys[keep], Z[keep]
    if len(Z) < 50:
        return None

    fx, fy, ppx, ppy = intr["fx"], intr["fy"], intr["ppx"], intr["ppy"]
    X = (xs - ppx) * Z / fx
    Y = (ys - ppy) * Z / fy
    pts = np.stack([X, Y, Z], axis=1)          # camera optical frame
    centroid = np.median(pts, axis=0)

    res = dict(
        centroid_cam=centroid.tolist(),        # [x_right, y_down, z_fwd] m
        distance=float(np.median(Z)),
        n_pts=int(pts.shape[0]),
    )
    if T_BASE_CAM is not None:
        Pb = (T_BASE_CAM[:3, :3] @ pts.T + T_BASE_CAM[:3, 3:4]).T
        res["centroid_base"] = np.median(Pb, axis=0).tolist()
    return res


def detect(model, color_bgr, depth_m, intr, target_ids, conf, device, depth_range):
    """Run YOLO11 on the BGR frame, keep target classes, localise each with depth.
    Returns a list of detections sorted by confidence (highest first)."""
    r = model.predict(color_bgr, classes=target_ids, conf=conf, device=device, verbose=False)[0]
    dets = []
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        d = dict(
            name=model.names[int(box.cls[0])],
            conf=float(box.conf[0]),
            bbox=bbox,
            center_px=(int((x1 + x2) / 2), int((y1 + y2) / 2)),
        )
        loc = localize(bbox, depth_m, intr, depth_range)
        if loc:
            d.update(loc)
        dets.append(d)
    dets.sort(key=lambda d: d["conf"], reverse=True)
    return dets


def draw(color_bgr, dets, fps=None):
    vis = color_bgr.copy()
    for d in dets:
        x, y, w, h = d["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cx, cy = d["center_px"]
        cv2.circle(vis, (cx, cy), 4, (0, 255, 255), -1)
        lines = [f"{d['name']} {d['conf']*100:.0f}%"]
        if "centroid_cam" in d:
            cm = d["centroid_cam"]
            lines.append(f"cam=({cm[0]:+.2f},{cm[1]:+.2f},{cm[2]:+.2f})m d={d['distance']:.2f}")
        if "centroid_base" in d:
            b = d["centroid_base"]
            lines.append(f"base=({b[0]:+.2f},{b[1]:+.2f},{b[2]:+.2f})")
        for i, t in enumerate(lines):
            yt = max(14, y - 6 - 15 * (len(lines) - 1 - i))
            cv2.putText(vis, t, (x, yt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, t, (x, yt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if fps is not None:
        cv2.putText(vis, f"{fps:.1f} FPS", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser(description="YOLOv11 microwave detector over the G1 image_server ZMQ stream")
    ap.add_argument("--host", default="127.0.0.1", help="image_server host (G1 IP from the laptop)")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--model", default="yolo11n.pt",
                    help="Ultralytics weights (yolo11n/s/m/l/x.pt); auto-downloads on first use")
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS,
                    help="COCO class names to keep (default: microwave)")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--device", default=None, help="torch device, e.g. 0 / cuda:0 / cpu (default: auto)")
    ap.add_argument("--depth-range", type=float, nargs=2, default=[0.2, 6.0], metavar=("MIN", "MAX"),
                    help="valid depth band (m) for 3D localisation")
    ap.add_argument("--once", action="store_true", help="one detection then exit")
    ap.add_argument("--emit", default=None, help="write the best (highest-conf) microwave detection to this JSON")
    ap.add_argument("--save", default=None, help="write the annotated view to this PNG")
    ap.add_argument("--no-display", action="store_true", help="headless (print only)")
    args = ap.parse_args()

    # ultralytics import is deferred so --help works without the heavy torch stack.
    from ultralytics import YOLO
    import torch
    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[yolo] loading {args.model} on device={device} ...")
    model = YOLO(args.model)
    target_ids = resolve_target_ids(model, args.targets)
    print(f"[yolo] targets {args.targets} -> class ids {target_ids}")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.CONFLATE, 1)            # keep only the latest message
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    addr = f"tcp://{args.host}:{args.port}"
    sock.connect(addr)
    poller = zmq.Poller(); poller.register(sock, zmq.POLLIN)
    print(f"[net] subscribed {addr}; waiting for frames ...")

    last_t = None
    printed_intr = False
    try:
        while True:
            payload = recv_latest(sock, poller)
            if payload is None:
                print("[net] no frame (is image_server.py running?)")
                if args.once:
                    sys.exit(2)
                continue
            color = split_color(payload)
            if color is None:
                print("[err] payload missing/undecodable color image")
                if args.once:
                    sys.exit(2)
                continue
            intr = get_intr(payload, frame_shape=color.shape)
            if not printed_intr:
                print(f"[intr] server published: {payload.get('intrinsics')}")
                print(f"[intr] using fx={intr['fx']:.1f} fy={intr['fy']:.1f} "
                      f"ppx={intr['ppx']:.1f} ppy={intr['ppy']:.1f} depth_scale={intr['depth_scale']}")
                printed_intr = True
            depth_raw = payload.get("depth_raw")
            depth_m = depth_raw.astype(np.float32) * float(intr["depth_scale"]) if depth_raw is not None else None

            dets = detect(model, color, depth_m, intr, target_ids, args.conf, device, tuple(args.depth_range))

            now = time.time()
            fps = (1.0 / (now - last_t)) if last_t else None
            last_t = now

            if dets:
                for d in dets:
                    extra = ""
                    if "centroid_cam" in d:
                        cm = d["centroid_cam"]
                        extra = f" | cam=({cm[0]:+.3f},{cm[1]:+.3f},{cm[2]:+.3f})m dist={d['distance']:.3f}"
                        if "centroid_base" in d:
                            b = d["centroid_base"]
                            extra += f" base=({b[0]:+.3f},{b[1]:+.3f},{b[2]:+.3f})"
                    else:
                        extra = " | no depth -> 2D only"
                    print(f"[detect] {d['name']} {d['conf']*100:5.1f}% bbox={d['bbox']}{extra}")
            else:
                print(f"[detect] no {'/'.join(args.targets)} (conf>={args.conf})")

            if args.emit and dets:
                import json
                best = dets[0]
                rec = {"class": best["name"], "confidence": best["conf"], "bbox": list(best["bbox"]),
                       "center_px": list(best["center_px"])}
                if "centroid_cam" in best:
                    rec["frame"] = "pelvis" if "centroid_base" in best else "camera_optical"
                    rec["cam_centroid"] = best["centroid_cam"]
                    rec["distance"] = best["distance"]
                    if "centroid_base" in best:
                        rec["center"] = best["centroid_base"]
                with open(args.emit, "w") as f:
                    json.dump(rec, f, indent=2)
                print(f"[emit] wrote best microwave -> {args.emit}")

            vis = draw(color, dets, fps)
            if args.save:
                cv2.imwrite(args.save, vis)
                print(f"[save] {args.save}")
            if not args.no_display:
                cv2.imshow("microwave_detection (q quits)", vis)
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
