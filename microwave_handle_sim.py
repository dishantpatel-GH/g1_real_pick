#!/usr/bin/env python3
# =============================================================================
# microwave_handle_sim.py  —  SIM validation of the REAL door-handle localiser.
#
# Goal (this is the SIM rehearsal; real comes later): from the G1 head camera
# RGB + DEPTH, localise the microwave DOOR HANDLE — and from it the microwave
# CENTRE — WITHOUT using any simulation ground truth, then COMPARE the recovered
# pose to the true sim pose to measure the method's error.
#
# Method (identical to what will run on the real robot):
#   1. YOLOv11 INSTANCE SEGMENTATION -> appliance region MASK (microwave/tv/oven;
#      the COCO class label does NOT matter, we only need the region). A mask, not
#      a bbox, because the bbox is bigger than the unit and our table is WHITE —
#      the bbox would sweep in white table pixels and ruin the white-strip search.
#   2. Inside that (hole-filled) mask, find the HANDLE by its WHITE STRIP: the unit
#      is black, the only bright thing on it is the chrome strip on the handle's
#      FRONT + TOP. Brightest blob in the region = handle.  (The white TABLE is
#      outside the mask, so it can't be confused for the strip.)
#   3. Deproject the handle pixels with the aligned DEPTH -> handle 3D centroid in
#      the camera optical frame -> pelvis/base frame (T_BASE_CAM).  [same maths as
#      object_detection.py]
#   4. The handle's front sits HANDLE_PROTRUDE (3.8 cm) ahead of the body, at a
#      known offset within the 49x31x27.5 cm body -> infer the microwave CENTRE
#      from the single handle detection (assumes the unit is axis-aligned / yaw
#      known, which holds in sim; in real, refine with the body mask if needed).
#
# The SIM CAMERA is MuJoCo: it renders the SAME scene geometry as sim_grasp_viz.py
# (table + microwave, dims/positions imported from it) from the G1 head-cam pose
# (T_BASE_CAM) at 1280x720, giving RGB + metric DEPTH.  Nothing here touches the
# real robot or the ZMQ image_server — the perception fns are written to be reused
# on the real stream later (they take rgb, depth_m, intr, T).
#
# RUN (cumotion_venv; needs an EGL/GL context for offscreen MuJoCo):
#   MUJOCO_GL=egl /home/dishant/cumotion_venv/bin/python microwave_handle_sim.py
#   ... --model yolo11m-seg.pt --save-dir /tmp/mwsim --white-thr 150
# =============================================================================
import argparse, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")        # offscreen GL for headless render
import numpy as np
import cv2

import sim_grasp_viz as sv                        # the single source of truth for the scene geometry

# camera_optical -> pelvis/base extrinsic (same nominal G1 head-d435 pose as object_detection.py).
T_BASE_CAM = np.array([[ 0.0, -0.737333,  0.67553,  0.05366],
                       [-1.0,  0.0,       0.0,      0.01753],
                       [ 0.0, -0.67553,  -0.737333, 0.47387],
                       [ 0.0,  0.0,       0.0,      1.0    ]])

# Real handle measurement the user supplied: the handle's front protrudes this far
# AHEAD of the main body's front face.
HANDLE_PROTRUDE = 0.038                            # 3.8 cm

# Sim head-cam image: 1280x720, D435-ish vertical FOV. fx=fy (square px), centre principal point.
RENDER_W, RENDER_H = 1280, 720
CAM_FOVY_DEG = 42.5
_FY = (RENDER_H / 2.0) / np.tan(np.deg2rad(CAM_FOVY_DEG) / 2.0)
SIM_INTR = {"fx": float(_FY), "fy": float(_FY), "ppx": RENDER_W / 2.0, "ppy": RENDER_H / 2.0,
            "width": RENDER_W, "height": RENDER_H, "depth_scale": 1.0}

# ---- known body layout (imported so it never drifts from sim_grasp_viz.py) ------------------
D = sv.MW_DIMS                                     # (depth x, width y, height z) = (0.31, 0.49, 0.275)
HALF = [d / 2 for d in D]
GT_MW_CENTER = np.array([sv.MW_CX, sv.MW_CY, sv.MW_CZ])
GT_HANDLE = np.array([sv.MW_FRONT_X - HANDLE_PROTRUDE, sv.MW_HANDLE_Y, sv.MW_CZ])   # front-strip centre, base frame
HANDLE_FROM_RIGHT = HALF[1] - sv.MW_HANDLE_FROM_LEFT + HALF[1]   # = MW_DIMS[1] - 0.39 = 0.10 (10 cm from right)
APPLIANCE_CLASSES = {"microwave", "tv", "oven", "refrigerator", "laptop", "book"}   # any dark-box class; label is irrelevant

# ---- GLASS (the green-wrapped cup the RIGHT arm grabs) -- validate its colour localiser too ----
GT_GLASS = np.array([sv.CX, sv.CY, sv.CZ])         # sim glass AXIS centre (pelvis frame), for the comparison
GLASS_R, GLASS_H = sv.R, sv.H                       # true glass radius / height
GREEN_LO, GREEN_HI = (35, 60, 40), (90, 255, 255)  # green-wrapper HSV band (same as object_detection.py)


# ===================== SIM CAMERA (MuJoCo) ==========================================
def build_mjcf():
    """MJCF mirroring sim_grasp_viz.py: white table + black microwave whose handle has a WHITE
    strip on its FRONT and TOP only; a head camera placed at the G1 head-cam pose (T_BASE_CAM)."""
    right = T_BASE_CAM[:3, 0]; up = -T_BASE_CAM[:3, 1]; pos = T_BASE_CAM[:3, 3]   # MuJoCo cam: +x right, +y up, looks -z
    fx_w = sv.MW_FRONT_X
    door_cy = (sv.MW_HINGE_Y + sv.MW_HANDLE_Y) / 2.0
    door_w = sv.MW_HINGE_Y - sv.MW_HANDLE_Y
    tbl_pos = sv.TABLE[1][:3]; tbl_h = [d / 2 for d in sv.TABLE[0]]
    hb_x = (fx_w - sv.DOOR_THICK + (fx_w - HANDLE_PROTRUDE)) / 2.0                 # handle bar centre x (door surface .. 3.8cm out)
    hb_hx = abs((fx_w - sv.DOOR_THICK) - (fx_w - HANDLE_PROTRUDE)) / 2.0
    hb_hz = HALF[2] * 0.6
    def s(p): return " ".join(f"{v:.5f}" for v in p)
    buttons = "".join(
        f'<geom name="btn{i}" type="box" size="0.004 0.020 0.013" '
        f'pos="{fx_w + 0.002:.5f} {sv.MW_CY - HALF[1] + 0.05:.5f} {sv.MW_CZ + 0.07 - 0.035 * i:.5f}" '
        f'rgba="0.45 0.45 0.5 1"/>' for i in range(4))
    return f"""
<mujoco>
  <visual>
    <global offwidth="{RENDER_W}" offheight="{RENDER_H}"/>
    <headlight ambient="0.55 0.55 0.55" diffuse="0.45 0.45 0.45" specular="0.1 0.1 0.1"/>
    <quality shadowsize="4096"/>
  </visual>
  <worldbody>
    <light pos="0.4 0.3 1.7" dir="-0.2 -0.1 -1" diffuse="0.6 0.6 0.6"/>
    <camera name="head" pos="{s(pos)}" xyaxes="{s(right)} {s(up)}" fovy="{CAM_FOVY_DEG:.4f}"/>
    <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 {tbl_pos[2]-tbl_h[2]:.4f}" rgba="0.25 0.25 0.28 1"/>
    <geom name="table" type="box" size="{s(tbl_h)}" pos="{s(tbl_pos)}" rgba="0.90 0.90 0.91 1"/>
    <geom name="mw_body" type="box" size="{s(HALF)}" pos="{s((sv.MW_CX, sv.MW_CY, sv.MW_CZ))}" rgba="0.05 0.05 0.06 1"/>
    <geom name="mw_door" type="box" size="{sv.DOOR_THICK/2:.5f} {door_w*0.49:.5f} {HALF[2]*0.90:.5f}"
          pos="{s((fx_w - sv.DOOR_THICK/2, door_cy, sv.MW_CZ))}" rgba="0.04 0.04 0.05 1"/>
    <geom name="mw_glass" type="box" size="0.003 {door_w*0.40:.5f} {HALF[2]*0.72:.5f}"
          pos="{s((fx_w - sv.DOOR_THICK - 0.003, door_cy + 0.01, sv.MW_CZ))}" rgba="0.15 0.16 0.19 1"/>
    <geom name="mw_panel" type="box" size="0.004 {sv.MW_BTN_W/2*0.85:.5f} {HALF[2]*0.90:.5f}"
          pos="{fx_w - 0.004:.5f} {sv.MW_CY - HALF[1] + sv.MW_BTN_W/2:.5f} {sv.MW_CZ:.5f}" rgba="0.20 0.20 0.23 1"/>
    {buttons}
    <!-- HANDLE bar: dark (NOT white) -->
    <geom name="mw_handle" type="box" size="{hb_hx:.5f} 0.013 {hb_hz:.5f}"
          pos="{hb_x:.5f} {sv.MW_HANDLE_Y:.5f} {sv.MW_CZ:.5f}" rgba="0.13 0.13 0.15 1"/>
    <!-- WHITE strip on the handle FRONT (faces the robot, -x) -->
    <geom name="hstrip_front" type="box" size="0.0015 0.012 {hb_hz:.5f}"
          pos="{fx_w - HANDLE_PROTRUDE + 0.0015:.5f} {sv.MW_HANDLE_Y:.5f} {sv.MW_CZ:.5f}" rgba="0.93 0.93 0.95 1"/>
    <!-- WHITE strip on the handle TOP -->
    <geom name="hstrip_top" type="box" size="{hb_hx:.5f} 0.012 0.0025"
          pos="{hb_x:.5f} {sv.MW_HANDLE_Y:.5f} {sv.MW_CZ + hb_hz:.5f}" rgba="0.93 0.93 0.95 1"/>
    <!-- GREEN-wrapped glass (cylinder) on the table, robot's right -- the RIGHT arm's grab target -->
    <geom name="glass" type="cylinder" size="{sv.R:.5f} {sv.H/2:.5f}" pos="{s((sv.CX, sv.CY, sv.CZ))}" rgba="0.13 0.62 0.20 1"/>
  </worldbody>
</mujoco>"""


def render_head_cam():
    """Render the sim scene from the head camera. Returns (bgr uint8, depth_m float32, intr)."""
    import mujoco
    m = mujoco.MjModel.from_xml_string(build_mjcf())
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    rr = mujoco.Renderer(m, RENDER_H, RENDER_W); rr.update_scene(d, camera="head")
    rgb = rr.render()                                   # HxWx3 uint8 RGB
    dr = mujoco.Renderer(m, RENDER_H, RENDER_W); dr.enable_depth_rendering(); dr.update_scene(d, camera="head")
    depth = dr.render().astype(np.float32)              # HxW metres (perpendicular z-depth)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), depth, dict(SIM_INTR)


# ===================== PERCEPTION (reused on the real stream later) =================
def fill_holes(mask_u8):
    """Fill ONLY truly-enclosed holes (e.g. the bright handle strip surrounded by the dark
    appliance), NOT concavities open to the image border (e.g. the white table wedge beside
    the tilted unit). Flood the background in from the border; whatever the flood can't reach
    is an interior hole -> add it back."""
    h, w = mask_u8.shape
    flood = mask_u8.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)          # fill reachable background from a corner
    holes = cv2.bitwise_not(flood)                      # unreachable background = enclosed holes
    return cv2.bitwise_or(mask_u8, holes)


def appliance_region(model, bgr, conf):
    """YOLO11 instance segmentation -> MASK of the microwave region. The raw YOLO mask is
    coarse (it can bleed onto the bright table), so we REFINE it: within the YOLO seed keep
    only the DARK pixels (the black appliance body), take the largest dark blob, then fill its
    enclosed holes so the bright handle strip inside the silhouette is recovered. The white
    table — bright AND mostly outside the dark blob — is dropped. Returns (region_u8, info)."""
    r = model.predict(bgr, conf=conf, verbose=False)[0]
    if r.masks is None or len(r.masks) == 0:
        return None, "no segmentation masks"
    H, W = bgr.shape[:2]
    cands = []
    for i in range(len(r.masks)):
        mk = r.masks.data[i].detach().cpu().numpy().astype(np.uint8)
        mk = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST)
        cls = model.names[int(r.boxes.cls[i])]; cf = float(r.boxes.conf[i])
        cands.append({"name": cls, "conf": cf, "mask": mk, "area": int(mk.sum())})
    appl = [c for c in cands if c["name"] in APPLIANCE_CLASSES]
    pool = appl if appl else cands
    seed = max(pool, key=lambda c: c["area"])
    seed_mask = seed["mask"] > 0

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    vals = gray[seed_mask].reshape(-1, 1).astype(np.uint8)        # split dark body vs bright table/strip
    thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = ((gray < thr) & seed_mask).astype(np.uint8) * 255      # the black appliance pixels
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark)
    if n > 1:                                                     # keep the largest dark blob (the body)
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        dark = (lab == big).astype(np.uint8) * 255
    # CLOSE absorbs the bright handle-strip NOTCH into the silhouette even when the strip touches the
    # body edge / panel gap (so it isn't a fully-enclosed hole that fill_holes could recover). Closing
    # grows inward only, so the bright white TABLE (outside the convex body) stays excluded.
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)))
    region = fill_holes(dark)                                     # + recover any fully-enclosed strip
    return region, {"name": seed["name"], "conf": seed["conf"], "dark_thr": int(thr),
                    "area": int((region > 0).sum()), "from": ("appliance-class" if appl else "largest-mask"),
                    "seed_mask": (seed_mask.astype(np.uint8) * 255)}      # raw YOLO mask, for visualization


def find_handle_strip(bgr, region_mask, bright_pct=99.0, abs_floor=90, min_area=60, min_contrast=20):
    """Find the handle's bright/chrome strip inside the appliance region, robust to viewing angle.

    Brightness alone isn't enough: the microwave also has a bright TOP RIM (a wide horizontal band) and
    door reflections that can outshine the chrome bar, and from an oblique view the chrome is dim. So we
    threshold the region's bright tail (percentile, floored at `abs_floor`) and then SELECT by shape +
    location: the handle is a COMPACT blob (not the full-width rim), NOT hugging the top edge (the rim),
    and on the RIGHT side of the unit (the handle sits at 39 cm from the +y/left edge -> robot's right ->
    image right). Among the survivors we take the largest. Returns (handle_mask_u8 | None, info)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    reg = region_mask > 0
    if reg.sum() < 50:
        return None, {"reason": "empty region", "bright": np.zeros_like(gray)}
    x0, y0, wbb, hbb = cv2.boundingRect((region_mask > 0).astype(np.uint8))
    vals = gray[reg]
    body = float(np.median(vals))
    thr = int(max(abs_floor, np.percentile(vals, bright_pct)))
    bright = ((gray >= thr) & reg).astype(np.uint8) * 255
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    info = {"thr": thr, "body": int(body), "bright": bright}
    if thr - body < min_contrast:
        info["reason"] = f"no bright strip (body={int(body)} thr={thr}; contrast < {min_contrast})"
        return None, info
    cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c); cx = x + w / 2.0
        if w > 0.45 * wbb:                       # reject the full-width top rim / horizontal bands
            continue
        if y < y0 + 0.12 * hbb:                  # reject blobs hugging the top edge (the rim)
            continue
        if cx < x0 + 0.45 * wbb:                 # handle is on the RIGHT (-y side -> image right)
            continue
        cands.append((a, c))
    if not cands:
        info["reason"] = f"no compact right-side bright blob (body={int(body)} thr={thr})"
        return None, info
    a, c = max(cands, key=lambda t: t[0])
    hmask = np.zeros_like(gray); cv2.drawContours(hmask, [c], -1, 255, cv2.FILLED)
    hmask = cv2.bitwise_and(hmask, region_mask)                      # keep on the appliance
    x, y, w, h = cv2.boundingRect(c)
    info.update(area=int(a), bbox=(x, y, w, h))
    return hmask, info


def find_handle_yellow(bgr, region_mask, hsv_lo=(15, 60, 45), hsv_hi=(45, 255, 255), min_area=150):
    """Find the handle by a YELLOW TAPE stuck on it — a colour cue far more robust than the chrome
    strip's view-dependent brightness. Detect yellow (HSV) CONSTRAINED to the microwave bounding box
    (so any background yellow can't fool it), take the largest blob. Returns (mask_u8 | None, info).
    info carries 'bright' (the yellow mask, for the debug save) like find_handle_strip."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ym = cv2.inRange(hsv, np.array(hsv_lo, np.uint8), np.array(hsv_hi, np.uint8))
    if region_mask is not None and (region_mask > 0).any():
        x0, y0, w, h = cv2.boundingRect((region_mask > 0).astype(np.uint8))
        box = np.zeros_like(ym); box[y0:y0 + h, x0:x0 + w] = 255       # constrain to the microwave bbox
        ym = cv2.bitwise_and(ym, box)
    ym = cv2.morphologyEx(ym, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    ym = cv2.morphologyEx(ym, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    info = {"bright": ym}
    cnts, _ = cv2.findContours(ym, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    if not cnts:
        info["reason"] = f"no yellow-tape blob >= {min_area}px"
        return None, info
    c = max(cnts, key=cv2.contourArea)
    hmask = np.zeros(ym.shape, np.uint8); cv2.drawContours(hmask, [c], -1, 255, cv2.FILLED)
    x, y, w, h = cv2.boundingRect(c)
    info.update(area=int(cv2.contourArea(c)), bbox=(x, y, w, h))
    return hmask, info


def find_glass_green(bgr, hsv_lo=GREEN_LO, hsv_hi=GREEN_HI, min_area=400):
    """Find the green-wrapped glass by HSV colour — same method as object_detection.py. Returns
    (filled_mask | None, info). info carries 'green' (the raw colour mask, for the debug save)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lo, np.uint8), np.array(hsv_hi, np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    info = {"green": mask}
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    if not cnts:
        info["reason"] = f"no green blob >= {min_area}px"
        return None, info
    c = max(cnts, key=cv2.contourArea)
    clean = np.zeros_like(mask); cv2.drawContours(clean, [c], -1, 255, cv2.FILLED)
    x, y, w, h = cv2.boundingRect(c)
    info.update(area=int(cv2.contourArea(c)), bbox=(x, y, w, h))
    return clean, info


def deproject(mask_u8, depth_m, intr, T, depth_range=(0.1, 5.0), vmid=True):
    """3D centroid of the masked pixels, in the camera optical frame and the base/pelvis frame.
    vmid: take the base-frame VERTICAL (z) coordinate as the extent MIDPOINT, not the centroid —
    the handle's white strip wraps the FRONT *and TOP*, and the top cap rides the centroid upward;
    the midpoint of the strip's vertical span sits at the handle (= body) mid-height. x,y stay median."""
    ys, xs = np.where(mask_u8 > 0)
    Z = depth_m[ys, xs]
    ok = np.isfinite(Z) & (Z > depth_range[0]) & (Z < depth_range[1])
    if ok.sum() < 20:
        return None, f"too few valid-depth px ({int(ok.sum())})"
    xs, ys, Z = xs[ok], ys[ok], Z[ok]
    zmed = np.median(Z); keep = np.abs(Z - zmed) < 0.05               # reject edge bleed
    xs, ys, Z = xs[keep], ys[keep], Z[keep]
    X = (xs - intr["ppx"]) * Z / intr["fx"]
    Y = (ys - intr["ppy"]) * Z / intr["fy"]
    pts = np.stack([X, Y, Z], 1)                                     # optical: x right, y down, z fwd
    base_pts = (T[:3, :3] @ pts.T + T[:3, 3:4]).T                    # all points in the pelvis/base frame
    base = np.median(base_pts, 0)
    if vmid:                                                        # de-bias height: midpoint of the z extent
        zlo, zhi = np.percentile(base_pts[:, 2], [5, 95])
        base[2] = 0.5 * (zlo + zhi)
    cam = np.median(pts, 0)                                          # camera-frame centroid (display only)
    return {"cam": cam, "base": base, "n": int(pts.shape[0]),
            "px": (int(np.median(xs)), int(np.median(ys)))}, "ok"


def region_center_z(region_mask, depth_m, intr, T, depth_range=(0.1, 5.0)):
    """Body-centre HEIGHT from the microwave front-face region. The unit rests on the table and
    its front face spans the full height, so the MIDPOINT of the region's vertical (base-z) extent
    is the body centre z. This is far more reliable than the handle z, whose lower half is hidden
    by the downward head-cam view (its visible extent is asymmetric -> ~2cm high bias). Returns z|None."""
    ys, xs = np.where(region_mask > 0)
    Z = depth_m[ys, xs]
    ok = np.isfinite(Z) & (Z > depth_range[0]) & (Z < depth_range[1])
    if ok.sum() < 50:
        return None
    xs, ys, Z = xs[ok], ys[ok], Z[ok]
    zmed = np.median(Z); keep = np.abs(Z - zmed) < 0.30              # drop far background; keep the whole face (spans depth)
    xs, ys, Z = xs[keep], ys[keep], Z[keep]
    X = (xs - intr["ppx"]) * Z / intr["fx"]
    Y = (ys - intr["ppy"]) * Z / intr["fy"]
    base = (T[:3, :3] @ np.stack([X, Y, Z], 1).T + T[:3, 3:4]).T
    zlo, zhi = np.percentile(base[:, 2], [2, 98])
    return float(0.5 * (zlo + zhi))


def microwave_center_from_handle(handle_base, center_z=None):
    """Infer the body centre from the handle 3D point + known geometry (axis-aligned body).
    Body front is HANDLE_PROTRUDE behind the handle front; handle is 39cm from the +y(left) edge.
    center_z: body height from region_center_z (preferred); else fall back to the handle z."""
    return np.array([handle_base[0] + HANDLE_PROTRUDE + HALF[0],          # +x: behind handle, then half-depth
                     handle_base[1] + (sv.MW_HANDLE_FROM_LEFT - HALF[1]),  # +0.145: left(+y) edge is 39cm from handle
                     center_z if center_z is not None else handle_base[2]])


# ===================== HARNESS / COMPARISON =========================================
def annotate(bgr, region, hmask, hd, save, gmask=None, gd=None):
    vis = bgr.copy()
    if region is not None:
        vis[region > 0] = (0.6 * vis[region > 0] + 0.4 * np.array([60, 160, 60])).astype(np.uint8)
    if hmask is not None:
        vis[hmask > 0] = (0, 0, 255)                                   # handle = red
    if hd is not None:
        cv2.circle(vis, hd["px"], 6, (0, 255, 255), 2)
    if gmask is not None:
        vis[gmask > 0] = (0.4 * vis[gmask > 0] + 0.6 * np.array([0, 200, 0])).astype(np.uint8)   # glass = green
    if gd is not None:
        cv2.circle(vis, gd["px"], 6, (255, 255, 0), 2)
    cv2.imwrite(save, vis)


def main():
    ap = argparse.ArgumentParser(description="SIM: localise microwave handle + centre from G1 head RGB-D (MuJoCo)")
    ap.add_argument("--model", default="yolo11n-seg.pt", help="Ultralytics SEG weights (auto-downloads)")
    ap.add_argument("--conf", type=float, default=0.05, help="YOLO confidence (low; label irrelevant)")
    ap.add_argument("--bright-pct", type=float, default=99.0, help="region percentile for the handle bright-strip threshold")
    ap.add_argument("--white-thr", type=int, default=0, help="optional absolute floor on the bright-strip threshold")
    ap.add_argument("--save-dir", default="/tmp/mwsim", help="dir for rgb/depth/region/handle pngs")
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("[sim] rendering G1 head camera (MuJoCo, %dx%d) ..." % (RENDER_W, RENDER_H))
    bgr, depth, intr = render_head_cam()
    cv2.imwrite(f"{args.save_dir}/rgb.png", bgr)
    dv = depth.copy(); dv[~np.isfinite(dv)] = 0; dv[dv > 4] = 0
    cv2.imwrite(f"{args.save_dir}/depth.png", cv2.applyColorMap(
        cv2.normalize(dv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET))

    from ultralytics import YOLO
    model = YOLO(args.model)

    region, rinfo = appliance_region(model, bgr, args.conf)
    if region is None:
        print(f"[FAIL] appliance region: {rinfo}"); sys.exit(2)
    print(f"[1] region: class='{rinfo['name']}' conf={rinfo['conf']*100:.1f}% ({rinfo['from']}) "
          f"area={rinfo['area']}px dark_thr={rinfo['dark_thr']}")
    # visualize what YOLO marked (raw seed) vs the refined dark region
    raw_vis = bgr.copy(); raw_vis[rinfo["seed_mask"] > 0] = (
        0.45 * raw_vis[rinfo["seed_mask"] > 0] + 0.55 * np.array([0, 200, 255])).astype(np.uint8)
    cv2.imwrite(f"{args.save_dir}/region_raw_yolo.png", raw_vis)
    reg_vis = bgr.copy(); reg_vis[region > 0] = (
        0.45 * reg_vis[region > 0] + 0.55 * np.array([60, 200, 60])).astype(np.uint8)
    cv2.imwrite(f"{args.save_dir}/region_refined.png", reg_vis)

    hmask, hinfo = find_handle_strip(bgr, region, bright_pct=args.bright_pct, abs_floor=args.white_thr)
    cv2.imwrite(f"{args.save_dir}/bright.png", hinfo["bright"])
    if hmask is None:
        annotate(bgr, region, None, None, f"{args.save_dir}/handle.png")
        print(f"[FAIL] handle strip: {hinfo.get('reason')}"); sys.exit(2)
    print(f"[2] handle strip: area={hinfo['area']}px body={hinfo['body']} thr={hinfo['thr']} bbox={hinfo['bbox']}")

    hd, dstat = deproject(hmask, depth, intr, T_BASE_CAM)
    if hd is None:
        print(f"[FAIL] deproject: {dstat}"); sys.exit(2)
    cz = region_center_z(region, depth, intr, T_BASE_CAM)         # body height from the front-face extent
    mw = microwave_center_from_handle(hd["base"], cz)

    # ===== GLASS: detect the green-wrapped cup (same HSV method as object_detection.py) =====
    gmask, ginfo = find_glass_green(bgr)
    gd = None
    cv2.imwrite(f"{args.save_dir}/green.png", ginfo["green"])
    if gmask is None:
        print(f"[glass] FAIL — {ginfo.get('reason')}")
    else:
        gd, gstat = deproject(gmask, depth, intr, T_BASE_CAM, vmid=False)   # cylinder: median centroid
        if gd is None:
            print(f"[glass] deproject FAIL — {gstat}")

    annotate(bgr, region, hmask, hd, f"{args.save_dir}/handle.png", gmask, gd)

    he = hd["base"] - GT_HANDLE
    ce = mw - GT_MW_CENTER
    print(f"[3] handle  detected base=({hd['base'][0]:+.3f},{hd['base'][1]:+.3f},{hd['base'][2]:+.3f}) "
          f"cam=({hd['cam'][0]:+.3f},{hd['cam'][1]:+.3f},{hd['cam'][2]:+.3f}) n={hd['n']}")
    print(f"    handle  GROUND  base=({GT_HANDLE[0]:+.3f},{GT_HANDLE[1]:+.3f},{GT_HANDLE[2]:+.3f})")
    print(f"    handle  ERROR   ({he[0]*100:+.1f},{he[1]*100:+.1f},{he[2]*100:+.1f}) cm  |err|={np.linalg.norm(he)*100:.1f} cm")
    print(f"[4] microwave detected center=({mw[0]:+.3f},{mw[1]:+.3f},{mw[2]:+.3f})")
    print(f"    microwave GROUND center=({GT_MW_CENTER[0]:+.3f},{GT_MW_CENTER[1]:+.3f},{GT_MW_CENTER[2]:+.3f})")
    print(f"    microwave ERROR ({ce[0]*100:+.1f},{ce[1]*100:+.1f},{ce[2]*100:+.1f}) cm  |err|={np.linalg.norm(ce)*100:.1f} cm")
    if gd is not None:
        gx, gy, gw, gh = ginfo["bbox"]; zg = float(gd["cam"][2])
        gr = (gw * zg / intr["fx"]) / 2.0; ghm = gh * zg / intr["fy"]   # approx radius/height from the bbox
        gle = gd["base"] - GT_GLASS
        print(f"[5] glass detected base=({gd['base'][0]:+.3f},{gd['base'][1]:+.3f},{gd['base'][2]:+.3f}) "
              f"r~{gr*100:.1f}cm h~{ghm*100:.1f}cm n={gd['n']}")
        print(f"    glass GROUND  base=({GT_GLASS[0]:+.3f},{GT_GLASS[1]:+.3f},{GT_GLASS[2]:+.3f}) (true r={GLASS_R*100:.1f} h={GLASS_H*100:.1f})")
        print(f"    glass ERROR   ({gle[0]*100:+.1f},{gle[1]*100:+.1f},{gle[2]*100:+.1f}) cm  |err|={np.linalg.norm(gle)*100:.1f} cm  "
              f"(detection sees the FRONT surface -> ~+r toward the camera)")
    print(f"[save] {args.save_dir}/{{rgb,depth,handle,green}}.png  (handle.png shows handle=red + glass=green)")


if __name__ == "__main__":
    main()
