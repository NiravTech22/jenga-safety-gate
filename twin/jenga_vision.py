#!/usr/bin/env python3
"""Camera calibration + tower pose in the ROBOT frame, for the hardware twin.

Run from the repository root: python twin/jenga_vision.py <cmd>. Outputs go to
data/calibration/ (gitignored). Full walkthrough: docs/CAMERA_CALIBRATION.md.

Relationship to src/perception/tower_state.py: that module works in the TOWER
frame (4x4_50 markers ids 0-3 around the tower base) for the Step 2 tower-state
CV. This tool works in the robot base frame (the MuJoCo world frame) using a
5x5_100 ChArUco board and marker id 40, so the dictionaries never collide. The
intrinsics it produces (data/calibration/front_intrinsics.json -> "K", "dist")
are what ArucoFrameCalibrator takes as camera_matrix / dist_coeffs on hardware.

Subcommands (run `python jenga_vision.py <cmd> -h` for options):
  make-targets   Printable ChArUco calibration board + tower ArUco marker (PNG + PDF at 300 DPI)
  capture        Live camera preview; SPACE saves a frame when the board is detected
  intrinsics     Calibrate focal length / lens distortion from the saved frames
  extrinsics     Find where the camera is relative to the robot, from the board lying at a measured spot
  tower          Live tower pose (x, y, yaw) in the robot frame from the marker on the tower's base plate
  mujoco-camera  Add/replace a camera in the MuJoCo scene that matches the real one exactly
  selftest       Synthetic camera images with known answers; checks every step above end to end

ROBOT FRAME used everywhere (same as the MuJoCo scene's world frame):
  origin = robot base origin. Physically: measure from the table point directly under the
           centre of the shoulder-pan axis, then this script adds the pan-axis offset (PAN_AXIS_XY).
  +X     = the direction the arm points when shoulder_pan = 0
  +Y     = to the LEFT when you stand behind the robot looking along +X
  +Z     = up; z = 0 is the surface the robot base sits on
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# ---------------------------------------------------------------- configuration
DICT = cv2.aruco.DICT_5X5_100
BOARD_COLS, BOARD_ROWS = 7, 5             # squares
SQUARE_MM, MARKER_MM = 30.0, 22.0          # as printed; re-measure after printing and pass --square-mm
TOWER_MARKER_ID, TOWER_MARKER_MM = 40, 50.0
PAN_AXIS_XY = np.array([0.0388, 0.0])      # shoulder-pan axis in the MuJoCo base frame (from your model)
DPI = 300


def board(square_mm=SQUARE_MM, marker_mm=MARKER_MM):
    d = cv2.aruco.getPredefinedDictionary(DICT)
    return cv2.aruco.CharucoBoard((BOARD_COLS, BOARD_ROWS), square_mm / 1000, marker_mm / 1000, d)


def aruco_detector():
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT), p)


def detect_board(gray, b, min_corners=6):
    """-> (obj_pts Nx3, img_pts Nx2) or None."""
    corners, ids, _, _ = cv2.aruco.CharucoDetector(b).detectBoard(gray)
    if ids is None or len(ids) < min_corners:
        return None
    obj, img = b.matchImagePoints(corners, ids)
    return obj.reshape(-1, 3).astype(np.float64), img.reshape(-1, 2).astype(np.float64)


# ---------------------------------------------------------------- small pose helpers
def T_from(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.ravel(t); return T


def T_inv(T):
    R, t = T[:3, :3], T[:3, 3]
    return T_from(R.T, -R.T @ t)


def rvec_T(rvec, tvec):
    return T_from(cv2.Rodrigues(np.asarray(rvec, float))[0], tvec)


def rotz(deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def average_poses(Ts):
    """Mean of nearby rigid transforms (translation mean + rotation via SVD projection)."""
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    U, _, Vt = np.linalg.svd(np.sum([T[:3, :3] for T in Ts], axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1; R = U @ Vt
    return T_from(R, t)


def circular_mean_deg(angles_deg):
    """Mean of angles that may straddle the -180/+180 wrap (179.9 and -179.9 -> 180, not 0)."""
    a = np.radians(np.asarray(angles_deg, float))
    return float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())))


def board_in_robot(origin_xyz, yaw_deg):
    """Board lying FACE UP. Board frame (OpenCV ChArUco): origin at the corner marked ORIGIN,
    +x along the edge marked '+X', +y across the board, +z into the table."""
    R = rotz(yaw_deg) @ np.diag([1.0, -1.0, -1.0])
    p = np.asarray(origin_xyz, float).copy(); p[:2] += PAN_AXIS_XY
    return T_from(R, p)


def load_json(p):
    return json.loads(Path(p).read_text())


def save_json(p, d):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(d, indent=2) + "\n")
    print(f"saved {p}")


def K_dist(intr):
    return np.array(intr["K"], float), np.array(intr["dist"], float).ravel()


def open_camera(index, width, height):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2) if sys.platform.startswith("linux") else cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {index}. List cameras with: v4l2-ctl --list-devices")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (w, h) != (width, height):
        print(f"WARNING: camera gave {w}x{h}, not {width}x{height}. Calibration is only valid at one resolution.")
    for _ in range(10):
        cap.read()  # let auto-exposure settle / flush stale frames
    return cap


# ---------------------------------------------------------------- make-targets
def cmd_make_targets(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pxmm = DPI / 25.4
    # page 1: board, landscape letter
    page = np.full((int(8.5 * DPI), int(11 * DPI)), 255, np.uint8)
    bw, bh = int(round(BOARD_COLS * SQUARE_MM * pxmm)), int(round(BOARD_ROWS * SQUARE_MM * pxmm))
    img = board().generateImage((bw, bh), marginSize=0, borderBits=1)
    y0, x0 = (page.shape[0] - bh) // 2 + 60, (page.shape[1] - bw) // 2
    page[y0:y0 + bh, x0:x0 + bw] = img
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(page, "ORIGIN", (x0 - 230, y0 - 25), f, 2.0, 0, 5)
    cv2.arrowedLine(page, (x0 - 60, y0 - 60), (x0 - 5, y0 - 5), 0, 6, tipLength=.3)
    cv2.arrowedLine(page, (x0 + 80, y0 - 50), (x0 + 600, y0 - 50), 0, 6, tipLength=.08)
    cv2.putText(page, "+X  (point this along robot +X)", (x0 + 640, y0 - 35), f, 1.6, 0, 4)
    cv2.putText(page, f"ChArUco {BOARD_COLS}x{BOARD_ROWS}, {SQUARE_MM:.0f} mm squares, {MARKER_MM:.0f} mm markers, "
                "5x5_100. PRINT AT 100% / ACTUAL SIZE. Measure a square after printing.",
                (150, page.shape[0] - 120), f, 1.15, 0, 3)
    # page 2: tower marker with cut line and centre cross
    page2 = np.full_like(page, 255)
    m = int(round(TOWER_MARKER_MM * pxmm))
    mk = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(DICT), TOWER_MARKER_ID, m)
    cy, cx = page2.shape[0] // 2, page2.shape[1] // 2
    page2[cy - m // 2:cy - m // 2 + m, cx - m // 2:cx - m // 2 + m] = mk
    q = int(8 * pxmm)
    cv2.rectangle(page2, (cx - m // 2 - q, cy - m // 2 - q), (cx + m // 2 + q, cy + m // 2 + q), 160, 2)
    cv2.arrowedLine(page2, (cx, cy + m // 2 + 2 * q), (cx, cy + m // 2 + 2 * q - 200), 0, 6, tipLength=.2)
    cv2.putText(page2, "marker +y", (cx + 20, cy + m // 2 + 2 * q - 100), f, 1.4, 0, 3)
    cv2.putText(page2, f"Tower marker id {TOWER_MARKER_ID}, {TOWER_MARKER_MM:.0f} mm. Cut on the grey line (keep the white border). "
                "PRINT AT 100%.", (150, page2.shape[0] - 120), f, 1.15, 0, 3)
    for name, p in (("charuco_board", page), ("tower_marker", page2)):
        cv2.imwrite(str(out / f"{name}.png"), p)
    try:
        from PIL import Image
        ims = [Image.fromarray(p) for p in (page, page2)]
        ims[0].save(out / "jenga_targets.pdf", resolution=DPI, save_all=True, append_images=ims[1:])
        print(f"wrote {out}/jenga_targets.pdf (2 pages) and PNGs")
    except ImportError:
        print(f"wrote PNGs to {out}/ (install pillow for a PDF)")


# ---------------------------------------------------------------- capture
def cmd_capture(a):
    b = board(a.square_mm, a.marker_mm); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cap = open_camera(a.camera, a.width, a.height)
    n = len(list(out.glob("*.png")))
    print("SPACE = save (needs >= 12 corners), Q = quit. Aim for 25+ frames: near/far, tilted, all image corners.")
    while True:
        ok, frame = cap.read()
        if not ok:
            sys.exit("camera read failed")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det = detect_board(gray, b, 1)
        view = frame.copy(); k = 0 if det is None else len(det[1])
        if det is not None:
            for u, v in det[1]:
                cv2.circle(view, (int(u), int(v)), 4, (0, 255, 0), -1)
        cv2.putText(view, f"corners {k}  saved {n}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .8,
                    (0, 255, 0) if k >= 12 else (0, 0, 255), 2)
        cv2.imshow("capture", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and k >= 12:
            cv2.imwrite(str(out / f"calib_{n:03d}.png"), frame); n += 1
    cap.release(); cv2.destroyAllWindows()


# ---------------------------------------------------------------- intrinsics
def calibrate_from_images(images, b):
    obj, img, names, size = [], [], [], None
    for p in images:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        size = (g.shape[1], g.shape[0])
        det = detect_board(g, b, 12)
        if det is not None:
            obj.append(det[0].astype(np.float32)); img.append(det[1].astype(np.float32)); names.append(Path(p).name)
    if len(obj) < 8:
        raise RuntimeError(f"Only {len(obj)} usable images (need >= 8, ideally 25+).")

    def run(o, i):
        rms, K, dist, rv, tv = cv2.calibrateCamera(o, i, size, None, None)
        errs = []
        for oo, ii, r, t in zip(o, i, rv, tv):
            proj, _ = cv2.projectPoints(oo, r, t, K, dist)
            errs.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - ii) ** 2, 1)))))
        return rms, K, dist, np.array(errs)

    rms, K, dist, errs = run(obj, img)
    keep = errs < max(3 * np.median(errs), .5)       # drop blurry / misdetected frames, then refit
    dropped = [n for n, k in zip(names, keep) if not k]
    if dropped and keep.sum() >= 8:
        rms, K, dist, errs = run([o for o, k in zip(obj, keep) if k], [i for i, k in zip(img, keep) if k])
    return dict(image_size=list(size), K=K.tolist(), dist=dist.ravel().tolist(), rms_px=float(rms),
                n_images=int(len(errs)), dropped=dropped)


def cmd_intrinsics(a):
    b = board(a.square_mm, a.marker_mm)
    images = sorted(Path(a.images).glob("*.png")) + sorted(Path(a.images).glob("*.jpg"))
    r = calibrate_from_images(images, b)
    r.update(square_mm=a.square_mm, marker_mm=a.marker_mm)
    K = np.array(r["K"])
    print(f"images used {r['n_images']}  dropped {r['dropped']}")
    print(f"fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}  dist={np.round(r['dist'],4)}")
    verdict = "GOOD" if r["rms_px"] < .5 else "OK" if r["rms_px"] < 1 else "POOR - recapture (blur? board not flat?)"
    print(f"reprojection RMS = {r['rms_px']:.3f} px  -> {verdict}")
    save_json(a.out, r)


# ---------------------------------------------------------------- extrinsics
def camera_in_robot(frames_gray, intr, b, origin_xyz, yaw_deg):
    K, dist = K_dist(intr)
    T_base_board = board_in_robot(origin_xyz, yaw_deg)
    Ts, errs = [], []
    for g in frames_gray:
        det = detect_board(g, b, 12)
        if det is None:
            continue
        ok, rv, tv = cv2.solvePnP(det[0], det[1], K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(det[0], rv, tv, K, dist)
        errs.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - det[1]) ** 2, 1)))))
        Ts.append(T_base_board @ T_inv(rvec_T(rv, tv)))     # T_base_cam
    if not Ts:
        raise RuntimeError("Board not detected (need >= 12 corners). Is the whole board visible and lit?")
    return average_poses(Ts), float(np.mean(errs)), len(Ts)


def cmd_extrinsics(a):
    intr = load_json(a.intrinsics); b = board(intr["square_mm"], intr["marker_mm"])
    if a.image:
        frames = [cv2.imread(a.image, cv2.IMREAD_GRAYSCALE)]
    else:
        w, h = intr["image_size"]; cap = open_camera(a.camera, w, h); frames = []
        for _ in range(a.frames):
            ok, f = cap.read()
            if ok:
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        cap.release()
    T, err, n = camera_in_robot(frames, intr, b, a.board_origin, a.board_yaw)
    fwd = T[:3, 2]
    print(f"camera position (robot frame, m): {T[:3,3]}   from {n} frames, reprojection {err:.2f} px")
    print(f"camera looks along {fwd}  (pitch {math.degrees(math.asin(-fwd[2])):.1f} deg down)")
    save_json(a.out, dict(T_base_cam=T.tolist(), board_origin=list(a.board_origin), board_yaw_deg=a.board_yaw,
                          reproj_px=err, intrinsics=str(a.intrinsics)))


# ---------------------------------------------------------------- tower pose
def tower_pose(gray, intr, T_base_cam, marker_mm=TOWER_MARKER_MM, marker_id=TOWER_MARKER_ID,
               offset=(0, 0), marker_z=0.0):
    """One frame -> dict(x, y, z, yaw_deg, edge_mm, tilt_deg) in the robot frame, or None.

    Position comes from intersecting the camera rays through the 4 marker corners with the
    known horizontal plane z = marker_z. With one small marker, solvePnP's depth estimate is the
    weakest number; the plane intersection removes it. PnP is kept only as a sanity check (tilt).
    """
    corners, ids, _ = aruco_detector().detectMarkers(gray)
    if ids is None or marker_id not in ids.ravel():
        return None
    c = corners[list(ids.ravel()).index(marker_id)].reshape(4, 2).astype(np.float64)   # TL, TR, BR, BL
    K, dist = K_dist(intr)
    rays = cv2.undistortPoints(c.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    R, o = T_base_cam[:3, :3], T_base_cam[:3, 3]
    P = []
    for x, y in rays:
        d = R @ np.array([x, y, 1.0])
        if abs(d[2]) < 1e-6:
            return None
        P.append(o + (marker_z - o[2]) / d[2] * d)
    P = np.array(P)
    xh = (P[1] + P[2] - P[0] - P[3]); xh[2] = 0; xh /= np.linalg.norm(xh)
    yh = np.array([-xh[1], xh[0], 0.0])                  # marker +y (marker lies face up)
    centre = P.mean(0) + offset[0] * xh + offset[1] * yh
    edge_mm = 1000 * float(np.mean([np.linalg.norm(P[i] - P[(i + 1) % 4]) for i in range(4)]))
    s = marker_mm / 2000
    ok, rv, _ = cv2.solvePnP(np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]]), c, K, dist,
                             flags=cv2.SOLVEPNP_IPPE_SQUARE)
    tilt = float(math.degrees(math.acos(np.clip((R @ cv2.Rodrigues(rv)[0])[2, 2], -1, 1)))) if ok else float("nan")
    return dict(x=float(centre[0]), y=float(centre[1]), z=float(marker_z),
                yaw_deg=float(math.degrees(math.atan2(xh[1], xh[0]))), edge_mm=edge_mm, tilt_deg=tilt)


def cmd_tower(a):
    intr = load_json(a.intrinsics); ext = load_json(a.extrinsics)
    T_bc = np.array(ext["T_base_cam"]); w, h = intr["image_size"]
    if a.image:
        r = tower_pose(cv2.imread(a.image, cv2.IMREAD_GRAYSCALE), intr, T_bc, a.marker_mm, a.marker_id,
                       a.offset, a.marker_z)
        print(json.dumps(r) if r else "tower marker not found"); return
    cap = open_camera(a.camera, w, h); hist = []
    while True:
        ok, frame = cap.read()
        if not ok:
            sys.exit("camera read failed")
        r = tower_pose(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), intr, T_bc, a.marker_mm, a.marker_id,
                       a.offset, a.marker_z)
        if r:
            hist = (hist + [r])[-30:]
            line = (f"tower x={r['x']*1000:7.1f} y={r['y']*1000:7.1f} mm  yaw={r['yaw_deg']:6.1f} deg  "
                    f"marker edge={r['edge_mm']:5.1f} mm (expect {a.marker_mm:.0f})")
            if abs(r["edge_mm"] - a.marker_mm) > 2:
                line += "  <- size off: check --marker-z, --marker-mm or redo extrinsics"
        else:
            line = "tower marker not visible"
        if a.once and len(hist) == 30:
            out = {k: float(np.median([x[k] for x in hist])) for k in ("x", "y", "z", "edge_mm", "tilt_deg")}
            out["yaw_deg"] = circular_mean_deg([x["yaw_deg"] for x in hist])   # a median breaks at +/-180
            print(json.dumps(out)); break
        if not a.once:
            print("\r" + line.ljust(110), end="", flush=True)
            if a.show:
                cv2.putText(frame, line[:60], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
                cv2.imshow("tower", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    cap.release()


# ---------------------------------------------------------------- mujoco camera
def mujoco_camera_attrs(intr, T_base_cam, name):
    """OpenCV camera (x right, y down, z forward) -> MuJoCo camera (x right, y up, looks along -z)."""
    K = np.array(intr["K"]); w, h = intr["image_size"]; R = T_base_cam[:3, :3]
    xa, ya = R[:, 0], -R[:, 1]
    pix = 3e-6                                            # nominal pixel pitch; only ratios matter
    return {"name": name, "pos": " ".join(f"{v:.6f}" for v in T_base_cam[:3, 3]),
            "xyaxes": " ".join(f"{v:.6f}" for v in (*xa, *ya)),
            "resolution": f"{w} {h}", "sensorsize": f"{w*pix:.9f} {h*pix:.9f}",
            "focalpixel": f"{K[0,0]:.4f} {K[1,1]:.4f}",
            # MuJoCo stores the principal-point OFFSET with both signs flipped relative to OpenCV (verified by render)
            "principalpixel": f"{(w-1)/2-K[0,2]:.4f} {(h-1)/2-K[1,2]:.4f}"}


def cmd_mujoco_camera(a):
    import shutil
    import xml.etree.ElementTree as ET
    intr = load_json(a.intrinsics); T = np.array(load_json(a.extrinsics)["T_base_cam"])
    attrs = mujoco_camera_attrs(intr, T, a.name)
    p = Path(a.scene); tree = ET.parse(p); world = tree.getroot().find("worldbody")
    for c in world.findall("camera"):
        if c.get("name") == a.name:
            world.remove(c)
    ET.SubElement(world, "camera", attrs)
    shutil.copy(p, str(p) + ".bak"); ET.indent(tree, space="  "); tree.write(p, encoding="utf-8", xml_declaration=True)
    print(f"camera '{a.name}' written to {p} (backup {p.name}.bak)\n  " +
          "\n  ".join(f"{k}={v}" for k, v in attrs.items()))
    print("MuJoCo renders without lens distortion: compare it with cv2.undistort()-ed real frames.")


# ---------------------------------------------------------------- selftest (synthetic images, known answers)
def _look_at(pos, target):
    z = np.asarray(target, float) - pos; z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
    return T_from(np.column_stack([x, np.cross(z, x), z]), pos)   # T_base_cam (OpenCV axes)


def _render_plane(tex, A, T_cam_plane, K, dist, size, rng):
    """Image of a flat textured plane. A maps texture pixels -> plane metres (3x3, z=0 plane)."""
    R, t = T_cam_plane[:3, :3], T_cam_plane[:3, 3]
    H = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ A
    bg = int(rng.integers(70, 140))
    ideal = cv2.warpPerspective(tex, H, size, flags=cv2.INTER_AREA, borderValue=bg)
    w, h = size
    grid = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).reshape(-1, 1, 2).astype(np.float32)
    und = cv2.undistortPoints(grid, K, dist, None, K)   # ideal pixel seen at each distorted pixel
    mx, my = und.reshape(h, w, 2)[..., 0], und.reshape(h, w, 2)[..., 1]
    img = cv2.remap(ideal, mx, my, cv2.INTER_LINEAR, borderValue=bg)
    img = cv2.GaussianBlur(img, (3, 3), .6).astype(np.float32) + rng.normal(0, 2.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def cmd_selftest(a):
    rng = np.random.default_rng(7)
    w, h = 640, 480
    K = np.array([[612.0, 0, 327.5], [0, 615.0, 236.0], [0, 0, 1]])
    dist = np.array([-0.08, 0.06, 0.0005, -0.0004, 0.0])
    b = board(); ppm = 12                                # texture pixels per mm
    bw, bh = int(BOARD_COLS * SQUARE_MM * ppm), int(BOARD_ROWS * SQUARE_MM * ppm)
    marg = int(SQUARE_MM * ppm * .6)
    tex = cv2.copyMakeBorder(b.generateImage((bw, bh), marginSize=0, borderBits=1), marg, marg, marg, marg,
                             cv2.BORDER_CONSTANT, value=245)
    s = 1 / (ppm * 1000)
    A_board = np.array([[s, 0, -marg * s], [0, s, -marg * s], [0, 0, 1]])
    results = {}

    # 1. intrinsics: board waved in front of the camera
    tmp = Path(a.workdir); tmp.mkdir(parents=True, exist_ok=True)
    for f in tmp.glob("calib_*.png"):
        f.unlink()
    bc = np.array([BOARD_COLS, BOARD_ROWS]) * SQUARE_MM / 2000
    n = 0
    while n < 40:
        R = cv2.Rodrigues(rng.uniform([-.85, -.85, -.5], [.85, .85, .5]))[0]
        c = np.array([rng.uniform(-.09, .09), rng.uniform(-.06, .06), rng.uniform(.33, .55)])
        T = T_from(R, c - R @ np.array([*bc, 0]))
        if T[:3, :3][2, 2] < .25:
            continue
        img = _render_plane(tex, A_board, T, K, dist, (w, h), rng)
        if detect_board(img, b, 12) is not None:
            cv2.imwrite(str(tmp / f"calib_{n:03d}.png"), img); n += 1
    r = calibrate_from_images(sorted(tmp.glob("calib_*.png")), b)
    r.update(square_mm=SQUARE_MM, marker_mm=MARKER_MM)
    Ke = np.array(r["K"])
    fe = 100 * max(abs(Ke[0, 0] / K[0, 0] - 1), abs(Ke[1, 1] / K[1, 1] - 1))
    ce = float(np.hypot(Ke[0, 2] - K[0, 2], Ke[1, 2] - K[1, 2]))
    results["intrinsics"] = (fe < .8 and ce < 5 and r["rms_px"] < .5,
                             f"rms {r['rms_px']:.3f}px, focal err {fe:.2f}%, principal-point err {ce:.2f}px")

    # 2. extrinsics: camera on a stand looking at the workcell, board flat on the table
    T_bc = _look_at(np.array([.62, .30, .42]), [.26, 0, .04])
    origin, yaw = [.12, .14, 0.0], -8.0
    T_bb = board_in_robot(origin, yaw)
    imgs = [_render_plane(tex, A_board, T_inv(T_bc) @ T_bb, K, dist, (w, h), rng) for _ in range(5)]
    T_est, err, nn = camera_in_robot(imgs, r, b, origin, yaw)
    pe = np.linalg.norm(T_est[:3, 3] - T_bc[:3, 3]) * 1000
    ae = math.degrees(np.linalg.norm(cv2.Rodrigues(T_est[:3, :3] @ T_bc[:3, :3].T)[0]))
    results["extrinsics"] = (pe < 4 and ae < .5, f"camera position err {pe:.2f} mm, angle err {ae:.3f} deg")

    # 3. tower marker on the table at a known pose
    mpx = int(TOWER_MARKER_MM * ppm); mq = int(mpx * .3)
    mk = cv2.copyMakeBorder(cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(DICT), TOWER_MARKER_ID, mpx),
                            mq, mq, mq, mq, cv2.BORDER_CONSTANT, value=245)
    half = mpx / 2 + mq
    A_mk = np.array([[s, 0, -half * s], [0, -s, half * s], [0, 0, 1]])
    worst_p = worst_y = 0.0
    for gx, gy, gyaw in [(.24, .02, 10), (.30, -.06, -25), (.20, .08, 40)]:
        T_bm = T_from(rotz(gyaw), [gx, gy, .004])
        img = _render_plane(mk, A_mk, T_inv(T_bc) @ T_bm, K, dist, (w, h), rng)
        tp = tower_pose(img, r, T_est, marker_z=.004)
        if tp is None:
            worst_p = worst_y = 1e9; break
        worst_p = max(worst_p, 1000 * math.hypot(tp["x"] - gx, tp["y"] - gy))
        worst_y = max(worst_y, abs((tp["yaw_deg"] - gyaw + 180) % 360 - 180))
    results["tower pose"] = (worst_p < 3 and worst_y < 1.5, f"worst xy err {worst_p:.2f} mm, worst yaw err {worst_y:.2f} deg")

    # 4. MuJoCo camera matches: render known points, compare with the OpenCV projection
    try:
        import mujoco
        attrs = mujoco_camera_attrs(dict(K=K.tolist(), image_size=[w, h]), T_bc, "real_cam")
        pts = np.array([[.26, 0, .04], [.20, .10, .02], [.32, -.08, .10], [.15, -.05, .0], [.30, .12, .15]])
        cam = " ".join(f'{k}="{v}"' for k, v in attrs.items())
        bodies = "".join(f'<geom type="sphere" size=".004" pos="{p[0]} {p[1]} {p[2]}" rgba="1 1 1 1"/>' for p in pts)
        xml = (f'<mujoco><visual><global offwidth="{w}" offheight="{h}"/><headlight ambient="1 1 1" diffuse="0 0 0" '
               f'specular="0 0 0"/></visual><worldbody><camera {cam}/>{bodies}</worldbody></mujoco>')
        m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        ren = mujoco.Renderer(m, h, w); errs = []
        T_cb = T_inv(T_bc)
        proj, _ = cv2.projectPoints(pts, cv2.Rodrigues(T_cb[:3, :3])[0], T_cb[:3, 3], K, None)
        for i, p in enumerate(pts):
            m.geom_rgba[:, 3] = 0; m.geom_rgba[i, 3] = 1
            ren.update_scene(d, "real_cam"); img = ren.render().sum(2).astype(float)
            ys, xs = np.nonzero(img > img.max() * .5)
            errs.append(float(np.hypot(xs.mean() - proj[i, 0, 0], ys.mean() - proj[i, 0, 1])))
        results["mujoco camera"] = (max(errs) < 1.0, f"worst pixel mismatch {max(errs):.2f} px over {len(pts)} points")
    except Exception as e:  # no OpenGL available: skip, don't fail
        results["mujoco camera"] = (None, f"skipped ({e.__class__.__name__}: {e})")

    print("\nSELFTEST (synthetic 640x480 camera with lens distortion and sensor noise)")
    for k, (ok, msg) in results.items():
        print(f"  {'PASS' if ok else 'SKIP' if ok is None else 'FAIL':4s}  {k:14s} {msg}")
    if any(ok is False for ok, _ in results.values()):
        sys.exit(1)


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("make-targets"); p.add_argument("--out", default="data/calibration/targets"); p.set_defaults(f=cmd_make_targets)

    def cam(p, size=True):
        p.add_argument("--camera", type=int, default=0, help="V4L2 index, e.g. 0 for /dev/video0")
        if size:
            p.add_argument("--width", type=int, default=640); p.add_argument("--height", type=int, default=480)

    def brd(p):
        p.add_argument("--square-mm", type=float, default=SQUARE_MM, help="measured square size after printing")
        p.add_argument("--marker-mm", type=float, default=MARKER_MM)

    p = sp.add_parser("capture"); cam(p); brd(p); p.add_argument("--out", default="data/calibration/front_images")
    p.set_defaults(f=cmd_capture)
    p = sp.add_parser("intrinsics"); brd(p); p.add_argument("--images", default="data/calibration/front_images")
    p.add_argument("--out", default="data/calibration/front_intrinsics.json"); p.set_defaults(f=cmd_intrinsics)
    p = sp.add_parser("extrinsics"); cam(p, False)
    p.add_argument("--intrinsics", default="data/calibration/front_intrinsics.json")
    p.add_argument("--board-origin", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"),
                   help="ORIGIN corner, metres, measured from the table point under the shoulder-pan axis")
    p.add_argument("--board-yaw", type=float, default=0.0, help="deg; 0 = '+X' arrow along robot +X")
    p.add_argument("--image", help="use a saved image instead of the live camera")
    p.add_argument("--frames", type=int, default=20); p.add_argument("--out", default="data/calibration/front_extrinsics.json")
    p.set_defaults(f=cmd_extrinsics)
    p = sp.add_parser("tower"); cam(p, False)
    p.add_argument("--intrinsics", default="data/calibration/front_intrinsics.json")
    p.add_argument("--extrinsics", default="data/calibration/front_extrinsics.json")
    p.add_argument("--marker-mm", type=float, default=TOWER_MARKER_MM)
    p.add_argument("--marker-id", type=int, default=TOWER_MARKER_ID)
    p.add_argument("--offset", type=float, nargs=2, default=[0, 0], metavar=("DX", "DY"),
                   help="tower centre relative to marker centre, metres, along marker +x / +y")
    p.add_argument("--marker-z", type=float, default=0.0,
                   help="height of the marker surface above the robot's base surface, metres (e.g. plate thickness)")
    p.add_argument("--once", action="store_true", help="print one median-of-30 JSON result and exit")
    p.add_argument("--image", help="use a saved image instead of the live camera")
    p.add_argument("--show", action="store_true"); p.set_defaults(f=cmd_tower)
    p = sp.add_parser("mujoco-camera")
    p.add_argument("--intrinsics", default="data/calibration/front_intrinsics.json")
    p.add_argument("--extrinsics", default="data/calibration/front_extrinsics.json")
    p.add_argument("--scene", default="twin/models/scene_custom_claw_physics.xml"); p.add_argument("--name", default="front_cam")
    p.set_defaults(f=cmd_mujoco_camera)
    p = sp.add_parser("selftest"); p.add_argument("--workdir", default="/tmp/jenga_selftest")
    p.set_defaults(f=cmd_selftest)

    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
