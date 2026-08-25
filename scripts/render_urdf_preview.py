#!/usr/bin/env python3
"""Render the SO-101 from its URDF, posed over a Jenga tower.

Isaac Sim segfaults on this host (see scripts/isaac/so101_jenga_scene.py), so
this rasterises the robot directly instead: parse the URDF, walk the
kinematic chain, load the visual STLs, and project everything through the
same pinhole camera the perception pipeline is validated against
(src/sim/camera.py). Forward kinematics and mesh geometry are real; only the
shading is invented.

Output is a transparent PNG so it can sit on either theme of a page.

  python scripts/render_urdf_preview.py --out /tmp/so101.png --width 1600
"""
from __future__ import annotations

import argparse
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.rig import BLOCK_H_M, BLOCK_L_M, BLOCK_W_M, SLOTS_PER_LAYER
from src.sim.camera import CameraIntrinsics, PinholeCamera

DEFAULT_URDF = Path(__file__).resolve().parents[1] / "data/assets/so101/so101_new_calib.urdf"

COLOR_LINK = np.array([196, 198, 203], float)     # RGB, light grey shell
COLOR_ACCENT = np.array([91, 42, 134], float)     # the site's accent purple
COLOR_BLOCK = np.array([206, 163, 112], float)
LIGHT = np.array([-0.35, -0.5, 0.78])
LIGHT /= np.linalg.norm(LIGHT)

# A pose that reads as "mid-extraction" rather than a factory home position.
POSE = {
    "shoulder_pan": 0.10,
    "shoulder_lift": -0.42,
    "elbow_flex": 0.78,
    "wrist_flex": 0.50,
    "wrist_roll": 0.10,
    "gripper": 0.30,
}


# --------------------------------------------------------------------------- #
# URDF
# --------------------------------------------------------------------------- #
def rpy_to_matrix(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx           # URDF fixed-axis convention


def axis_angle_to_matrix(axis, angle) -> np.ndarray:
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    x, y, z = axis / n
    c, s, C = np.cos(angle), np.sin(angle), 1 - np.cos(angle)
    return np.array([
        [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
    ])


def transform(R, t) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def parse_urdf(path: Path):
    root = ET.parse(path).getroot()

    def origin_of(node):
        o = node.find("origin")
        if o is None:
            return np.eye(4)
        xyz = [float(v) for v in (o.get("xyz") or "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()]
        return transform(rpy_to_matrix(rpy), xyz)

    links = {}
    for link in root.findall("link"):
        visuals = []
        for vis in link.findall("visual"):
            mesh = vis.find("geometry/mesh")
            if mesh is None:
                continue
            scale = mesh.get("scale")
            visuals.append({
                "file": mesh.get("filename"),
                "scale": np.array([float(v) for v in scale.split()]) if scale else np.ones(3),
                "origin": origin_of(vis),
            })
        links[link.get("name")] = visuals

    joints = []
    for j in root.findall("joint"):
        axis = j.find("axis")
        joints.append({
            "name": j.get("name"),
            "type": j.get("type"),
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "origin": origin_of(j),
            "axis": [float(v) for v in (axis.get("xyz") if axis is not None else "0 0 1").split()],
        })
    return links, joints


def forward_kinematics(joints, pose: dict, root_link="base_link") -> dict:
    """World transform per link. Straightforward tree walk: the chain is short
    and strictly serial apart from the gripper jaw."""
    poses = {root_link: np.eye(4)}
    remaining = list(joints)
    while remaining:
        progressed = False
        for j in list(remaining):
            if j["parent"] not in poses:
                continue
            local = j["origin"]
            if j["type"] in ("revolute", "continuous"):
                angle = pose.get(j["name"], 0.0)
                local = local @ transform(axis_angle_to_matrix(j["axis"], angle), np.zeros(3))
            poses[j["child"]] = poses[j["parent"]] @ local
            remaining.remove(j)
            progressed = True
        if not progressed:
            break            # disconnected subtree; ignore rather than hang
    return poses


def load_stl(path: Path) -> np.ndarray:
    """Binary STL -> (n, 3, 3) triangle vertex array."""
    data = path.read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    if len(data) != 84 + 50 * n:
        raise ValueError(f"{path} is not a binary STL")
    rec = np.frombuffer(data, dtype=np.dtype([
        ("normal", "<3f4"), ("v", "<3f4", (3,)), ("attr", "<u2")]), count=n, offset=84)
    return np.array(rec["v"], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Geometry assembly
# --------------------------------------------------------------------------- #
def box_triangles(centre, half) -> np.ndarray:
    cx, cy, cz = centre
    hx, hy, hz = half
    v = np.array([[cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
                  [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz],
                  [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
                  [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz]])
    faces = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 5, 1), (0, 4, 5),
             (2, 6, 7), (2, 7, 3), (1, 5, 6), (1, 6, 2), (0, 3, 7), (0, 7, 4)]
    return np.array([[v[a], v[b], v[c]] for a, b, c in faces])


def build_tower(n_layers: int, centre) -> list:
    out = []
    for layer in range(n_layers):
        z = BLOCK_H_M * (layer + 0.5)
        along_x = layer % 2 == 0
        for slot in range(SLOTS_PER_LAYER):
            off = (slot - 1) * BLOCK_W_M
            if along_x:
                c = (centre[0], centre[1] + off, z)
                h = (BLOCK_L_M / 2, BLOCK_W_M / 2, BLOCK_H_M / 2)
            else:
                c = (centre[0] + off, centre[1], z)
                h = (BLOCK_W_M / 2, BLOCK_L_M / 2, BLOCK_H_M / 2)
            shade = 1.0 - 0.04 * ((layer + slot) % 3)
            out.append((box_triangles(c, h), COLOR_BLOCK * shade))
    return out


def gather_robot(urdf: Path, pose: dict) -> list:
    links, joints = parse_urdf(urdf)
    link_poses = forward_kinematics(joints, pose)
    base_dir = urdf.parent

    groups = []
    for link_name, visuals in links.items():
        if link_name not in link_poses:
            continue
        world = link_poses[link_name]
        for vis in visuals:
            mesh_path = base_dir / vis["file"]
            if not mesh_path.exists():
                continue
            tris = load_stl(mesh_path) * vis["scale"]
            M = world @ vis["origin"]
            flat = tris.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
            colour = COLOR_ACCENT if "jaw" in link_name else COLOR_LINK
            groups.append((flat.reshape(-1, 3, 3), colour))
    return groups


# --------------------------------------------------------------------------- #
# Rasteriser
# --------------------------------------------------------------------------- #
def render(groups, camera: PinholeCamera, width: int, height: int,
           supersample: int = 2) -> np.ndarray:
    """Painter's algorithm with backface culling, into a BGRA canvas."""
    W, H = width * supersample, height * supersample
    intr = CameraIntrinsics(camera.intrinsics.fx * supersample,
                            camera.intrinsics.fy * supersample,
                            camera.intrinsics.cx * supersample,
                            camera.intrinsics.cy * supersample, W, H)
    cam = PinholeCamera(intr, camera.R, camera.t)
    canvas = np.zeros((H, W, 4), np.uint8)

    tris, colours = [], []
    for group_tris, colour in groups:
        tris.append(group_tris)
        colours.append(np.repeat(colour[None, :], len(group_tris), axis=0))
    tris = np.concatenate(tris)
    colours = np.concatenate(colours)

    centroids = tris.mean(axis=1)
    depths = cam.to_camera_frame(centroids)[:, 2]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, np.maximum(lengths, 1e-12))

    eye = cam.position
    facing = np.einsum("ij,ij->i", normals, eye - centroids)
    keep = (depths > 1e-4) & (np.abs(facing) > 0)
    normals[facing < 0] *= -1          # STL winding is not reliable

    order = np.argsort(-depths)
    order = order[keep[order]]

    lambert = np.abs(normals @ LIGHT)
    shade = 0.35 + 0.65 * lambert       # ambient + key
    projected = cam.project(tris.reshape(-1, 3)).reshape(-1, 3, 2)

    for i in order:
        pts = projected[i]
        if not np.isfinite(pts).all():
            continue
        colour = np.clip(colours[i] * shade[i], 0, 255)
        bgra = (float(colour[2]), float(colour[1]), float(colour[0]), 255.0)
        cv2.fillConvexPoly(canvas, pts.round().astype(np.int32), bgra, lineType=cv2.LINE_AA)

    return cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)


def crop_to_content(img: np.ndarray, pad_frac: float = 0.04) -> np.ndarray:
    """Trim transparent margin so the subject fills the file.

    A framing choice made in the camera would have to be redone every time
    the pose changes; cropping to the alpha channel is stable across poses.
    """
    ys, xs = np.nonzero(img[:, :, 3])
    if ys.size == 0:
        return img
    pad = int(round(pad_frac * max(img.shape[:2])))
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, img.shape[1])
    return img[y0:y1, x0:x1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1100)
    ap.add_argument("--layers", type=int, default=14)
    ap.add_argument("--tower-x", type=float, default=0.325)
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--no-tower", action="store_true")
    args = ap.parse_args()

    if not args.urdf.exists():
        print(f"URDF not found at {args.urdf}")
        sys.exit(1)

    groups = gather_robot(args.urdf, POSE)
    n_robot = sum(len(g) for g, _ in groups)
    if not args.no_tower:
        groups += build_tower(args.layers, centre=(args.tower_x, 0.0))
    total = sum(len(g) for g, _ in groups)
    print(f"triangles: robot {n_robot}, total {total}")

    intr = CameraIntrinsics.from_fov(args.width, args.height, args.fov)
    camera = PinholeCamera.look_at(intr, eye=(0.78, -0.72, 0.46),
                                   target=(0.16, 0.0, 0.13), up=(0, 0, 1))
    img = render(groups, camera, args.width, args.height)

    img = crop_to_content(img, pad_frac=0.04)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), img)
    covered = (img[:, :, 3] > 0).mean()
    print(f"wrote {args.out} ({args.width}x{args.height}, {covered:.1%} covered)")


if __name__ == "__main__":
    main()
