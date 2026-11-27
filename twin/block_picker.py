#!/usr/bin/env python3
"""Pick the next Jenga block to pull, from static equilibrium and where the arm can reach.

The arm works one side of the tower and pulls a block straight out toward itself by
its end, so only blocks in layers whose length points AT the arm are candidates.
Each candidate is removed in the load-path model (twin/tower_dataset.equilibrium:
rigid blocks, force and moment balance, top down); the ones that keep every block's
load over its supports are ranked by their equilibrium margin (mm, larger = safer).
Statics cannot tell tight from loose blocks (real thickness differences): that is the
wiggle check's job, after this picker.

Blocks are named from the ARM's point of view:
    pointing layers (length toward the arm):  L / M / R   (left, middle, right)
    crossing layers (length across):          N / M / F   (near, middle, far)
Layer 0 is the bottom. Example: --removed 3M 5L 4N

Usage
  python twin/block_picker.py --layer0 pointing                          # intact tower, tower at the calibrated spot
  python twin/block_picker.py --layer0 pointing --removed 3M 5M --from-camera
  python twin/block_picker.py --layer0 crossing --tower 409 0 -90 --removed 2M

--layer0: does the BOTTOM layer's blocks point at the arm ('pointing') or lie across ('crossing')?
Prints a side view of the tower and the ranked picks with the grasp point (the block
end facing the arm) in the arm base frame, mm: the same frame as arm_twin's tip readout.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

TWIN = Path(__file__).resolve().parent
sys.path.insert(0, str(TWIN))
import tower_dataset as td  # noqa: E402
import tower_physics as TP  # noqa: E402

PAN_AXIS_XY = np.array([0.0388, 0.0])      # shoulder-pan axis in the arm base frame (jenga_vision)
BASE_BOARD_M = 0.010                       # the tower stands on the 10 mm base board
POINTING_NAMES = {"L": 2, "M": 1, "R": 0}  # see slot_index(): tower frame +y is the arm's left
CROSSING_NAMES = {"N": 0, "M": 1, "F": 2}


def layer_points_at_arm(layer, layer0):
    return (layer % 2 == 0) == (layer0 == "pointing")


def slot_index(layer, name, layer0):
    """Arm's-view name -> model slot. Model (tower_physics): even layers lie along tower x
    with slots along +y; odd layers along y with slots along +x. We take the pointing
    layers' length axis as 'away from the arm'; in a right-handed frame that makes the
    crossing layers' slot axis also 'away' (near -> far) and the pointing layers' slot
    axis the arm's left (right -> left). Statics are mirror-symmetric, so only this
    consistency matters."""
    names = POINTING_NAMES if layer_points_at_arm(layer, layer0) else CROSSING_NAMES
    if name not in names:
        raise ValueError(f"layer {layer} {'points at' if layer_points_at_arm(layer, layer0) else 'crosses'} "
                         f"the arm: use one of {'/'.join(names)}")
    return names[name]


def slot_name(layer, slot, layer0):
    names = POINTING_NAMES if layer_points_at_arm(layer, layer0) else CROSSING_NAMES
    return {v: k for k, v in names.items()}[slot]


def parse_removed(items, layer0):
    out = set()
    for it in items:
        layer, name = int(it[:-1]), it[-1].upper()
        out.add((layer, slot_index(layer, name, layer0)))
    return out


def allowed_layers(layers, removed):
    """Jenga rule: never the top layer; not the layer below it while the top is incomplete."""
    top = layers - 1
    top_complete = all((top, s) not in removed for s in range(3))
    return range(0, top - (0 if top_complete else 1))


def rank(p, removed, layer0, min_layer=2):
    """Safe first; then larger equilibrium margin (rounded, it is a whole-tower minimum and
    often ties); then middle blocks (both neighbours keep supporting); then mid-height
    layers (comfortable reach, away from the base board and the top)."""
    L = int(p.v("layers"))
    picks = []
    for layer in allowed_layers(L, removed):
        if layer < min_layer or not layer_points_at_arm(layer, layer0):
            continue
        for s in range(3):
            if (layer, s) in removed:
                continue
            stands, margin = td.equilibrium(p, removed | {(layer, s)})
            picks.append(dict(layer=layer, slot=s, name=f"{layer}{slot_name(layer, s, layer0)}",
                              stands=stands, margin_mm=margin))
    mid = (L - 1) / 2
    picks.sort(key=lambda r: (not r["stands"], -round(r["margin_mm"], 1), r["slot"] != 1, abs(r["layer"] - mid)))
    return picks


def grasp_point(p, tower_xy_mm, layer, slot):
    """End of the block facing the arm, arm base frame, mm. The pointing direction is the
    line from the pan axis to the tower centre; slot offsets run across it."""
    c = np.asarray(tower_xy_mm, float) / 1000
    away = c - PAN_AXIS_XY
    away /= np.linalg.norm(away)
    left = np.array([-away[1], away[0]])
    pitch = TP.slot_pitch(p)
    offset = (slot - 1) * pitch          # pointing layers: slot axis = arm's left (see slot_index)
    end = c + offset * left - away * p.v("block_length") / 2
    z = BASE_BOARD_M + (layer + .5) * p.v("block_height")
    return 1000 * np.array([end[0], end[1], z])


def side_view(L, removed, layer0, best=None):
    rows = []
    for layer in range(L - 1, -1, -1):
        pointing = layer_points_at_arm(layer, layer0)
        cells = []
        for s in (2, 1, 0) if pointing else (0, 1, 2):   # pointing: drawn L M R; crossing: N M F
            mark = "  " if (layer, s) in removed else ("<>" if best == (layer, s) else "##")
            cells.append(mark)
        kind = "points at arm " + "/".join("LMR") if pointing else "crosses       " + "/".join("NMF")
        rows.append(f"  {layer:2d} [{' '.join(cells)}]  {kind}")
    return "\n".join(rows)


def tower_from_camera(camera, intr, extr, offset, marker_z, marker_mm):
    import cv2
    import jenga_vision as jv
    intr_d, ext = jv.load_json(intr), jv.load_json(extr)
    T = np.array(ext["T_base_cam"]); w, h = intr_d["image_size"]
    cap = jv.open_camera(camera, w, h); rs = []
    for _ in range(200):
        ok, f = cap.read()
        r = jv.tower_pose(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), intr_d, T, marker_mm=marker_mm,
                          offset=offset, marker_z=marker_z) if ok else None
        if r:
            rs.append(r)
        if len(rs) >= 30:
            break
    cap.release()
    if not rs:
        sys.exit("tower marker not visible")
    return (1000 * float(np.median([r["x"] for r in rs])), 1000 * float(np.median([r["y"] for r in rs])),
            jv.circular_mean_deg([r["yaw_deg"] for r in rs]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer0", choices=["pointing", "crossing"], required=True,
                    help="do the bottom layer's blocks point at the arm, or lie across?")
    ap.add_argument("--removed", nargs="*", default=[], help="e.g. 3M 5L 4N (layer + L/M/R or N/M/F)")
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--tower", type=float, nargs=3, metavar=("X_MM", "Y_MM", "YAW_DEG"), default=[409.0, 0.0, -90.0],
                    help="tower centre in the arm base frame (default: the calibrated spot)")
    ap.add_argument("--from-camera", action="store_true", help="measure the tower position with the overhead camera")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--intrinsics", default="data/calibration/overhead_intrinsics.json")
    ap.add_argument("--extrinsics", default="data/calibration/overhead_extrinsics.json")
    ap.add_argument("--offset", type=float, nargs=2, default=[0, -0.09])
    ap.add_argument("--marker-z", type=float, default=0.010)
    ap.add_argument("--marker-mm", type=float, default=45.3)
    ap.add_argument("--min-layer", type=int, default=2, help="lowest layer to pull from (claw clearance above the board)")
    ap.add_argument("--top", type=int, default=5, help="how many picks to list")
    ap.add_argument("--json", action="store_true", help="print the ranked picks as JSON")
    a = ap.parse_args(argv)

    p = TP.TowerPhysics(); p.layers.value = a.layers
    try:
        removed = parse_removed(a.removed, a.layer0)
    except (ValueError, IndexError) as exc:
        ap.error(str(exc) or "bad --removed entry; use layer + letter, e.g. 4M")
    tower = tower_from_camera(a.camera, a.intrinsics, a.extrinsics, a.offset, a.marker_z, a.marker_mm) \
        if a.from_camera else tuple(a.tower)
    stands_now, margin_now = td.equilibrium(p, removed)
    picks = rank(p, removed, a.layer0, a.min_layer)
    for r in picks:
        r["grasp_mm"] = [round(float(v), 1) for v in grasp_point(p, tower[:2], r["layer"], r["slot"])]
    safe = [r for r in picks if r["stands"]]
    if a.json:
        print(json.dumps(dict(tower_mm_deg=tower, stands_now=stands_now, picks=picks[:a.top]), indent=1))
        return 0
    print(f"tower at x={tower[0]:.0f} y={tower[1]:.0f} mm, yaw {tower[2]:.0f} deg; {a.layers} layers; "
          f"removed: {' '.join(a.removed) or 'none'}")
    print(f"current tower: {'stands' if stands_now else 'SHOULD NOT STAND'} (margin {margin_now:.1f} mm)\n")
    best = (safe[0]["layer"], safe[0]["slot"]) if safe else None
    print("side view from the arm (top first, ## = block, <> = best pick):")
    print(side_view(a.layers, removed, a.layer0, best))
    print(f"\n{'pick':>5}  {'margin':>8}  grasp point (block end facing arm), arm frame mm")
    for r in safe[:a.top]:
        x, y, z = r["grasp_mm"]
        print(f"{r['name']:>5}  {r['margin_mm']:7.1f}mm  x={x:6.1f} y={y:6.1f} z={z:6.1f}")
    unsafe = [r["name"] for r in picks if not r["stands"]]
    if unsafe:
        print(f"\nwould topple the tower (statics): {' '.join(unsafe)}")
    if not safe:
        print("\nno reachable block keeps the tower in equilibrium")
    print("\nStatics assume identical blocks. Wiggle-test the pick before pulling: tight = carrying load.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
