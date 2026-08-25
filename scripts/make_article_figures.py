#!/usr/bin/env python3
"""Generate the figures used in the write-ups.

Every figure here comes out of the actual pipeline rather than being drawn to
illustrate it: the camera views are real renders, the occupancy overlay is the
real detector's output with its real decisions, and the stability curve is
measured by pulling blocks out of the simulation until it falls over. If the
pipeline regresses, these figures change, which is the point.

  python scripts/make_article_figures.py --out-dir /path/to/figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.perception.tower_state import (
    ArucoFrameCalibrator,
    detect_block_occupancy,
    estimate_lean_from_silhouette,
    fit_occupancy_model,
)
from src.rig import SLOTS_PER_LAYER, slot_face_quad
from src.sim.render import RenderConfig, default_rig, render_empty_table, render_frame_pair
from src.sim.tower import ExtractionOutcome, TowerSim, TowerSimConfig, stability_score

N_LAYERS = 12
OK_GREEN = (120, 220, 150)
GAP_RED = (110, 120, 250)


FONT = cv2.FONT_HERSHEY_DUPLEX


def label(img, text, org, colour=(245, 245, 245), scale=0.62):
    """Caption on a translucent plate.

    A stroked outline was unreadable against the tower: at this size the
    stroke is thick enough relative to the glyphs that it reads as a second
    copy of the text.
    """
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    x, y = org
    pad = 8
    box = img[max(y - th - pad, 0):y + base + pad, max(x - pad, 0):x + tw + pad]
    if box.size:
        box[:] = cv2.addWeighted(box, 0.28, np.full_like(box, 18), 0.72, 0)
    cv2.putText(img, text, (x, y), FONT, scale, colour, 1, cv2.LINE_AA)


def save_raster(path: Path, img) -> Path:
    """JPEG for the rendered scenes: they are photographic in character and a
    lossless encode of one costs about a megabyte for no visible gain."""
    out = path.with_suffix(".jpg")
    cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return out


def calibrate(rig):
    empty_over, empty_side = render_empty_table(rig, np.random.default_rng(0))
    K = rig.side.intrinsics.matrix()
    cal_side = ArucoFrameCalibrator(K, layout=rig.markers)
    cal_over = ArucoFrameCalibrator(K, layout=rig.markers)
    cal_side.calibrate(empty_side)
    cal_over.calibrate(empty_over)
    refs = []
    for i in range(4):
        sim = TowerSim(TowerSimConfig(seed=700 + i, num_layers=N_LAYERS))
        refs.append(render_frame_pair(sim.state, rig, np.random.default_rng(800 + i))[1])
    return cal_side, cal_over, fit_occupancy_model(refs, cal_side, N_LAYERS)


def figure_cameras(rig, out: Path):
    """The two fixed views the whole pipeline runs on."""
    sim = TowerSim(TowerSimConfig(seed=3, num_layers=N_LAYERS,
                                  pre_removed=((3, 1), (5, 0), (7, 2)),
                                  initial_lean_deg=1.5))
    over, side = render_frame_pair(sim.state, rig, np.random.default_rng(11))
    label(over, "overhead: marker frame, top layer", (22, 44))
    label(side, "side, corner view: every layer resolvable", (22, 44))
    gap = np.full((over.shape[0], 10, 3), 24, np.uint8)
    return save_raster(out, np.hstack([over, gap, side]))


def figure_occupancy(rig, cal_side, model, out: Path):
    """The detector's actual decisions, drawn on the frame it decided from."""
    sim = TowerSim(TowerSimConfig(seed=21, num_layers=N_LAYERS,
                                  pre_removed=((2, 1), (4, 0), (4, 2), (6, 1), (9, 0)),
                                  initial_lean_deg=2.5))
    _, side = render_frame_pair(sim.state, rig, np.random.default_rng(31))
    lean, iou = estimate_lean_from_silhouette(side, cal_side, N_LAYERS)
    occ, _ = detect_block_occupancy(side, cal_side, N_LAYERS, model, lean)

    canvas = side.copy()
    overlay = canvas.copy()
    for layer in range(N_LAYERS):
        for slot in range(SLOTS_PER_LAYER):
            quad = cal_side.project(slot_face_quad(layer, slot, lean))
            if not np.isfinite(quad).all():
                continue
            pts = quad.round().astype(np.int32)
            colour = OK_GREEN if occ[layer, slot] else GAP_RED
            cv2.fillConvexPoly(overlay, pts, colour)
            cv2.polylines(canvas, [pts], True, colour, 1, cv2.LINE_AA)
    canvas = cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0)

    wrong = int((occ != sim.state.occupancy).sum())
    label(canvas, "slot regions projected from the recovered pose", (22, 44))
    label(canvas, "green = read occupied,  red = read empty", (22, 78))
    label(canvas, f"lean {lean:+.1f} deg (true {sim.state.lean_deg:+.1f})   "
                  f"silhouette IoU {iou:.2f}   errors {wrong}/{occ.size}",
          (22, canvas.shape[0] - 26))
    return save_raster(out, canvas)


def figure_stability_svg(out: Path):
    """Measured, not illustrative: pull blocks until the tower goes over."""
    rng = np.random.default_rng(5)
    runs = []
    for trial in range(24):
        sim = TowerSim(TowerSimConfig(seed=int(rng.integers(1e6)), num_layers=N_LAYERS))
        series = []
        for step in range(26):
            actions = sim.legal_actions()
            if not actions:
                break
            margin = stability_score(sim.state)
            risk = max(sim.risk(l, s) for l, s in actions)
            series.append((step, margin, risk))
            layer, slot = actions[int(rng.integers(len(actions)))]
            if sim.extract(layer, slot) is ExtractionOutcome.TOPPLE:
                break
        runs.append(series)

    max_step = max(len(r) for r in runs)
    margin_by_step, risk_by_step = [], []
    for step in range(max_step):
        m = [r[step][1] for r in runs if len(r) > step]
        k = [r[step][2] for r in runs if len(r) > step]
        margin_by_step.append(float(np.mean(m)))
        risk_by_step.append(float(np.mean(k)))

    W, H = 720, 300
    pad_l, pad_r, pad_t, pad_b = 52, 52, 26, 40
    plot_w, plot_h = W - pad_l - pad_r, H - pad_t - pad_b
    m_max = max(margin_by_step)

    def pts(values, vmax):
        out = []
        for i, v in enumerate(values):
            x = pad_l + plot_w * i / max(len(values) - 1, 1)
            y = pad_t + plot_h * (1 - (v / vmax if vmax else 0))
            out.append(f"{x:.1f},{y:.1f}")
        return " ".join(out)

    gridlines = "".join(
        f'<line x1="{pad_l}" y1="{pad_t + plot_h * f:.1f}" x2="{pad_l + plot_w}" '
        f'y2="{pad_t + plot_h * f:.1f}" class="grid"/>'
        for f in (0, 0.25, 0.5, 0.75, 1.0))
    ticks = "".join(
        f'<text x="{pad_l + plot_w * i / max(max_step - 1, 1):.1f}" y="{H - 14}" '
        f'class="tick" text-anchor="middle">{i}</text>'
        for i in range(0, max_step, 4))

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img"
     aria-label="Mean stability margin falls and worst-case topple probability rises as blocks are removed, over 24 simulated runs.">
  <style>
    .grid {{ stroke: currentColor; stroke-opacity: 0.12; stroke-width: 1; }}
    .axis {{ stroke: currentColor; stroke-opacity: 0.35; stroke-width: 1; }}
    .tick, .lbl {{ fill: currentColor; fill-opacity: 0.62; font: 11px 'JetBrains Mono', monospace; }}
    .margin {{ fill: none; stroke: #5b8cff; stroke-width: 2.4; stroke-linejoin: round; }}
    .risk {{ fill: none; stroke: #ff6b6b; stroke-width: 2.4; stroke-linejoin: round; }}
    .key {{ font: 11px 'JetBrains Mono', monospace; fill: currentColor; fill-opacity: 0.8; }}
  </style>
  {gridlines}
  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" class="axis"/>
  <line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" y2="{pad_t + plot_h}" class="axis"/>
  <polyline class="margin" points="{pts(margin_by_step, m_max)}"/>
  <polyline class="risk" points="{pts(risk_by_step, 1.0)}"/>
  {ticks}
  <text x="{pad_l}" y="{H - 2}" class="lbl">blocks removed</text>
  <text x="{pad_l + 6}" y="{pad_t + 14}" class="key" fill="#5b8cff">stability margin</text>
  <text x="{pad_l + 6}" y="{pad_t + 30}" class="key" fill="#ff6b6b">worst-case topple probability</text>
</svg>'''
    out.write_text(svg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Figures render at 2x the pipeline's working resolution purely so they
    # hold up on a high-density display. Intrinsics follow the config, and
    # calibration runs against this same rig, so the geometry stays consistent.
    rig = default_rig(RenderConfig(width=1280, height=960))
    cal_side, cal_over, model = calibrate(rig)

    print(figure_cameras(rig, args.out_dir / "fig-cameras.png"))
    print(figure_occupancy(rig, cal_side, model, args.out_dir / "fig-occupancy.png"))
    print(figure_stability_svg(args.out_dir / "fig-stability.svg"))


if __name__ == "__main__":
    main()
