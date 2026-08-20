# Progress tracker

Update this file as steps complete. Keep PLAN.md unmodified — it's the spec,
this file is the log.

| Step | Status | Notes | Date |
|---|---|---|---|
| 0 — Hardware split | ⬜ Not started | Confirm host PC GPU in config.yaml | |
| 1 — Camera rig | ⬜ Not started | | |
| 2 — Tower-state CV | ⬜ Not started | | |
| 3 — Recording pipeline | ⬜ Not started | Repo scaffolding for this exists (`src/data/recorder.py`) | |
| 4 — Data campaign | ⬜ Not started | 0 / 22 sessions | |
| 5 — Baseline policy | ⬜ Not started | | |
| 6 — Anomaly detector | ⬜ Not started | | |
| 7 — Wire gate | ⬜ Not started | Gate logic scaffolding exists (`src/safety/gate.py`) | |
| 8 — Gated run | ⬜ Not started | | |
| 9 — Comparison | ⬜ Not started | Table generator exists (`src/eval/compare.py`) | |

Legend: ⬜ not started · 🟨 in progress · ✅ done · ❌ blocked (note why)

## Known open issues (carried over from teleop setup)

- `wrist_flex` calibration is asymmetric (743–3157, resting near the top of
  range) — recalibrate before Step 4 data collection, not after, or every
  episode inherits the same directional bias.
- `z_floor` in the VR control config was previously set to `-0.05` to work
  around an incorrectly-calibrated ground plane. This is a workaround, not a
  fix — it removes a real safety constraint rather than correcting its
  value. Before Step 3 recording begins: measure true table height in the
  base frame (jog down to contact, read FK Z) and set `z_floor` a few mm
  below that measured value, then delete the `-0.05` override.
