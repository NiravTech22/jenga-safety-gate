# Progress tracker

Update this file as steps complete. Keep PLAN.md unmodified — it's the spec,
this file is the log.

| Step | Status | Notes | Date |
|---|---|---|---|
| 0 — Hardware split | ❌ Blocked | `has_gpu: false` with `distillation_machine: host_pc`. Needs a real decision before Step 6 — `scripts/00_check_hardware.py` exits 1 until then | |
| 1 — Camera rig | ✅ Done (sim) | `scripts/01_capture_reference.py`: 21 reference pairs, both cameras calibrated from the empty table, occupancy model fitted. Hardware capture still to do | 2026-08-21 |
| 2 — Tower-state CV | ✅ Done (sim) | `scripts/02_validate_tower_state.py`: 20/20 trials, 720/720 slots, mean lean error 0.23° | 2026-08-21 |
| 3 — Recording pipeline | 🟨 In progress | `src/data/recorder.py` works dry-run; needs wiring to the sim/hardware frame source | |
| 4 — Data campaign | ⬜ Not started | 0 / 22 sessions | |
| 5 — Baseline policy | ⬜ Not started | | |
| 6 — Anomaly detector | ⬜ Not started | Validation harness exists; distillation blocked on Step 4 | |
| 7 — Wire gate | ⬜ Not started | Gate logic + threshold locking implemented (`src/safety/gate.py`) | |
| 8 — Gated run | ⬜ Not started | | |
| 9 — Comparison | ⬜ Not started | Table generator + verdict logic exist (`src/eval/compare.py`) | |

**Backend note:** steps marked "(sim)" were completed against the simulation
backend (`docs/SIMULATION.md`), which substitutes for hardware at frame
capture only. They must be re-run on the physical rig before any number
produced downstream counts as a result.

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
