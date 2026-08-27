# Progress tracker

Update this file as steps complete. Keep PLAN.md unmodified — it's the spec,
this file is the log.

| Step | Status | Notes | Date |
|---|---|---|---|
| 0 — Hardware split | ✅ Done | RTX 3050 8 GB (CC 8.6), Ryzen 5 5600X, 15.5 GB RAM — measured, not assumed. Distillation stays on host_pc, constrained to the smallest V-JEPA ViT with embeddings cached to disk. `scripts/00_check_hardware.py` exits 0 | 2026-08-22 |
| 1 — Camera rig | ✅ Done (sim) | `scripts/01_capture_reference.py`: 21 reference pairs, both cameras calibrated from the empty table, occupancy model fitted. Hardware capture still to do | 2026-08-21 |
| 2 — Tower-state CV | ✅ Done (sim) | `scripts/02_validate_tower_state.py`: 20/20 trials, 720/720 slots, mean lean error 0.23° | 2026-08-21 |
| 3 — Recording pipeline | 🟨 In progress | `src/data/recorder.py` works dry-run; needs wiring to the sim/hardware frame source | |
| 4 — Data campaign | ⬜ Not started | 0 / 22 sessions | |
| 5 — Baseline policy | ⬜ Not started | | |
| 6 — Anomaly detector | ⬜ Not started | Validation harness exists; distillation blocked on Step 4 | |
| 7 — Wire gate | ⬜ Not started | Gate logic + threshold locking implemented (`src/safety/gate.py`) | |
| 8 — Gated run | ⬜ Not started | | |
| 9 — Comparison | ⬜ Not started | Table generator + verdict logic exist (`src/eval/compare.py`) | |

**Step 6 risk carried forward:** 8 GB VRAM is enough for ViT-S inference and
a shallow student, and nothing larger. If measurement at Step 6 says
otherwise, move distillation to a rented GPU rather than shrinking the
experiment to fit the machine.

**Backend note:** steps marked "(sim)" were completed against the simulation
backend (`docs/SIMULATION.md`), which substitutes for hardware at frame
capture only. They must be re-run on the physical rig before any number
produced downstream counts as a result.

Legend: ⬜ not started · 🟨 in progress · ✅ done · ❌ blocked (note why)

## Teleop accuracy

Before Step 4 collection begins, characterise the arm rather than adjusting
it by feel: `python scripts/diagnose_teleop.py run --hardware`. The three
issues below are all things that report reads directly, and it distinguishes
faults that feel identical in the headset but are fixed in different layers
(a constant offset, a span error, lost motion, and command latency).

`scripts/diagnose_teleop.py z-floor --hardware` measures the table height so
the disabled safety floor can be restored to a real number.

Blocking on the hardware side: `src/hardware/arm.py:VRTeleopArm` has two
unimplemented methods. Point them at whatever the existing teleop stack uses
to read joint state and command joint positions, and both tools work.

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
