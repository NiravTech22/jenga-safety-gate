# Progress tracker

## Milestone 1 software evidence — 2026-10-17

Implemented passive schema-v2 overhead/moving-wrist recording, independently
drained cameras, host-receipt monotonic timestamps and sequence IDs, age/skew
validation, failed-read/sequence-gap reports, measured-state/sent-action telemetry
contract, session/tower/calibration metadata, removal/collapse/interruption
events, atomic per-tick manifests, and dual-view replay/video export. Version-1
side-camera recordings remain readable without relabeling or rewriting them.
Recorder/replay issue no motor commands. Existing datasets/checkpoints untouched.

Evidence on this checkout (Python 3.12.10):

- Focused recording/schema/gate/event tests: **38 passed**. Includes injected
  missing/stale/future/skewed frames, state/action freshness, replayed sequences,
  missing images, failed writes, malformed manifests, interrupted runs, drop
  accounting and legacy side-camera replay.
- Broader suite before the final additional malformed-manifest test: **119
  passed, 1 skipped, 3 failed**. All three failures require the absent
  `data/assets/so101/so101_new_calib.urdf`; they are kinematics asset failures,
  not recording regressions. Existing TorchScript deprecation warnings remain.
- CLI synthetic episode:
  `logs/milestone1-smoke/68301b2f-57fb-4c24-9f7e-6851e736a4c8/episode.json`:
  **15 observations**, both views, zero integrity issues, zero failed reads or
  sequence gaps; removal/collapse markers generated explicitly as synthetic.
- Headless replay export `logs/milestone1-replay.mp4`: **15 decoded frames**;
  preview visually inspected for overhead/wrist labels and state/action text.
  `docs/figures/milestone1-evidence.png` generated and visually checked. First
  synthetic frame has about **16 ms host age** versus near-zero subsequent
  frames; this is startup/software work, not measured sensor latency.
- `git diff --check` passed. Installed pytest/matplotlib in the user Python
  environment for tests/plots; no training or checkpoint changes performed.

Documentation reconciliation: MuJoCo implements frozen pre-distilled V-JEPA
features + supervised risk-head training and AccelSight descriptor/velocity
training. Older blanket claims that training is stubbed are obsolete. Physical
ACT policy training, masked-error uncertainty evidence and autonomous gated
extractions remain pending. README highlights historical **2 missed failures /
3 false alarms**, plotted from the documented 48-episode simulation evaluation.
Its source JSON/checkpoints and the documented `.venv` are absent here, so these
numbers were **not reproduced**. Fixed-side training is not moving-wrist evidence.

**Hardware pending (arm disconnected):** external teleop hook providing encoder
feedback and the command actually sent, correct joint ordering/units and
calibration provenance; camera device identification; wrist hand-eye calibration
and moving-view visibility; physical buffering/exposure skew and sustained frame
rate; cable/device/telemetry disconnect behavior; physical event timing; verify
recording start/stop causes no movement. The server API is unavailable locally;
the exact implemented bridge contract and first-run commands are in
[RECORDING.md](RECORDING.md). No dummy feedback is allowed in hardware mode.
All validation above is software/synthetic, **not hardware validation**.

The older chronological entries below retain their historical context; newer
MuJoCo and milestone-1 entries supersede their implementation-status claims.

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
