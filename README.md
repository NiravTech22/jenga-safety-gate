# jenga-safety-gate

Structural anomaly detection as a safety gate for a learned manipulation policy,
built on an SO-101 arm playing Jenga.

This repo is the ground-truth experiment for a broader thesis: **a learned world
model's uncertainty, not its prediction, is what should be allowed to veto an
action.** Jenga extraction is a deliberately small, physically legible domain to
test that end to end: collect data, train a baseline policy, break it, detect the
break, gate it, and honestly report whether gating helped.

- [`docs/PLAN.md`](docs/PLAN.md): the sequential build plan (the spec; not edited)
- [`docs/PROGRESS.md`](docs/PROGRESS.md): live status of every step

<img width="960" height="720" alt="run" src="https://github.com/user-attachments/assets/a1774427-0043-465f-98b0-6129e5930350" />

## Architecture

| Component | What it does | Where | Status |
|---|---|---|---|
| Tower-state CV | Classical CV (ArUco frame, block occupancy, lean); ground truth that is independent of any learned model | `src/perception/` | Done in sim (20/20); hardware pending |
| Camera calibration | Intrinsics, camera-to-robot extrinsics, tower pose in the robot frame, matching MuJoCo camera | `twin/jenga_vision.py` | Verified on synthetic images; hardware pending |
| Anomaly / risk detector | Frozen distilled V-JEPA 2.1 features + trained risk head; AccelSight motion student | `src/runtime/`, `src/safety/anomaly_detector.py` | Trained in MuJoCo sim |
| Safety gate | Blocks actions on anomaly score with a locked threshold; fails closed | `src/safety/gate.py`, `src/runtime/` | Implemented, dry-run only |
| Policy | ACT baseline (Step 5); the digital twin's scripted IK controller generates demonstrations | `scripts/pipeline/05_*`, `twin/` | Scripted controller works in sim; ACT pending data |
| Digital twin | SO-101 + custom claw in MuJoCo, physics pick-and-place | `twin/` | Pick-and-place PASS in sim |
| Tower physics | 18-layer tower that stands in equilibrium and collapses when it is lost, validated against theory | `twin/tower_physics.py` | 11/11 validation tests pass |
| Gated extraction (sim) | Arm test-pushes Jenga blocks; a MuJoCo world model conditioned on the push gates the pull | `twin/jenga_push.py` | 12/12 towers fall ungated vs 4/12 gated (sim) |

Tower state uses classical CV rather than an object detector on purpose: it gives
ground truth that the policy and the anomaly detector can both be scored against.

### Custom gripper

<img width="859" height="619" alt="Screenshot from 2026-11-10 12-39-47" src="https://github.com/user-attachments/assets/9c9c5b6c-948d-42b0-85a4-3f365f7ae21e" />

A servo turns a central pinion between two racks, so the fingers open and close
symmetrically (a prismatic joint), for precise, fine-tuned motion.

<img width="1900" height="933" alt="Screenshot from 2026-11-06 13-22-09" src="https://github.com/user-attachments/assets/04c7ce76-1f98-4965-86fe-9b12136dab09" />

RViz view:

<img width="1201" height="845" alt="image" src="https://github.com/user-attachments/assets/3d709a69-571a-443d-9d4e-061f2bd83a38" />

## Repository layout

```
config/config.yaml        hardware, camera, policy and gate settings
docs/                     plan, progress, and how-to guides (index below)
scripts/
  pipeline/00-09_*.py     one script per PLAN.md step, run in order
  tools/                  diagnostics, asset fetching, figures, training wrappers
  archive/isaac/          Isaac Sim attempt (host segfaults; kept for the record)
src/
  data/                   episode schema, recorder, replay
  eval/                   metrics and the Step 9 comparison
  hardware/               arm interface, kinematics, teleop diagnostics
  perception/             classical tower-state CV
  runtime/                live inference: V-JEPA risk head, motion student, gate pipeline
  safety/                 anomaly detector, safety gate, Invariant event schema
  sim/                    tower model, renderer, MuJoCo Jenga benchmark
  rig.py                  marker layout and tower geometry shared by sim and perception
twin/                     SO-101 + custom-claw digital twin (MuJoCo)
  so101_pickplace.py      physics pick-and-place with viewer / video / headless modes
  jenga_vision.py         camera calibration and tower pose
  models/                 MJCF scenes; custom-claw meshes committed, upstream fetched
tests/                    pytest suite
vendor/motion_perception/ imported AccelSight perception model (own license)
requirements/             base, gpu, mujoco, twin
```

## Quick start (Linux)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/base.txt -r requirements/twin.txt
python scripts/tools/fetch_so101_assets.py     # upstream SO-101 URDF/MJCF/meshes, once
python -m pytest tests -q

python twin/so101_pickplace.py                 # watch the twin pick and place
python twin/jenga_vision.py selftest           # check the calibration tool
python twin/jenga_push.py                      # watch a gated Jenga game
```

GPU work (V-JEPA features, risk head, motion student) needs a CUDA PyTorch wheel
first, then `pip install -r requirements/mujoco.txt`; see
[`docs/MUJOCO_TRAINING.md`](docs/MUJOCO_TRAINING.md).

## Docs

| Doc | Covers |
|---|---|
| [PLAN.md](docs/PLAN.md) | The 10-step build plan and the thesis |
| [PROGRESS.md](docs/PROGRESS.md) | What is done, with evidence and dates |
| [TOWER_DATASET.md](docs/TOWER_DATASET.md) | Database of simulated tower runs: generate, query, replay |
| [TOWER_PHYSICS.md](docs/TOWER_PHYSICS.md) | Tower physics vs static equilibrium: tests, parameters, limits |
| [DIGITAL_TWIN.md](docs/DIGITAL_TWIN.md) | The MuJoCo twin: running it, frames, what was fixed |
| [CAMERA_CALIBRATION.md](docs/CAMERA_CALIBRATION.md) | Calibrating real cameras and locating the tower |
| [SIMULATION.md](docs/SIMULATION.md) | The 2D tower simulation backend and its limits |
| [MUJOCO_TRAINING.md](docs/MUJOCO_TRAINING.md) | MuJoCo + V-JEPA training recipe |
| [REAL_DATA.md](docs/REAL_DATA.md) | Collecting real episodes: cameras, calibration, arm telemetry, what to record |
| [RECORDING.md](docs/RECORDING.md) | Overhead + wrist recording and replay (format details) |
| [INTEGRATION.md](docs/INTEGRATION.md) | Parallel inference runtime and gate semantics |

## Still needed for autonomous Jenga

Hardware camera calibration, the teleop hook (encoder feedback and sent
commands), the Step 4 data campaign, a trained ACT checkpoint, a calibrated gate
threshold, and physical validation. Everything gated today is dry-run only: no
code path in this repo sends motor commands autonomously.
