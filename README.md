# jenga-safety-gate

Structural anomaly detection as a safety gate for a learned manipulation policy,
built on an SO-101 arm with VR teleoperation.

This repo is the ground-truth experiment for a broader thesis: **a learned world
model's uncertainty, not its prediction, is what should be allowed to veto an
action.** The Jenga-extraction task is a deliberately small, physically legible
domain to test that thesis end to end — collect data, train a baseline policy,
break it, detect the break, gate it, and honestly report whether gating helped.

See [`docs/PLAN.md`](docs/PLAN.md) for the full sequential build plan and
[`docs/PROGRESS.md`](docs/PROGRESS.md) for current status. Do not skip ahead —
each step in the plan produces something the next step consumes, and exit
conditions are load-bearing, not decorative.

## Repo layout

```
config/                 Hardware + experiment configuration
src/hardware/           Arm interface, camera rig control
src/perception/         Camera calibration, classical-CV tower-state extraction
src/data/                Episode schema, LeRobot-compatible recorder
src/policy/              ACT training + inference wrappers
src/safety/              Invariant event schema, anomaly detector, execute/block gate
src/eval/                Metrics + the Step 9 comparison table
scripts/                 One script per plan step (00-09), run in order
tests/                   Tests for the parts that don't require hardware
docs/                    PLAN.md (source of truth), PROGRESS.md (status tracker)
```

## Running order

Scripts in `scripts/` are numbered to match the plan steps exactly. `00` through
`03` can be run and iterated on without any Jenga-specific data. `04` onward
requires the physical rig. Do not run a script until the previous step's exit
condition (see PLAN.md) is actually met — the pipeline is intentionally not
forgiving about this, because silent data misalignment is the most common way
projects like this quietly fail.

## What's real vs. stubbed right now

This initial commit implements the parts that don't require the physical rig or
a GPU: the episode schema, the Invariant event schema, the ArUco-based
tower-state pipeline skeleton, the recorder, the gate logic, and the metrics /
comparison-table code. Policy training, V-JEPA distillation, and the Orange Pi
deployment are stubbed with clear interfaces and TODOs — they need real camera
frames and real demonstration data before they can be anything but scaffolding.
See each module's docstring for exactly what's implemented vs. stubbed.

## Hardware assumptions (Step 0)

| Component | Runs on |
|---|---|
| VR teleop / IK / arm control | Host PC (Quest-paired) |
| Dataset recording | Same host PC |
| Policy training (ACT) | Host PC or GPU machine |
| V-JEPA distillation | GPU machine (one-time, offline) |
| Live inference (policy + detector + gate) | Orange Pi Zero 2W |

Confirm this split in `config/config.yaml` before running anything past `01`.
