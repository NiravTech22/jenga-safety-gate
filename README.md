# jenga-safety-gate:

The Arm's Machine Learning Architecture is based on 3 main existing models:
- YOLOv3 for detecting Jenga blocks and their approximate locations
- V-JEPA 2.1 for learning representations of visual motion and tower dynamics
- ACT for generating the Arm's actuation
- Safety Gate: a custom trained policy that rejects actions that violate set safety constraints (calculated with a failure risk equation)
- VLA for interpreting language instructions and visual observations from YOLOv3 and V-JEPA to generate robot actions.

<img width="1900" height="933" alt="Screenshot from 2026-11-06 13-22-09" src="https://github.com/user-attachments/assets/04c7ce76-1f98-4965-86fe-9b12136dab09" />

In need of development:
Safety-gate runtime, CUDA setup,
local student training, and human-turn action checks  The integrated runtime is dry-run only;
trained checkpoints, calibrated risk thresholds, and physical validation are
still required for autonomous Jenga.

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
