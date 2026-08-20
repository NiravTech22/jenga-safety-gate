# Ground-Up Build Plan: Structural Anomaly Detection as a Safety Gate

Starting point: VR teleoperation working. Nothing else exists yet.

This document is sequential. Do not start a step before the step above it is
checked off — each one produces something the next step needs.

## Current state (update as you go — see PROGRESS.md for the live tracker)

| Component | Status |
|---|---|
| SO-101 follower arm | Built, calibrated |
| VR teleoperation | Working |
| Camera pipeline | Scaffolded (this repo), not physically mounted |
| Dataset | Does not exist |
| Policy | Does not exist |
| Anomaly detector | Does not exist |
| Invariant safety gate | Scaffolded (this repo), not wired to hardware |

## Step 0 — Fix the target hardware question before anything else

| Component | Runs on | Why |
|---|---|---|
| VR teleop → IK → arm control | Host PC (Quest-paired) | Needs low latency, close to the arm |
| Dataset recording (LeRobot) | Same host PC | LeRobot records on the machine driving the arm |
| Policy training (ACT) | Host PC or GPU machine | One-time offline cost |
| V-JEPA distillation | GPU machine | V-JEPA itself never runs on the Pi |
| Live inference (policy + detector + gate) | Orange Pi Zero 2W | Edge deployment |

**Decision needed before Step 6:** confirm the host PC's GPU situation in
`config/config.yaml`. If none, plan a separate machine for distillation now.

## Step 1 — Camera rig and calibration (no software logic yet)

Goal: two fixed camera angles that don't move for the rest of the project.

1. Mount one overhead camera (full tower view) and one side camera (lean/tilt
   visibility), both rigidly fixed.
2. Fix lighting — a cheap fixed-position light beats ambient light.
3. Take 20 reference photos of an empty table and a built tower from both
   angles.

**Exit condition:** cameras physically locked down, lighting consistent,
reference photos saved to `data/reference_photos/`.

## Step 2 — Tower-state detection (classical CV, no AI)

1. Calibrate a grid/reference frame (ArUco markers at the tower base).
2. Block-edge detection: occupied/empty per layer.
3. Lean/tilt estimate from the side camera.
4. Validate against 20+ manual trials.

**Exit condition:** 20/20 on manual validation. If not, do not proceed —
everything downstream depends on this being trustworthy.

## Step 3 — VR teleop data recording pipeline

1. Wire LeRobot dataset recording into the existing teleop loop: camera
   frames (both angles), joint state, action, per timestep.
2. Confirm frames and actions are time-aligned — the single most common
   silent failure in teleop data collection.
3. Add episode metadata: scenario category + outcome label.

**Exit condition:** one full recorded episode, played back, is a clean,
correctly time-aligned (image, state, action) sequence.

## Step 4 — Full data collection campaign

| Scenario category | Description | Sessions |
|---|---|---|
| Stable tower, easy extraction | Standard tower, loose blocks | 5 |
| Stable tower, tight extraction | Standard tower, snug/friction-fit blocks | 4 |
| Leaning tower | Intentional lean | 4 |
| Near-collapse tower | Several blocks removed, high instability | 3 |
| Off-center block placement | Irregular geometry | 3 |
| Recovery scenarios | Induced wobble mid-extraction, corrected | 3 |
| **Total** | | **22 sessions minimum** |

**Exit condition:** 22+ labeled episodes, spot-checked, bad ones discarded
and re-recorded.

## Step 5 — Baseline policy (ungated) — control group

1. Train ACT via LeRobot on the Step 4 dataset.
2. Deploy autonomously, no gate.
3. Run across held-out test scenarios (reserved, not trained on).
4. Log: success rate, catastrophic failure rate, near-miss rate.

**Exit condition:** baseline numbers recorded. This table is frozen once
recorded — it's what Step 9 compares against.

## Step 6 — V-JEPA-distilled anomaly detector

**Part A (GPU machine):**
1. Pull smallest published V-JEPA ViT checkpoint.
2. Extract embeddings over Step 4 frames (stable + hand-labeled risky).
3. Distill a small student model (shallow CNN/MLP) to regress an anomaly
   score from raw frames, trained against V-JEPA's embedding-space
   prediction error.

**Part B (manual, parallel with A):**
1. Hand-label ~75 held-out frames: normal / risky.
2. Score the student model's precision/recall against these labels.

**Exit condition:** student model runs in real time on the Orange Pi
(measured, not assumed) and clears a precision/recall bar above random
guessing on the 75-frame validation set.

## Step 7 — Wire the safety gate (Invariant)

1. Deploy Step 5 policy + Step 6 detector to the Orange Pi.
2. Invariant DAG node: input = proposed action + anomaly score, output =
   execute/block against a fixed threshold.
3. **Lock the threshold before Step 8.** Tune only against Step 6's
   validation set, never against live test runs.

**Exit condition:** dry run — gate correctly blocks an obviously-bad
proposed action and passes an obviously-fine one.

## Step 8 — Gated policy run — treatment group

Same policy, same held-out scenarios as Step 5, now behind the gate. Log the
Step 5 metrics plus gate false-positive rate and added latency per action.

**Exit condition:** full metrics table across the same scenarios as Step 5.

## Step 9 — The comparison

| Metric | Ungated (Step 5) | Gated (Step 8) | Δ |
|---|---|---|---|
| Success rate | | | |
| Catastrophic failure rate | | | |
| Near-miss rate | | | |
| Gate false-positive rate | n/a | | |
| Added latency (ms/action) | n/a | | |

**Thesis supported if:** catastrophic failure rate drops meaningfully without
success rate collapsing and without the gate just blocking everything.

**Thesis not supported (still publishable) if:** the gate doesn't meaningfully
help, or only helps by being overly conservative. Document honestly — a clean
negative result with good methodology beats a forced positive one.

## Sequencing

```
0 hardware split → 1 camera rig → 2 tower-state CV (20/20) → 3 recording pipeline
  → 4 data campaign (22+ episodes)
        ├─→ 5 baseline policy ──┐
        └─→ 6 anomaly detector ─┴─→ 7 wire gate → 8 gated run → 9 comparison
```

Steps 5 and 6 can run in parallel once Step 4 is done. Everything else is
strictly sequential.

---

## Origin and connection to the broader research thread

This build is downstream of a longer line of reasoning worth keeping attached
to the repo, not just in chat history:

- **Invariant** started as a general execution-observability layer for the
  SO-101 (event schema: node_id, stage, status, timestamp, latency,
  input/output refs, failure_reason) and was extended toward embodied
  diagnostics: sim-to-real divergence per node, spatially-embedded DAGs, and
  a proprioceptive veto comparing high-level policy intent against low-level
  sensor/motor reality.
- Applied to construction robotics, the same structure reframes as an
  **attestation layer**: independent, diverse monitors (not sharing sensors,
  clocks, or models with the policy) that make autonomous execution
  auditable to a third party — a GC, a surety, an inspector — which is the
  actual blocker on unattended work, not raw capability.
- The safety thesis this repo tests is the general one: **a world model's
  epistemic uncertainty, not its point prediction, is the trustworthy safety
  signal**, because the model is least accurate exactly when it matters most.
  This follows Nakamura/Bajcsy's latent safety filters (HJ reachability in
  the latent space of a generative world model) and the uncertainty-aware
  extension (conformal-calibrated epistemic uncertainty as an OOD proxy),
  and is meant to sit alongside runtime monitors like Sentinel (STAC +
  VLM progress checks) and SAFE (learned failure likelihood from VLA
  internal features).
- "Pause and think even in the worst conditions" decomposes into three
  distinct mechanisms that must not be conflated: (1) a continuously-valid
  **abort reachability** guarantee — the hard part, and mostly unaddressed
  in the literature; (2) spending more inference compute only when
  uncertainty is high; (3) **deferring to a human** with a calibrated,
  conformal-guaranteed intervention rate (cf. KnowNo). The Jenga gate in
  this repo is a concrete, physically legible testbed for (1) and (2) before
  attempting the general manipulation case.
- The camera-fixed, classical-CV-first approach in Steps 1-2 is deliberate:
  it gives ground-truth tower state independent of any learned model, so the
  anomaly detector and the policy can both be scored against something that
  isn't itself learned.
