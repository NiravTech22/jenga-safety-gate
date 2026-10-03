# MuJoCo + pretrained distilled V-JEPA

This Windows host now has MuJoCo, the upstream SO-101 MJCF/meshes, Meta's
official distilled V-JEPA 2.1 ViT-B checkpoint, and a runnable simulation
training pipeline. All commands below run from the repository root.

## View and run

Open the interactive MuJoCo scene (close the viewer to stop):

```powershell
.\.venv\Scripts\python.exe -m src.sim.mujoco_jenga --viewer
```

Replay held-out simulation clips through both trained CUDA branches:

```powershell
.\.venv\Scripts\python.exe -m src.parallel.mujoco_demo --steps 12
```

An optional video reruns those same physics seeds, overlays each gate decision,
and continues the simulation to show the counterfactual outcome:

```powershell
.\.venv\Scripts\python.exe -m src.parallel.mujoco_demo --steps 12 --video logs/mujoco_demo.mp4
```

Rebuild data, features, both trained components, and the parallel smoke run:

```powershell
.\scripts\train_mujoco.ps1
```

Use different output paths and a new seed range for independent experiments:

```powershell
.\scripts\train_mujoco.ps1 -Episodes 1000 -Seed 10000 -Dataset data/mujoco_jenga_v2 -Output checkpoints/mujoco_vjepa_v2
```

## What is trained

1. MuJoCo simulates 36 free-body blocks, contact, friction and gravity, with
   an SO-101 arm parked away from the tower. Block mass, friction, small placement
   errors and wood color vary by episode. Side, overhead and overview cameras
   are available; this training recipe uses the side camera.
2. A brief lateral force acts on the top layer. Four RGB frames spaced 20 ms
   apart form the observation. Episodes already beyond the failure criterion
   at the end of the observation period are rejected. Physics then runs for
   another second to produce the binary outcome. Failure means any block loses
   over 3 cm height or moves over 4 cm horizontally from its settled position.
3. The official **V-JEPA 2.1 ViT-B distilled from ViT-G** encoder is loaded with
   `strict=True` using `ema_encoder`. The pretrained backbone stays frozen;
   a linear risk head learns from its cached pooled features. This is supervised
   adaptation of a pre-distilled checkpoint, not training V-JEPA from scratch
   or estimating its masked-prediction uncertainty.
4. AccelSight learns descriptors from those pretrained features and pooled
   top-layer velocity from the simulator. Only embedding and velocity outputs
   are exposed. Object detection, boxes, and arm controls are not trained.
5. Seed-disjoint episodes are split 60/20/20 into train/validation/test. Only
   training features set normalization statistics; validation selects the risk
   head and threshold. Test outcomes are reported once after those are fixed.
   A simulation bar checks precision >=0.6, recall >=0.8 and normal-pass >=0.5.
   This is a development criterion, not a hardware release condition.

The runtime executes AccelSight and V-JEPA scoring + SafetyGate concurrently.
The default 500 ms deadline still blocks stale/failed computation. Simulation
is paused while processing recorded clips; this does not prove real-time arm
control. `execute` is a dry-run gate decision; no motor commands are issued.

## First measured risk model (2026-10-03)

240 episodes: 144 train, 48 validation, 48 test. Validation selected a threshold
of 0.72. The held-out test results are:

| Metric | Result |
|---|---:|
| Accuracy | 89.58% |
| Precision | 83.33% |
| Failure recall | 88.24% (15 of 17 failures caught) |
| Normal pass rate | 90.32% (28 of 31 stable towers passed) |
| Missed failures / false alarms | 2 / 3 |
| Majority-class accuracy baseline | 64.58% |

The model meets the stated simulation development bar on this small test set.
The test uses the same camera and disturbance family as training, with different
seeds; it does not establish robustness to new cameras, extraction actions,
human hands or physical hardware. No test metrics were used to change the
threshold. Risk-head scores are not calibrated physical failure probabilities.

The AccelSight student's held-out mean-velocity RMSE is 0.0299 m/s, compared
with 0.1052 m/s for the constant training-mean baseline. This measures pooled
top-layer velocity, not per-block tracking or gripper control accuracy.

CUDA/JIT warm-up runs on the actual inference worker threads before any timed
observation. Warm-up does not log an execute decision or relax the 500 ms
deadline. The first un-warmed threaded run correctly latched a timeout; the
worker warm-up addresses that startup behavior.

The exported models reproduced the same confusion matrix through the parallel
runtime on all 48 test episodes: 110-141 ms per observation, 125 ms median,
zero runtime faults, and zero motor commands. See
`checkpoints/mujoco_vjepa/runtime_evaluation.json`. The full regression suite
passed 108 tests. `logs/mujoco_demo.mp4` contains a verified 12-second, 20 FPS
visual replay of 12 test episodes with the gate decisions overlaid.

## Files

- `data/mujoco_jenga/manifest.json`: episode seeds, split, physics outcomes.
- `checkpoints/pretrained/vjepa2_1_vitb_dist_vitG_384.pt`: official base checkpoint.
- `checkpoints/mujoco_vjepa/encoder.ts`: frozen encoder, fixed 4-frame/384px input.
- `checkpoints/mujoco_vjepa/risk_head.ts`: trained risk head.
- `checkpoints/mujoco_vjepa/motion.pt`: trained AccelSight student.
- `checkpoints/mujoco_vjepa/evaluation.json`: validation threshold and test metrics.
- `checkpoints/mujoco_vjepa/motion_evaluation.json`: velocity errors versus a
  constant training-mean baseline.
- `checkpoints/mujoco_vjepa/provenance.json`: source commit and checkpoint SHA256.
- `logs/mujoco_parallel.jsonl`: gate events.

TorchScript traces the frozen encoder at a fixed shape; the adapter enforces
the same input dimensions. This 4-frame recipe is an adaptation of the released
model's longer video training setup. It is not the published benchmark protocol.
TorchScript is deprecated in the installed PyTorch version but functional here.

## Scope before hardware

This benchmark learns **post-disturbance early warning**. It does not yet learn
which block to extract, a gripper trajectory, or human-hand avoidance. Forces
are applied directly to blocks, not through gripper contact. Train extraction
and action-conditioned risk on actual arm-contact trajectories next, with
separate test scenarios for removals, leaning towers and human interference.
ArUco can calibrate real camera views later; it does not replace physics.

## Dependencies and upstream assets

The verified environment uses PyTorch 2.11.0+cu128 and torchvision 0.26.0+cu128.
Install matching CUDA wheels first, then `requirements-mujoco.txt`.
The host GPU reports 4 GB VRAM, despite an older 8 GB entry in config.yaml.

SO-101 assets under `data/assets/so101` are copied from
`TheRobotStudio/SO-ARM100/Simulation/SO101` (XML and the entire assets directory).
V-JEPA source under `data/assets/vjepa2` is pinned by the generated provenance.
The extractor imports that source in an isolated process to avoid its `src`
package colliding with this project's own `src` package. It loads the checkpoint
directly, bypassing an upstream Hub download URL currently pointing to localhost.

Sources: [Meta checkpoint listing](https://github.com/facebookresearch/vjepa2#models),
[SO-101 assets](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101),
[MuJoCo Python API](https://mujoco.readthedocs.io/en/stable/python.html).
