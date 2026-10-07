# Parallel SO-101 integration

**Update:** the [MuJoCo training recipe](docs/MUJOCO_TRAINING.md) now uses an
official pretrained distilled V-JEPA 2.1 checkpoint, simulated failure labels,
and a trained AccelSight descriptor/velocity student. The original generic
cached-target distillation interface below remains available separately.

This checkout combines jenga-safety-gate at the root with
motion-aware-perception-model under vendor/motion_perception. Original clones
are unchanged. The perception remote retains the imported Git history.

## Run on this Windows host

    .\.venv\Scripts\python.exe -m src.parallel.run --smoke --steps 6
    .\.venv\Scripts\python.exe -m pytest tests -q

The smoke run uses the real AccelSight architecture on CUDA with RANDOM weights
and a synthetic alternating anomaly score. It only tests wiring and execution.
Every output explicitly reports smoke=true and motor_commands_sent=0.

Install on another machine:

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
    .\.venv\Scripts\python.exe -m pip install -r requirements-parallel.txt

PyTorch provides CUDA kernels for preprocessing and neural inference. Building
the optional custom CUDA extension is not required. Raw model tensors are
returned; the upstream postprocessor's placeholder depth is not used as a
metric position estimate.

## Real local inference

    .\.venv\Scripts\python.exe -m src.parallel.run --perception-checkpoint checkpoints/motion.pt --student checkpoints/student.ts --student-metadata checkpoints/student.json --video 0 --threshold YOUR_VALIDATED_THRESHOLD

Use a validated numeric threshold. The runner is dry-run only, including camera
mode: it never connects to an arm. Frames from OpenCV are converted BGR to RGB.

Perception checkpoint: torch.save a dictionary with state_dict, model_kwargs
(AccelSightNet constructor parameters), and preprocessing. Only weights-only
loading is used.

Student: trusted local TorchScript module whose forward returns one scalar,
higher means more anomalous. Export the trained distilled model INCLUDING its
anomaly scoring head. A base V-JEPA embedding model is not this artifact.

Example student JSON (replace with training-time preprocessing):

    {"frames": 5, "size": [128, 128], "mean": [0, 0, 0], "std": [1, 1, 1], "layout": "BTCHW"}

The perception preprocessing uses the same fields except layout; its input is
BTCHW. Student layouts supported: BTCHW, BCTHW, BCHW (latest frame).
Both adapters require the same clip length. Resize is bilinear, RGB values are
scaled to [0,1], then normalized with mean/std. Match training exactly.

## Parallel execution and gate semantics

Each observation fans out to two Python worker threads: CUDA motion inference
and anomaly scoring + the existing SafetyGate. Both must finish for the SAME
observation before an execute decision is returned. One request is in flight;
there is no unbounded frame backlog. GPU kernel overlap is not guaranteed on
a shared GPU; measure end-to-end latency with both models loaded.

Nonfinite scores, failed inference, old/future timestamps, repeated sequence
numbers, and deadlines block. A timeout latches a fault until restart.
Log records include the branch gate result followed by a final block when the
other branch fails; only ParallelPipeline's final Result is authoritative.
The default 500 ms deadline is an engineering starting point, not a validated
physical safety bound. Camera timestamps mark host receipt, not sensor exposure.

Python cannot kill a stuck inference thread, and executor threads can delay
process exit. Production arm control needs its own watchdog and verified hold/
stop behavior, independent of inference. An execute result is a per-observation
decision, not a reusable motor authorization.

## Remaining hardware/model work

Supply the trained motion checkpoint and distilled student artifact plus its
preprocessing and validated threshold. No trained model was found in the local
project folders. The student trainer below is implemented; extracting real
V-JEPA teacher targets and training/validating on recorded episodes remain pending.

The physical SO-101 adapter and ACT policy remain unimplemented upstream. AccelSight's
three control outputs are NOT a six-joint SO-101 command. Integrate an actual
LeRobot policy/teleop action source, calibrated joint units/limits, and a motor
watchdog before enabling actuation. Existing study/data-validation requirements
in docs/PLAN.md still apply; smoke outputs are not experimental results.

## Local student training

The compact 3D CNN training/export path is now implemented:

    .\.venv\Scripts\python.exe -m src.parallel.distill --cache data/teacher_cache.npz --provenance data/teacher.json --output checkpoints/student --epochs 20 --batch-size 4

The NPZ cache contains RGB uint8 `clips` [N,T,H,W,3], nonnegative float `scores`
[N], string `episode_ids` [N], and string `split` [N] containing `train` or
`validation`. Splits must have disjoint episode IDs; adjacent windows from one
episode must never cross splits. Use exactly the same frame sampling/camera
configuration for training and inference. A single camera clip is the current
contract; synchronized multi-camera fusion is not implemented.

`teacher.json` must identify `teacher_checkpoint`, `score_definition`, and
`dataset_id`. Generate scores offline from a pretrained V-JEPA teacher's
masked prediction error with a fixed, recorded mask protocol. Do not substitute
embedding magnitude or claim synthetic labels are teacher predictions. Teacher
loading, masking and cache extraction are still external prerequisites.

Training runs locally on CUDA, exports `student.ts` and `student.json` directly
compatible with LocalStudent, and records validation regression MSE and
provenance in `training.json`. This is score distillation, not a calibrated
failure probability. Evaluate separate held-out risk labels and choose the
gate threshold before treatment runs. Training outputs explicitly record
`safety_validated=false`. Keep a final test split outside this training cache.
The RTX 3050's 4 GB VRAM was sufficient for the integration smoke run; teacher
memory requirements have not been measured. Cache teacher outputs separately
so the teacher and both live models need not share GPU memory.

## Proposed actions and human turns

`src.parallel.actions.ActionChecks` binds a six-joint proposal to the inference
observation, checks freshness, joint limits and per-step displacement, rejects
replays, and requires explicit robot-turn, workspace-clear and stop-state inputs.
It defaults to blocked. This is a dry-run API and does not send commands.
Joint values use radians in `JOINT_NAMES` order; do not feed LeRobot's normalized
joint values into this API without an explicit calibrated conversion.

Workspace clearance must come from a validated external monitor. No human-hand
detector or action-conditioned collision predictor is implemented here. The
current anomaly scorer observes video only; it cannot evaluate the consequence
of a particular proposed extraction. Continuous monitoring during movement,
verified stopping behavior, contact/force feedback, a trained action policy and
hardware validation remain necessary for human-versus-robot Jenga.

## Source revisions and setup

GitHub remotes checked on 2026-10-03:

- jenga-safety-gate main: `fce828c7050d120fbfd11fb23ea1fadcd21130e5`
- motion-aware-perception-model main: `95e5489860e360187e7c1cbcc82c7f3dd819a223`

The imported source is under `vendor/motion_perception`; its license is retained.
The local adapter removes an unused PostProcessor import to avoid requiring its
optional tracker dependencies. Python PyTorch CUDA operations back this runtime;
the separate upstream CUDA_integration branch is not automatically imported.

The kinematics tests require the upstream SO-101 URDF at
`data/assets/so101/so101_new_calib.urdf`. Copy it from
`TheRobotStudio/SO-ARM100/Simulation/SO101/so101_new_calib.urdf` before testing.
The full simulation renderer additionally requires its referenced mesh assets.

Reference implementations: https://github.com/facebookresearch/vjepa2 and
https://huggingface.co/docs/lerobot/so101 .

## Verified on this host (2026-10-03)

- Repository tests: 104 passed. TorchScript emits deprecation warnings in the
  installed PyTorch version; the existing runtime format is still functional.
- CUDA perception plus synthetic gate: six observations, alternating pass/block,
  15-83 ms observed latency after warmup, zero motor commands.
- One epoch of CUDA student training, export, and CUDA reload passed with
  explicitly synthetic test targets. Artifacts are under ignored
  `logs/distillation-smoke/`, not production checkpoints.

These checks demonstrate software integration and training mechanics only.
They do not measure fault-prediction accuracy or establish safe human interaction.
