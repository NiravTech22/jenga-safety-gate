# Milestone 1: passive overhead and moving wrist recording

Run commands from the repository root with Python containing NumPy and OpenCV.
The new entry point writes version 2 episodes; the old `EpisodeRecorder` and
`Episode` API remain version 1 for existing consumers. `load_recording` and the
replay command accept both. A historical `side` image stays `side`, never `wrist`.
No data migration, checkpoint replacement, policy inference or motor commands
are performed by the recorder or replay.

## First real episode

Supply actual overhead/wrist device indices, a session ID, tower ID, operator,
calibration version and explicit joint units. CLI options are authoritative;
the recorder does not silently use `runtime.backend: sim` or camera placeholders.
The cameras are installed; the wrist camera moves with the arm. Wrist extrinsics
require hand-eye calibration plus the measured arm configuration. Existing fixed
side-camera calibration and MuJoCo side-view risk training do not transfer to it.

The external VR server implementation was not found in this repository or the
searched local Documents/Desktop/robot_arm_sim sources. `VRTeleopArm.read_joints`
and `command_joints` are unimplemented. The precise missing integration is a hook
in the existing teleop loop that provides **encoder feedback** and the **joint
command actually sent**, their host monotonic timestamps, and a sequence number.
The passive bridge is implemented in `src/hardware/passive.py`; no network API
or servo bus access has been guessed. Add this hook on the same Windows host:

```python
from src.hardware.passive import publish_telemetry
from src.hardware.kinematics import JOINT_NAMES

# After the EXISTING teleop read and send operations; do not send again here.
publish_telemetry(
    "logs/teleop_latest.json",
    joint_state=measured_encoder_positions,
    action=joint_command_actually_sent,
    state_timestamp=encoder_read_receipt_monotonic,
    action_timestamp=command_send_monotonic,
    sequence_id=teleop_tick,
    joint_names=JOINT_NAMES,
    joint_units="radians",  # use the actual units, not an assumed conversion
    calibration_version="so101-cal-001",
)
```

Create `logs` before publishing. Both timestamps must use `time.monotonic()`
on this host. Remote-clock timestamps need an independently implemented clock
mapping; do not pass them directly. Joint ordering is shoulder_pan, shoulder_lift,
elbow_flex, wrist_flex, wrist_roll, gripper. Copy actual values without substituting
IK targets for feedback. Record per-joint units in the units string if mixed.

With the hook running, replace the example indices and identifiers:

```powershell
python scripts/03_record_episode.py --hardware --overhead 0 --wrist 1 --telemetry logs/teleop_latest.json --session-id session-001 --tower-id tower-001 --calibration-version so101-cal-001 --joint-units radians --operator Nirav --seconds 60 --preview
python -m src.data.replay data/episodes/PRINTED_EPISODE_ID/episode.json
```

Press `r` for removal, `c` for collapse onset, `q` for normal finish; Ctrl-C saves
an interrupted run. These are operator receipt-time annotations, not inferred
physical onset times. The manifest is atomically checkpointed after each tick;
an unclean termination leaves status `interrupted`. Frames written just before
a crash can be orphaned but never replace prior episodes. A failed disk itself
cannot guarantee persistence. Replay shows both views, measured state, sent
actions, camera sequence/age, units, timestamps and the latest event markers;
space pauses, q quits. The manifest retains all events and their details.

## Timing, integrity and acceptance

Each camera drains on its own thread. Its timestamp is host receipt immediately
after OpenCV `read`, **not sensor exposure**. The recorder samples each latest
frame at a host tick. This is bounded-skew software pairing, not hardware trigger
synchronization; driver buffering and exposure latency require physical testing.
Missing or stale images are retained as explicit anomalies, never black-image
substitutes. Default age limit is 200 ms; camera skew limit is 50 ms. Override
with `--max-age`/`--max-skew` after measurement, not to hide failures.

`integrity.json` reports missing/unreadable images, stale/future timestamps,
camera skew, repeated/reversed sequence IDs, invalid state/actions and count
mismatches. Sequence gaps count camera frames not represented in recorded ticks,
including intentional downsampling; failed reads are a separate cumulative
counter. Sensor-internal drops cannot be measured with this OpenCV interface.
Legacy timing/drop information is explicitly unknown.

Acceptance for the first connected run: correct distinct overhead/moving-wrist
views; changing encoder feedback and sent actions in the declared units/order;
both ages <=200 ms and skew <=50 ms; no missing/repeated frames or telemetry;
zero failed reads; all sequence gaps explained by acquisition/recording rates;
`integrity.json` has `ok: true`; r/c markers appear at the observed event times;
replay frame count matches recording; Ctrl-C preserves a readable interrupted
episode; recording and replay issue zero motor commands. An integrity failure
does not delete evidence. Replay exits 2 for an invalid episode.

Pending with arm disconnected: implement/verify external teleop hook, identify
devices, verify calibrated joint units and actual command semantics, wrist
hand-eye calibration, motion-dependent visibility, sustained acquisition rate,
exposure/buffering skew, camera/teleop disconnect recovery, physical event timing,
and manual confirmation that starting/stopping recording does not move the arm.
No physical safety or autonomous-policy claim follows from these software checks.

## Synthetic validation

```powershell
python scripts/03_record_episode.py --dry-run --root logs/milestone1-smoke --session-id synthetic-m1 --tower-id synthetic-tower --calibration-version synthetic-v1 --joint-units radians --seconds 1 --fps 15
python -m src.data.replay logs/milestone1-smoke/PRINTED_EPISODE_ID/episode.json --headless --video logs/milestone1-replay.mp4
python -m pytest tests/test_recording_v2.py tests/test_schema.py tests/test_gate.py tests/test_invariant_schema.py -q
```

Synthetic frames are visibly labeled; synthetic state/actions and removal/collapse
markers only exercise storage/replay. Video export uses constant `--fps` (default
15); interactive replay follows recorded tick spacing, capped at 1 s per gap.
