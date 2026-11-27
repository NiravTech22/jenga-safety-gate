# Real data: overhead + wrist camera episodes

The sim dataset (`docs/TOWER_DATASET.md`) gives physics under known conditions.
Real episodes give what the sim cannot: what a real collapse looks like on camera,
which blocks are actually tight, and the servo current while pulling them.
Target: **~100 episodes, about 60 normal and 40 failures** (PROGRESS / Notion Phase 10).

Each episode = one extraction attempt, recorded as:

| Signal | Source | Used for |
|---|---|---|
| Overhead frames | fixed camera, calibrated to the robot frame | tower state (lean, block poses) by classical CV; V-JEPA input |
| Wrist frames | camera on the claw | close view of the block being pulled; V-JEPA input |
| Joint positions + goal positions | `so101_telemetry.py` (read-only) | where the arm was, what it was told |
| Servo load (`load_pct`) | same | pull resistance, **only when the arm does the pulling** (see step 5) |
| `r` / `c` key presses | operator | removal time, collapse-onset time |

Nothing here trains a model yet. Training starts once there are enough labelled
episodes (step 6).

## 1. Fix the cameras (once)

```bash
v4l2-ctl --list-devices                   # find /dev/videoN for each camera
v4l2-ctl -d /dev/video0 -l                # list controls; names vary by camera
# lock exposure, white balance and focus (use the names your camera lists):
v4l2-ctl -d /dev/video0 -c auto_exposure=1 -c exposure_time_absolute=200 \
         -c white_balance_automatic=0 -c focus_automatic_continuous=0
```

Do the same for the wrist camera. For Nirav's two cameras this is scripted:
`./scripts/tools/lock_cameras.sh` (run it before every session; settings reset on replug). Cameras are found by their stable `/dev/v4l/by-id/` names (overhead = Jieli USB PHY 2.0, wrist = Generic USB Camera), so `/dev/videoN` swaps after a replug no longer matter; the recorder refuses to start if either picture is black. The overhead camera must not move after
calibration; if it is bumped, redo step 2. Use fixed lighting.

## 2. Calibrate

Calibrate at **1280×720**, the resolution the recorder uses (`CameraStreams`). Intrinsics
from a different resolution do not apply to the episode frames.

Overhead (intrinsics + where it is in the robot frame):

```bash
python twin/jenga_vision.py capture --camera 0 --width 1280 --height 720 --out data/calibration/overhead_images
python twin/jenga_vision.py intrinsics --images data/calibration/overhead_images \
       --square-mm <measured> --out data/calibration/overhead_intrinsics.json
python twin/jenga_vision.py extrinsics --camera 0 --intrinsics data/calibration/overhead_intrinsics.json \
       --board-origin X Y 0 --out data/calibration/overhead_extrinsics.json
python twin/jenga_vision.py tower --camera 0 --intrinsics data/calibration/overhead_intrinsics.json \
       --extrinsics data/calibration/overhead_extrinsics.json --marker-z 0.006 --show
```

Wrist (intrinsics only, for now):

```bash
python twin/jenga_vision.py capture --camera 2 --width 1280 --height 720 --out data/calibration/wrist_images
python twin/jenga_vision.py intrinsics --images data/calibration/wrist_images \
       --square-mm <measured> --out data/calibration/wrist_intrinsics.json
```

The wrist camera moves with the arm, so it has no fixed extrinsics. Its pose
needs hand-eye calibration (camera relative to the claw) plus forward kinematics
from the joint readings; that comes with the arm bring-up. Wrist frames are
still useful before then (V-JEPA input, block close-ups).

Give the calibration a version string, e.g. `cal-2026-11-21`, and use it below.
Change it whenever you recalibrate.

## 3. Start the arm telemetry (terminal 1)

```bash
python scripts/tools/so101_telemetry.py --port /dev/ttyACM0 --calibration-version cal-2026-11-21
```

Read-only: it never writes to the servos, so it does not move the arm or change
torque. With torque off you can move the arm by hand. Test without the arm:
`--mock`.

### 3b. Or drive the arm with a gamepad (instead of step 3)

With a gamepad the servos do the pulling, so `load_pct` (Present_Load, 0.1 % of max torque per unit, signed) becomes a real measure of
how hard the block resists. `gamepad_teleop.py` both drives the arm and publishes the
same telemetry file, so do **not** run `so101_telemetry.py` at the same time.

```bash
pip install pygame
python scripts/tools/gamepad_teleop.py --show-inputs          # check your pad's axis/button numbers
python scripts/tools/gamepad_teleop.py --port /dev/ttyACM0 --calibration-version jenga_follower --dry-run
python scripts/tools/gamepad_teleop.py --port /dev/ttyACM0 --calibration-version jenga_follower --speed 150
```

8BitDo in X-input mode (Xbox layout). Left stick: shoulder pan / lift. Right stick:
elbow / wrist roll. D-pad up/down: wrist flex. RB/LB: claw open/close. START arms,
B freezes (arm holds still), BACK goes limp (hold the arm first). Speed is capped and
every joint stays inside its calibrated range.

## 4. Record an episode (terminal 2)

```bash
python scripts/pipeline/03_record_episode.py --hardware \
  --telemetry logs/arm_latest.json --calibration-version cal-2026-11-21 \
  --joint-units sts3215_ticks --session-id s001 --tower-id t001 --operator Nirav \
  --scenario stable_easy --seconds 60 --preview
```

In the preview window: `r` when the block comes out, `c` the moment the tower
starts to fall, `q` to finish. Ctrl-C saves the run as interrupted.
Use a new `--tower-id` for every rebuild, and a new `--session-id` per sitting.

Check and replay:

```bash
python -m src.data.replay data/episodes/<EPISODE_ID>/episode.json
```

The recorder prints an integrity report (dropped, stale or out-of-sync frames).
Discard episodes it marks as not ok.

## 5. What to record

Until the scripted controller runs on the arm, extract blocks **by hand**, with
the arm's torque off and the claw moved by hand to face the block. The wrist
camera then sees what it will see later. One extraction per episode.

These are **visual + kinematics episodes only**. With torque off the servos supply
no force, so `load_pct` and `current_ma` in them does not measure block resistance and must not
be used as a tightness label. Tightness ground truth for now is your own note of
which blocks were tight or loose (write it in the episode, e.g. with the `note`
field, or a per-session text file). Force data comes once the arm pulls the blocks.

| `--scenario` | What to do | Count (min) |
|---|---|---|
| `stable_easy` | loose middle block, slow pull | 12 |
| `stable_tight` | tight block, careful pull, tower survives | 12 |
| `off_center` | outer block from a layer that still has two blocks | 12 |
| `leaning` | tower already leaning (several removals), pull a loose block | 12 |
| `near_collapse` | pull that should topple it: last outer support, tight load-bearing block | 40 |
| `recovery` | wobble but stands | 12 |

Before each episode, note which blocks were tight or loose (as you did for
layers 2-7). That is ground truth the sim cannot provide.

## 6. After ~50 episodes: labels and training

1. Tower state per frame from the overhead camera (lean, block positions), by
   classical CV against the calibration: the same quantities the sim produces.
2. V-JEPA 2 features per clip: `scripts/tools/extract_vjepa_features.py`.
3. Fine-tune the risk head (not the V-JEPA backbone) on sim + real features,
   then set the gate threshold on a held-out set of real episodes.

**Why load, not current.** Probe at 5.3 V supply (2026-11-27): holding pose, the elbow read
Present_Current 3 units (~20 mA, 6.5 mA/unit) while Present_Load read 84; squeezing the claw moved
current 0→4 but load 0→80. Current is too coarse for Jenga-sized forces, so `load_pct` is the
force signal; `current_ma` is still logged. Episodes recorded before this change have no `load_pct`.
