# SO-101 digital twin (`twin/`)

A MuJoCo model of the SO-101 follower with the custom rack-and-pinion claw, a
two-platform workcell and a 75 x 25 x 25 mm block, plus a scripted IK
controller that picks the block up and places it on the second platform.

```bash
python scripts/tools/fetch_so101_assets.py      # once: upstream meshes into twin/models/meshes/
python twin/so101_pickplace.py                  # window opens and plays the attempt; P replays
python twin/so101_pickplace.py --headless       # PASS/FAIL only
python twin/so101_pickplace.py --record run.mp4 # offscreen video (needs imageio[ffmpeg])
python twin/so101_pickplace.py --manual         # keyboard teleop, no auto-play
```

Viewer keys: W/S, A/D, R/F move the gripper along world X, Y, Z; Q/E wrist roll;
T/G wrist flex; O/C open/close; P scripted pick-and-place; H reset.

Result log: `logs/twin/pickplace_run.json`. `tests/test_twin.py` runs the full
pick-and-place headless on every test run (skips if meshes are not fetched).

## Files

| File | Role |
|---|---|
| `models/so101_custom_claw_physics.xml` | Arm + claw, actuators, finger equality constraint |
| `models/scene_custom_claw_physics.xml` | Scene wrapper (floor, lights, cameras). Edit this one |
| `models/scene_pickplace.generated.xml` | Rebuilt by `so101_pickplace.py` each run; gitignored |
| `models/meshes/custom_claw_*.stl` | Custom claw meshes (committed) |
| `models/meshes/*` (other) | Upstream SO-ARM100 meshes (fetched, gitignored) |

## Frame

The MuJoCo world frame is the robot base frame: +X is where the arm points at
`shoulder_pan = 0`, +Y is to the robot's left, +Z up, z = 0 the surface the base
sits on. The shoulder-pan axis is at (0.0388, 0) in this frame. The calibration
tool (`twin/jenga_vision.py`) reports positions in the same frame.

## Current workcell

Platform tops at z = 0.040 m, pick at (0.263, 0.000), place 60 mm closer to the
base. The claw approaches tilted 35 degrees from vertical: with the pads about
170 mm past the wrist-flex joint, a straight-down approach is out of reach at
hover height. Measured jaw gap: 84 mm open, 24 mm closed.

## Fixes that made pick-and-place work (2026-11-14)

1. **Grip pads were behind the claw.** The pad collision boxes sat beside the
   wrist motor, about 50 mm behind the claw base, so IK drove the wrist into the
   block. They were moved to the measured fingertips.
2. **The block fell through the platform.** Platforms were moved with
   `model.geom_pos` after compiling; MuJoCo's precomputed collision data for
   static geoms stayed at the old height. Platform height is now written into
   the XML before compiling.
3. **Grip too weak.** The close target squeezed 1 mm under the block width and
   the block slid out in transit; it now targets 16 mm under (12 mm was the
   smallest that passed).
4. **Viewer idled.** The window waited for a key press; it now plays the
   attempt on open.

Robustness check: the block's start pose was offset by up to 4 mm and rotated
up to 5 degrees in eight variations; all eight passed.

## Phase 2: gated Jenga extraction (`twin/jenga_push.py`)

The twin pushes Jenga blocks with one fingertip, and a MuJoCo world model
decides whether the extraction is safe before it is finished.

```bash
python twin/jenga_push.py                        # viewer: one gated game on tower 101
python twin/jenga_push.py --ungated --seed 105   # same, no gate
python twin/jenga_push.py --record push.mp4      # offscreen video
python twin/jenga_push.py --compare --episodes 12   # gated vs ungated table (~1 h on 2 cores)
```

### Two worlds

| | TRUTH ("real" tower) | MODEL (the gate) |
|---|---|---|
| Arm | Custom-claw SO-101, joint-space motion, servo limits | None: a kinematic fingertip |
| Blocks | Hidden per-block thickness (±0.3 mm) and friction (0.3-0.7) | Nominal sizes; thickness and friction **sampled** from the same prior |
| Block poses | Exact | What the camera would report: truth + 0.5 mm noise |

### One turn

1. **Test push with a reflex.** The arm puts one fingertip on the block's near
   end and pushes up to 12 mm, stopping early if the camera sees any other
   block move 3 mm (what a person does when a block feels tight). It records
   how deep the push got and how much the tower moved.
2. **Gate.** The world model replays the same reflex push on 16 sampled
   towers, weights each by how well it reproduces the observed depth and tower
   movement, and runs the pull on the ones that explain it. P(fall) is the weighted fraction that
   topple. The gate **blocks** if P(fall) > 0.25, or if fewer than 2.5 samples
   (effective sample size) explain the observation: the model does not
   understand this tower, so it does not get a vote.
3. **Pull.** If allowed, the block is pulled out from the far side.

Thresholds were fixed on calibration towers 1-2 before any evaluation tower
was run (with the reflex: 9/9 toppling pulls blocked, 5/11 safe pulls allowed). Every
blocked pull also gets a counterfactual: the same pull on a copy of the truth
state, so wrong blocks and correct blocks are both counted.

### Why the push looks like this

- **Only the fingertip goes in.** The claw's fingers are hooks: about 10 mm
  tall at the tip, then rising into the layer above about 15 mm back. So the
  push is a 12 mm test push, and fingertip collision boxes matching the real
  tip replace the taller pick-and-place pads in this scene.
- **Only even layers, two slots.** A 5-DOF arm can hold the claw horizontal
  only along a line through the shoulder-pan axis. One fingertip sits on the
  block's end while the other stays outside the tower, which fixes the tower at
  y = -45 mm and makes the middle and inner slots of even layers reachable
  (layers 4-12 of 15: below layer 4 the claw would hit the table).
- **The pull is idealized.** The arm cannot reach behind the tower, so the
  block is pulled out by a capped 3 N force on the block itself (a pinned block
  still drags its neighbours, as a real pull would). On hardware this needs a
  turntable or a second approach; it is the open item for Phase 3.

### Results (simulation, 12 evaluation towers, seeds 201-212)

| Metric | Ungated | Gated |
|---|---|---|
| Towers that fell | 12 / 12 | 4 / 12 |
| ... of those, during a test push | 0 | 3 |
| Games ended safely (no block judged safe / 10 attempts used) | n/a | 8 |
| Blocks removed (total) | 10 | 16 |

Per decision (79): of 46 pulls that would have toppled the tower, the gate
blocked 45; of 33 safe pulls, it wrongly blocked 17. It is cautious by design:
a block it cannot explain counts as unsafe. Raw results:
`docs/results/jenga_compare_v2_reflex.json` (regenerate with `--compare --seed0 201`).

**A first version failed, and the fix was not tuned on these towers.** Without
the reflex (towers 101-112, `docs/results/jenga_compare_v1_no_reflex.json`) the
gate blocked 32 of 34 toppling pulls, yet 11/12 gated towers still fell:
every full 12 mm test push on a tight block shifted the upper tower about
10 mm, and those shifts added up until a test push brought it down. The reflex
was added, re-checked only on calibration towers 1-2, and then evaluated on the
fresh towers 201-212 above.

The simulated tower is harsh: about half of all first pulls topple it, which is
why every ungated game ends in a fall. Remaining failures are mostly test
pushes, so the next safety step is a gate on the test push itself.

### What it does not show

Both worlds are simulations from the same physics engine, so the gate's model
is better matched to its "reality" than it will be to wood. The numbers below
show that the observation-conditioned gate works as designed in simulation,
not that it will on the physical tower.

## Live twin of the real arm (`twin/arm_twin.py`)

`mirror` makes the MuJoCo arm follow the real one, from the telemetry file that
`scripts/tools/gamepad_teleop.py` publishes (it opens no serial port, so it runs
next to the teleop). It prints every joint angle and the claw tip position in the
arm base frame, the same frame as the overhead camera calibration. `sim` lets the
gamepad drive the simulated arm alone, with the same controls, limits and
telemetry.

```bash
python scripts/tools/gamepad_teleop.py --port /dev/ttyACM0 --calibration-version jenga_follower   # terminal 1
python twin/arm_twin.py mirror                                                                    # terminal 2
python twin/arm_twin.py sim          # no hardware: gamepad drives the model
```

Ticks map to model angles as `q = sign * (ticks - zero_tick) * 2*pi/4096`
(`data/calibration/joint_map.json`). Defaults: zero_tick 2048 (LeRobot's middle
of range), sign +1. Line the model up with the real arm once:

1. `ref` shows the model's zero pose. Drive the real arm into the same pose, `zero`, `ref` again.
2. Move one joint at a time. If the model moves the opposite way, `flip <joint>`.
3. Fine-tune with `nudge <joint> <deg>`; claw: `claw open` / `claw closed` at its two ends.
4. `save`. Re-do after any `lerobot-calibrate`.

Note: in the model, positive `shoulder_pan` turns clockwise seen from above.
