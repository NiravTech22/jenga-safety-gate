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
