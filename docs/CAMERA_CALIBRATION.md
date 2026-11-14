# Camera calibration and tower pose

`twin/jenga_vision.py` calibrates a real camera, finds where it sits relative to
the robot, reports the tower's position and yaw in the robot frame, and writes a
matching camera into the MuJoCo scene. Outputs go to `data/calibration/`
(gitignored).

Two different needs, two different requirements:

- **ACT / learned policies** need no geometric calibration, only consistency:
  cameras bolted down, auto-exposure and autofocus off (`v4l2-ctl`), the same
  resolution, frame rate and camera names when recording and when running.
- **The safety gate, tower-state CV and the twin** need metric calibration:
  intrinsics (focal length, distortion) and extrinsics (camera pose in the
  robot frame).

## Steps

Print `make-targets` output at 100% / actual size. Mount the board on something
rigid and flat. Measure a square; pass the measurement as `--square-mm`. Cut out
the tower marker (id 40) and glue it flat on a plate under the tower.

```bash
python twin/jenga_vision.py selftest           # 4x PASS confirms your OpenCV build
python twin/jenga_vision.py make-targets       # data/calibration/targets/jenga_targets.pdf
python twin/jenga_vision.py capture --camera 0 # 25-40 frames: near, far, tilted, all image corners
python twin/jenga_vision.py intrinsics --square-mm 30.0          # aim for RMS < 0.5 px
python twin/jenga_vision.py extrinsics --camera 0 --board-origin 0.12 0.14 0
python twin/jenga_vision.py tower --camera 0 --marker-z 0.006 --show
python twin/jenga_vision.py mujoco-camera      # adds front_cam to twin/models/scene_custom_claw_physics.xml
```

**Board placement for `extrinsics`:** flat, face up, the printed "+X" arrow
along robot +X; the board then extends toward robot -Y. `--board-origin` is the
ORIGIN corner in metres, measured from the table point under the centre of the
shoulder-pan axis (the script adds the pan-axis offset). `--board-yaw` covers a
board that is not square to the robot.

**`--marker-z`** is the height of the marker surface above the robot's base
surface (for example the plate thickness).

## Sanity checks

- `tower` prints the marker's measured edge length; it should read close to
  50 mm. If not, check `--marker-z`, the printed marker size, or redo extrinsics.
- MuJoCo renders without lens distortion. Compare sim views with real frames
  passed through `cv2.undistort()`.

## Accuracy (synthetic test, 640x480 with distortion and noise)

| Step | Result |
|---|---|
| Intrinsics | 0.25 px RMS, focal length within 0.05% |
| Extrinsics | about 1-2 mm, 0.35 degrees |
| Tower pose | within 0.6 mm and 0.4 degrees |
| MuJoCo camera vs OpenCV | within 0.13 px |

Tower position comes from intersecting the camera rays through the marker
corners with the known table plane, not from the marker's own depth estimate,
which is the weakest number for a small marker (that change took the error from
5 mm to 0.6 mm). Real-world accuracy depends on print scale and tape-measure care.

## Relationship to `src/perception/tower_state.py`

That module works in the tower frame (4x4_50 markers ids 0-3 around the tower
base) for Step 2. This tool works in the robot frame with a 5x5_100 board and
marker id 40, so the dictionaries never collide. On hardware, the intrinsics
from `data/calibration/front_intrinsics.json` (`K`, `dist`) are the
`camera_matrix` / `dist_coeffs` that `ArucoFrameCalibrator` expects.
