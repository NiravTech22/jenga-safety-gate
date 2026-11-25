# sage_so101 — make the sim follow your real SO-101

Built from NVIDIA's SAGE page (paired real/sim data → per-joint gap → fix). That page stops at *measuring*
the gap and defers the fix to GapONet / "parameter tuning" with no fitting method given, so the fitter here is added.

## Loop
1. `python -m sage_so101 gen --scale 0.3`           → motion files (one per joint + random_waypoints), SAGE format
2. `python -m sage_so101 record --motion output/motions/excite_elbow_flex.txt --out output/real/excite_elbow_flex.csv --port /dev/ttyACM0 --scale 0.3`
   (repeat per joint, then `random_waypoints`; keep names `excite_<joint>.csv`)
3. `python -m sage_so101 fit`                        → output/fitted_params.json (kp, kd, armature, damping, friction, backlash, delay)
4. `python -m sage_so101 analyze --real output/real/random_waypoints.csv --params output/fitted_params.json`  (held-out check + plots)
5. `python -m sage_so101 export --params output/fitted_params.json`  → Isaac Lab ImplicitActuatorCfg snippet

## Before real use
- `configs/so101.json`: point `mjcf` at your twin MJCF (assets/so101_standin.xml is a stand-in), set per-joint
  `sign`/`offset_rad` (LeRobot calibration zero → sim zero; the #1 sim-to-real bug), `center`/`amp`/`lo`/`hi` for a SAFE workspace.
- Start at `--scale 0.3`. `record_real.py` is untested on hardware (LeRobot API drifts); it ramps in/out and clamps to limits.
- Gripper excluded (being rebuilt). Add it as a joint later; fit it separately.
- Validation done without hardware: `python scripts/demo_synthetic.py` (hidden arm → recover params → held-out RMSE 0.6-0.9° → 0.05-0.15°).
- Caveat: kp / kd / armature trade off against each other (fitted values drift 10-40% from truth while behaviour matches);
  trust the *behaviour*, don't read them as physical constants. tau_max is fixed (set to your supply voltage's STS3215 limit).

## In this repo
Lives at `twin/sage_so101/`. `configs/so101.json` points at `twin/models/so101_custom_claw_physics.xml`; its `baseline_params`
are the twin's shipped actuator values (kp≈998, kd 2.73, ±3.35 Nm), so "sim default" in the plots = what the twin does today.
The loader switches the twin's built-in `<position>` actuators off (the fitted servo model drives joints instead) and, if
mesh files are missing, drops them (dynamics come from `<inertial>`). `configs/standin.json` is the hardware-free test arm.
Run everything from `twin/sage_so101/` (see commands in the PR/commit message).
