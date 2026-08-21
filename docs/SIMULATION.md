# The simulation backend

## What it is

A deterministic Jenga-tower model plus a renderer for the two camera views,
added so that Steps 1-9 of `PLAN.md` can be executed, tested and
regression-checked before the physical rig exists.

It substitutes for hardware at exactly one seam: frame capture. Everything
downstream of capture — calibration, tower-state extraction, recording,
training, the gate, the metrics — is the same code on both backends. Swapping
to hardware means replacing `capture_sim()` in `scripts/01_capture_reference.py`
with `cv2.VideoCapture` reads and setting `runtime.backend: hardware`.

## What it is not

**It is not evidence for the thesis.** A gate that reduces catastrophic
failures against a tower model I wrote has demonstrated that the pipeline is
wired correctly and nothing else. The physics here is a static stability
margin with a hand-tuned collapse probability, not a rigid-body contact
simulation, and the anomaly detector's job is made easier by a renderer whose
failure modes are the ones I happened to think of.

Every artifact records the backend that produced it, and the Step 9
comparison table carries a provenance banner. Simulated numbers must never be
reported as experimental results.

## What the model actually contains

`src/sim/tower.py`

- Occupancy grid, lean angle, per-block extraction friction.
- Stability is `support_margin - structural_penalty`: centre of mass against
  the supporting footprint, penalised for hollowed-out and single-block
  layers, which a pure centre-of-mass check treats as fine.
- Extraction outcomes are drawn from a per-(state, action) RNG stream keyed
  by the occupancy grid and step count. This makes `preview_extract` return
  exactly what `extract` would, so the counterfactual "would this blocked
  action actually have toppled the tower?" has a real answer. That is what
  makes an honest gate false-positive rate computable in Step 8, and it is
  the main thing the simulation buys that hardware cannot easily provide.

`src/sim/render.py`

- Real pinhole projection (verified against `cv2.projectPoints`), so the
  ArUco pose recovery in Step 2 is genuine rather than assumed. A renderer
  that drew a convenient top-down layout would let the occupancy detector
  cheat in a way that would only surface when the real rig arrived.
- Flat-shaded boxes with visible seams, four ArUco markers on the table, a
  dark recess for each removed block, per-frame lighting jitter and sensor
  noise.
- The side camera sits at a corner rather than square-on. A layer's three
  slots are only countable on the face perpendicular to their offset, and
  those faces alternate with layer parity, so a square-on view can resolve
  only half the layers. This is a rig design constraint, not a rendering
  convenience: the physical side camera needs the same placement.

## Known ways it is easier than reality

Worth stating plainly, because these are where a sim-validated pipeline will
first disappoint on hardware:

- No motion blur, rolling shutter, autofocus hunting or exposure drift.
- Blocks are perfectly rectangular, identically coloured and never rotate
  slightly in place as they are disturbed.
- Lighting is uniform; there are no cast shadows between blocks and no
  specular highlights.
- The tower's base never shifts on the table, and the camera never moves.
- Collapse is rendered as scattered blocks, not simulated dynamically.

The Step 2 exit condition (20/20 on known configurations) is therefore
necessary but not sufficient. It has to be re-run against real imagery, and
`segment_wood_mask`'s saturation thresholds and the occupancy brightness
model are the two places most likely to need refitting.
