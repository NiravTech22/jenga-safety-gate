# Tower physics (`twin/tower_physics.py`)

A Jenga tower in MuJoCo that stands while it is in static equilibrium and
collapses completely once it is not, checked against closed-form predictions.

```bash
python twin/tower_physics.py --validate        # 12 tests, ~6 min; table + docs/results/tower_physics_validation.json
python twin/tower_physics.py --view neck       # watch a tower tip over a one-block neck
python twin/tower_physics.py --record t.mp4    # video: standing, removal collapse, tilt collapse
```

## The law being tested

Everything above a layer acts as one weight at its centre of mass. The tower
stands while the line of action of that weight passes through the layer's
remaining support; once it passes outside, there is an unbalanced moment and the
upper part rotates off. Two consequences, used as tests:

- **Tilting:** a tower tips when gravity's line through its centre of mass
  reaches the support edge. For a rigid tower, tan θ = (base half-width) /
  (height of the centre of mass).
- **Removing blocks:** the load above is centred, so a layer holds it while its
  remaining blocks span the centre line (middle block present, or both outer
  blocks present). One outer block alone puts the centre of mass 25 mm outside
  its support, so the tower falls.

## Results (all pass, measured parameters, 2026-11-20)

| Test | Theory | Simulation |
|---|---|---|
| Intact 16-layer tower, 10 s | no movement | 0.000 mm |
| Tilt, friction raised to 0.6 (tipping isolated) | tips when the equilibrium margin reaches 0 (rigid: 18.0°) | tips at 15.5°; margin -0.3 / -0.8 mm; 83% down |
| Tilt, measured friction μ = 0.23 | slides when tan θ = μ: 13.0° (before tipping at 18.0°) | upper tower slides off at 11.7°; 83% down |
| Tilt on a one-block neck (friction 0.6) | margin reaches 0 (rigid: 7.5°) | tips at 6.4°; margin 0.0 / -0.2 mm; 85% down |
| Remove middle / one outer / both outer blocks | stands | stands |
| Remove middle + one outer (two cases) | collapses | collapses, 72-78% down |
| Pull a load-bearing block (layer 8) | 0.92-1.80 N (dragged / pinned limits) | always out from 0.87 N |
| Pull a loose block | 0.042 N (μmg) | 0.042 N |
| Creep under 50% of the friction limit | 0 (real wood) | 0.26 mm/s (known limitation) |

**Two failure modes under tilt.** Tilted by θ, every block-on-block interface
carries tangential / normal force = tan θ, so the tower slides when tan θ = μ;
it tips when tan θ = (half-width) / (centre-of-mass height). With the measured
μ = 0.23 sliding comes first (13.0° vs 18.0°). The real tower agreed: in 3
tilt-board trials the upper part slid off twice (8°, 10°) and the whole tower
tipped once (18°). The real slides came earlier than tan⁻¹(0.23) because the
weakest of 15 interfaces goes first: with μ spread ±0.04, the lowest of 15
draws is about 0.16, i.e. 9°.

The sim slides a little early (11.7° vs 13.0°) because MuJoCo's soft friction
creeps near the limit.

## Parameters

All values are in one table (`TowerPhysics` at the top of the module), each with
its source. Replace `estimate:` with `measured:` as your numbers come in.

| Parameter | Value | Source |
|---|---|---|
| Block size | 74.72 × 24.83 × 14.54 mm | measured: calipers, 5 blocks (thickness 14.38-14.77, sd 0.19 mm) |
| Block mass | 18.5 g | measured: 5 blocks = 92.70 g |
| Layers | 16 | measured: the real set |
| Gap between blocks in a layer | 0.5 mm | estimate |
| Wood-wood friction μ | 0.23 (sd 0.04 per block) | measured: block-on-block tilt, 10 trials, 10-17° |
| Table friction | 0.50 | estimate (the base never slid in the tilt trials) |
| Contact time constant | 3 ms | tuned (stiffest stable at a 2 ms step) |
| Contact stiffening width (solimp) | 0.01 mm | tuned |
| impratio | 10 | tuned: at 1, loaded blocks slipped at 61% of μN |
| Time step | 2 ms | tuned: 0.000 mm creep, 0.3 s of computing per simulated second |
| Friction cone | elliptic | more accurate than pyramidal |
| No-slip solver | off | tested: it added creep and let blocks slide at 90% of μN |

## What was wrong before (the Phase 2 sim)

- **Friction was soft** (`impratio` 1 by default): loaded blocks slid at about
  60% of the friction limit, so tight blocks were too easy to drag.
- **Contacts were too compliant** (default stiffening over 0.5 mm): a tower
  tilted 8° bent 11 mm at the top.
- **Tip detection** compared positions, which bending alone could trigger.
  It now detects growing rotation.

## Known limitations

- **Varied block thickness is not stable in MuJoCo.** `Tower(thickness=...)` still
  exists, but towers of uneven blocks rock and walk apart under the stiff contacts
  (details in `docs/TOWER_DATASET.md`). The dataset uses identical geometry.

- **Creep:** under a sustained sideways load below the friction limit, MuJoCo's
  soft friction lets blocks slide slowly (0.28 mm/s at 50% of the limit). An
  upright tower has no sideways load, so it does not creep; neighbours of a
  block being pulled will move by up to about a millimetre over a few seconds.
- **Heavy-load friction is 10-20% low:** a block carrying nine layers comes out
  at 80-90% of μN. Unloaded blocks match μmg exactly. This is inside the spread
  of real wood friction (±30%) but means tight blocks are slightly easier to
  pull than the textbook value.
- **One friction coefficient:** MuJoCo does not separate static from kinetic
  friction (wood: about 0.43 vs 0.30).
- **Compliance:** the tower leans about 0.5° at 10° of tilt. Real wood is
  stiffer; stiffer settings went unstable or locked blocks.
