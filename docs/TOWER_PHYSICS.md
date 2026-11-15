# Tower physics (`twin/tower_physics.py`)

A Jenga tower in MuJoCo that stands while it is in static equilibrium and
collapses completely once it is not, checked against closed-form predictions.

```bash
python twin/tower_physics.py --validate        # 11 tests, ~3 min; table + docs/results/tower_physics_validation.json
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

## Results (all pass)

| Test | Theory | Simulation |
|---|---|---|
| Intact tower, 10 s | no movement | 0.000 mm |
| Tilt, whole tower | tips when the equilibrium margin reaches 0 (rigid: 15.7°) | tips at 13.0°; margin +0.9 mm at the last stable angle, +0.3 mm one step later; 85% of blocks down |
| Tilt on a one-block neck | margin reaches 0 (rigid: 6.3°) | tips at 5.4°; margin +0.2 / 0.0 mm; 87% down |
| Remove middle block | stands | stands |
| Remove one outer block | stands | stands |
| Remove both outer blocks | stands | stands |
| Remove middle + one outer (two cases) | collapses | collapses, 71% of blocks down |
| Pull a load-bearing block (layer 8) | 1.10-2.16 N (dragged / pinned limits) | always out from 0.93 N; stick-slip from 0.90 N |
| Pull a loose block | 0.039 N (μmg) | 0.040 N |
| Creep under 50% of the friction limit | 0 (real wood) | 0.28 mm/s (known limitation) |

The **equilibrium margin** is the signed distance from the weight's line of
action to the support edge, computed from the tower's actual (slightly bent)
shape. It is the direct test of the law: the tower stands while it is positive
and tips within a millimetre of it reaching zero, out of a 38 mm half-width.

The tower tips earlier than the rigid-body angle (13.0° vs 15.7°) because it
leans about 1.5° under load before tipping: MuJoCo contacts are slightly
compliant and load moves to the corner contact points. The lean shifts the
centre of mass outward, and the margin shows the tower still tips exactly when
equilibrium is lost.

## Parameters

All values are in one table (`TowerPhysics` at the top of the module), each with
its source. Replace `estimate:` with `measured:` as your numbers come in.

| Parameter | Value | Source |
|---|---|---|
| Block size | 75 × 25 × 15 mm | estimate: standard Jenga |
| Block mass | 10 g | estimate: 540 g set / 54 |
| Layers | 18 | estimate: standard tower |
| Gap between blocks in a layer | 0.5 mm | estimate |
| Wood-wood friction μ | 0.40 | estimate: published 0.25-0.50, lab value 0.43 |
| Table friction | 0.50 | estimate |
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
