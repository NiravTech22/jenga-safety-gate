# Tower-run database (`twin/tower_dataset.py`)

Many simulated towers, many ways to stand or fall, one SQLite file. Built on the
validated tower physics in [TOWER_PHYSICS.md](TOWER_PHYSICS.md).

```bash
python twin/tower_dataset.py generate --runs 600 --workers 6   # resumable; skips ids already stored
python twin/tower_dataset.py summary                           # outcomes by scenario + predictor accuracy
python twin/tower_dataset.py show 17                           # one run
python twin/tower_dataset.py replay 17                         # watch it in MuJoCo (re-simulated from its id)
python twin/tower_dataset.py export data/tower_runs/runs.csv   # flat table
python twin/tower_dataset.py check --fix                       # find runs MuJoCo silently reset; then re-run generate
```

If MuJoCo hits a numerical blow-up it resets the simulation and carries on.
`generate` detects this and redoes the run with a smaller time step (recorded
in the `timestep` column). `check` finds such runs in databases made before
that, and `generate` also redoes runs whose scenario definition has changed
(per-scenario `version`).

Each run takes about 8 s of one CPU core. Runs are deterministic from their id,
so `replay` and re-generation reproduce them exactly.

## Scenarios (run id mod 6)

| id mod 6 | Scenario | What happens |
|---|---|---|
| 0 | removal | 1-12 random blocks vanish one at a time (perfect extraction) |
| 1 | pull | a random block is pulled along its length, force ramp to 0.1-4 N |
| 2 | poke | a 12 mm fingertip moves into the upper half at 2-40 cm/s, 5-40 mm past the face, then back |
| 3 | tilt | the table tilts slowly to 0-22° in a random direction |
| 4 | shake | the table shakes sideways, 0-4 m/s² at 1-6 Hz for 2 s |
| 5 | game | random extractions in sequence until it falls or 15 are out |

Per run, randomised around the measured blocks (2026-11-20): 10-16 layers,
friction 0.18-0.30, block mass 17.6-19.5 g. In half the runs friction also
varies block to block (sd 0.04, the measured spread). **Block geometry is
identical in every run** (see below for why).

## What is stored

`data/tower_runs/runs.sqlite`, table `runs`, one row per run: physics, scenario
parameters (JSON, including which blocks were removed), outcome (`stood`,
`partial` = some blocks fell, `collapsed` = at least 25% came down), collapse
onset time, blocks fallen, top displacement, and for removal/game runs a
**static-equilibrium prediction** with its margin.
`data/tower_runs/traj/run_XXXXXX.npz`: every block's position and orientation at
25 Hz (about 90 KB per run).

## The static-equilibrium predictor

A load-path model, top down: each block is a rigid beam on the crossing blocks
below it; the loads on it (its own weight plus where the blocks above rest on
it) have a resultant along its length, which must lie over its supports; the
load is passed down by force and moment balance. It is evaluated after every
removal, and the least stable state counts.

With identical blocks it agrees with the simulation in nearly every run
(43/44 in the first database, 19/20 in the current sample), with uniform or
per-block friction.

### Why block thickness is not varied (2026-11-21)

Real blocks differ in thickness (measured sd 0.19 mm), so in a real tower some
blocks carry the load and others are loose: in a 16-layer test, the layer-3
block was tight and the layer-2, 4 and 7 blocks were loose, and the tower stood
through all four removals. Modelling that geometrically does not work in MuJoCo:
a block resting on two supports of different heights rocks and slowly walks
under MuJoCo's stiff box contacts, and towers of varied blocks drifted 9-400 mm
in 3 s **with nothing removed** (identical towers: under 1 µm/s). Eight fixes
were tried (contact damping, time step, torsional friction, no-slip, rotational
damping, settle-then-stiffen, contact margin, pyramidal cone); none settled the
tower without making it far softer than the real one (tipping at ~10° instead of
18°). Earlier databases with `thickness_tol_mm > 0` are invalid for that reason,
and `summary` reports them separately.

So which blocks are tight is treated as **uncertainty the gate must sense**
(wiggle force, servo current), not something the simulation predicts. A
regression test checks that every dataset tower comes to rest untouched.

## Limits

Poke version 1 (a force on one block) launched blocks unrealistically fast and
could blow up the simulation; version 2 uses a moving fingertip.

Inherits the tower physics limits (soft-friction creep, heavy-load friction
10-20% low). Removals are instantaneous, not slid out. The "partial" outcome
counts any block dropping 3 cm, so a single top block knocked off is a partial.
