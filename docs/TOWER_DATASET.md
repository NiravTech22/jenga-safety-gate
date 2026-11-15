# Tower-run database (`twin/tower_dataset.py`)

Many simulated towers, many ways to stand or fall, one SQLite file. Built on the
validated tower physics in [TOWER_PHYSICS.md](TOWER_PHYSICS.md).

```bash
python twin/tower_dataset.py generate --runs 600 --workers 6   # resumable; skips ids already stored
python twin/tower_dataset.py summary                           # outcomes by scenario + predictor accuracy
python twin/tower_dataset.py show 17                           # one run
python twin/tower_dataset.py replay 17                         # watch it in MuJoCo (re-simulated from its id)
python twin/tower_dataset.py export data/tower_runs/runs.csv   # flat table
```

Each run takes about 8 s of one CPU core. Runs are deterministic from their id,
so `replay` and re-generation reproduce them exactly.

## Scenarios (run id mod 6)

| id mod 6 | Scenario | What happens |
|---|---|---|
| 0 | removal | 1-12 random blocks vanish one at a time (perfect extraction) |
| 1 | pull | a random block is pulled along its length, force ramp to 0.1-4 N |
| 2 | poke | a 0.5-8 N sideways impulse (30-300 ms) on a block in the upper half |
| 3 | tilt | the table tilts slowly to 0-22° in a random direction |
| 4 | shake | the table shakes sideways, 0-4 m/s² at 1-6 Hz for 2 s |
| 5 | game | random extractions in sequence until it falls or 15 are out |

Per run, randomised: 12-18 layers, friction 0.25-0.50, block mass 9-12 g, block
thickness tolerance 0 / ±0.1 / ±0.2 / ±0.3 mm.

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

First runs: with **identical blocks it agreed with the simulation every time**;
with blocks varying by ±0.1-0.3 mm it missed some falls. That is a real effect,
not a bug: a few tenths of a millimetre decide which blocks carry the load, so
a tower that is stable on paper can be unstable with real blocks. This is the
gap the wiggle sensing is meant to close.

## Limits

Inherits the tower physics limits (soft-friction creep, heavy-load friction
10-20% low). Removals are instantaneous, not slid out. The "partial" outcome
counts any block dropping 3 cm, so a single top block knocked off is a partial.
