"""Tower-run database: static-equilibrium predictor, determinism, database round trip."""
import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
TWIN = Path(__file__).resolve().parents[1] / "twin"
sys.path.insert(0, str(TWIN))
_spec = importlib.util.spec_from_file_location("tower_dataset", TWIN / "tower_dataset.py")
td = importlib.util.module_from_spec(_spec); sys.modules["tower_dataset"] = td; _spec.loader.exec_module(td)
TP = td.TP


@pytest.mark.parametrize("removed,stands", [
    (set(), True), ({(3, 1)}, True), ({(3, 0)}, True), ({(3, 0), (3, 2)}, True),
    ({(3, 0), (3, 1)}, False), ({(3, 1), (3, 2)}, False), ({(2, 0), (2, 2)}, True)])
def test_load_path_predictor(removed, stands):
    assert td.equilibrium(TP.TowerPhysics(), removed)[0] == stands


def test_predictor_matches_sim_for_identical_blocks():
    p = TP.TowerPhysics()
    for removed in ({(4, 0), (4, 1)}, {(4, 1), (5, 0)}):
        t = TP.Tower(p, removed=removed); t.run(3.0)
        assert (not t.collapsed()) == td.equilibrium(p, removed)[0]


def test_runs_are_deterministic_and_stored(tmp_path):
    a = td._job((3, 0, None)); b = td._job((3, 0, None))      # run 3 is a "tilt" run
    assert a["unstable"] == 0 and a["timestep"] == .002
    assert a["scenario"] == "tilt" and a["outcome"] == b["outcome"] and a["onset_s"] == b["onset_s"]
    db = td.connect(tmp_path)
    cols = [c[1] for c in db.execute("PRAGMA table_info(runs)")]
    db.execute(f"INSERT INTO runs ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", [a.get(c) for c in cols])
    db.commit()
    assert sqlite3.connect(tmp_path / "runs.sqlite").execute("SELECT scenario FROM runs").fetchone()[0] == "tilt"


def test_poke_is_a_finger_and_stays_stable():
    rec = td._job((2, 0, None))                                    # run 2 is a "poke" run
    assert rec["scenario"] == "poke" and rec["version"] == td.VERSIONS["poke"]
    assert rec["unstable"] == 0 and rec["max_top_disp_mm"] < 200
