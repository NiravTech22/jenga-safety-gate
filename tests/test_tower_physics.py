"""Jenga tower physics against static equilibrium (fast subset of
`python twin/tower_physics.py --validate`; the full suite takes ~3 min)."""
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
_spec = importlib.util.spec_from_file_location("tower_physics", Path(__file__).resolve().parents[1] / "twin/tower_physics.py")
tp = importlib.util.module_from_spec(_spec); sys.modules["tower_physics"] = tp; _spec.loader.exec_module(tp)


def test_intact_tower_does_not_creep():
    assert tp.test_stand(seconds=3.0)["passed"]


@pytest.mark.parametrize("slots,stands", [((1,), True), ((0, 2), True), ((0, 1), False)])
def test_removal_matches_equilibrium(slots, stands):
    t = tp.Tower(removed={(3, s) for s in slots})
    t.run(3.0)
    assert (not t.collapsed()) == stands == tp.predict_removal_stands(slots)


def test_neck_tips_at_the_equilibrium_boundary():
    r = tp.test_neck()
    assert r["passed"], r
    m_ok, m_next = r["margin_mm"]
    assert m_ok >= -2 and m_next <= 2


def test_loose_block_pull_is_mu_m_g():
    dragged, pinned, loose = tp.predict_pull_n()
    f = tp._breakaway(tp.P, 8, -.0003, 10 * loose)
    assert f is not None and abs(f - loose) <= .3 * loose
