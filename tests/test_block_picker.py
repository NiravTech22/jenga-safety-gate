"""Block picker: arm's-view naming, Jenga top rule, equilibrium ranking, grasp geometry."""
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
TWIN = Path(__file__).resolve().parents[1] / "twin"
sys.path.insert(0, str(TWIN))
_spec = importlib.util.spec_from_file_location("block_picker", TWIN / "block_picker.py")
bp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bp)
TP = bp.TP


def p16():
    p = TP.TowerPhysics(); p.layers.value = 16
    return p


def test_only_layers_pointing_at_the_arm_are_offered():
    picks = bp.rank(p16(), set(), "pointing")
    assert picks and all(r["layer"] % 2 == 0 for r in picks)
    picks = bp.rank(p16(), set(), "crossing")
    assert all(r["layer"] % 2 == 1 for r in picks)


def test_middle_block_of_a_mid_layer_comes_first_on_an_intact_tower():
    best = bp.rank(p16(), set(), "pointing")[0]
    assert best["stands"] and best["slot"] == 1 and 4 <= best["layer"] <= 10


def test_a_pick_that_leaves_one_outer_block_is_flagged_unsafe():
    removed = bp.parse_removed(["6M"], "pointing")
    by_name = {r["name"]: r for r in bp.rank(p16(), removed, "pointing")}
    assert not by_name["6L"]["stands"] and not by_name["6R"]["stands"]


def test_top_rule():
    assert 15 not in bp.allowed_layers(16, set())
    assert 14 not in bp.allowed_layers(16, {(15, 0)})


def test_names_round_trip_and_bad_names_fail():
    for layer in (4, 5):
        for s in range(3):
            assert bp.slot_index(layer, bp.slot_name(layer, s, "pointing"), "pointing") == s
    with pytest.raises(ValueError):
        bp.slot_index(5, "L", "pointing")              # layer 5 crosses: N/M/F


def test_grasp_point_is_the_block_end_facing_the_arm():
    p = p16()
    x, y, z = bp.grasp_point(p, (409.0, 0.0), 6, 1)
    assert x == pytest.approx(409 - 1000 * p.v("block_length") / 2, abs=.1) and y == pytest.approx(0, abs=1e-6)
    assert z == pytest.approx(1000 * (bp.BASE_BOARD_M + 6.5 * p.v("block_height")), abs=.1)
    _, yl, _ = bp.grasp_point(p, (409.0, 0.0), 6, 2)    # slot 2 = arm's left = +y
    assert yl > 20
