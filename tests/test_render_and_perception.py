"""Round-trip tests: render a known tower, read it back with the Step 2
perception pipeline, and require the reconstruction to match.

This is the Step 2 exit condition in miniature. The full 20-trial version
lives in scripts/02_validate_tower_state.py; these run on every commit so a
regression in either the renderer or the reader surfaces immediately.
"""
import numpy as np
import pytest

from src.perception.tower_state import (
    ArucoFrameCalibrator,
    OccupancyModel,
    detect_block_occupancy,
    estimate_lean_angle,
    estimate_lean_from_silhouette,
    extract_tower_edges,
    fit_occupancy_model,
    get_tower_state,
    segment_wood_mask,
)
from src.sim.render import default_rig, render_empty_table, render_frame_pair
from src.sim.tower import TowerSim, TowerSimConfig

N_LAYERS = 12


@pytest.fixture(scope="module")
def calibrated():
    """Rig, calibrated cameras and a fitted occupancy model — the artifacts
    Step 1 produces, built once for the whole module since rendering is the
    slow part."""
    rig = default_rig()
    empty_over, empty_side = render_empty_table(rig, np.random.default_rng(0))
    K = rig.side.intrinsics.matrix()

    cal_side = ArucoFrameCalibrator(K, layout=rig.markers)
    cal_over = ArucoFrameCalibrator(K, layout=rig.markers)
    assert cal_side.calibrate(empty_side)
    assert cal_over.calibrate(empty_over)

    refs = []
    for i in range(3):
        sim = TowerSim(TowerSimConfig(seed=300 + i, num_layers=N_LAYERS))
        refs.append(render_frame_pair(sim.state, rig, np.random.default_rng(400 + i))[1])
    model = fit_occupancy_model(refs, cal_side, N_LAYERS)
    return rig, cal_side, cal_over, model


def render_state(rig, pre_removed=(), lean=0.0, seed=11):
    sim = TowerSim(TowerSimConfig(seed=seed, num_layers=N_LAYERS,
                                  pre_removed=tuple(pre_removed), initial_lean_deg=lean))
    over, side = render_frame_pair(sim.state, rig, np.random.default_rng(seed + 1))
    return sim, over, side


# -- renderer ---------------------------------------------------------------
def test_render_shapes_and_determinism():
    rig = default_rig()
    sim = TowerSim(TowerSimConfig(seed=1, num_layers=N_LAYERS))
    a = render_frame_pair(sim.state, rig, np.random.default_rng(5))
    b = render_frame_pair(sim.state, rig, np.random.default_rng(5))
    assert a[0].shape == (rig.config.height, rig.config.width, 3)
    assert a[0].dtype == np.uint8
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_removed_block_renders_darker_than_its_neighbours():
    """The occupancy reader keys on brightness, so a gap must actually be
    darker in the image, not merely absent from the model."""
    rig = default_rig()
    _, _, full_side = render_state(rig)
    _, _, gap_side = render_state(rig, pre_removed=[(5, 1)])
    assert gap_side.mean() < full_side.mean()


def test_empty_table_shows_all_four_markers():
    rig = default_rig()
    over, _ = render_empty_table(rig, np.random.default_rng(0))
    cal = ArucoFrameCalibrator(rig.overhead.intrinsics.matrix(), layout=rig.markers)
    assert cal.calibrate(over)
    assert cal.n_markers_seen == 4


# -- calibration ------------------------------------------------------------
def test_calibration_recovers_true_pose(calibrated):
    rig, cal_side, _, _ = calibrated
    _, tvec_true = rig.side.rvec_tvec()
    recovered = cal_side._tvec.ravel()
    assert np.abs(recovered - tvec_true.ravel()).max() < 2e-3      # under 2 mm


def test_calibration_fails_on_a_frame_with_no_markers(calibrated):
    rig, _, _, _ = calibrated
    blank = np.full((rig.config.height, rig.config.width, 3), 120, np.uint8)
    cal = ArucoFrameCalibrator(rig.side.intrinsics.matrix(), layout=rig.markers)
    assert cal.calibrate(blank) is False
    assert cal.is_calibrated() is False


def test_projection_before_calibration_raises(calibrated):
    rig, _, _, _ = calibrated
    cal = ArucoFrameCalibrator(rig.side.intrinsics.matrix(), layout=rig.markers)
    with pytest.raises(Exception):
        cal.project(np.zeros((1, 3)))


def test_calibration_survives_save_load(tmp_path, calibrated):
    _, cal_side, _, _ = calibrated
    path = cal_side.save(tmp_path / "side.json")
    reloaded = ArucoFrameCalibrator.load(path)
    pts = np.array([[0.0, 0.0, 0.0], [0.03, 0.01, 0.09]])
    assert np.allclose(reloaded.project(pts), cal_side.project(pts), atol=1e-6)


def test_occupancy_model_survives_save_load(tmp_path, calibrated):
    *_, model = calibrated
    reloaded = OccupancyModel.load(model.save(tmp_path / "model.json"))
    assert reloaded.threshold == pytest.approx(model.threshold)
    assert reloaded.classify(model.occupied_mean) is True


# -- occupancy --------------------------------------------------------------
@pytest.mark.parametrize("pre_removed", [
    (),
    ((2, 1),),
    ((3, 0), (6, 2)),
    ((5, 0), (5, 1), (5, 2)),          # fully hollowed layer
    ((1, 0), (4, 2), (7, 0), (9, 2)),
])
def test_occupancy_reconstructed_exactly(calibrated, pre_removed):
    rig, cal_side, _, model = calibrated
    sim, _, side = render_state(rig, pre_removed=pre_removed, seed=21)
    lean, _ = estimate_lean_from_silhouette(side, cal_side, N_LAYERS)
    occ, _ = detect_block_occupancy(side, cal_side, N_LAYERS, model, lean)
    assert np.array_equal(occ, sim.state.occupancy)


def test_occupancy_margin_is_larger_for_confident_reads(calibrated):
    rig, cal_side, _, model = calibrated
    sim, _, side = render_state(rig, pre_removed=[(4, 1)], seed=23)
    lean, _ = estimate_lean_from_silhouette(side, cal_side, N_LAYERS)
    _, margins = detect_block_occupancy(side, cal_side, N_LAYERS, model, lean)
    assert np.isfinite(margins).all()
    assert margins.max() > 0


# -- lean -------------------------------------------------------------------
@pytest.mark.parametrize("true_lean", [0.0, 2.0, 4.0, -3.0])
def test_lean_estimate_tracks_truth(calibrated, true_lean):
    rig, cal_side, _, _ = calibrated
    sim, _, side = render_state(rig, pre_removed=[(3, 1)], lean=true_lean, seed=31)
    est, iou = estimate_lean_from_silhouette(side, cal_side, N_LAYERS)
    assert abs(est - sim.state.lean_deg) <= 1.0
    assert iou > 0.5


def test_estimate_lean_angle_is_median_not_mean():
    """One bad edge detection should not move the estimate much."""
    edges = np.array([[0, 0, 0, 10], [1, 0, 1, 10], [2, 0, 2, 10], [0, 0, 40, 10]])
    assert abs(estimate_lean_angle(edges)) < 5.0


def test_estimate_lean_angle_rejects_empty_and_horizontal():
    with pytest.raises(ValueError):
        estimate_lean_angle(np.empty((0, 4)))
    with pytest.raises(ValueError):
        estimate_lean_angle(np.array([[0, 5, 10, 5]]))


def test_tower_edges_are_found_in_a_rendered_frame(calibrated):
    rig, *_ = calibrated
    _, _, side = render_state(rig, seed=41)
    assert len(extract_tower_edges(side)) > 0


def test_wood_mask_covers_tower_but_not_whole_frame(calibrated):
    rig, *_ = calibrated
    _, _, side = render_state(rig, seed=43)
    frac = np.count_nonzero(segment_wood_mask(side)) / side[:, :, 0].size
    assert 0.02 < frac < 0.5


# -- top level --------------------------------------------------------------
def test_get_tower_state_round_trip(calibrated):
    rig, cal_side, cal_over, model = calibrated
    sim, over, side = render_state(rig, pre_removed=[(2, 0), (5, 2)], lean=1.5, seed=51)
    state = get_tower_state(over, side, cal_side, cal_over, N_LAYERS, model)
    assert np.array_equal(state.occupancy_array(), sim.state.occupancy)
    assert abs(state.lean_angle_deg - sim.state.lean_deg) <= 1.0
    assert state.num_layers_detected == N_LAYERS
    assert 0.0 <= state.confidence <= 1.0
    assert state.confidence > 0.5
