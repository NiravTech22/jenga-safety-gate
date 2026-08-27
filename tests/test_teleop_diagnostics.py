"""The diagnostics are only useful if they recover a fault they were not told
about, so each test injects a known defect and checks the measurement finds it.

A diagnostic that cannot be validated is just a number generator, and these
numbers are meant to decide which layer of the stack someone spends a weekend
on.
"""
import numpy as np
import pytest

from src.hardware.arm import ArmDefects, N_JOINTS, SimulatedArm, VRTeleopArm
from src.hardware.diagnostics import (
    measure_calibration,
    measure_latency,
    measure_lost_motion,
    measure_repeatability,
    run_diagnostics,
)
from src.hardware.kinematics import JOINT_NAMES, ArmChain


def defects(**kw):
    base = dict(offset_rad=np.zeros(N_JOINTS), scale=np.ones(N_JOINTS),
                backlash_rad=np.zeros(N_JOINTS), deadband_rad=np.zeros(N_JOINTS))
    base.update(kw)
    return ArmDefects(**base)


# -- calibration ------------------------------------------------------------
def test_clean_arm_reports_no_error():
    report = run_diagnostics(SimulatedArm(), joints=[0])
    j = report.joints[0]
    assert abs(j.offset_deg) < 0.05
    assert abs(j.scale_error_pct) < 0.5
    assert j.lost_motion_deg < 0.05
    assert j.worst_fault() == "within tolerance"


def test_offset_is_recovered():
    off = np.zeros(N_JOINTS); off[1] = np.radians(2.5)
    scale, offset, _ = measure_calibration(SimulatedArm(defects(offset_rad=off)), 1)
    assert np.degrees(offset) == pytest.approx(2.5, abs=0.1)
    assert scale == pytest.approx(1.0, abs=0.01)


def test_scale_error_is_recovered_and_not_confused_with_offset():
    sc = np.ones(N_JOINTS); sc[3] = 1.12
    scale, offset, _ = measure_calibration(SimulatedArm(defects(scale=sc)), 3)
    assert scale == pytest.approx(1.12, abs=0.01)
    assert abs(np.degrees(offset)) < 0.2      # a span error is not a bias


def test_offset_and_scale_are_separable_when_both_present():
    off = np.zeros(N_JOINTS); off[2] = np.radians(-1.5)
    sc = np.ones(N_JOINTS); sc[2] = 0.9
    scale, offset, _ = measure_calibration(SimulatedArm(defects(offset_rad=off, scale=sc)), 2)
    assert scale == pytest.approx(0.9, abs=0.01)
    assert np.degrees(offset) == pytest.approx(-1.5, abs=0.15)


# -- lost motion ------------------------------------------------------------
def test_backlash_is_recovered():
    bl = np.zeros(N_JOINTS); bl[2] = np.radians(3.0)
    lost = measure_lost_motion(SimulatedArm(defects(backlash_rad=bl)), 2)
    assert np.degrees(lost) == pytest.approx(3.0, abs=0.15)


def test_backlash_does_not_leak_into_the_calibration_fit():
    """A single-direction sweep sits half a band low at every point and reports
    that as an offset. The two-sided sweep must cancel it."""
    bl = np.zeros(N_JOINTS); bl[2] = np.radians(4.0)
    scale, offset, _ = measure_calibration(SimulatedArm(defects(backlash_rad=bl)), 2)
    assert abs(np.degrees(offset)) < 0.3
    assert scale == pytest.approx(1.0, abs=0.02)


def test_deadband_shows_up_as_lost_motion():
    """Documented behaviour, not an oversight: a servo deadband and mechanical
    backlash both make the joint settle short, and joint position alone cannot
    tell them apart."""
    db = np.zeros(N_JOINTS); db[4] = np.radians(1.5)
    lost = measure_lost_motion(SimulatedArm(defects(deadband_rad=db)), 4)
    assert np.degrees(lost) > 1.0


def test_repeatability_degrades_with_lost_motion():
    bl = np.zeros(N_JOINTS); bl[0] = np.radians(3.0)
    clean = measure_repeatability(SimulatedArm(), 0)
    sloppy = measure_repeatability(SimulatedArm(defects(backlash_rad=bl)), 0)
    assert sloppy > clean + np.radians(0.5)


# -- latency ----------------------------------------------------------------
@pytest.mark.parametrize("latency_s", [0.0, 0.05, 0.2])
def test_latency_is_recovered(latency_s):
    measured = measure_latency(SimulatedArm(ArmDefects(latency_s=latency_s, tau_s=0.02)))
    assert measured == pytest.approx(latency_s, abs=0.03)


def test_high_latency_is_called_out_in_the_report():
    report = run_diagnostics(SimulatedArm(ArmDefects(latency_s=0.25)), joints=[0])
    assert any("latency" in n for n in report.notes)


# -- report -----------------------------------------------------------------
def test_report_round_trips_to_disk(tmp_path):
    report = run_diagnostics(SimulatedArm(), joints=[0, 1])
    path = report.save(tmp_path / "r.json")
    import json
    raw = json.loads(path.read_text())
    assert len(raw["joints"]) == 2
    assert raw["joints"][0]["joint"] == JOINT_NAMES[0]


def test_dominant_fault_picks_the_injected_one():
    off = np.zeros(N_JOINTS); off[1] = np.radians(3.0)
    bl = np.zeros(N_JOINTS); bl[2] = np.radians(3.0)
    sc = np.ones(N_JOINTS); sc[3] = 1.15
    report = run_diagnostics(SimulatedArm(defects(offset_rad=off, backlash_rad=bl, scale=sc)),
                             joints=[1, 2, 3])
    assert report.joints[0].worst_fault() == "offset"
    assert report.joints[1].worst_fault() == "lost motion"
    assert report.joints[2].worst_fault() == "scale"


# -- hardware seam ----------------------------------------------------------
def test_vr_adapter_refuses_to_pretend():
    """The seam must fail loudly rather than return plausible zeros, which
    would silently characterise nothing."""
    arm = VRTeleopArm("https://example.invalid:8443")
    with pytest.raises(NotImplementedError):
        arm.read_joints()
    with pytest.raises(NotImplementedError):
        arm.command_joints(np.zeros(N_JOINTS))


# -- kinematics -------------------------------------------------------------
def test_fk_matches_known_reach_at_zero_pose():
    chain = ArmChain()
    pos = chain.tool_position(np.zeros(6))
    assert pos[0] == pytest.approx(0.3914, abs=1e-3)
    assert pos[2] == pytest.approx(0.2265, abs=1e-3)


def test_fk_moves_the_tool_when_a_joint_moves():
    chain = ArmChain()
    a = chain.tool_position(np.zeros(6))
    b = chain.tool_position([0.0, -0.5, 0.0, 0.0, 0.0, 0.0])
    assert np.linalg.norm(a - b) > 0.05


def test_urdf_limits_are_present_for_every_actuated_joint():
    limits = ArmChain().limits()
    for name in JOINT_NAMES:
        assert name in limits, f"{name} has no declared limit in the URDF"
        lo, hi = limits[name]
        assert lo < hi
