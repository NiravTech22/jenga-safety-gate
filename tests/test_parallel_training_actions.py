import json
import time

import numpy as np
import pytest

from src.hardware.kinematics import JOINT_NAMES
from src.parallel.actions import ActionChecks, ActionProposal
from src.parallel.runtime import Observation, Result


def check_inputs():
    return ActionProposal(0, (0.05,) * 6), Observation(0, time.monotonic(), None), Result(0, "execute", "ok"), (0,) * 6


def checker():
    return ActionChecks({name: (-1, 1) for name in JOINT_NAMES}, 0.1)


def test_action_requires_explicit_clearance_and_robot_turn():
    checks = checker()
    assert not checks.evaluate(*check_inputs())[0]
    assert not checks.evaluate(*check_inputs(), robot_turn=True, emergency_stop=False)[0]
    assert checks.evaluate(*check_inputs(), robot_turn=True, workspace_clear=True, emergency_stop=False)[0]
    assert not checks.evaluate(*check_inputs(), robot_turn=True, workspace_clear=True, emergency_stop=False)[0]


@pytest.mark.parametrize("case", ["stale", "mismatch", "blocked", "step", "limit", "nan", "shape"])
def test_unsafe_actions_block(case):
    proposal, obs, result, current = check_inputs()
    if case == "stale":
        obs = Observation(0, time.monotonic() - 2, None)
    elif case == "mismatch":
        proposal = ActionProposal(1, proposal.target_rad)
    elif case == "blocked":
        result = Result(0, "block", "anomaly")
    else:
        targets = {"step": (0.2,) * 6, "limit": (2,) * 6,
                   "nan": (float("nan"),) * 6, "shape": (0,) * 3}
        proposal = ActionProposal(0, targets[case])
    assert not checker().evaluate(proposal, obs, result, current, robot_turn=True,
                                  workspace_clear=True, emergency_stop=False)[0]


def write_cache(path, leak=False):
    np.savez(path, clips=np.zeros((4, 2, 16, 16, 3), dtype=np.uint8),
             scores=np.array([0.1, 0.2, 0.1, 0.2], np.float32),
             episode_ids=np.array(["a", "a", "a" if leak else "b", "b"]),
             split=np.array(["train", "train", "validation", "validation"]))


def test_distillation_rejects_episode_leakage(tmp_path):
    from src.parallel.distill import load_cache
    write_cache(tmp_path / "data.npz", leak=True)
    with pytest.raises(ValueError, match="leakage"):
        load_cache(tmp_path / "data.npz")


def test_training_exports_loadable_local_student(tmp_path):
    from src.parallel.distill import train
    from src.parallel.models import LocalStudent
    write_cache(tmp_path / "data.npz")
    provenance = tmp_path / "teacher.json"
    provenance.write_text(json.dumps(dict(teacher_checkpoint="synthetic-test-only",
        score_definition="synthetic constant targets", dataset_id="test-only")))
    report = train(tmp_path / "data.npz", provenance, tmp_path / "out",
                   device="cpu", epochs=1, batch_size=2, size=16)
    student = LocalStudent(tmp_path / "out/student.ts", tmp_path / "out/student.json", "cpu")
    assert np.isfinite(student.score(np.zeros((2, 16, 16, 3), np.uint8)).value)
    assert report["safety_validated"] is False
