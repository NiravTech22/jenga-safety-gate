import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from src.sim.mujoco_jenga import JengaPhysics, ROBOT
from src.parallel.train_mujoco_risk import metrics


@pytest.mark.skipif(not ROBOT.exists(), reason="SO-101 simulation assets not installed")
def test_tower_stable_without_disturbance():
    env = JengaPhysics(1001)
    try:
        assert env.model.nu == 6
        assert len(env.block_ids) == 36
        env.step(1)
        assert not env.failed()
    finally:
        env.close()


@pytest.mark.skipif(not ROBOT.exists(), reason="SO-101 simulation assets not installed")
def test_force_causes_future_failure_after_observation_window():
    env = JengaPhysics(1001)
    try:
        for i in range(4):
            env.data.xfrc_applied[:] = 0
            if i < 2:
                for bid in env.top_ids:
                    env.data.xfrc_applied[bid, 0] = .6 / 3
            env.step(.02)
        assert not env.failed()
        env.data.xfrc_applied[:] = 0
        env.step(1)
        assert env.failed()
    finally:
        env.close()


def test_metrics_penalize_block_everything():
    result = metrics(np.ones(4), np.array([0, 0, 1, 1]), .5)
    assert result["recall"] == 1
    assert result["precision"] == .5
    assert result["normal_pass_rate"] == 0
