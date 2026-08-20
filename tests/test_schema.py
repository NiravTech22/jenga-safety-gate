import tempfile
from pathlib import Path

from src.data.schema import Episode, EpisodeOutcome, ScenarioCategory, TimestepRecord


def make_clean_episode():
    ep = Episode.new(ScenarioCategory.STABLE_EASY, operator="test")
    for i in range(5):
        ep.add_timestep(TimestepRecord(
            t=i * 0.05,
            overhead_frame_path=f"frames/{i:06d}_overhead.jpg",
            side_frame_path=f"frames/{i:06d}_side.jpg",
            joint_state=[0.0] * 6,
            action=[0.0] * 6,
        ))
    return ep


def test_new_episode_has_uuid_and_metadata():
    ep = Episode.new(ScenarioCategory.LEANING)
    assert ep.metadata.episode_id
    assert ep.metadata.scenario_category == ScenarioCategory.LEANING
    assert ep.metadata.outcome == EpisodeOutcome.UNLABELED


def test_add_timestep_updates_metadata():
    ep = make_clean_episode()
    assert ep.metadata.num_timesteps == 5
    assert ep.metadata.duration_s == 4 * 0.05


def test_clean_episode_has_no_validation_problems():
    ep = make_clean_episode()
    problems = ep.validate_time_alignment()
    assert problems == []


def test_detects_non_increasing_timestamp():
    ep = Episode.new(ScenarioCategory.STABLE_EASY)
    ep.add_timestep(TimestepRecord(t=0.1, overhead_frame_path="a", side_frame_path="a",
                                    joint_state=[0.0], action=[0.0]))
    ep.add_timestep(TimestepRecord(t=0.05, overhead_frame_path="b", side_frame_path="b",
                                    joint_state=[0.0], action=[0.0]))
    problems = ep.validate_time_alignment()
    assert any("non-increasing" in p for p in problems)


def test_detects_large_gap():
    ep = Episode.new(ScenarioCategory.STABLE_EASY)
    ep.add_timestep(TimestepRecord(t=0.0, overhead_frame_path="a", side_frame_path="a",
                                    joint_state=[0.0], action=[0.0]))
    ep.add_timestep(TimestepRecord(t=5.0, overhead_frame_path="b", side_frame_path="b",
                                    joint_state=[0.0], action=[0.0]))
    problems = ep.validate_time_alignment(max_gap_s=0.2)
    assert any("gap" in p for p in problems)


def test_detects_mismatched_state_action_length():
    ep = Episode.new(ScenarioCategory.STABLE_EASY)
    ep.add_timestep(TimestepRecord(t=0.0, overhead_frame_path="a", side_frame_path="a",
                                    joint_state=[0.0, 0.0], action=[0.0]))
    problems = ep.validate_time_alignment()
    assert any("len" in p for p in problems)


def test_empty_episode_flagged():
    ep = Episode.new(ScenarioCategory.STABLE_EASY)
    problems = ep.validate_time_alignment()
    assert any("zero timesteps" in p for p in problems)


def test_save_and_load_roundtrip():
    ep = make_clean_episode()
    ep.finalize(EpisodeOutcome.SUCCESS, notes="test episode")
    with tempfile.TemporaryDirectory() as tmp:
        saved_path = ep.save(Path(tmp))
        assert saved_path.exists()
        loaded = Episode.load(saved_path)
        assert loaded.metadata.episode_id == ep.metadata.episode_id
        assert loaded.metadata.outcome == EpisodeOutcome.SUCCESS
        assert len(loaded.timesteps) == len(ep.timesteps)
        assert loaded.timesteps[0].joint_state == ep.timesteps[0].joint_state
