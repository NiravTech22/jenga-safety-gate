import json
import time

import cv2
import numpy as np
import pytest

from src.data.recording_v2 import Recording, inspect_episode, load_episode
from src.data.replay import render_row
from src.hardware.passive import TelemetryFile, publish_telemetry, CameraStreams


def record(tmp_path):
    rec = Recording(tmp_path, backend="synthetic", session_id="s", tower_id="t",
                    calibration_version="c", joint_names=["a"]*6, joint_units="radians")
    now = time.monotonic()
    frames = {k: (np.zeros((24,32,3),np.uint8), now, 0, 0) for k in ("overhead","wrist")}
    telemetry = dict(joint_state=[.1]*6, action=[.2]*6, state_timestamp=now, action_timestamp=now, sequence_id=0)
    rec.append(frames, telemetry, now)
    return rec


def test_roundtrip_events_and_replay(tmp_path):
    rec = record(tmp_path)
    rec.event("removal", "block 1", t=0)
    rec.event("collapse_onset", t=.01)
    path = rec.close()
    assert inspect_episode(path)["ok"]
    data = load_episode(path)
    assert data["metadata"]["cameras"]["wrist"]["mount"] == "arm"
    assert data["metadata"]["motor_commands_sent"] == 0
    assert render_row(data, path.parent, 0).shape == (530,960,3)


@pytest.mark.parametrize("fault,expected", [
    ("missing", "missing wrist"), ("stale", "stale/future wrist"),
    ("skew", "camera skew"), ("future", "stale/future wrist"),
    ("state", "stale/future state"), ("action", "stale/future action"),
    ("count", "count mismatch"), ("nan", "invalid joint_state"),
    ("path", "unsafe wrist"), ("sequence", "repeated/reversed"),
])
def test_integrity_faults(tmp_path, fault, expected):
    rec = record(tmp_path)
    row = rec.data["timesteps"][0]
    if fault == "missing":
        row["cameras"]["wrist"] = None
    elif fault in ("stale", "skew", "future"):
        row["cameras"]["wrist"]["timestamp"] += {"stale": -1, "skew": -.1, "future": 1}[fault]
    elif fault in ("state", "action"):
        row["telemetry"][fault+"_timestamp"] -= 1
    elif fault == "count":
        rec.data["metadata"]["num_timesteps"] = 99
    elif fault == "nan":
        row["telemetry"]["joint_state"][0] = float("nan")
    elif fault == "path":
        row["cameras"]["wrist"]["path"] = "../../escape.jpg"
    elif fault == "sequence":
        rec.data["timesteps"].append(row.copy())
    path = rec.directory / "episode.json"
    path.write_text(json.dumps(rec.data))
    assert any(expected in issue for issue in inspect_episode(path)["issues"])


def test_missing_image(tmp_path):
    rec = record(tmp_path)
    (rec.directory / rec.data["timesteps"][0]["cameras"]["wrist"]["path"]).unlink()
    assert not inspect_episode(rec.close())["ok"]


def test_interrupted_and_partial_manifest(tmp_path):
    rec = record(tmp_path)
    assert load_episode(rec.directory / "episode.json")["metadata"]["status"] == "interrupted"
    assert "interrupted run" in inspect_episode(rec.close(True, "disconnect"))["warnings"]


def test_drop_report(tmp_path):
    rec = record(tmp_path)
    row = json.loads(json.dumps(rec.data["timesteps"][0]))
    row["t"] += .01
    row["telemetry"]["sequence_id"] = 1
    for sample in row["cameras"].values():
        sample.update(sequence_id=4, failed_reads=2)
    rec.data["timesteps"].append(row)
    report = inspect_episode(rec.close())
    assert report["dropped_frames"]["wrist"] == {"sequence_gaps": 3, "failed_reads": 2}


def test_legacy_side_not_wrist(tmp_path):
    cv2.imwrite(str(tmp_path / "frame.jpg"), np.zeros((24,32,3),np.uint8))
    path = tmp_path / "episode.json"
    path.write_text(json.dumps(dict(metadata={}, timesteps=[dict(t=0,
        overhead_frame_path="frame.jpg", side_frame_path="frame.jpg", joint_state=[0]*6, action=[1]*6)])))
    data = load_episode(path)
    assert "side" in data["metadata"]["cameras"] and "wrist" not in data["metadata"]["cameras"]
    assert inspect_episode(path)["warnings"]
    assert render_row(data, tmp_path, 0).shape == (530,960,3)


def test_bridge_contract_and_no_commands(tmp_path):
    path = tmp_path / "telemetry.json"
    names = ["a"]*6
    source = TelemetryFile(path, names, "ticks", "c")
    with pytest.raises(FileNotFoundError):
        source.read()
    publish_telemetry(path, joint_state=[1]*6, action=[2]*6, state_timestamp=1,
                      action_timestamp=1, sequence_id=1, joint_names=names,
                      joint_units="ticks", calibration_version="c")
    assert source.read()["action"] == [2]*6
    assert not hasattr(source, "command_joints")
    with pytest.raises(ValueError, match="units"):
        TelemetryFile(path, names, "radians", "c").read()
    with pytest.raises(ValueError, match="distinct"):
        CameraStreams({"overhead": 0, "wrist": 0})


def test_failed_write_and_invalid_state(tmp_path, monkeypatch):
    rec = record(tmp_path)
    now = time.monotonic()
    telemetry = rec.data["timesteps"][0]["telemetry"].copy()
    telemetry["joint_state"] = []
    with pytest.raises(ValueError):
        rec.append({}, telemetry, now)
    telemetry["joint_state"] = [0]*6
    monkeypatch.setattr(cv2, "imwrite", lambda *args: False)
    with pytest.raises(IOError):
        rec.append({"overhead": (np.zeros((2,2,3),np.uint8), now, 1, 0)}, telemetry, now)


def test_malformed_and_empty(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"schema_version": 2}')
    assert not inspect_episode(path)["ok"]
    rec = Recording(tmp_path, backend="synthetic", session_id="s", tower_id="t",
                    calibration_version="c", joint_names=["a"], joint_units="radians")
    assert "empty episode" in inspect_episode(rec.close())["issues"]
