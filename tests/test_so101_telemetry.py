"""Read-only arm telemetry bridge feeds the v2 recorder (mock bus, no hardware)."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("so101_telemetry", ROOT / "scripts/tools/so101_telemetry.py")
tel = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(tel)

from src.hardware.kinematics import JOINT_NAMES
from src.hardware.passive import TelemetryFile


def test_mock_bridge_publishes_valid_telemetry(tmp_path):
    out = tmp_path / "arm.json"
    assert tel.main(["--mock", "--out", str(out), "--seconds", "0.2", "--hz", "50"]) == 0
    v = TelemetryFile(out, JOINT_NAMES, tel.UNITS, "so101-raw-ticks").read()
    assert len(v["joint_state"]) == len(v["action"]) == len(v["current_ma"]) == 6
    assert v["sequence_id"] > 0 and v["mock"] is True


def test_bridge_never_writes_registers():
    src = (ROOT / "scripts/tools/so101_telemetry.py").read_text()
    assert ".write(" not in src and "disable_torque=False" in src


def test_bridge_survives_bus_errors(tmp_path, monkeypatch):
    calls = {"n": 0}
    real = tel.MockBus.sync_read

    def flaky(self, name, normalize=False):
        calls["n"] += 1
        if calls["n"] % 4 == 0:
            raise RuntimeError("no status packet")
        return real(self, name, normalize)

    monkeypatch.setattr(tel.MockBus, "sync_read", flaky)
    out = tmp_path / "arm.json"
    assert tel.main(["--mock", "--out", str(out), "--seconds", "0.3", "--hz", "50"]) == 0
    v = TelemetryFile(out, JOINT_NAMES, tel.UNITS, "so101-raw-ticks").read()
    assert v["sequence_id"] > 3
