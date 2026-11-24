#!/usr/bin/env python3
"""Drive the SO-101 joint by joint with a gamepad (8BitDo in X-input mode, Xbox layout),
and publish telemetry for the episode recorder at the same time.

This REPLACES so101_telemetry.py while it runs: only one process may own the servo bus.
The recorder reads the same telemetry file, now with the servos doing the pulling, so
Present_Current measures how hard the block resists.

Safety
  * --dry-run (do this first): reads the arm and the pad, prints the goals it WOULD send,
    never enables torque or writes a register.
  * Starts DISARMED. Goals start at the arm's present position, so arming never jumps.
  * START arms (torque on, holding the present pose). B = FREEZE: stop all motion and
    disarm (torque stays on, the arm holds where it is, so it does not drop onto the tower).
    BACK/SELECT = LIMP: torque off. Hold the arm before pressing it.
  * Speed capped (--speed, ticks/s, default 250 = about 22 deg/s); every joint clamped
    inside its calibrated range minus a margin, from the LeRobot calibration file.

Controls (Xbox layout; run --show-inputs once to check your pad's numbering)
  left stick X / Y   shoulder_pan / shoulder_lift
  right stick Y / X  elbow_flex / wrist_roll
  D-pad up / down    wrist_flex
  RB / LB            claw open / close
  START arm   B freeze   BACK limp   Ctrl-C quit (freezes first)

Usage
  python scripts/tools/gamepad_teleop.py --show-inputs
  python scripts/tools/gamepad_teleop.py --port /dev/ttyACM0 --calibration-version jenga_follower --dry-run
  python scripts/tools/gamepad_teleop.py --port /dev/ttyACM0 --calibration-version jenga_follower
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from so101_telemetry import CURRENT_MA_PER_UNIT, UNITS, MockBus, _sync_read, open_bus  # noqa: E402
from src.hardware.kinematics import JOINT_NAMES  # noqa: E402
from src.hardware.passive import publish_telemetry  # noqa: E402

CAL_DIR = Path.home() / ".cache/huggingface/lerobot/calibration/robots"
# action -> (kind, index, sign). kind: axis | hat | buttons (+button, -button)
DEFAULT_MAP = {
    "shoulder_pan": ("axis", 0, +1.0),
    "shoulder_lift": ("axis", 1, -1.0),
    "elbow_flex": ("axis", 4, -1.0),
    "wrist_roll": ("axis", 3, +1.0),
    "wrist_flex": ("hat", 1, +1.0),
    "gripper": ("buttons", (5, 4), +1.0),          # RB open, LB close
}
BUTTONS = dict(start=7, freeze=1, limp=6)
SHORT = dict(shoulder_pan="pan", shoulder_lift="lift", elbow_flex="elbow",
             wrist_flex="wflex", wrist_roll="wroll", gripper="claw")


def load_limits(robot_id, margin):
    """{joint: (lo, hi)} raw ticks from the LeRobot calibration file, shrunk by margin."""
    hits = list(CAL_DIR.glob(f"*/{robot_id}.json"))
    if not hits:
        raise SystemExit(f"no calibration '{robot_id}' under {CAL_DIR}; run lerobot-calibrate first")
    cal = json.loads(hits[0].read_text())
    return {n: (cal[n]["range_min"] + margin, cal[n]["range_max"] - margin) for n in JOINT_NAMES}


class Pad:
    """pygame joystick, read without opening a window."""

    def __init__(self, index=0):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame
        pygame.init(); pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise SystemExit("no gamepad found: connect the 8BitDo (X-input mode) by USB or Bluetooth")
        self.pg = pygame
        self.js = pygame.joystick.Joystick(index); self.js.init()
        print(f"gamepad: {self.js.get_name()} ({self.js.get_numaxes()} axes, "
              f"{self.js.get_numbuttons()} buttons, {self.js.get_numhats()} hats)")

    def read(self):
        self.pg.event.pump()
        js = self.js
        return dict(axes=[js.get_axis(i) for i in range(js.get_numaxes())],
                    buttons=[js.get_button(i) for i in range(js.get_numbuttons())],
                    hats=[js.get_hat(i) for i in range(js.get_numhats())])


def command(state, mapping, deadzone):
    """Pad state -> {joint: -1..1}."""
    out = {}
    for joint, (kind, idx, sign) in mapping.items():
        v = 0.0
        if kind == "axis" and idx < len(state["axes"]):
            v = state["axes"][idx]
            v = 0.0 if abs(v) < deadzone else (v - deadzone * (1 if v > 0 else -1)) / (1 - deadzone)
        elif kind == "hat" and state["hats"]:
            v = float(state["hats"][0][idx])
        elif kind == "buttons":
            p, n = idx
            b = state["buttons"]
            v = float((b[p] if p < len(b) else 0) - (b[n] if n < len(b) else 0))
        out[joint] = sign * v
    return out


def pressed(state, name):
    i = BUTTONS[name]
    return i < len(state["buttons"]) and bool(state["buttons"][i])


class Teleop:
    def __init__(self, bus, limits, speed, dry_run, out, cal_version, mapping=DEFAULT_MAP, deadzone=.15):
        self.bus, self.limits, self.speed, self.dry = bus, limits, speed, dry_run
        self.out, self.cal_version, self.mapping, self.deadzone = out, cal_version, mapping, deadzone
        pos = _sync_read(bus, "Present_Position")
        self.goal = {n: float(pos[n]) for n in JOINT_NAMES}     # start where the arm is: no jump
        self.armed, self.seq, self.writes = False, 0, 0
        self.prev = {k: False for k in BUTTONS}       # for press edges
        self.limp_done = False

    def clamp(self, n, v):
        lo, hi = self.limits[n]
        return min(max(v, lo), hi)

    def arm(self):
        pos = _sync_read(self.bus, "Present_Position")
        self.goal = {n: self.clamp(n, float(pos[n])) for n in JOINT_NAMES}
        if not self.dry:
            self.bus.sync_write("Goal_Position", {n: int(round(v)) for n, v in self.goal.items()}, normalize=False)
            self.bus.enable_torque()
        self.armed = True
        print("\nARMED (holding present pose)")

    def freeze(self, announce=True):
        """Disarm FIRST, so no further motion is sent even if the bus calls below fail."""
        was_armed, self.armed = self.armed, False
        if was_armed:
            pos = _sync_read(self.bus, "Present_Position")
            self.goal = {n: float(pos[n]) for n in JOINT_NAMES}
            if not self.dry:
                self.bus.sync_write("Goal_Position", {n: int(round(v)) for n, v in self.goal.items()},
                                    normalize=False)
        if announce:
            print("\nFROZEN (disarmed; arm holds its pose)")

    def limp(self, announce=True):
        self.armed = False
        if not self.dry:
            self.bus.disable_torque()          # raises on a bus error; retried while BACK is held
        self.limp_done = True
        if announce:
            print("\nLIMP (torque off)")

    def step(self, state, dt):
        now = {k: pressed(state, k) for k in BUTTONS}
        edge = {k: now[k] and not self.prev[k] for k in BUTTONS}
        self.prev = now
        # Safety buttons act every tick they are held (a failed bus call is retried); they only
        # print on the press. START arms on the press only.
        if now["freeze"]:
            self.freeze(announce=edge["freeze"])
        elif now["limp"]:
            if edge["limp"]:
                self.limp_done = False
            if not self.limp_done:
                self.limp(announce=edge["limp"])
            else:
                self.armed = False
        elif edge["start"] and not self.armed:
            self.arm()
        cmd = command(state, self.mapping, self.deadzone)
        if self.armed and any(cmd.values()):
            for n, v in cmd.items():
                self.goal[n] = self.clamp(n, self.goal[n] + v * self.speed * dt)
            if not self.dry:
                self.bus.sync_write("Goal_Position", {n: int(round(v)) for n, v in self.goal.items()},
                                    normalize=False)
            self.writes += 1
        return cmd

    def publish(self):
        pos = _sync_read(self.bus, "Present_Position"); t_pos = time.monotonic()
        cur = _sync_read(self.bus, "Present_Current")
        publish_telemetry(self.out, joint_state=[float(pos[n]) for n in JOINT_NAMES],
                          action=[float(self.goal[n]) for n in JOINT_NAMES],
                          state_timestamp=t_pos, action_timestamp=time.monotonic(), sequence_id=self.seq,
                          joint_names=list(JOINT_NAMES), joint_units=UNITS, calibration_version=self.cal_version,
                          extra=dict(current_ma=[float(cur[n]) * CURRENT_MA_PER_UNIT for n in JOINT_NAMES],
                                     armed=self.armed, dry_run=self.dry, source="gamepad_teleop"))
        self.seq += 1


def show_inputs():
    pad = Pad()
    print("move sticks / press buttons; Ctrl-C to stop")
    try:
        while True:
            s = pad.read()
            ax = " ".join(f"{i}:{v:+.2f}" for i, v in enumerate(s["axes"]))
            bt = " ".join(str(i) for i, v in enumerate(s["buttons"]) if v)
            print(f"\raxes {ax} | buttons [{bt}] | hats {s['hats']}".ljust(120), end="", flush=True)
            time.sleep(.05)
    except KeyboardInterrupt:
        print()


def main(argv=None, pad=None, bus=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port")
    ap.add_argument("--mock", action="store_true", help="mock arm (testing)")
    ap.add_argument("--show-inputs", action="store_true", help="print raw pad axes/buttons and exit")
    ap.add_argument("--dry-run", action="store_true", help="never enable torque or write a register")
    ap.add_argument("--calibration-version", default="jenga_follower", help="LeRobot robot id")
    ap.add_argument("--out", default="logs/arm_latest.json")
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--speed", type=float, default=250.0, help="max joint speed, ticks/s (4096 per turn)")
    ap.add_argument("--margin", type=int, default=40, help="ticks kept away from each calibrated end")
    ap.add_argument("--seconds", type=float, default=float("inf"))
    a = ap.parse_args(argv)
    if a.show_inputs:
        show_inputs(); return 0
    if not (a.port or a.mock or bus):
        ap.error("--port or --mock required")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    limits = ({n: (500, 3600) for n in JOINT_NAMES} if (a.mock or bus)
              else load_limits(a.calibration_version, a.margin))
    bus = bus or (MockBus() if a.mock else open_bus(a.port))
    pad = pad or Pad()
    t = Teleop(bus, limits, a.speed, a.dry_run, a.out, a.calibration_version)
    print(("DRY RUN: nothing will move. " if a.dry_run else "") + "Disarmed. START = arm, B = freeze, BACK = limp.")
    t_end, last = time.monotonic() + a.seconds, time.monotonic()
    try:
        while time.monotonic() < t_end:
            tick = time.monotonic(); dt = min(tick - last, .1); last = tick
            try:
                cmd = t.step(pad.read(), dt)
                t.publish()
            except Exception as exc:          # one missed servo reply must not end the session
                print(f"\nbus error (continuing): {exc}")
                continue
            if t.seq % max(1, int(a.hz / 2)) == 0:
                active = " ".join(f"{SHORT[n]}{v:+.1f}" for n, v in cmd.items() if abs(v) >= .05)
                print(f"\r{'ARMED' if t.armed else 'disarmed'}{' (dry)' if a.dry_run else ''}  {active}".ljust(100),
                      end="", flush=True)
            time.sleep(max(0.0, 1 / a.hz - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        t.freeze()
        bus.disconnect(disable_torque=False)     # arm keeps holding; BACK before quitting to go limp
    print(f"\nstopped: {t.seq} telemetry samples, {t.writes} motion commands")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
