#!/usr/bin/env python3
"""Digital twin of the real SO-101: live mirror of the real arm, or gamepad-driven sim.

  mirror  The MuJoCo arm follows the REAL arm, read from the telemetry file that
          gamepad_teleop.py (or so101_telemetry.py) publishes. No serial port is
          opened, so it runs alongside the teleop. Prints each joint angle and the
          claw tip position (x, y, z in metres, arm base frame: the same frame the
          overhead camera calibration uses). Type commands in the terminal to line
          the model up with the real arm, then `save`.
  sim     The gamepad drives the SIMULATED arm only (no hardware), with the same
          controls, limits and telemetry as gamepad_teleop.py.

Joint map (data/calibration/joint_map.json): servo ticks -> model joint angle
    q = sign * (ticks - zero_tick) * 2*pi / 4096           (4096 ticks per turn)
    claw: closed_tick -> slide 0.03 m (closed), open_tick -> 0 (open)
LeRobot calibration puts the middle of each joint's range at tick ~2048 and the SO-101
model's zero pose is that middle pose, so the defaults are zero_tick 2048, sign +1.
Check each joint with `ref` and fix any that disagree.

Mirror commands (type + Enter):
  ref               toggle: show the model's zero pose (put the real arm in it) / live mirror
  zero [joint]      real arm is in the zero pose: set zero_tick so it reads 0 (all hinge joints, or one)
  flip <joint>      reverse a joint's direction
  nudge <joint> <deg>   shift a joint by a few degrees
  claw open|closed  the real claw is fully open/closed now: record that tick
  show              print the map        save   write data/calibration/joint_map.json

Usage
  python twin/arm_twin.py mirror                       # with gamepad_teleop.py running
  python twin/arm_twin.py sim                          # gamepad drives the sim only
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

TWIN = Path(__file__).resolve().parent
REPO = TWIN.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts/tools"))
from src.hardware.kinematics import JOINT_NAMES  # noqa: E402

SCENE = TWIN / "models/scene_custom_claw_physics.xml"
MAP_PATH = REPO / "data/calibration/joint_map.json"
HINGES = [n for n in JOINT_NAMES if n != "gripper"]
TICKS = 4096
CLAW_SLIDE_M = 0.03          # model gripper slide: 0 = open (88 mm pad gap), 0.03 = closed (28 mm)


# ------------------------------------------------------------------ joint map
class JointMap:
    def __init__(self, d=None):
        d = d or {}
        self.j = {n: {"sign": 1, "zero_tick": 2048.0, **d.get(n, {})} for n in HINGES}
        g = d.get("gripper", {})
        self.claw = dict(closed_tick=float(g.get("closed_tick", 2048)), open_tick=float(g.get("open_tick", 3500)))

    @classmethod
    def load(cls, path=MAP_PATH, lerobot_cal=None):
        if Path(path).exists():
            return cls(json.loads(Path(path).read_text()))
        m = cls()
        if lerobot_cal and Path(lerobot_cal).exists():           # claw ends from the LeRobot sweep
            g = json.loads(Path(lerobot_cal).read_text()).get("gripper")
            if g:
                m.claw = dict(closed_tick=float(g["range_min"]), open_tick=float(g["range_max"]))
        return m

    def to_dict(self):
        return {**{n: dict(v) for n, v in self.j.items()}, "gripper": dict(self.claw)}

    def save(self, path=MAP_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def q(self, name, ticks):
        if name == "gripper":                 # model slide: 0 = fully open, 0.03 m = closed
            c, o = self.claw["closed_tick"], self.claw["open_tick"]
            opening = float(np.clip((ticks - c) / (o - c), 0, 1)) if o != c else 0.0
            return (1.0 - opening) * CLAW_SLIDE_M
        v = self.j[name]
        return v["sign"] * (ticks - v["zero_tick"]) * 2 * math.pi / TICKS

    def ticks(self, name, q):
        if name == "gripper":
            c, o = self.claw["closed_tick"], self.claw["open_tick"]
            return c + (1.0 - float(np.clip(q / CLAW_SLIDE_M, 0, 1))) * (o - c)
        v = self.j[name]
        return v["zero_tick"] + v["sign"] * q * TICKS / (2 * math.pi)


def lerobot_cal_path(robot_id):
    hits = list((Path.home() / ".cache/huggingface/lerobot/calibration/robots").glob(f"*/{robot_id}.json"))
    return hits[0] if hits else None


# ------------------------------------------------------------------ model helpers
class Arm:
    def __init__(self, scene=SCENE):
        self.m = mujoco.MjModel.from_xml_path(str(scene))
        self.d = mujoco.MjData(self.m)
        self.qadr = {n: int(self.m.jnt_qposadr[self.m.joint(n).id]) for n in JOINT_NAMES}
        self.mimic = int(self.m.jnt_qposadr[self.m.joint("gripper_mimic").id])
        self.act = {n: self.m.actuator(n).id for n in JOINT_NAMES}
        self.pads = (self.m.site("pad_marker_finger_left").id, self.m.site("pad_marker_finger_right").id)
        self.base = self.m.site("baseframe").id
        mujoco.mj_forward(self.m, self.d)

    def set_q(self, q):
        for n, v in q.items():
            lo, hi = self.m.jnt_range[self.m.joint(n).id]
            v = float(np.clip(v, lo, hi))
            self.d.qpos[self.qadr[n]] = v
            if n == "gripper":
                self.d.qpos[self.mimic] = v
        self.d.qvel[:] = 0
        mujoco.mj_forward(self.m, self.d)

    def get_q(self):
        return {n: float(self.d.qpos[self.qadr[n]]) for n in JOINT_NAMES}

    def tip_world(self):
        """Custom claw grasp point: midway between the two finger pads (as twin/so101_pickplace.py)."""
        return (self.d.site_xpos[self.pads[0]] + self.d.site_xpos[self.pads[1]]) / 2

    def tip_xyz(self):
        """Claw grasp point in the arm base frame (metres)."""
        R = self.d.site_xmat[self.base].reshape(3, 3)
        return R.T @ (self.tip_world() - self.d.site_xpos[self.base])

    def joint_ticks_limits(self, jm, margin=40):
        """Model joint ranges as tick limits for the teleop clamp."""
        out = {}
        for n in JOINT_NAMES:
            lo, hi = self.m.jnt_range[self.m.joint(n).id]
            a, b = sorted((jm.ticks(n, lo), jm.ticks(n, hi)))
            out[n] = (a + margin, b - margin) if n != "gripper" else (a, b)
        return out


def apply_telemetry(arm, jm, tele):
    arm.set_q({n: jm.q(n, t) for n, t in zip(JOINT_NAMES, tele["joint_state"])})


def status_line(arm, ticks=None):
    q = arm.get_q()
    short = dict(shoulder_pan="pan", shoulder_lift="lift", elbow_flex="elbow", wrist_flex="wflex", wrist_roll="wroll")
    ang = " ".join(f"{short[n]}:{math.degrees(q[n]):+6.1f}" for n in HINGES)
    x, y, z = arm.tip_xyz()
    claw = (1 - q["gripper"] / CLAW_SLIDE_M) * 100     # % open
    return f"{ang} claw:{claw:3.0f}% open  tip x={x*1000:6.1f} y={y*1000:6.1f} z={z*1000:6.1f} mm"


# ------------------------------------------------------------------ mirror
def mirror(a):
    jm = JointMap.load(a.map, lerobot_cal_path(a.robot_id))
    arm = Arm()
    state = dict(ref=False, last=None, quit=False)

    def commands():
        while not state["quit"]:
            try:
                words = input().split()
            except EOFError:
                return
            if not words:
                continue
            cmd, args = words[0], words[1:]
            t = state["last"]
            ticks = dict(zip(JOINT_NAMES, t["joint_state"])) if t else None
            try:
                if cmd == "ref":
                    state["ref"] = not state["ref"]
                    print("showing ZERO POSE: put the real arm in this pose, then `zero`" if state["ref"]
                          else "live mirror")
                elif cmd == "zero":
                    for n in (args or HINGES):
                        jm.j[n]["zero_tick"] = float(ticks[n])
                    print("zero set for", args or "all hinge joints")
                elif cmd == "flip":
                    jm.j[args[0]]["sign"] *= -1; print(args[0], "sign", jm.j[args[0]]["sign"])
                elif cmd == "nudge":
                    n, deg = args[0], float(args[1])
                    jm.j[n]["zero_tick"] -= jm.j[n]["sign"] * deg / 360 * TICKS
                    print(n, f"shifted {deg:+.1f} deg")
                elif cmd == "claw":
                    jm.claw["open_tick" if args[0] == "open" else "closed_tick"] = float(ticks["gripper"])
                    print("claw", args[0], "=", ticks["gripper"])
                elif cmd == "show":
                    print(json.dumps(jm.to_dict(), indent=1))
                elif cmd == "save":
                    jm.save(a.map); print("saved", a.map)
                else:
                    print("commands: ref | zero [joint] | flip <joint> | nudge <joint> <deg> | claw open|closed | show | save")
            except (KeyError, IndexError, ValueError, TypeError) as exc:
                print("bad command:", exc, "(joints:", ", ".join(HINGES) + ")")

    threading.Thread(target=commands, daemon=True).start()
    import mujoco.viewer
    print(f"mirroring {a.telemetry}; type `help` for commands")
    with mujoco.viewer.launch_passive(arm.m, arm.d) as v:
        last_print = 0.0
        while v.is_running():
            try:
                state["last"] = json.loads(Path(a.telemetry).read_text())
            except (OSError, json.JSONDecodeError):
                pass
            if state["ref"]:
                arm.set_q({n: 0.0 for n in HINGES} | {"gripper": 0.0})
            elif state["last"]:
                apply_telemetry(arm, jm, state["last"])
            v.sync()
            if time.monotonic() - last_print > .25 and not state["ref"]:
                age = time.monotonic() - state["last"]["state_timestamp"] if state["last"] else float("inf")
                print("\r" + status_line(arm) + ("  [telemetry stale]" if age > 1 else ""), end="", flush=True)
                last_print = time.monotonic()
            time.sleep(1 / 60)
    state["quit"] = True


# ------------------------------------------------------------------ sim (gamepad drives the model)
class SimBus:
    """Looks like the LeRobot bus to gamepad_teleop.Teleop, backed by the MuJoCo arm."""

    def __init__(self, arm, jm):
        self.arm, self.jm = arm, jm
        self.gain = arm.m.actuator_gainprm.copy(); self.bias = arm.m.actuator_biasprm.copy()
        self.disable_torque()

    def sync_read(self, reg, normalize=False):
        q = self.arm.get_q()
        if reg == "Present_Position":
            return {n: self.jm.ticks(n, q[n]) for n in JOINT_NAMES}
        if reg == "Goal_Position":
            return {n: self.jm.ticks(n, float(self.arm.d.ctrl[self.arm.act[n]])) for n in JOINT_NAMES}
        if reg == "Present_Current":                 # actuator effort, scaled to servo current units
            f = self.arm.d.actuator_force
            return {n: abs(float(f[self.arm.act[n]])) / 3.35 * 400 for n in JOINT_NAMES}
        raise KeyError(reg)

    def sync_write(self, reg, values, normalize=True):
        assert reg == "Goal_Position"
        for n, t in values.items():
            self.arm.d.ctrl[self.arm.act[n]] = self.jm.q(n, t)

    def enable_torque(self):
        self.torque = True
        self.arm.m.actuator_gainprm[:] = self.gain; self.arm.m.actuator_biasprm[:] = self.bias

    def disable_torque(self):
        """Torque off like the real servos: actuators produce no force at all."""
        self.torque = False
        self.arm.m.actuator_gainprm[:] = 0; self.arm.m.actuator_biasprm[:] = 0

    def step(self, seconds):
        for _ in range(max(1, round(seconds / self.arm.m.opt.timestep))):
            mujoco.mj_step(self.arm.m, self.arm.d)

    def disconnect(self, disable_torque=False):
        pass


def sim(a):
    import gamepad_teleop as gt
    jm = JointMap.load(a.map, lerobot_cal_path(a.robot_id))
    arm = Arm()
    arm.set_q({n: 0.0 for n in JOINT_NAMES})
    for n in JOINT_NAMES:
        arm.d.ctrl[arm.act[n]] = 0.0
    bus = SimBus(arm, jm)
    pad = gt.Pad()
    Path(a.telemetry).parent.mkdir(parents=True, exist_ok=True)
    t = gt.Teleop(bus, arm.joint_ticks_limits(jm), a.speed, False, a.telemetry, "sim")
    import mujoco.viewer
    print("SIM: START arms, B freezes, BACK limp. Close the viewer to quit.")
    with mujoco.viewer.launch_passive(arm.m, arm.d) as v:
        last, last_print = time.monotonic(), 0.0
        while v.is_running():
            now = time.monotonic(); dt = min(now - last, .1); last = now
            t.step(pad.read(), dt)
            bus.step(dt)
            t.publish()
            v.sync()
            if now - last_print > .25:
                print("\r" + ("ARMED " if t.armed else "disarmed ") + status_line(arm), end="", flush=True)
                last_print = now
            time.sleep(max(0.0, 1 / 30 - (time.monotonic() - now)))
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["mirror", "sim"])
    ap.add_argument("--telemetry", default="logs/arm_latest.json")
    ap.add_argument("--map", default=str(MAP_PATH))
    ap.add_argument("--robot-id", default="jenga_follower", help="LeRobot calibration name (claw ends)")
    ap.add_argument("--speed", type=float, default=250.0, help="sim: max joint speed, ticks/s")
    a = ap.parse_args(argv)
    (mirror if a.mode == "mirror" else sim)(a)


if __name__ == "__main__":
    main()
