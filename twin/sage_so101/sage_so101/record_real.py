"""Play a motion file on the physical SO-101 and log commanded vs measured joint angles.
UNTESTED ON HARDWARE (written against the current LeRobot SO101Follower API: robot.get_observation()
keys '<joint>.pos', robot.send_action({...}); LeRobot's API has moved between versions - adjust imports if needed).
Real arm frame: degrees with LeRobot calibration zero; converted to sim frame with cfg sign/offset_rad."""
import time, numpy as np
from .common import load_motion, Trial, save_trial

def record(cfg, motion_path, out_csv, port, robot_id="my_so101", scale=1.0, settle_s=2.0):
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    names, mot = load_motion(motion_path); J = cfg["joints"]; hz = cfg["control_hz"]; dt = 1.0 / hz
    assert names == [j["name"] for j in J], "motion joints must match config"
    sign = np.array([j["sign"] for j in J]); off = np.array([j["offset_rad"] for j in J])
    lo = np.array([j["lo"] for j in J]); hi = np.array([j["hi"] for j in J]); cen = np.array([j["center"] for j in J])
    mot = np.clip(cen + scale * (mot - cen), lo, hi)
    to_deg = lambda q: np.degrees((q - off) * sign); to_sim = lambda d: np.radians(d) * sign + off
    robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id, use_degrees=True)); robot.connect()
    send = lambda q: robot.send_action({f"{n}.pos": float(v) for n, v in zip(names, to_deg(q))})
    read = lambda: to_sim(np.array([robot.get_observation()[f"{n}.pos"] for n in names]))
    q_start = read()
    def ramp(a, b, secs):
        for i in range(int(secs * hz)): t0 = time.perf_counter(); send(a + (b - a) * (i + 1) / (secs * hz)); time.sleep(max(0, dt - (time.perf_counter() - t0)))
    try:
        ramp(q_start, mot[0], 3.0); time.sleep(settle_s)             # go to first waypoint slowly, settle
        N = len(mot); pos = np.zeros_like(mot); ts = np.zeros(N); t0 = time.perf_counter()
        for i in range(N):
            send(mot[i]); tn = t0 + (i + 1) * dt
            while time.perf_counter() < tn - 0.002: time.sleep(0.0005)
            pos[i] = read(); ts[i] = time.perf_counter() - t0        # pos at END of tick i == sim logging convention
        jit = np.diff(ts); print(f"tick jitter: mean {jit.mean()*1e3:.2f} ms, max {jit.max()*1e3:.2f} ms (target {dt*1e3:.1f})")
        save_trial(out_csv, Trial(np.arange(N) * dt, mot, pos, names))
    finally:
        try: ramp(read(), q_start, 2.0)                               # return so torque-off doesn't drop the arm
        finally: robot.disconnect()
