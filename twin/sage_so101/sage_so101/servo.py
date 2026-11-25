"""Servo + physics model: the thing being fitted so sim follows the real SO-101.

Per joint:  command -> (transport delay) -> (backlash/deadband hysteresis) -> PD -> torque clip -> MuJoCo.
 kp,kd        position-loop gains of the STS3215 (effective, Nm/rad and Nm*s/rad)
 tau_max      torque saturation (depends on supply voltage)
 armature     effective reflected rotor/gear inertia  (absorbs link-inertia modelling error too)
 damping      viscous friction;  frictionloss: Coulomb friction (Nm)
 backlash     total deadband width (rad) between command and output
 delay        bus + servo latency (s)
The same pipeline (delay+backlash) can wrap policy actions in Isaac Lab, which has no native delay/backlash.
"""
import os
from dataclasses import dataclass, asdict, fields
import numpy as np, mujoco

PNAMES = ["kp", "kd", "tau_max", "armature", "damping", "frictionloss", "backlash", "delay"]

@dataclass
class JointParams:
    kp: float = 12.0; kd: float = 0.4; tau_max: float = 2.0; armature: float = 0.01
    damping: float = 0.05; frictionloss: float = 0.02; backlash: float = 0.0; delay: float = 0.02
    def vec(self): return np.array([getattr(self, n) for n in PNAMES])
    @staticmethod
    def from_vec(v): return JointParams(**dict(zip(PNAMES, map(float, v))))

def _load_model(path):
    """Load MJCF; if some mesh files are missing (e.g. a partial checkout), drop those meshes/geoms so
    dynamics (explicit <inertial>, joints) still load. With all meshes present this is a plain load."""
    import xml.etree.ElementTree as ET, tempfile
    try: return mujoco.MjModel.from_xml_path(path)
    except ValueError: pass
    d = os.path.dirname(path); tree = ET.parse(path); root = tree.getroot()
    cm = root.find("compiler"); md = os.path.join(d, cm.get("meshdir", "") if cm is not None else "")
    missing = {m.get("name") or os.path.splitext(os.path.basename(m.get("file")))[0] for m in root.iter("mesh") if not os.path.exists(os.path.join(md, m.get("file")))}
    for par in list(root.iter()):
        for ch in list(par):
            if (ch.tag == "mesh" and (ch.get("name") or os.path.splitext(os.path.basename(ch.get("file")))[0]) in missing) or (ch.tag == "geom" and ch.get("mesh") in missing):
                par.remove(ch)
    tmp = os.path.join(d, "_sage_tmp.xml"); tree.write(tmp)
    try: return mujoco.MjModel.from_xml_path(tmp)
    finally: os.remove(tmp)

class ServoSim:
    def __init__(self, cfg, params=None, physics_dt=None):
        self.cfg = cfg
        path = cfg["mjcf"] if os.path.isabs(cfg["mjcf"]) else os.path.join(cfg["_dir"], "..", cfg["mjcf"])
        self.m = _load_model(os.path.normpath(path))
        self.m.opt.timestep = physics_dt or cfg["physics_dt"]
        self.d = mujoco.MjData(self.m)
        self.names = [j["name"] for j in cfg["joints"]]
        jid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j["mjcf_joint"]) for j in cfg["joints"]]
        self.qadr = np.array([self.m.jnt_qposadr[i] for i in jid]); self.dadr = np.array([self.m.jnt_dofadr[i] for i in jid])
        self.sign = np.array([j["sign"] for j in cfg["joints"]], float)
        self.off = np.array([j["offset_rad"] for j in cfg["joints"]], float)
        self.nsub = int(round(1.0 / cfg["control_hz"] / self.m.opt.timestep))
        # the twin's built-in <position> actuators would fight our torque loop: switch actuation off
        self.m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
        ip = cfg.get("baseline_params", cfg["init_params"])   # 'before' = what the twin ships with
        self.params = params or [JointParams(**ip) for _ in self.names]

    # sim frame (calibrated real-arm frame) <-> mujoco joint frame
    def to_mj(self, q): return (q - self.off) * self.sign
    def from_mj(self, q): return q * self.sign + self.off

    def set_params(self, params):
        self.params = params
        for k, p in enumerate(params):
            a = self.dadr[k]; self.m.dof_armature[a] = p.armature
            self.m.dof_damping[a] = p.damping; self.m.dof_frictionloss[a] = p.frictionloss

    def run(self, cmd, joints=None):
        """cmd (N,J) rad in sim frame, one row per control tick. Returns pos (N,J) at the END of each tick."""
        self.set_params(self.params)
        N, J = cmd.shape; hz = self.cfg["control_hz"]; dt = self.m.opt.timestep; S = N * self.nsub
        P = np.array([p.vec() for p in self.params]); kp, kd, tmax, bl, dly = P[:, 0], P[:, 1], P[:, 2], P[:, 6], P[:, 7]
        tc = np.arange(N) / hz; ts = np.arange(S) * dt
        tgt = np.stack([np.interp(ts - dly[k], tc, cmd[:, k]) for k in range(J)], 1)  # delayed, first-order-hold
        tgt = self.to_mj(tgt)
        mujoco.mj_resetData(self.m, self.d)
        q0 = self.to_mj(cmd[0]); self.d.qpos[self.qadr] = q0; mujoco.mj_forward(self.m, self.d)
        eff = q0.copy(); out = np.zeros((N, J)); h = bl / 2
        for s in range(S):
            dl = tgt[s] - eff
            eff = np.where(dl > h, tgt[s] - h, np.where(dl < -h, tgt[s] + h, eff))   # backlash hysteresis
            q = self.d.qpos[self.qadr]; qd = self.d.qvel[self.dadr]
            self.d.qfrc_applied[self.dadr] = np.clip(kp * (eff - q) - kd * qd, -tmax, tmax)
            mujoco.mj_step(self.m, self.d)
            if (s + 1) % self.nsub == 0: out[s // self.nsub] = self.d.qpos[self.qadr]
        return self.from_mj(out)

def save_params(path, names, params):
    import json; json.dump({n: asdict(p) for n, p in zip(names, params)}, open(path, "w"), indent=2)

def load_params(path):
    import json; d = json.load(open(path)); return list(d), [JointParams(**v) for v in d.values()]
