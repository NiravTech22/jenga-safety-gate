"""System identification: find per-joint servo/physics params so sim trials match real trials.
SAGE (NVIDIA doc) stops at measuring/visualising the gap; this is the 'parameter tuning' step it points to."""
import numpy as np
from scipy.optimize import least_squares
from .common import load_trial, Trial
from .servo import ServoSim, JointParams, PNAMES

# name -> (lower, upper); fitted in log space where positive-scale
BOUNDS = dict(kp=(1, 80), kd=(0.01, 5), tau_max=(0.5, 4), armature=(1e-4, 0.2), damping=(1e-4, 1.0),
              frictionloss=(0, 0.5), backlash=(0, 0.1), delay=(0, 0.15))
LOG = {"kp", "kd", "tau_max", "armature", "damping"}

def _enc(name, v): return np.log(v) if name in LOG else v
def _dec(name, x): return np.exp(x) if name in LOG else x

def fit_joint(cfg, k, trials, free=("kp", "kd", "armature", "damping", "frictionloss", "backlash", "delay"), init=None, verbose=0):
    """Fit joint k using trials where it moves. Other joints get cfg init params (they're held still)."""
    sim = ServoSim(cfg); base = [JointParams(**cfg["init_params"]) for _ in sim.names]
    p0 = init or base[k]
    lo = [_enc(n, max(BOUNDS[n][0], 1e-9) if n in LOG else BOUNDS[n][0]) for n in free]
    hi = [_enc(n, BOUNDS[n][1]) for n in free]
    x0 = np.clip([_enc(n, max(getattr(p0, n), 1e-6) if n in LOG else getattr(p0, n)) for n in free], np.array(lo) + 1e-9, np.array(hi) - 1e-9)
    refs = [(t.cmd, t.pos[:, k], t.vel()[:, k]) for t in trials]

    def make(x):
        p = JointParams(**{**p0.__dict__, **{n: _dec(n, v) for n, v in zip(free, x)}})
        ps = list(base); ps[k] = p; return ps

    def resid(x):
        sim.params = make(x); r = []
        for cmd, pos, vel in refs:
            out = sim.run(cmd); from scipy.signal import savgol_filter
            sv = savgol_filter(out[:, k], 7, 2, deriv=1, delta=1 / cfg["control_hz"])
            r += [(out[:, k] - pos) / 0.01, 0.3 * (sv - vel) / 0.2]
        return np.concatenate(r)

    sol = least_squares(resid, x0, bounds=(lo, hi), x_scale=1.0, diff_step=1e-2, loss="soft_l1", f_scale=2.0, verbose=verbose, max_nfev=60)
    return make(sol.x)[k], float(np.sqrt(np.mean(sol.fun[::1] ** 2)))

def fit_all(cfg, trial_paths, joints=None, **kw):
    names = [j["name"] for j in cfg["joints"]]; trials = {n: load_trial(p) for n, p in trial_paths.items()}
    result = []
    for k, n in enumerate(names):
        tr = [t for key, t in trials.items() if key == f"excite_{n}"]
        if not tr: result.append(JointParams(**cfg["init_params"])); print(f"[skip] {n}: no excite file"); continue
        p, c = fit_joint(cfg, k, tr, **kw); result.append(p)
        print(f"[fit] {n:14s} cost={c:6.2f}  " + "  ".join(f"{a}={getattr(p, a):.4g}" for a in PNAMES))
    return result
