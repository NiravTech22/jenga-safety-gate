"""End-to-end check with NO hardware: a hidden 'real' arm (different params, finer physics dt, encoder
quantisation + noise) is excited, then the pipeline must recover its params and predict a held-out motion."""
import os, sys, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sage_so101.common import load_config, joint_names, Trial, save_trial, load_trial
from sage_so101 import motions, analyze
from sage_so101.servo import ServoSim, JointParams, save_params, PNAMES
from sage_so101.fit import fit_all
from sage_so101.isaac_export import isaac_lab_snippet

out = sys.argv[1] if len(sys.argv) > 1 else "output_demo"; os.makedirs(f"{out}/real", exist_ok=True); os.makedirs(f"{out}/plots", exist_ok=True)
cfg = load_config(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "standin.json")); names = joint_names(cfg); rng = np.random.default_rng(1)
motions.generate(cfg, f"{out}/motions")
hidden = [JointParams(kp=k, kd=d, tau_max=1.8, armature=a, damping=v, frictionloss=f, backlash=b, delay=dl) for k, d, a, v, f, b, dl in [
    (22, .7, .02, .15, .05, .012, .045), (30, 1.0, .03, .20, .06, .010, .040), (18, .6, .015, .10, .04, .015, .050),
    (14, .4, .008, .08, .03, .020, .035), (9, .25, .004, .05, .02, .008, .050)]]
real_sim = ServoSim(cfg, hidden, physics_dt=0.001)   # different dt than the fitter's model (no inverse crime)
def fake_real(cmd):
    pos = real_sim.run(cmd); pos = np.round(pos / 0.00153) * 0.00153 + rng.normal(0, 0.0007, pos.shape)  # 4096-tick encoder + noise
    return Trial(np.arange(len(cmd)) / cfg["control_hz"], cmd, pos, names)
paths = {}
for f in sorted(os.listdir(f"{out}/motions")):
    n = f[:-4]; cmd = np.loadtxt(f"{out}/motions/{f}", delimiter=",", skiprows=1)
    save_trial(f"{out}/real/{n}.csv", fake_real(cmd)); paths[n] = f"{out}/real/{n}.csv"

t0 = time.time(); fitted = fit_all(cfg, {k: v for k, v in paths.items() if k.startswith("excite_")}); print(f"fit time {time.time()-t0:.0f}s")
save_params(f"{out}/fitted_params.json", names, fitted)
print("\nhidden vs fitted"); 
for n, h, p in zip(names, hidden, fitted): print(f"{n:14s}", "  ".join(f"{a}:{getattr(h,a):.3g}->{getattr(p,a):.3g}" for a in ["kp","kd","armature","damping","frictionloss","backlash","delay"]))

# held-out validation on all-joint random waypoints
real = load_trial(paths["random_waypoints"]); cmd = real.cmd
before = ServoSim(cfg).run(cmd); after = ServoSim(cfg, fitted).run(cmd)
mk = lambda p: Trial(real.t, cmd, p, names)
mb, ma = analyze.metrics(real, mk(before), names), analyze.metrics(real, mk(after), names)
print("\n" + analyze.table(mb, "HELD-OUT random_waypoints: default params vs real") + "\n" + analyze.table(ma, "HELD-OUT random_waypoints: fitted params vs real"))
analyze.plot_overlay(real, {"sim default": mk(before), "sim fitted": mk(after)}, names, f"{out}/plots/validation_random_waypoints.png", "Held-out validation")
analyze.plot_bars(mb, ma, f"{out}/plots/rmse_before_after.png")
print(isaac_lab_snippet(names, fitted))
