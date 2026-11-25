import argparse, os, numpy as np
from .common import load_config, joint_names, load_trial, save_trial, Trial
from . import motions, analyze
from .servo import ServoSim, save_params, load_params
from .fit import fit_all
from .isaac_export import isaac_lab_snippet

def main():
    ap = argparse.ArgumentParser(prog="sage_so101"); ap.add_argument("--config"); sub = ap.add_subparsers(dest="c", required=True)
    g = sub.add_parser("gen"); g.add_argument("--out", default="output/motions"); g.add_argument("--scale", type=float, default=0.5)
    r = sub.add_parser("record"); r.add_argument("--motion", required=True); r.add_argument("--out", required=True); r.add_argument("--port", required=True); r.add_argument("--id", default="my_so101"); r.add_argument("--scale", type=float, default=0.5)
    f = sub.add_parser("fit"); f.add_argument("--real-dir", default="output/real"); f.add_argument("--out", default="output/fitted_params.json")
    a = sub.add_parser("analyze"); a.add_argument("--real", required=True); a.add_argument("--params"); a.add_argument("--out", default="output/plots")
    x = sub.add_parser("export"); x.add_argument("--params", required=True)
    A = ap.parse_args(); cfg = load_config(A.config); names = joint_names(cfg)
    if A.c == "gen": print("wrote", motions.generate(cfg, A.out, A.scale))
    elif A.c == "record":
        from .record_real import record; record(cfg, A.motion, A.out, A.port, A.id, A.scale)
    elif A.c == "fit":
        paths = {f[:-4]: os.path.join(A.real_dir, f) for f in os.listdir(A.real_dir) if f.endswith(".csv")}
        p = fit_all(cfg, {k: v for k, v in paths.items() if k.startswith("excite_")}); save_params(A.out, names, p); print("saved", A.out)
    elif A.c == "analyze":
        real = load_trial(A.real); os.makedirs(A.out, exist_ok=True); sims = {"sim default": Trial(real.t, real.cmd, ServoSim(cfg).run(real.cmd), names)}
        if A.params: _, ps = load_params(A.params); sims["sim fitted"] = Trial(real.t, real.cmd, ServoSim(cfg, ps).run(real.cmd), names)
        for k, s in sims.items(): print(analyze.table(analyze.metrics(real, s, names), k))
        analyze.plot_overlay(real, sims, names, os.path.join(A.out, os.path.basename(A.real)[:-4] + ".png"), "real vs sim")
    elif A.c == "export": n, ps = load_params(A.params); print(isaac_lab_snippet(n, ps))
if __name__ == "__main__": main()
