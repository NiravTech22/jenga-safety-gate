"""SAGE-style gap analysis: per-joint RMSE / max error / lag, plus overlay + before/after plots."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def lag_ms(a, b, dt, maxlag=25):
    """positive = b lags a. Cross-correlation of de-meaned signals."""
    a = a - a.mean(); b = b - b.mean()
    cs = [np.dot(a[max(0, -l):len(a) - max(0, l)], b[max(0, l):len(b) - max(0, -l)]) for l in range(-maxlag, maxlag + 1)]
    return (np.argmax(cs) - maxlag) * dt * 1000

def metrics(real, sim, joints):
    dt = real.dt; out = {}
    for k, j in enumerate(joints):
        e = sim.pos[:, k] - real.pos[:, k]
        out[j] = dict(rmse_deg=np.degrees(np.sqrt(np.mean(e ** 2))), max_deg=np.degrees(np.abs(e).max()),
                      lag_ms=lag_ms(real.pos[:, k], sim.pos[:, k], dt))
    return out

def table(m, title=""):
    s = f"{title}\n{'joint':14s}{'RMSE°':>8s}{'max°':>8s}{'lag ms':>8s}\n"
    for j, v in m.items(): s += f"{j:14s}{v['rmse_deg']:8.2f}{v['max_deg']:8.2f}{v['lag_ms']:8.1f}\n"
    return s

def plot_overlay(real, sims, joints, path, title=""):
    """sims: dict label->Trial. One row per joint that actually moves."""
    mv = [k for k in range(len(joints)) if np.ptp(real.cmd[:, k]) > 1e-3] or list(range(len(joints)))
    fig, ax = plt.subplots(len(mv), 1, figsize=(10, 2.4 * len(mv)), sharex=True, squeeze=False)
    for r, k in enumerate(mv):
        a = ax[r, 0]; a.plot(real.t, np.degrees(real.cmd[:, k]), color="0.7", lw=1, label="command")
        a.plot(real.t, np.degrees(real.pos[:, k]), "k", lw=1.4, label="real")
        for lab, s in sims.items(): a.plot(s.t, np.degrees(s.pos[:, k]), lw=1.1, label=lab)
        a.set_ylabel(f"{joints[k]} (°)"); a.grid(alpha=.3)
    ax[0, 0].legend(ncol=4, fontsize=8); ax[-1, 0].set_xlabel("time (s)"); fig.suptitle(title); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)

def plot_bars(before, after, path):
    js = list(before); x = np.arange(len(js)); fig, a = plt.subplots(figsize=(7, 3.5))
    a.bar(x - .2, [before[j]["rmse_deg"] for j in js], .4, label="before fit"); a.bar(x + .2, [after[j]["rmse_deg"] for j in js], .4, label="after fit")
    a.set_xticks(x); a.set_xticklabels(js, rotation=20); a.set_ylabel("RMSE vs real (°)"); a.legend(); a.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
