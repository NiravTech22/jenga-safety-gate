"""Plot explicitly historical MuJoCo gate counts and new synthetic timing."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


episode = Path(sys.argv[1])
output = Path("docs/figures")
output.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
counts = [15, 2, 3, 28]
axes[0].bar(["Caught\nfailures", "MISSED\nfailures", "False\nalarms", "Stable\npasses"], counts,
            color=["#247e70", "#d13d45", "#dc9931", "#247e70"])
for i, count in enumerate(counts):
    axes[0].text(i, count+.4, str(count), ha="center")
axes[0].set_ylim(0, 32)
axes[0].set_ylabel("Held-out episodes (48 total)")
axes[0].set_title("Historical MuJoCo risk gate, threshold 0.72\nSource: docs/MUJOCO_TRAINING.md (2026-10-03)")
axes[0].text(.5, -.25, "2/17 failures passed the gate; no action policy evaluated.\nSource evaluation JSON absent in this checkout; not rerun.",
             transform=axes[0].transAxes, ha="center", fontsize=9)
data = json.loads(episode.read_text())
rows = data["timesteps"]
for name in ("overhead", "wrist"):
    axes[1].plot([r["t"] for r in rows],
        [(r["receipt_timestamp"]-r["cameras"][name]["timestamp"])*1000 for r in rows], label=name, marker=".")
axes[1].axhline(200, color="#d13d45", linestyle="--", label="age limit")
axes[1].set_ylim(-5, 220)
axes[1].set_xlabel("Episode seconds")
axes[1].set_ylabel("Host receipt frame age (ms)")
axes[1].set_title("New synthetic recording smoke test\nNot camera exposure or hardware latency")
axes[1].legend()
fig.tight_layout()
fig.subplots_adjust(bottom=.25)
fig.savefig(output / "milestone1-evidence.png", dpi=160)
