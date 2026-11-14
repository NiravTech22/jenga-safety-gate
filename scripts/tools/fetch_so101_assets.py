#!/usr/bin/env python3
"""Download the upstream SO-101 robot description (URDF, MJCF, meshes).

The upstream files are third-party (TheRobotStudio/SO-ARM100, Apache-2.0) and
are fetched rather than committed, matching .gitignore. They land in two places:

  data/assets/so101/            URDF + MJCF + assets/ for src/hardware/kinematics.py,
                                src/sim/mujoco_jenga.py and the URDF tests
  twin/models/meshes/           the upstream meshes the digital twin needs, next to
                                the custom-claw meshes that ARE committed

Usage:
  python scripts/tools/fetch_so101_assets.py               # upstream main branch
  python scripts/tools/fetch_so101_assets.py --ref <sha>   # pin a commit
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = "TheRobotStudio/SO-ARM100"
SUBDIR = "Simulation/SO101"
TOP_FILES = ["so101_new_calib.urdf", "so101_new_calib.xml", "scene.xml", "joints_properties.xml"]
DEST = REPO_ROOT / "data/assets/so101"
TWIN_MESHES = REPO_ROOT / "twin/models/meshes"
TWIN_MODEL = REPO_ROOT / "twin/models/so101_custom_claw_physics.xml"


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def resolve_ref(ref: str) -> str:
    """Turn a branch name into a commit SHA so the download is reproducible."""
    try:
        data = json.loads(get(f"https://api.github.com/repos/{UPSTREAM}/commits/{ref}"))
        return data["sha"]
    except Exception:
        return ref  # API unreachable/rate-limited: fall back to the name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="main", help="upstream branch, tag or commit")
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    a = ap.parse_args()

    ref = resolve_ref(a.ref)
    base = f"https://raw.githubusercontent.com/{UPSTREAM}/{ref}/{SUBDIR}"
    (DEST / "assets").mkdir(parents=True, exist_ok=True)
    TWIN_MESHES.mkdir(parents=True, exist_ok=True)

    def fetch(rel: str) -> Path:
        out = DEST / rel
        if out.exists() and not a.force:
            return out
        print(f"  {rel}")
        out.write_bytes(get(f"{base}/{rel}"))
        return out

    print(f"Fetching {UPSTREAM}@{ref[:12]} {SUBDIR} -> {DEST.relative_to(REPO_ROOT)}")
    meshes: set[str] = set()
    for name in TOP_FILES:
        text = fetch(name).read_text(errors="ignore")
        meshes |= {Path(m).name for m in re.findall(r'(?:file|filename)="([^"]+\.(?:stl|STL))"', text)}

    # The twin needs these too (its custom-claw meshes are committed, the rest are upstream).
    if TWIN_MODEL.exists():
        twin_meshes = set(re.findall(r'file="([^"]+\.stl)"', TWIN_MODEL.read_text()))
        upstream = {m for m in twin_meshes if not m.startswith("custom_claw_")}   # custom claw is committed
        # --force refreshes every upstream twin mesh, not only the absent ones, so a new --ref takes effect.
        missing = upstream if a.force else {m for m in upstream if not (TWIN_MESHES / m).exists()}
        meshes |= missing
    else:
        missing = set()

    for m in sorted(meshes):
        fetch(f"assets/{m}")
    for m in sorted(missing):
        shutil.copy2(DEST / "assets" / m, TWIN_MESHES / m)

    (DEST / "PROVENANCE.json").write_text(json.dumps(
        {"upstream": UPSTREAM, "ref": ref, "subdir": SUBDIR, "license": "Apache-2.0",
         "files": TOP_FILES + [f"assets/{m}" for m in sorted(meshes)]}, indent=2) + "\n")
    print(f"Done: {len(meshes)} meshes; {len(missing)} copied/refreshed in {TWIN_MESHES.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
