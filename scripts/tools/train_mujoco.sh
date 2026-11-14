#!/usr/bin/env bash
# Linux equivalent of train_mujoco.ps1: rebuild MuJoCo data, features, both
# trained components, and the parallel smoke run. Run from anywhere.
set -euo pipefail
EPISODES=${EPISODES:-240}; SEED=${SEED:-1000}
DATASET=${DATASET:-data/mujoco_jenga}; OUTPUT=${OUTPUT:-checkpoints/mujoco_vjepa}
cd "$(dirname "$0")/../.."
PY=${PYTHON:-.venv/bin/python}
"$PY" -m src.sim.mujoco_jenga --episodes "$EPISODES" --seed "$SEED" --output "$DATASET"
"$PY" scripts/tools/extract_vjepa_features.py --dataset "$DATASET" --output "$OUTPUT"
"$PY" -m src.runtime.train_mujoco_risk --directory "$OUTPUT"
"$PY" -m src.runtime.train_motion_student --dataset "$DATASET" --directory "$OUTPUT"
"$PY" -m src.runtime.mujoco_demo --dataset "$DATASET" --directory "$OUTPUT"
