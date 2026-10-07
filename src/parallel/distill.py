"""Train a compact scalar student from cached V-JEPA prediction-error targets.

No teacher is downloaded. Cache generation must use a real pretrained teacher
and record its checkpoint and masking/error definition. Synthetic targets only
exercise the training code; they do not produce a validated safety detector.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.parallel.models import device_for, preprocess


class VideoStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 12, 3, padding=1, stride=(1, 2, 2)), nn.ReLU(),
            nn.Conv3d(12, 24, 3, padding=1, stride=(1, 2, 2)), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(24, 1), nn.Softplus())

    def forward(self, x):
        return self.features(x.permute(0, 2, 1, 3, 4)).flatten()


def load_cache(path):
    with np.load(path, allow_pickle=False) as data:
        clips, scores = data["clips"], data["scores"]
        episodes, split = data["episode_ids"], data["split"]
    if (clips.dtype != np.uint8 or clips.ndim != 5 or clips.shape[-1] != 3
            or min(clips.shape) <= 0 or scores.shape != (len(clips),)
            or episodes.shape != scores.shape or split.shape != scores.shape
            or not np.isfinite(scores).all() or (scores < 0).any()):
        raise ValueError("Invalid teacher cache")
    if set(split.tolist()) != {"train", "validation"}:
        raise ValueError("Cache needs train and validation splits")
    train, val = np.flatnonzero(split == "train"), np.flatnonzero(split == "validation")
    if set(episodes[train].tolist()) & set(episodes[val].tolist()):
        raise ValueError("Episode leakage between train and validation")
    return clips, scores.astype(np.float32), train, val


def train(cache, provenance, output, device="cuda", epochs=20, batch_size=4, size=128):
    if epochs <= 0 or batch_size <= 0 or size <= 0:
        raise ValueError("epochs, batch_size and size must be positive")
    info = json.loads(Path(provenance).read_text())
    for key in ("teacher_checkpoint", "score_definition", "dataset_id"):
        if not isinstance(info.get(key), str) or not info[key].strip():
            raise ValueError(f"Missing provenance: {key}")
    clips, scores, train_ids, val_ids = load_cache(cache)
    device = device_for(device)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    config = dict(frames=clips.shape[1], size=[size, size], mean=[0, 0, 0],
                  std=[1, 1, 1], layout="BTCHW")
    model = VideoStudent().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []

    def batch(ids):
        x = torch.cat([preprocess(clips[i], config, device) for i in ids])
        y = torch.as_tensor(scores[ids], device=device)
        return x, y

    for epoch in range(epochs):
        model.train()
        ids = rng.permutation(train_ids)
        for start in range(0, len(ids), batch_size):
            x, y = batch(ids[start:start + batch_size])
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
        model.eval()
        total = 0.0
        with torch.inference_mode():
            for start in range(0, len(val_ids), batch_size):
                x, y = batch(val_ids[start:start + batch_size])
                total += nn.functional.mse_loss(model(x), y, reduction="sum").item()
        mse = total / len(val_ids)
        if not np.isfinite(mse):
            raise ValueError("Nonfinite validation loss")
        history.append(dict(epoch=epoch + 1, validation_mse=mse))
        if mse < best:
            best = mse
            torch.jit.script(model).save(str(output / "student.ts"))
    (output / "student.json").write_text(json.dumps(config, indent=2))
    report = dict(provenance=info, history=history, best_validation_mse=best,
                  device=str(device), safety_validated=False,
                  note="Teacher-score regression only; validate against held-out risk labels before gating.")
    (output / "training.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(train(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
