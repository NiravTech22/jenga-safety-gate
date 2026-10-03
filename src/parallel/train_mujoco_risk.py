"""Train and evaluate a risk head on cached pretrained V-JEPA features."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class RiskHead(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.linear = nn.Linear(len(mean), 1)

    def forward(self, x):
        return self.linear((x - self.mean) / self.std).flatten()


def metrics(scores, labels, threshold):
    pred, positive = scores >= threshold, labels.astype(bool)
    tp, fp = int((pred & positive).sum()), int((pred & ~positive).sum())
    fn, tn = int((~pred & positive).sum()), int((~pred & ~positive).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(precision=precision, recall=recall,
        f1=2 * precision * recall / max(precision + recall, 1e-9),
        accuracy=(tp + tn) / len(labels), tp=tp, fp=fp, fn=fn, tn=tn,
        n=len(labels), normal_pass_rate=tn / max(tn + fp, 1))


def train(directory, epochs=200):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    directory = Path(directory)
    with np.load(directory / "features.npz", allow_pickle=False) as data:
        features, labels, split, seeds = [data[k] for k in ("features", "labels", "split", "seeds")]
    if (not np.isfinite(features).all() or not np.isfinite(labels).all()
            or set(labels.tolist()) != {0., 1.} or len(set(seeds.tolist())) != len(seeds)):
        raise ValueError("Invalid features, labels or duplicate episode seeds")
    ids = {s: np.flatnonzero(split == s) for s in ("train", "validation", "test")}
    if any(set(labels[ix].tolist()) != {0., 1.} for ix in ids.values()):
        raise ValueError("Every split needs both successful and failed rollouts")
    x, y = torch.tensor(features, device="cuda"), torch.tensor(labels, device="cuda")
    train_x = x[ids["train"]]
    head = RiskHead(train_x.mean(0), train_x.std(0).clamp_min(.05)).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=.005, weight_decay=.1)
    best, history = float("inf"), []
    for epoch in range(epochs):
        head.train()
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(head(train_x), y[ids["train"]])
        loss.backward()
        opt.step()
        head.eval()
        with torch.inference_mode():
            val = nn.functional.binary_cross_entropy_with_logits(head(x[ids["validation"]]), y[ids["validation"]]).item()
        if not np.isfinite(val):
            raise RuntimeError("Nonfinite validation loss")
        history.append(val)
        if val < best:
            best = val
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    head.eval()
    with torch.inference_mode():
        scores = head(x).sigmoid().cpu().numpy()
    val_ids = ids["validation"]
    # Tune only on validation; reward recall subject to useful normal passes.
    candidates = [(float(t), metrics(scores[val_ids], labels[val_ids], t)) for t in np.linspace(.05, .95, 91)]
    eligible = [(t, m) for t, m in candidates if m["recall"] >= .8 and m["normal_pass_rate"] >= .5]
    threshold, val_metrics = max(eligible or candidates, key=lambda pair: (pair[1]["f1"], pair[1]["recall"]))
    test_metrics = metrics(scores[ids["test"]], labels[ids["test"]], threshold)
    scripted = torch.jit.script(head)
    scripted.save(str(directory / "risk_head.ts"))
    report = dict(backend="mujoco", encoder="pretrained distilled V-JEPA 2.1 ViT-B (frozen)",
        threshold=threshold, threshold_selection="validation only", epochs=epochs,
        best_validation_loss=best, validation=val_metrics, test=test_metrics,
        majority_baseline_test_accuracy=float(max(labels[ids["test"]].mean(), 1-labels[ids["test"]].mean())),
        simulation_bar_met=bool(test_metrics["precision"] >= .6 and test_metrics["recall"] >= .8
                                and test_metrics["normal_pass_rate"] >= .5),
        hardware_validated=False, train_episodes=len(ids["train"]))
    (directory / "evaluation.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", default="checkpoints/mujoco_vjepa")
    p.add_argument("--epochs", type=int, default=200)
    args = p.parse_args()
    if args.epochs <= 0:
        p.error("epochs must be positive")
    print(json.dumps(train(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
