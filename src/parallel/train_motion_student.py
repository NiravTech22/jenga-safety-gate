"""Distill AccelSight descriptors from pretrained V-JEPA; supervise tower velocity.

Only descriptor and pooled velocity are trained. Detection, boxes, and controls
are deliberately excluded from the exported runtime output contract.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.parallel.models import preprocess
from vendor.motion_perception.src.models.accelsight_net import AccelSightNet


def train(dataset="data/mujoco_jenga", directory="checkpoints/mujoco_vjepa", epochs=30):
    torch.manual_seed(7)
    torch.set_num_threads(4)
    dataset, directory = Path(dataset), Path(directory)
    manifest = json.loads((dataset / "manifest.json").read_text())
    with np.load(directory / "features.npz", allow_pickle=False) as cached:
        targets = torch.tensor(cached["features"], device="cuda")
        seeds = cached["seeds"]
    if not np.array_equal(seeds, [r["seed"] for r in manifest["records"]]):
        raise ValueError("Teacher features do not match dataset episode order")
    config = dict(frames=4, size=[96, 96], mean=[0, 0, 0], std=[1, 1, 1])
    images, velocities = [], []
    for record in manifest["records"]:
        with np.load(dataset / record["file"], allow_pickle=False) as sample:
            images.append(preprocess(sample["clip"], config, "cpu"))
            velocities.append(sample["velocity"])
    x = torch.cat(images).cuda()
    y = torch.tensor(np.stack(velocities), device="cuda")
    targets = F.normalize(targets, dim=1)
    ids = {s: torch.tensor([i for i, r in enumerate(manifest["records"]) if r["split"] == s], device="cuda")
           for s in ("train", "validation", "test")}
    kwargs = dict(num_frames=4, base_channels=8, embedding_dim=targets.shape[1])
    model = AccelSightNet(**kwargs).cuda()
    # Heads without supervision must not be optimized or exposed as predictions.
    for head in (model.det_head, model.control_head):
        head.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    best = float("inf")

    def loss_on(indices):
        out = model(x[indices])
        embedding = F.normalize(out["embedding"].mean((2, 3)), dim=1)
        velocity = out["velocity"].mean((2, 3))
        loss = (1 - (embedding * targets[indices]).sum(1)).mean() + 20 * F.mse_loss(velocity, y[indices])
        return loss, velocity

    for epoch in range(epochs):
        model.train()
        order = ids["train"][torch.randperm(len(ids["train"]), device="cuda")]
        for batch in order.split(4):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = loss_on(batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite motion distillation loss")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val = sum(loss_on(b)[0].item() * len(b) for b in ids["validation"].split(4)) / len(ids["validation"])
        if val < best:
            best = val
            torch.save(dict(state_dict=model.state_dict(), model_kwargs=kwargs, preprocessing=config,
                output_keys=["embedding", "velocity"], trained_heads=["embedding", "velocity"],
                provenance="distilled V-JEPA 2.1 teacher descriptors + MuJoCo top-block mean velocity"), directory / "motion.pt")
        if epoch % 5 == 0:
            print(json.dumps(dict(epoch=epoch + 1, validation_loss=val)), flush=True)
    model.load_state_dict(torch.load(directory / "motion.pt", weights_only=True)["state_dict"])
    model.eval()
    with torch.inference_mode():
        predicted = torch.cat([loss_on(b)[1] for b in ids["test"].split(4)])
        truth = y[ids["test"]]
        rmse = (predicted - truth).square().mean().sqrt().item()
        baseline = (y[ids["train"]].mean(0) - truth).square().mean().sqrt().item()
    report = dict(test_velocity_rmse_m_s=rmse, constant_baseline_rmse_m_s=baseline,
                  trained_outputs=["embedding", "velocity"], hardware_validated=False)
    (directory / "motion_evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="data/mujoco_jenga")
    p.add_argument("--directory", default="checkpoints/mujoco_vjepa")
    p.add_argument("--epochs", type=int, default=30)
    args = p.parse_args()
    if args.epochs <= 0:
        p.error("epochs must be positive")
    train(**vars(args))
