"""Isolated upstream V-JEPA importer, avoiding its top-level src name collision.

Run directly with python scripts/extract_vjepa_features.py, not as a module.
Loads the official distilled ViT-B checkpoint strictly, then exports a fixed
4-frame encoder and caches features for supervised simulation risk training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F


class PooledEncoder(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x.permute(0, 2, 1, 3, 4)).mean(dim=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", default="data/assets/vjepa2")
    p.add_argument("--checkpoint", default="checkpoints/pretrained/vjepa2_1_vitb_dist_vitG_384.pt")
    p.add_argument("--dataset", default="data/mujoco_jenga")
    p.add_argument("--output", default="checkpoints/mujoco_vjepa")
    args = p.parse_args()
    torch.set_num_threads(4)
    upstream = Path(args.upstream).resolve()
    sys.path.insert(0, str(upstream))
    from app.vjepa_2_1.models.vision_transformer import vit_base
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this local recipe")
    encoder = vit_base(img_size=(384, 384), patch_size=16, num_frames=64,
        tubelet_size=2, use_sdpa=True, use_SiLU=False, wide_SiLU=True,
        uniform_power=False, use_rope=True, img_temporal_dim_size=1,
        interpolate_rope=True)
    checkpoint = Path(args.checkpoint)
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {k.removeprefix("module.").removeprefix("backbone."): v for k, v in bundle["ema_encoder"].items()}
    encoder.load_state_dict(state, strict=True)
    del bundle, state
    model = PooledEncoder(encoder).eval().cuda()
    model.requires_grad_(False)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(frames=4, size=[384, 384], mean=[.485, .456, .406],
                  std=[.229, .224, .225], layout="BTCHW")
    dataset = Path(args.dataset)
    manifest = json.loads((dataset / "manifest.json").read_text())
    features = []
    with torch.inference_mode():
        for i, record in enumerate(manifest["records"]):
            with np.load(dataset / record["file"], allow_pickle=False) as sample:
                x = torch.from_numpy(sample["clip"].copy()).cuda().float().permute(0, 3, 1, 2) / 255.
            x = F.interpolate(x, (384, 384), mode="bilinear", align_corners=False)
            x = ((x - x.new_tensor(config["mean"])[:, None, None]) / x.new_tensor(config["std"])[:, None, None]).unsqueeze(0)
            feature = model(x)
            if not torch.isfinite(feature).all():
                raise ValueError("Nonfinite pretrained feature")
            features.append(feature.cpu().numpy()[0])
            if i == 0:
                # Fixed input contract; trace only the frozen feature encoder.
                traced = torch.jit.trace(model, x, check_trace=False)
                if not torch.allclose(feature, traced(x), atol=1e-4, rtol=1e-4):
                    raise RuntimeError("Encoder export mismatch")
                traced.save(str(output / "encoder.ts"))
                del traced
            if i % 10 == 0:
                print(json.dumps(dict(extracted=i + 1, total=len(manifest["records"]),
                    cuda_peak_mb=torch.cuda.max_memory_allocated() / 2**20)), flush=True)
    np.savez_compressed(output / "features.npz", features=np.stack(features),
        labels=np.array([r["risky"] for r in manifest["records"]], np.float32),
        split=np.array([r["split"] for r in manifest["records"]]),
        seeds=np.array([r["seed"] for r in manifest["records"]]))
    (output / "encoder.json").write_text(json.dumps(config, indent=2))
    with checkpoint.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    (output / "provenance.json").write_text(json.dumps(dict(
        checkpoint=str(checkpoint), sha256=digest, checkpoint_key="ema_encoder",
        upstream_revision=subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip(),
        source_url="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt",
        strict_load=True, frozen_encoder=True, clip_frames=4,
        note="Short-clip adaptation of a pretrained distilled video encoder; risk labels come from MuJoCo, not JEPA prediction error.",
        dataset=str(dataset.resolve())), indent=2))


if __name__ == "__main__":
    main()
