"""Python/PyTorch CUDA adapters. Preprocessing is specified per checkpoint."""
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.safety.anomaly_detector import AnomalyScore
from vendor.motion_perception.src.models.accelsight_net import AccelSightNet


def device_for(name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; install a CUDA PyTorch wheel")
    return device


def preprocess(clip, config, device):
    if clip.dtype != np.uint8 or clip.ndim != 4 or clip.shape[-1] != 3:
        raise ValueError("Expected RGB uint8 clip [T,H,W,3]")
    if len(clip) != config["frames"]:
        raise ValueError("Clip length does not match checkpoint metadata")
    x = torch.from_numpy(np.ascontiguousarray(clip)).to(device=device, dtype=torch.float32)
    x = x.permute(0, 3, 1, 2) / 255.0
    x = F.interpolate(x, size=config["size"], mode="bilinear", align_corners=False)
    mean = x.new_tensor(config["mean"]).view(1, 3, 1, 1)
    std = x.new_tensor(config["std"]).view(1, 3, 1, 1)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid normalization metadata")
    return ((x - mean) / std).unsqueeze(0)


class MotionPerception:
    def __init__(self, checkpoint=None, device="cuda", smoke=False):
        self.device = device_for(device)
        self.output_keys = None
        if checkpoint is None:
            if not smoke:
                raise ValueError("A trained perception checkpoint is required")
            self.config = dict(frames=5, size=[128, 128], mean=[0, 0, 0], std=[1, 1, 1])
            self.model = AccelSightNet(num_frames=5)
        else:
            bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.config = bundle["preprocessing"]
            self.model = AccelSightNet(**bundle["model_kwargs"])
            self.model.load_state_dict(bundle["state_dict"], strict=True)
            self.output_keys = bundle.get("output_keys")
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, clip):
        x = preprocess(clip, self.config, self.device)
        outputs = self.model(x)
        if self.output_keys is not None:
            outputs = {key: outputs[key] for key in self.output_keys}
        if not all(torch.isfinite(value).all().item() for value in outputs.values()):
            raise ValueError("Perception returned nonfinite values")
        # CPU transfer completes GPU work before the runtime considers it finished.
        return {key: value.cpu().numpy() for key, value in outputs.items()}


class LocalStudent:
    """Trusted local TorchScript scalar anomaly regressor + JSON metadata.

    This loader does not claim arbitrary V-JEPA weights are a trained detector.
    Export the distilled student's forward pass including its scoring head.
    """
    def __init__(self, checkpoint, metadata, device="cuda"):
        self.device = device_for(device)
        self.config = json.loads(Path(metadata).read_text())
        if self.config["layout"] not in ("BTCHW", "BCTHW", "BCHW"):
            raise ValueError("Unsupported student layout")
        self.model = torch.jit.load(str(checkpoint), map_location=self.device).eval()

    @torch.inference_mode()
    def score(self, clip):
        start = time.perf_counter()
        x = preprocess(clip, self.config, self.device)
        layout = self.config["layout"]
        if layout == "BCTHW":
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        elif layout == "BCHW":
            x = x[:, -1]
        value = self.model(x)
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise ValueError("Student must return one scalar anomaly score")
        scalar = value.item()
        if not np.isfinite(scalar):
            raise ValueError("Student returned a nonfinite score")
        return AnomalyScore(scalar, (time.perf_counter() - start) * 1000, "student_model")


class VjepaRiskScorer:
    """Frozen pretrained distilled encoder with a MuJoCo-trained risk head."""
    def __init__(self, directory, device="cuda"):
        directory = Path(directory)
        self.device = device_for(device)
        self.config = json.loads((directory / "encoder.json").read_text())
        self.encoder = torch.jit.load(str(directory / "encoder.ts"), map_location=self.device).eval()
        self.head = torch.jit.load(str(directory / "risk_head.ts"), map_location=self.device).eval()
        self.threshold = json.loads((directory / "evaluation.json").read_text())["threshold"]
        if self.config["frames"] != 4 or self.config["size"] != [384, 384]:
            raise ValueError("Exported V-JEPA encoder requires fixed 4-frame 384px input")

    @torch.inference_mode()
    def score(self, clip):
        start = time.perf_counter()
        x = preprocess(clip, self.config, self.device)
        score = self.head(self.encoder(x)).sigmoid().item()
        if not np.isfinite(score):
            raise ValueError("Nonfinite V-JEPA risk score")
        return AnomalyScore(score, (time.perf_counter() - start) * 1000, "vjepa_simulation_risk_head")
