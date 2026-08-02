#!/usr/bin/env python3
"""Smoke-test Variant A: frozen RGB features."""

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vla.load import load_vla


def pure_vla(checkpoint, episode_path, frames=8):
    vla = load_vla(
        model_id_or_path=checkpoint,
        load_for_training=False,
        use_bf16=True,
    )
    vision = vla.vlm.vision_backbone
    image_transform = vision.image_transform
    vision.requires_grad_(False)
    vision.eval()
    del vla
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision = vision.to(device)
    dtype = next(vision.parameters()).dtype

    with np.load(episode_path, allow_pickle=False) as data:
        rgb_frames = data["rgb"][:frames]

    transformed = [image_transform(Image.fromarray(rgb)) for rgb in rgb_frames]
    pixel_values = {
        name: torch.stack([frame[name] for frame in transformed]).to(
            device=device, dtype=dtype
        )
        for name in transformed[0]
    }

    with torch.inference_mode():
        patch_tokens = vision(pixel_values)
        rgb_features = patch_tokens.mean(dim=1).float()
    return patch_tokens, rgb_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/workspace/multimodal/models/model_b/checkpoints/memvla-bridge.pt"),
    )
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()

    patch_tokens, rgb_features = pure_vla(args.checkpoint, args.episode, args.frames)
    print("patch tokens:", tuple(patch_tokens.shape))
    print("pooled RGB features:", tuple(rgb_features.shape))
    print("finite:", torch.isfinite(rgb_features).all().item())


if __name__ == "__main__":
    main()
