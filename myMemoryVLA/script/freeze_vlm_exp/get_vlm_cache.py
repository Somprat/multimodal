#!/usr/bin/env python3
"""Cache frozen VLM vision features for every recorded probe episode."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from probe_splits import train_test_val_split
from vla.load import load_vla


def transform_batch(rgb_frames, image_transform, device, dtype):
    transformed = [image_transform(Image.fromarray(rgb)) for rgb in rgb_frames]
    if isinstance(transformed[0], dict):
        return {
            name: torch.stack([frame[name] for frame in transformed]).to(
                device=device, dtype=dtype
            )
            for name in transformed[0]
        }
    return torch.stack(transformed).to(device=device, dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, help="Directory containing probe NPZ episodes")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/workspace/multimodal/models/model_b/checkpoints/memvla-bridge.pt"),
    )
    
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    cache_dir = (args.cache_dir or data_dir / "feature_cache").resolve()
    splits_path = (args.splits or data_dir / "splits.json").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    splits = train_test_val_split(data_dir, random_seed=args.seed)
    splits_path.write_text(json.dumps(splits, indent=2) + "\n")
    episode_names = [
        name
        for split_name in ("train", "validation", "test")
        for name in splits[split_name]
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vla = load_vla(
        model_id_or_path=args.checkpoint,
        load_for_training=False,
        use_bf16=True,
    )
    vision = vla.vlm.vision_backbone
    image_transform = vision.image_transform
    vision.requires_grad_(False)
    vision.eval()
    del vla
    gc.collect()

    vision = vision.to(device)
    dtype = next(vision.parameters()).dtype

    for episode_name in episode_names:
        episode_path = data_dir / episode_name
        cache_path = cache_dir / f"{episode_path.stem}.pt"
        if cache_path.exists() and not args.overwrite:
            print(f"Skipping existing cache: {cache_path.name}")
            continue
    
        with np.load(episode_path, allow_pickle=False) as data:
            rgb_frames = data["rgb"].copy()

        feature_batches = []
        with torch.inference_mode():
            for start in range(0, len(rgb_frames), args.batch_size):
                batch = transform_batch(
                    rgb_frames[start:start + args.batch_size],
                    image_transform,
                    device,
                    dtype,
                )
                patch_tokens = vision(batch)
                feature_batches.append(patch_tokens.mean(dim=1).float().cpu())

        rgb_features = torch.cat(feature_batches, dim=0)
        if len(rgb_features) != len(rgb_frames):
            raise RuntimeError(f"Frame/cache mismatch for {episode_name}")
        torch.save(
            {"episode": episode_name, "rgb_features": rgb_features},
            cache_path,
        )
        print(f"Cached {episode_name}: {tuple(rgb_features.shape)}")

    print(f"Splits: {splits_path}")
    print(f"Feature cache: {cache_dir}")


if __name__ == "__main__":
    main()
