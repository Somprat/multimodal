#!/usr/bin/env python3
"""Smoke-test Variant B: frozen RGB plus trainable spatial features."""

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vla.load import load_vla
from vla.spatial.encoder import PointCloudSpatialEncoder
from vla.spatial.geometry import depth_to_points, transform_points


def spatial_vla(checkpoint, episode_path, frames=8):
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
        frame_count = min(frames, len(data["rgb"]))
        rgb_frames = data["rgb"][:frame_count]
        depth = torch.from_numpy(data["depth"][:frame_count]).float().to(device)
        intrinsics = torch.from_numpy(
            data["camera_intrinsics"][:frame_count]
        ).float().to(device)
        extrinsics = torch.from_numpy(
            data["camera_extrinsics"][:frame_count]
        ).float().to(device)

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

    points, point_mask = depth_to_points(depth=depth, intrinsics=intrinsics)
    points_world = transform_points(points, torch.linalg.inv(extrinsics))
    spatial_encoder = PointCloudSpatialEncoder(
        spatial_token_size=256,
        num_spatial_tokens=16,
        hidden_dim=128,
        num_heads=4,
        max_points=1024,
    ).to(device)
    spatial_tokens = spatial_encoder(points=points_world, point_mask=point_mask)
    spatial_features = spatial_tokens.mean(dim=1)
    combined_features = torch.cat([rgb_features.detach(), spatial_features], dim=-1)
    return rgb_features, spatial_tokens, spatial_features, combined_features


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

    rgb, tokens, spatial, combined = spatial_vla(
        args.checkpoint, args.episode, args.frames
    )
    print("RGB features:", tuple(rgb.shape))
    print("spatial tokens:", tuple(tokens.shape))
    print("spatial features:", tuple(spatial.shape))
    print("combined features:", tuple(combined.shape))
    print("finite:", torch.isfinite(combined).all().item())


if __name__ == "__main__":
    main()
