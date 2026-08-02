#!/usr/bin/env python3
"""Smoke-test Variant C: point-cloud spatial features without RGB features."""

import argparse
from pathlib import Path

import numpy as np
import torch

from vla.spatial.encoder import PointCloudSpatialEncoder
from vla.spatial.geometry import depth_to_points, transform_points


def spatial(episode, frames=8):

    with np.load(episode, allow_pickle=False) as data:
        frame_count = min(frames, len(data["depth"]))
        # no rgb
        depth = torch.from_numpy(data["depth"][:frame_count]).float()
        intrinsics = torch.from_numpy(
            data["camera_intrinsics"][:frame_count]
        ).float()
        extrinsics = torch.from_numpy(
            data["camera_extrinsics"][:frame_count]
        ).float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth = depth.to(device)
    intrinsics = intrinsics.to(device)
    extrinsics = extrinsics.to(device)

    # depth_to_points produces OpenCV camera-frame XYZ coordinates.
    points_camera, point_mask = depth_to_points(
        depth=depth,
        intrinsics=intrinsics,
    )

    # SimplerEnv's extrinsic_cv maps world coordinates to camera coordinates.
    camera_to_world = torch.linalg.inv(extrinsics)
    points_world = transform_points(points_camera, camera_to_world)

    spatial_encoder = PointCloudSpatialEncoder(
        spatial_token_size=256,
        num_spatial_tokens=16,
        hidden_dim=128,
        num_heads=4,
        max_points=1024,
    ).to(device)

    with torch.no_grad():
        spatial_tokens = spatial_encoder(
            points=points_world,
            point_mask=point_mask,
        )
        spatial_features = spatial_tokens.mean(dim=1)

    assert spatial_tokens.shape == (frame_count, 16, 256)
    assert spatial_features.shape == (frame_count, 256)
    assert torch.isfinite(spatial_features).all()

    # print(f"device: {device}")
    # print(f"camera points: {tuple(points_camera.shape)}")
    # print(f"valid points: {int(point_mask.sum())}")
    # print(f"world points: {tuple(points_world.shape)}")
    # print(f"spatial tokens: {tuple(spatial_tokens.shape)}")
    # print(f"spatial-only probe input: {tuple(spatial_features.shape)}")
    # print("finite: True")
    return spatial_tokens, spatial_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    tokens, features = spatial(args.episode, args.frames)
    print("spatial tokens:", tuple(tokens.shape))
    print("spatial-only probe input:", tuple(features.shape))
    print("finite:", torch.isfinite(features).all().item())
