from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PointCloudSpatialEncoder(nn.Module):
    """Encode point-cloud observations into a fixed set of spatial tokens."""

    def __init__(
        self,
        spatial_token_size: int,
        num_spatial_tokens: int,
        point_dim: int = 3,
        proprio_dim: Optional[int] = None,
        camera_dim: Optional[int] = None,
        hidden_dim: int = 128,
        num_heads: int = 4,
        max_points: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.spatial_token_size = spatial_token_size
        self.num_spatial_tokens = num_spatial_tokens
        self.max_points = max_points

        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.proprio_proj = (
            nn.Sequential(
                nn.Linear(proprio_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if proprio_dim is not None
            else None
        )
        self.camera_proj = (
            nn.Sequential(
                nn.Linear(camera_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if camera_dim is not None
            else None
        )

        self.query_tokens = nn.Parameter(torch.randn(num_spatial_tokens, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, spatial_token_size)

    def forward(
        self,
        points: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        camera: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            points: Point cloud tensor shaped [B, N, point_dim].
            proprio: Optional proprioception vector shaped [B, proprio_dim].
            camera: Optional camera metadata vector shaped [B, camera_dim].
            point_mask: Optional boolean mask shaped [B, N], where True marks valid points.

        Returns:
            Spatial tokens shaped [B, num_spatial_tokens, spatial_token_size].
        """

        if points.ndim != 3:
            raise ValueError(f"points must be shaped [B, N, point_dim], got {tuple(points.shape)}")

        points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
        points, point_mask = self._maybe_subsample(points, point_mask)

        point_tokens = self.point_mlp(points)

        context = self._encode_context(
            batch_size=points.shape[0],
            device=points.device,
            dtype=point_tokens.dtype,
            proprio=proprio,
            camera=camera,
        )
        if context is not None:
            point_tokens = point_tokens + context[:, None, :]

        queries = self.query_tokens[None, :, :].expand(points.shape[0], -1, -1)
        key_padding_mask = None if point_mask is None else ~point_mask.bool()

        spatial_tokens, _ = self.cross_attn(
            query=queries,
            key=point_tokens,
            value=point_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        spatial_tokens = self.output_norm(spatial_tokens)
        return self.output_proj(spatial_tokens)

    def _maybe_subsample(
        self,
        points: torch.Tensor,
        point_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.max_points is None or points.shape[1] <= self.max_points:
            return points, point_mask

        # Deterministic stride sampling keeps this module lightweight and reproducible.
        indices = torch.linspace(
            0,
            points.shape[1] - 1,
            steps=self.max_points,
            device=points.device,
        ).long()
        points = points.index_select(dim=1, index=indices)
        if point_mask is not None:
            point_mask = point_mask.index_select(dim=1, index=indices)
        return points, point_mask

    def _encode_context(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        proprio: Optional[torch.Tensor],
        camera: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        context = None

        if self.proprio_proj is not None and proprio is not None:
            context = self.proprio_proj(proprio.to(device=device, dtype=dtype))

        if self.camera_proj is not None and camera is not None:
            camera_context = self.camera_proj(camera.to(device=device, dtype=dtype))
            context = camera_context if context is None else context + camera_context

        if context is not None and context.shape[0] != batch_size:
            raise ValueError(f"context batch size must be {batch_size}, got {context.shape[0]}")

        return context
