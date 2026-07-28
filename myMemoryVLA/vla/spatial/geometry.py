from __future__ import annotations

from typing import Mapping, Optional, Tuple, Union

import torch

Bounds = Union[Tuple[float, float, float, float, float, float], torch.Tensor]


def make_pixel_grid(
    height: int,
    width: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pixel-coordinate grids shaped [H, W] as (u, v)."""

    v, u = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return u, v

# intrinsic + depth = points
# can put points in PointCloudSpatialEncoder
def parse_intrinsics(
    intrinsics: Union[Mapping[str, torch.Tensor], torch.Tensor],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parse camera intrinsics from a dict or [B, 3, 3] camera matrix."""

    if isinstance(intrinsics, Mapping):
        fx = _as_batch_vector(intrinsics["fx"], batch_size, device, dtype)
        fy = _as_batch_vector(intrinsics["fy"], batch_size, device, dtype)
        cx = _as_batch_vector(intrinsics["cx"], batch_size, device, dtype)
        cy = _as_batch_vector(intrinsics["cy"], batch_size, device, dtype)
        return fx, fy, cx, cy

    if not torch.is_tensor(intrinsics):
        raise TypeError("intrinsics must be a mapping with fx/fy/cx/cy or a tensor shaped [B, 3, 3]")

    intrinsics = intrinsics.to(device=device, dtype=dtype)
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0).expand(batch_size, -1, -1)
    if intrinsics.shape != (batch_size, 3, 3):
        raise ValueError(f"intrinsics must be shaped [B, 3, 3], got {tuple(intrinsics.shape)}")

    return intrinsics[:, 0, 0], intrinsics[:, 1, 1], intrinsics[:, 0, 2], intrinsics[:, 1, 2]


def depth_to_points(
    depth: torch.Tensor,
    intrinsics: Union[Mapping[str, torch.Tensor], torch.Tensor],
    mask: Optional[torch.Tensor] = None,
    flatten: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert depth images to camera-frame point clouds.

    Args:
        depth: Depth tensor shaped [B, H, W], [B, 1, H, W], or [B, H, W, 1].
        intrinsics: Dict with fx/fy/cx/cy or camera matrix shaped [B, 3, 3].
        mask: Optional valid-pixel mask shaped [B, H, W].
        flatten: If True, return points as [B, H * W, 3].

    Returns:
        points and valid_mask. Flattened outputs are [B, N, 3] and [B, N].
    """

    depth = _normalize_depth(depth)
    B, H, W = depth.shape
    device, dtype = depth.device, depth.dtype

    fx, fy, cx, cy = parse_intrinsics(intrinsics, B, device, dtype)
    u, v = make_pixel_grid(H, W, device=device, dtype=dtype)

    z = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    x = (u.unsqueeze(0) - cx[:, None, None]) * z / fx[:, None, None].clamp_min(1e-6)
    y = (v.unsqueeze(0) - cy[:, None, None]) * z / fy[:, None, None].clamp_min(1e-6)

    points = torch.stack([x, y, z], dim=-1)
    valid_mask = torch.isfinite(depth) & (depth > 0)

    if mask is not None:
        valid_mask = valid_mask & _normalize_mask(mask, B, H, W, device)

    points = torch.where(valid_mask[..., None], points, torch.zeros_like(points))

    if not flatten:
        return points, valid_mask

    return points.reshape(B, H * W, 3), valid_mask.reshape(B, H * W)


def transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Apply [B, 4, 4] transforms to points shaped [B, N, 3]."""

    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must be shaped [B, N, 3], got {tuple(points.shape)}")

    B, N, _ = points.shape
    transform = transform.to(device=points.device, dtype=points.dtype)
    if transform.ndim == 2:
        transform = transform.unsqueeze(0).expand(B, -1, -1)
    if transform.shape != (B, 4, 4):
        raise ValueError(f"transform must be shaped [B, 4, 4], got {tuple(transform.shape)}")

    ones = torch.ones(B, N, 1, device=points.device, dtype=points.dtype)
    points_h = torch.cat([points, ones], dim=-1)
    transformed = torch.matmul(points_h, transform.transpose(1, 2))
    return transformed[..., :3]


def crop_points(points: torch.Tensor, bounds: Bounds) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero points outside xyz bounds and return the boolean crop mask."""

    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must be shaped [B, N, 3], got {tuple(points.shape)}")

    bounds = _bounds_tensor(bounds, points.device, points.dtype)
    mins = bounds[:3]
    maxs = bounds[3:]
    crop_mask = ((points >= mins) & (points <= maxs)).all(dim=-1)
    cropped = torch.where(crop_mask[..., None], points, torch.zeros_like(points))
    return cropped, crop_mask


def normalize_points(
    points: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    center: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Center and scale points per batch item."""

    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must be shaped [B, N, 3], got {tuple(points.shape)}")

    B, N, _ = points.shape
    if mask is None:
        mask = torch.ones(B, N, device=points.device, dtype=torch.bool)
    else:
        mask = mask.to(device=points.device).bool()

    weights = mask.to(dtype=points.dtype).unsqueeze(-1)

    if center is None:
        denom = weights.sum(dim=1).clamp_min(1.0)
        center = (points * weights).sum(dim=1) / denom
    else:
        center = center.to(device=points.device, dtype=points.dtype)

    centered = points - center[:, None, :]

    if scale is None:
        distances = torch.linalg.norm(centered, dim=-1)
        distances = torch.where(mask, distances, torch.zeros_like(distances))
        scale = distances.max(dim=1).values.clamp_min(eps)
    else:
        scale = scale.to(device=points.device, dtype=points.dtype).reshape(B).clamp_min(eps)

    normalized = centered / scale[:, None, None]
    normalized = torch.where(mask[..., None], normalized, torch.zeros_like(normalized))
    return normalized, center, scale


def voxelize_points(
    points: torch.Tensor,
    voxel_size: float,
    bounds: Bounds,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert points to a binary occupancy grid.

    Returns:
        voxel_grid: [B, X, Y, Z]
        voxel_indices: [B, N, 3], with invalid points set to -1
    """

    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must be shaped [B, N, 3], got {tuple(points.shape)}")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")

    B, N, _ = points.shape
    bounds = _bounds_tensor(bounds, points.device, points.dtype)
    mins = bounds[:3]
    maxs = bounds[3:]
    grid_size = torch.ceil((maxs - mins) / voxel_size).long()

    if (grid_size <= 0).any():
        raise ValueError("bounds must define a positive volume")

    voxel_float = torch.floor((points - mins) / voxel_size)
    voxel_indices = voxel_float.long()
    valid = ((voxel_indices >= 0) & (voxel_indices < grid_size)).all(dim=-1)
    if mask is not None:
        valid = valid & mask.to(device=points.device).bool()

    voxel_indices = torch.where(valid[..., None], voxel_indices, torch.full_like(voxel_indices, -1))
    voxel_grid = torch.zeros(
        B,
        int(grid_size[0].item()),
        int(grid_size[1].item()),
        int(grid_size[2].item()),
        device=points.device,
        dtype=torch.bool,
    )

    batch_idx, point_idx = valid.nonzero(as_tuple=True)
    if batch_idx.numel() > 0:
        idx = voxel_indices[batch_idx, point_idx]
        voxel_grid[batch_idx, idx[:, 0], idx[:, 1], idx[:, 2]] = True

    return voxel_grid, voxel_indices


def _normalize_depth(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]
    elif depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    if depth.ndim != 3:
        raise ValueError(f"depth must be [B, H, W], [B, 1, H, W], or [B, H, W, 1], got {tuple(depth.shape)}")

    return depth


def _normalize_mask(mask: torch.Tensor, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    elif mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]

    if mask.shape != (batch_size, height, width):
        raise ValueError(f"mask must be shaped [B, H, W], got {tuple(mask.shape)}")

    return mask.to(device=device).bool()


def _as_batch_vector(value: Union[float, int, torch.Tensor], batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    value = torch.as_tensor(value, device=device, dtype=dtype)
    if value.ndim == 0:
        value = value.expand(batch_size)
    if value.shape != (batch_size,):
        raise ValueError(f"intrinsic values must be scalar or shaped [B], got {tuple(value.shape)}")
    return value


def _bounds_tensor(bounds: Bounds, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    bounds = torch.as_tensor(bounds, device=device, dtype=dtype)
    if bounds.shape != (6,):
        raise ValueError(f"bounds must be (x_min, y_min, z_min, x_max, y_max, z_max), got {tuple(bounds.shape)}")
    return bounds
