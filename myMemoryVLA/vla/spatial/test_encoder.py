from __future__ import annotations

from copy import deepcopy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def _load_encoder_module():
    module_path = Path(__file__).with_name("encoder.py")
    spec = importlib.util.spec_from_file_location("spatial_encoder_test_module", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


encoder_module = _load_encoder_module()


def _make_encoder():
    return encoder_module.PointCloudSpatialEncoder(
        spatial_token_size=32,
        num_spatial_tokens=4,
        point_dim=3,
        hidden_dim=16,
        num_heads=4,
        max_points=32,
    )


def test_point_cloud_encoder_parameters_receive_gradients():
    encoder = _make_encoder()
    points = torch.randn(2, 48, 3)
    point_mask = torch.ones(2, 48, dtype=torch.bool)

    output = encoder(points=points, point_mask=point_mask)
    output.square().mean().backward()

    assert encoder.query_tokens.grad is not None
    assert torch.count_nonzero(encoder.query_tokens.grad).item() > 0
    point_weight = encoder.point_mlp[0].weight
    assert point_weight.grad is not None
    assert torch.count_nonzero(point_weight.grad).item() > 0


def test_point_cloud_encoder_state_dict_round_trip():
    torch.manual_seed(7)
    source = _make_encoder().eval()
    points = torch.randn(2, 24, 3)

    with torch.no_grad():
        expected = source(points=points)

    restored = _make_encoder().eval()
    restored.load_state_dict(deepcopy(source.state_dict()), strict=True)

    with torch.no_grad():
        actual = restored(points=points)

    torch.testing.assert_close(actual, expected)


def test_point_cloud_encoder_rejects_all_invalid_depth_points():
    encoder = _make_encoder()
    points = torch.zeros(1, 8, 3)
    point_mask = torch.zeros(1, 8, dtype=torch.bool)

    with pytest.raises(ValueError, match="no valid depth points"):
        encoder(points=points, point_mask=point_mask)
