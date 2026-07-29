#!/usr/bin/env python3
"""Print raw observation fields from one step of a local RLDS TFDS dataset."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds


def describe(value: object) -> str:
    if isinstance(value, tf.Tensor):
        return f"shape={tuple(value.shape)} dtype={value.dtype.name}"
    return f"type={type(value).__name__}"


def flatten_fields(value: object, prefix: str = ""):
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}/{key}" if prefix else str(key)
            yield from flatten_fields(child, path)
    else:
        yield prefix, value


def candidate_kind(path: str, value: object) -> str | None:
    name = path.lower()
    shape = tuple(value.shape) if isinstance(value, tf.Tensor) else ()

    if any(token in name for token in ("intrinsic", "camera_matrix", "calibration")):
        return "intrinsics"
    if shape[-2:] == (3, 3):
        return "possible intrinsics (3x3 tensor)"
    if "depth" in name or "range_image" in name:
        return "depth"
    return None


def first_step(episode: Mapping[str, object]) -> Mapping[str, object]:
    steps = episode.get("steps")
    if steps is None:
        # Some exported datasets are already step-level rather than episodic.
        return episode
    if not isinstance(steps, tf.data.Dataset):
        raise TypeError(f"episode['steps'] is {type(steps).__name__}, expected tf.data.Dataset")
    return next(iter(steps.take(1)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect one raw RLDS observation and flag likely depth/intrinsics keys."
    )
    parser.add_argument("dataset_dir", type=Path, help="TFDS version directory, e.g. dataset/1.0.0")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        parser.error(f"dataset directory does not exist: {dataset_dir}")

    builder = tfds.builder_from_directory(str(dataset_dir))
    dataset = builder.as_dataset(split=args.split, shuffle_files=False)
    episode = next(iter(dataset.take(1)))
    step = first_step(episode)

    observation = step.get("observation")
    if not isinstance(observation, Mapping):
        raise KeyError("sample has no mapping-valued 'observation' field")

    print(f"dataset: {builder.info.full_name}")
    print(f"split: {args.split}")
    print("\nObservation fields:")
    candidates = []
    for path, value in flatten_fields(observation):
        print(f"  {path}: {describe(value)}")
        kind = candidate_kind(path, value)
        if kind:
            candidates.append((kind, path, describe(value)))

    print("\nDepth/intrinsics candidates:")
    if candidates:
        for kind, path, description in candidates:
            print(f"  [{kind}] {path}: {description}")
    else:
        print("  none found")


if __name__ == "__main__":
    main()
