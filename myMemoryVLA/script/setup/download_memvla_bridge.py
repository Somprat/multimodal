#!/usr/bin/env python3
"""Download the official MemoryVLA Bridge model into this repository."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    repo_root = project_root.parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "models" / "model_b",
        help="Model directory (default: <repo>/models/model_b)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=repo_root / ".cache" / "huggingface" / "hub",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        "shihao1895/memvla-bridge",
        cache_dir=args.cache_dir,
        local_dir=args.output,
        allow_patterns=[
            "config.yaml",
            "config.json",
            "dataset_statistics.json",
            "checkpoints/memvla-bridge.pt",
        ],
        max_workers=4,
    )

    checkpoint_dir = args.output / "checkpoints"
    alias = checkpoint_dir / "step-2.pt"
    if not alias.exists():
        alias.symlink_to("memvla-bridge.pt")
    print(f"Model ready at {checkpoint_dir / 'memvla-bridge.pt'}")


if __name__ == "__main__":
    main()
