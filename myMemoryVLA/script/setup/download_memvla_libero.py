#!/usr/bin/env python3
"""Download and verify an official MemoryVLA LIBERO checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = {
    "spatial": (
        "shihao1895/memvla-libero-spatial",
        "memvla-libero-spatial.pt",
        33_507_487_606,
    ),
}


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    repo_root = project_root.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", choices=MODELS, nargs="?", default="spatial")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    repo_id, checkpoint_name, expected_size = MODELS[args.suite]
    output = args.output or repo_root / "models" / f"memvla-libero-{args.suite}"
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id,
        cache_dir=repo_root / ".cache" / "huggingface" / "hub",
        local_dir=output,
        allow_patterns=[
            "config.yaml",
            "config.json",
            "dataset_statistics.json",
            f"checkpoints/{checkpoint_name}",
        ],
        max_workers=4,
    )
    checkpoint = output / "checkpoints" / checkpoint_name
    actual_size = checkpoint.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(f"Expected {expected_size} bytes, found {actual_size}: {checkpoint}")
    print(f"Verified {checkpoint} ({actual_size} bytes)")



if __name__ == "__main__":
    main()

