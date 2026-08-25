"""Download the Bridge RLDS dataset with automatic resume support."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ID = "shihao1895/bridge-rlds"
RECOMMENDED_FREE_BYTES = 150 * 1024**3


def human_size(value: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("datasets/bridge-rlds"),
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-space-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.output).free
    print(f"Destination: {args.output.resolve()} ({human_size(free)} free)")
    if free < RECOMMENDED_FREE_BYTES and not args.skip_space_check:
        raise SystemExit(
            f"Only {human_size(free)} is free. Bridge RLDS is about 132 GB; "
            "free 150 GiB or pass --skip-space-check."
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Run: pip install -U huggingface_hub") from exc

    print("Downloading; rerunning the same command resumes partial files.")
    snapshot_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.output,
        allow_patterns=["bridge_orig/**", "README.md", ".gitattributes"],
        max_workers=args.workers,
    )
    dataset_dir = Path(snapshot_path) / "bridge_orig" / "1.0.0"
    train = list(dataset_dir.glob("bridge_orig-train.tfrecord-*"))
    val = list(dataset_dir.glob("bridge_orig-val.tfrecord-*"))
    print(f"Complete: {snapshot_path}")
    print(f"Found {len(train)} train shards and {len(val)} validation shards")
    if not train:
        raise SystemExit(f"No TFRecord shards found in {dataset_dir}")


if __name__ == "__main__":
    main()
