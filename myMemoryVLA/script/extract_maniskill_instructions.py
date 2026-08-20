#!/usr/bin/env python3
"""Extract unique language instructions from a ManiSkill RLDS dataset.

The script can read an already-materialized TFDS version directory or stream
randomized shards from the public Open X-Embodiment GCS copy. It counts each
instruction once per episode and writes a CSV ready for manual router labels.
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping
from pathlib import Path


LABEL_COLUMNS = (
    "primary_mode",
    "secondary_mode",
    "confidence",
    "notes",
    "split",
)

DEFAULT_REMOTE_URI = (
    "gs://gresearch/robotics/"
    "maniskill_dataset_converted_externally_to_rlds/0.1.0"
)


def normalize_instruction(instruction: str) -> str:
    """Normalize harmless textual differences without changing semantics."""
    normalized = unicodedata.normalize("NFKC", instruction)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.casefold()


def decode_instruction(value: object) -> str:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def episode_instruction_counts(episode: Mapping[str, object]) -> Counter[str]:
    """Count normalized instructions within one episodic or step-level sample."""
    steps = episode.get("steps")
    if steps is None:
        raw = episode.get("language_instruction")
        if raw is None:
            return Counter()
        instruction = normalize_instruction(decode_instruction(raw))
        return Counter({instruction: 1}) if instruction else Counter()

    counts: Counter[str] = Counter()
    for step in steps:
        raw = step.get("language_instruction")
        if raw is None:
            continue
        instruction = normalize_instruction(decode_instruction(raw))
        if instruction:
            counts[instruction] += 1
    return counts


def write_labeling_csv(
    output_path: Path,
    episode_counts: Counter[str],
    step_counts: Counter[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(episode_counts, key=lambda text: (-episode_counts[text], text))

    fieldnames = (
        "instruction_id",
        "instruction",
        "episode_count",
        "step_count",
        *LABEL_COLUMNS,
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, instruction in enumerate(ordered, start=1):
            writer.writerow(
                {
                    "instruction_id": index,
                    "instruction": instruction,
                    "episode_count": episode_counts[instruction],
                    "step_count": step_counts[instruction],
                    **{column: "" for column in LABEL_COLUMNS},
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract normalized, deduplicated language instructions from a local "
            "or public-GCS ManiSkill RLDS/TFDS dataset and create a router-labeling CSV."
        )
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        nargs="?",
        help=(
            "Local TFDS version directory, for example "
            "data/maniskill_dataset_converted_externally_to_rlds/0.1.0. "
            "Omit when using --remote."
        ),
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Stream randomized shards from the public ManiSkill RLDS GCS dataset.",
    )
    parser.add_argument(
        "--remote-uri",
        default=DEFAULT_REMOTE_URI,
        help="TFDS version URI used by --remote.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=5000,
        help="Maximum episodes to scan; use 0 for no episode limit (default: 5000).",
    )
    parser.add_argument(
        "--target-unique",
        type=int,
        default=200,
        help="Stop after finding this many unique instructions; use 0 for no target (default: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed controlling remote shard order (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("maniskill_instructions_to_label.csv"),
    )
    args = parser.parse_args()

    if args.max_episodes < 0 or args.target_unique < 0:
        parser.error("--max-episodes and --target-unique must be nonnegative")
    if args.remote and args.dataset_dir is not None:
        parser.error("provide either a local dataset_dir or --remote, not both")
    if not args.remote and args.dataset_dir is None:
        parser.error("dataset_dir is required unless --remote is used")

    dataset_location: str
    if args.remote:
        dataset_location = args.remote_uri
    else:
        dataset_dir = args.dataset_dir.expanduser().resolve()
        if not dataset_dir.is_dir():
            parser.error(f"dataset directory does not exist: {dataset_dir}")
        dataset_location = str(dataset_dir)

    try:
        import tensorflow_datasets as tfds
    except ImportError as error:
        parser.error(
            "tensorflow-datasets is required; run this with the MemoryVLA environment"
        )
        raise AssertionError("unreachable") from error

    try:
        builder = tfds.builder_from_directory(dataset_location)
        read_config = tfds.ReadConfig(shuffle_seed=args.seed)
        dataset = builder.as_dataset(
            split=args.split,
            shuffle_files=args.remote,
            read_config=read_config,
        )
    except Exception as error:
        if args.remote:
            raise RuntimeError(
                "Could not stream the public GCS dataset. Confirm outbound network "
                "access and TensorFlow GCS support, or copy selected/full TFDS data "
                "locally and run without --remote. "
                f"Remote URI: {dataset_location}"
            ) from error
        raise
    if args.max_episodes:
        dataset = dataset.take(args.max_episodes)

    episode_counts: Counter[str] = Counter()
    step_counts: Counter[str] = Counter()
    episodes_scanned = 0
    episodes_without_language = 0

    for episode in dataset:
        episodes_scanned += 1
        within_episode = episode_instruction_counts(episode)
        if not within_episode:
            episodes_without_language += 1
            continue
        step_counts.update(within_episode)
        episode_counts.update(within_episode.keys())
        if args.target_unique and len(episode_counts) >= args.target_unique:
            break

    output_path = args.output.expanduser().resolve()
    write_labeling_csv(output_path, episode_counts, step_counts)

    print(f"dataset={builder.info.full_name}")
    print(f"source={dataset_location}")
    print(f"split={args.split}")
    print(f"episodes_scanned={episodes_scanned}")
    print(f"episodes_without_language={episodes_without_language}")
    print(f"unique_instructions={len(episode_counts)}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
