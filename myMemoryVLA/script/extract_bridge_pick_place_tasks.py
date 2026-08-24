#!/usr/bin/env python3
"""Extract simple pick-and-place instructions from BridgeData V2 RLDS.

This script reads an already-downloaded TFDS version directory with
``tfds.builder_from_directory``. It records each normalized instruction once
per episode, parses only unambiguous ``put/place X in/on Y`` commands, and
marks evaluation-object overlap instead of silently dropping it.

Example:

    python script/extract_bridge_pick_place_tasks.py \
        /path/to/bridge_dataset/1.0.0 \
        --output outputs/bridge_pick_place_tasks.csv \
        --max-episodes 1000

Pass ``--max-episodes 0`` to scan the complete split.
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EVALUATION_OBJECT_ALIASES = {
    "spoon": "spoon",
    "towel": "towel",
    "table cloth": "towel",
    "tablecloth": "towel",
    "carrot": "carrot",
    "plate": "plate",
    "green cube": "green cube",
    "green block": "green cube",
    "yellow cube": "yellow cube",
    "yellow block": "yellow cube",
    "eggplant": "eggplant",
    "egg plant": "eggplant",
    "basket": "basket",
}

RELATION_ALIASES = {
    "in": "in",
    "into": "in",
    "inside": "in",
    "inside of": "in",
    "on": "on",
    "onto": "on",
    "on top of": "on",
}

PICK_PLACE_PATTERN = re.compile(
    r"^(?:put|place)\s+(?:the\s+)?(?P<source>.+?)\s+"
    r"(?P<relation>on\s+top\s+of|inside\s+of|inside|into|onto|in|on)\s+"
    r"(?:the\s+)?(?P<target>.+?)$"
)

MULTI_TASK_PATTERN = re.compile(r"\b(?:and|then|after|before)\b")


@dataclass(frozen=True)
class ParsedInstruction:
    source: str = ""
    target: str = ""
    relation: str = ""
    status: str = "not_pick_place"
    notes: str = ""


def normalize_instruction(value: str) -> str:
    """Normalize textual differences while preserving instruction meaning."""
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"\s+", " ", value).strip().casefold()
    return value.rstrip(" .!?")


def _decoded_strings(value: Any) -> Iterable[str]:
    """Yield strings from a scalar Tensor/bytes value or a tensor-like array."""
    if hasattr(value, "numpy"):
        value = value.numpy()

    if isinstance(value, bytes):
        yield value.decode("utf-8", errors="replace")
        return
    if isinstance(value, str):
        yield value
        return

    # NumPy and TensorFlow scalars expose item(); arrays expose flat.
    if hasattr(value, "shape") and getattr(value, "shape", None) == ():
        yield from _decoded_strings(value.item())
        return
    if hasattr(value, "flat"):
        for item in value.flat:
            yield from _decoded_strings(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _decoded_strings(item)


def _first_nonempty(value: Any) -> str | None:
    for decoded in _decoded_strings(value):
        normalized = normalize_instruction(decoded)
        if normalized:
            return normalized
    return None


def _get_nested(mapping: Mapping[str, Any], *path: str) -> Any | None:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def instruction_from_step(step: Mapping[str, Any]) -> str | None:
    """Read the instruction from known Bridge/OXE step layouts."""
    candidates = (
        _get_nested(step, "language_instruction"),
        _get_nested(step, "observation", "natural_language_instruction"),
        _get_nested(step, "observation", "language_instruction"),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        instruction = _first_nonempty(candidate)
        if instruction:
            return instruction
    return None


def instruction_from_episode(episode: Mapping[str, Any]) -> str | None:
    """Return the first non-empty instruction stored in an RLDS episode."""
    for path in (
        ("language_instruction",),
        ("episode_metadata", "language_instruction"),
    ):
        candidate = _get_nested(episode, *path)
        if candidate is not None:
            instruction = _first_nonempty(candidate)
            if instruction:
                return instruction

    steps = episode.get("steps")
    if steps is None:
        return None

    # Some TFDS decoders return a mapping of time-major tensors rather than a
    # nested tf.data.Dataset. Handle that layout without materializing images.
    if isinstance(steps, Mapping):
        return instruction_from_step(steps)

    for step in steps:
        if not isinstance(step, Mapping):
            continue
        instruction = instruction_from_step(step)
        if instruction:
            return instruction
    return None


def _clean_object_phrase(value: str) -> str:
    value = re.sub(r"^(?:the|a|an)\s+", "", value.strip())
    return value.rstrip(" .!?")


def _evaluation_object_matches(value: str) -> list[str]:
    matches: set[str] = set()
    for alias, canonical in EVALUATION_OBJECT_ALIASES.items():
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", value):
            matches.add(canonical)
    return sorted(matches)


def parse_pick_place_instruction(instruction: str) -> ParsedInstruction:
    """Parse only simple, single-stage put/place instructions."""
    instruction = normalize_instruction(instruction)
    if not instruction:
        return ParsedInstruction(notes="empty instruction")
    if MULTI_TASK_PATTERN.search(instruction):
        return ParsedInstruction(
            status="manual_review",
            notes="possible multi-stage instruction",
        )

    match = PICK_PLACE_PATTERN.fullmatch(instruction)
    if match is None:
        status = "manual_review" if instruction.startswith(("put ", "place ")) else "not_pick_place"
        return ParsedInstruction(status=status, notes="unsupported instruction form")

    source = _clean_object_phrase(match.group("source"))
    target = _clean_object_phrase(match.group("target"))
    relation_text = re.sub(r"\s+", " ", match.group("relation"))
    relation = RELATION_ALIASES[relation_text]

    if not source or not target:
        return ParsedInstruction(status="manual_review", notes="missing source or target")

    excluded = sorted(
        set(_evaluation_object_matches(source) + _evaluation_object_matches(target))
    )
    if excluded:
        return ParsedInstruction(
            source=source,
            target=target,
            relation=relation,
            status="excluded_eval_object",
            notes="evaluation overlap: " + ", ".join(excluded),
        )

    return ParsedInstruction(
        source=source,
        target=target,
        relation=relation,
        status="accepted",
    )


def load_bridge_dataset(dataset_dir: Path, split: str):
    """Load a materialized BridgeData V2 TFDS/RLDS version directory."""
    try:
        import tensorflow_datasets as tfds
    except ImportError as error:
        raise RuntimeError(
            "tensorflow-datasets is required. Run this script from the "
            "MemoryVLA environment created by script/setup/bootstrap_runpod_eval.sh."
        ) from error

    builder = tfds.builder_from_directory(builder_dir=str(dataset_dir))
    dataset = builder.as_dataset(split=split, shuffle_files=False)
    return builder, dataset


def write_results(output_path: Path, counts: Counter[str]) -> Counter[str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_counts: Counter[str] = Counter()
    fieldnames = (
        "instruction_id",
        "instruction",
        "episode_count",
        "source",
        "target",
        "relation",
        "status",
        "notes",
    )

    ordered = sorted(counts, key=lambda text: (-counts[text], text))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for instruction_id, instruction in enumerate(ordered, start=1):
            parsed = parse_pick_place_instruction(instruction)
            status_counts[parsed.status] += 1
            writer.writerow(
                {
                    "instruction_id": instruction_id,
                    "instruction": instruction,
                    "episode_count": counts[instruction],
                    "source": parsed.source,
                    "target": parsed.target,
                    "relation": parsed.relation,
                    "status": parsed.status,
                    "notes": parsed.notes,
                }
            )
    return status_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Local TFDS version directory, for example /data/bridge_dataset/1.0.0",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=1000,
        help="Maximum episodes to scan; use 0 for the complete split (default: 1000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/bridge_pick_place_tasks.csv"),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress after this many episodes; use 0 to disable",
    )
    args = parser.parse_args()

    args.dataset_dir = args.dataset_dir.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.dataset_dir.is_dir():
        parser.error(f"dataset directory does not exist: {args.dataset_dir}")
    if args.max_episodes < 0 or args.progress_every < 0:
        parser.error("--max-episodes and --progress-every must be nonnegative")
    return args


def main() -> None:
    args = parse_args()
    builder, dataset = load_bridge_dataset(args.dataset_dir, args.split)
    if args.max_episodes:
        dataset = dataset.take(args.max_episodes)

    instruction_counts: Counter[str] = Counter()
    episodes_scanned = 0
    episodes_without_instruction = 0

    for episode in dataset:
        episodes_scanned += 1
        instruction = instruction_from_episode(episode)
        if instruction is None:
            episodes_without_instruction += 1
        else:
            instruction_counts[instruction] += 1

        if args.progress_every and episodes_scanned % args.progress_every == 0:
            print(
                f"episodes_scanned={episodes_scanned} "
                f"unique_instructions={len(instruction_counts)}"
            )

    status_counts = write_results(args.output, instruction_counts)
    print(f"dataset={builder.info.full_name}")
    print(f"source={args.dataset_dir}")
    print(f"split={args.split}")
    print(f"episodes_scanned={episodes_scanned}")
    print(f"episodes_without_instruction={episodes_without_instruction}")
    print(f"unique_instructions={len(instruction_counts)}")
    for status in sorted(status_counts):
        print(f"{status}={status_counts[status]}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
