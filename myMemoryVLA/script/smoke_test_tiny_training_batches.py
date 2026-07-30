#!/usr/bin/env python3
"""Exercise Tiny ManiSkill RLDS samples through the real training collator."""

import argparse
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaTokenizerFast

from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from vla import get_vla_dataset_and_collator


def image_transform(image):
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    tokenizer = LlamaTokenizerFast.from_pretrained(args.tokenizer, local_files_only=True)
    dataset, _, collator = get_vla_dataset_and_collator(
        data_root_dir=args.data_root.resolve(),
        data_mix="tiny_maniskill_spatial",
        image_transform=image_transform,
        tokenizer=tokenizer,
        prompt_builder_fn=PurePromptBuilder,
        default_image_resolution=(3, 64, 64),
        shuffle_buffer_size=1,
        train=True,
        image_aug=False,
        future_action_window_size=15,
        load_all_data_for_training=True,
        dataloader_type="group",
        group_size=20,
        load_depth=True,
        load_proprio=True,
        use_spatial_features=True,
    )

    examples = list(islice(iter(dataset), args.samples))
    if len(examples) != args.samples:
        raise RuntimeError(f"requested {args.samples} samples, received {len(examples)}")

    episode_ids = set()
    batches = 0
    for start in range(0, len(examples), args.batch_size):
        batch = collator(examples[start : start + args.batch_size])
        batches += 1
        required = {
            "images", "instructions", "depth", "intrinsic", "proprio",
            "actions", "action_masks", "episode_ids", "timesteps",
        }
        missing = required.difference(batch)
        if missing:
            raise RuntimeError(f"training batch is missing fields: {sorted(missing)}")
        if batch["depth"].shape[-3:] != (64, 64, 1):
            raise RuntimeError(f"unexpected depth shape: {tuple(batch['depth'].shape)}")
        if batch["intrinsic"].shape[-2:] != (3, 3):
            raise RuntimeError(f"unexpected intrinsics shape: {tuple(batch['intrinsic'].shape)}")
        if batch["actions"].shape[-2:] != (16, 7):
            raise RuntimeError(f"unexpected action-window shape: {tuple(batch['actions'].shape)}")
        if len(batch["images"]) != len(batch["instructions"]):
            raise RuntimeError("retrieval image/instruction batch sizes differ")
        episode_ids.update(int(value) for value in batch["episode_ids"])

    final = collator(examples[: args.batch_size])
    print(
        f"samples={len(examples)} batches={batches} episodes_seen={len(episode_ids)}\n"
        f"pixel_values={tuple(final['pixel_values'].shape)} "
        f"depth={tuple(final['depth'].shape)} "
        f"intrinsic={tuple(final['intrinsic'].shape)}\n"
        f"actions={tuple(final['actions'].shape)} "
        f"action_masks={tuple(final['action_masks'].shape)} "
        f"retrieval_images={len(final['images'])} "
        f"instructions={len(final['instructions'])}\n"
        "Tiny ManiSkill training-batch workflow: OK"
    )


if __name__ == "__main__":
    main()
