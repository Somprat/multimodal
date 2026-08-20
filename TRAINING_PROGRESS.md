# ManiSkill training progress — 2026-08-20

## Goal

Run MemoryVLA ManiSkill training from `/workspace/multimodal` with the local
MemoryVLA bridge checkpoint and the full TFDS ManiSkill RLDS dataset.

## Completed

- Downloaded the complete TFDS dataset to:
  `/workspace/datasets/maniskill_dataset_converted_externally_to_rlds/0.1.0`.
- Confirmed the dataset contains all 1,024 TFRecord shards plus
  `dataset_info.json` and `features.json`.
- Compared all 1,026 local objects with the Google Cloud Storage manifest.
- Found one same-size but checksum-corrupt shard:
  `maniskill_dataset_converted_externally_to_rlds-train.tfrecord-00585-of-01024`.
- Re-downloaded shard 00585 without resume, verified its authoritative MD5
  (`KMl3Vqkb1T6QZIIPOlYJsw==`), and atomically replaced the corrupt copy.
- Installed/repaired the training dependencies in `myMemoryVLA/.venv`, notably:
  Draccus, TensorFlow 2.15, TFDS 4.9.3, dlimp, WandB, and FlashAttention 2.5.5.
- Downloaded/cached the DINOv2 and SigLIP vision backbones.
- Confirmed the local checkpoint exists at:
  `/workspace/multimodal/models/model_b/checkpoints/memvla-bridge.pt`.
- Memory-mapped and inspected the 33.5 GB checkpoint. It contains 291 Llama
  tensors and all 6,738,939,904 Llama parameters (embeddings, 32 layers, final
  norm, and LM head).
- Verified that downloading a separate gated Llama-2 7B checkpoint is
  unnecessary.
- Confirmed the real launcher reaches model restoration and real ManiSkill
  dataset iteration from `/workspace/multimodal`.

## Code changes

### `myMemoryVLA/script/train/maniskill/train_maniskill.sh`

- Sources `script/setup/env.sh` automatically.
- Prepends `myMemoryVLA/.venv/bin` to `PATH` automatically.
- This prevents the launcher from silently using `/usr/bin/python`, which
  caused the original `ModuleNotFoundError: draccus`.
- The working tree also contains a default-checkpoint change to
  `../models/model_b/checkpoints/memvla-bridge.pt`; that change was already
  present before today's launcher hardening.

### `myMemoryVLA/vla/load.py`

- For a complete MemoryVLA checkpoint, constructs Llama from the local config
  instead of downloading gated `meta-llama/Llama-2-7b-hf` weights first.
- Restores training behavior after loading the checkpoint by disabling the LLM
  cache and enabling input gradients.
- A smoke run confirmed the intended message:
  `Building empty llama2 LLM from /workspace/multimodal/models/llama2-7b-public`.

## Validation reached

The guarded real run successfully completed:

1. Draccus and dependency imports.
2. Torch distributed startup on one A100 80 GB.
3. DINOv2 and SigLIP backbone loading.
4. Local Llama architecture construction without gated downloads.
5. Restoration of the 33.5 GB MemoryVLA checkpoint.
6. Creation of the ManiSkill TFDS/RLDS pipeline.
7. Iteration over real ManiSkill trajectories for normalization statistics.

The run reported 8,749.736 million total parameters and 1,057.221 million
trainable parameters for the requested scope.

## Warnings observed

- TensorFlow prints duplicate cuDNN/cuFFT/cuBLAS factory warnings. These did not
  stop startup.
- The bridge checkpoint has no trained weights for the new spatial modules
  (`per_spatial_gate`, `point_cloud_spatial_encoder`, `spatial_mem_bank`,
  `spatial_to_per_fusion`), so those modules remain randomly initialized. This
  appears intentional for ManiSkill adaptation but should be reviewed before a
  long production run.
- First startup is slow because the 33.5 GB checkpoint is read from a
  network-mounted filesystem and can use roughly 60 GB of host RAM while
  restoring.

## Remaining work

The one-time normalization-statistics pass was intentionally stopped when work
ended for the day. It must finish once before the first optimizer step. After
the corrupt shard was repaired, the restarted pass was stopped during imports,
so no incomplete statistics cache was written.

Then run a one-step smoke test and confirm an optimizer step completes. The
full 100-step job should only be started after that succeeds.

## Next commands

From `/workspace/multimodal`, the launcher now activates its own runtime, so the
intended training command is:

```bash
WANDB_MODE=disabled \
DATA_ROOT_DIR=/workspace/datasets \
PRETRAINED_CKPT=/workspace/multimodal/models/model_b/checkpoints/memvla-bridge.pt \
EXPERIMENT_MODE=full \
MAX_STEPS=1 \
SAVE_INTERVAL=9999 \
BATCH_SIZE=1 \
N_GPU=1 \
RUN_ID=maniskill_one_step_smoke \
bash myMemoryVLA/script/train/maniskill/train_maniskill.sh
```

Allow the first run to complete normalization statistics. Once the one-step
