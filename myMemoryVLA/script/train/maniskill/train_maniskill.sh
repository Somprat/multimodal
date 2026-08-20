#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${project_root}"

pretrained_ckpt="${PRETRAINED_CKPT:-./pretrained/CogACT-Large/checkpoints/CogACT-Large.pt}"
data_root_dir="${DATA_ROOT_DIR:-./data}"
run_root_dir="${RUN_ROOT_DIR:-./log/maniskill}"
run_id="${RUN_ID:-memvla_maniskill_spatial}"

n_gpu="${N_GPU:-1}"
per_device_batch_size="${BATCH_SIZE:-1}"
max_steps="${MAX_STEPS:-100}"
save_interval="${SAVE_INTERVAL:-100}"
shuffle_buffer_size="${SHUFFLE_BUFFER_SIZE:-4096}"
cuda_devices="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -f "${pretrained_ckpt}" ]]; then
    echo "Missing pretrained checkpoint: ${pretrained_ckpt}" >&2
    echo "Set PRETRAINED_CKPT to the CogACT-Large checkpoint." >&2
    exit 1
fi

dataset_info="${data_root_dir}/maniskill_dataset_converted_externally_to_rlds/0.1.0/dataset_info.json"
if [[ ! -f "${dataset_info}" ]]; then
    echo "Missing ManiSkill RLDS dataset metadata: ${dataset_info}" >&2
    echo "Set DATA_ROOT_DIR to the TFDS data root containing the ManiSkill dataset directory." >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="${cuda_devices}" \
torchrun --nproc_per_node="${n_gpu}" train.py \
  --pretrained_checkpoint "${pretrained_ckpt}" \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix maniskill \
  --vla.expected_world_size "${n_gpu}" \
  --vla.per_device_batch_size "${per_device_batch_size}" \
  --vla.global_batch_size "$((n_gpu * per_device_batch_size))" \
  --vla.learning_rate "${LEARNING_RATE:-2e-5}" \
  --vla.max_steps "${max_steps}" \
  --vla.shuffle_buffer_size "${shuffle_buffer_size}" \
  --data_root_dir "${data_root_dir}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --image_aug "${IMAGE_AUG:-true}" \
  --save_interval "${save_interval}" \
  --repeated_diffusion_steps "${DIFFUSION_STEPS:-4}" \
  --future_action_window_size "${FUTURE_ACTION_WINDOW_SIZE:-15}" \
  --action_model_type DiT-L \
  --dataloader_type stream \
  --is_resume false \
  --resume_step 0 \
  --resume_epoch 0 \
  --wandb_project "${WANDB_PROJECT:-memvla}" \
  --wandb_entity "${WANDB_ENTITY:-}" \
  --hf_token "${HF_TOKEN_PATH:-.hf_token}"
