#!/usr/bin/env bash


set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${project_root}"

# Use the repository runtime configuration and Python environment even when
# this launcher is called from a fresh shell.
source "${project_root}/script/setup/env.sh"
if [[ -x "${project_root}/.venv/bin/python" ]]; then
    export PATH="${project_root}/.venv/bin:${PATH}"
fi

pretrained_ckpt="${PRETRAINED_CKPT:-../models/model_b/checkpoints/memvla-bridge.pt}"
data_root_dir="${DATA_ROOT_DIR:-../datasets}"
run_root_dir="${RUN_ROOT_DIR:-./log/maniskill}"
experiment_mode="${EXPERIMENT_MODE:-full}"
freeze_vlm="${FREEZE_VLM:-true}"
freeze_action_model="${FREEZE_ACTION_MODEL:-true}"
if [[ "${freeze_vlm}" == "true" && "${freeze_action_model}" == "true" ]]; then
    scope_tag="frozen_vlm_action"
elif [[ "${freeze_vlm}" == "true" ]]; then
    scope_tag="frozen_vlm"
else
    scope_tag="full_finetune"
fi
run_id="${RUN_ID:-memvla_maniskill_${experiment_mode}_${scope_tag}}"

n_gpu="${N_GPU:-1}"
per_device_batch_size="${BATCH_SIZE:-1}"
max_steps="${MAX_STEPS:-100}"
save_interval="${SAVE_INTERVAL:-100}"
shuffle_buffer_size="${SHUFFLE_BUFFER_SIZE:-4096}"
cuda_devices="${CUDA_VISIBLE_DEVICES:-0}"
dry_run="${DRY_RUN:-false}"

if [[ "${experiment_mode}" != "baseline" && "${experiment_mode}" != "episodic" && "${experiment_mode}" != "query" && "${experiment_mode}" != "query_episodic" && "${experiment_mode}" != "full" ]]; then
    echo "EXPERIMENT_MODE must be baseline, episodic, query, query_episodic, or full, got: ${experiment_mode}" >&2
    exit 1
fi

for boolean_value in "${freeze_vlm}" "${freeze_action_model}" "${dry_run}"; do
    if [[ "${boolean_value}" != "true" && "${boolean_value}" != "false" ]]; then
        echo "FREEZE_VLM, FREEZE_ACTION_MODEL, and DRY_RUN must be true or false" >&2
        exit 1
    fi
done
if [[ "${dry_run}" != "true" && "${freeze_vlm}" == "true" && "${freeze_action_model}" == "true" && ( "${experiment_mode}" == "baseline" || "${experiment_mode}" == "query" ) ]]; then
    echo "${experiment_mode} is an evaluation-only ablation with the pretrained path frozen; no training is needed." >&2
    exit 2
fi

dataset_info="${data_root_dir}/maniskill_dataset_converted_externally_to_rlds/0.1.0/dataset_info.json"

echo "ManiSkill training: mode=${experiment_mode}, freeze_vlm=${freeze_vlm}, freeze_action_model=${freeze_action_model}, steps=${max_steps}, GPUs=${n_gpu}"
if [[ "${dry_run}" != "true" && ! -f "${pretrained_ckpt}" ]]; then
    echo "Missing pretrained checkpoint: ${pretrained_ckpt}" >&2
    echo "Set PRETRAINED_CKPT to the CogACT-Large checkpoint." >&2
    exit 1
fi

if [[ "${dry_run}" != "true" && ! -f "${dataset_info}" ]]; then
    echo "Missing ManiSkill RLDS dataset metadata: ${dataset_info}" >&2
    echo "Set DATA_ROOT_DIR to the TFDS data root containing the ManiSkill dataset directory." >&2
    exit 1
fi

train_command=(
    torchrun --nproc_per_node="${n_gpu}" train.py
    --pretrained_checkpoint "${pretrained_ckpt}"
    --vla.type prism-dinosiglip-224px+oxe+diffusion
    --vla.data_mix maniskill
    --vla.expected_world_size "${n_gpu}"
    --vla.per_device_batch_size "${per_device_batch_size}"
    --vla.global_batch_size "$((n_gpu * per_device_batch_size))"
    --vla.learning_rate "${LEARNING_RATE:-2e-5}"
    --vla.max_steps "${max_steps}"
    --vla.shuffle_buffer_size "${shuffle_buffer_size}"
    --data_root_dir "${data_root_dir}"
    --run_root_dir "${run_root_dir}"
    --run_id "${run_id}"
    --experiment_mode "${experiment_mode}"
    --freeze_vlm "${freeze_vlm}"
    --freeze_action_model "${freeze_action_model}"
    --image_aug "${IMAGE_AUG:-true}"
    --save_interval "${save_interval}"
    --repeated_diffusion_steps "${DIFFUSION_STEPS:-4}"
    --future_action_window_size "${FUTURE_ACTION_WINDOW_SIZE:-15}"
    --action_model_type DiT-L
    --dataloader_type stream
    --is_resume false
    --resume_step 0
    --resume_epoch 0
    --wandb_project "${WANDB_PROJECT:-memvla}"
    --wandb_entity "${WANDB_ENTITY:-}"
    --hf_token "${HF_TOKEN_PATH:-.hf_token}"
    --query_retrieval_mode query
    --query_retrieval_top_k "${QUERY_RETRIEVAL_TOP_K:-4}"
    --episodic_max_steps "${EPISODIC_MAX_STEPS:-10}"
    --episodic_top_k "${EPISODIC_TOP_K:-2}"
)

if [[ "${dry_run}" == "true" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${cuda_devices}"
    printf '%q ' "${train_command[@]}"
    printf '\n'
else
    CUDA_VISIBLE_DEVICES="${cuda_devices}" "${train_command[@]}"
fi
