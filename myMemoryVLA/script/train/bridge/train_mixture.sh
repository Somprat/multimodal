#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
python_bin="${repo_root}/.venv/bin/python"

if [[ ! -x "${python_bin}" ]]; then
  echo "Missing virtual environment: ${python_bin}" >&2
  exit 1
fi

pretrained_ckpt="${PRETRAINED_CKPT:-../models/model_b/checkpoints/memvla-bridge.pt}"
hf_token="${HF_TOKEN:-YOUR_HF_TOKEN}"

data_root_dir="${DATA_ROOT_DIR:-./data/bridge-rlds}"
experiment_mode="${EXPERIMENT_MODE:-full}"

if [[ "${experiment_mode}" != "baseline" && "${experiment_mode}" != "episodic" && "${experiment_mode}" != "query" && "${experiment_mode}" != "query_episodic" && "${experiment_mode}" != "full" ]]; then
  echo "EXPERIMENT_MODE must be baseline, episodic, query, query_episodic, or full, got: ${experiment_mode}" >&2
  exit 1
fi

if [[ -n "${DATA_MIX:-}" ]]; then
  data_mix="${DATA_MIX}"
elif [[ "${experiment_mode}" == "full" ]]; then
  data_mix='bridge_widowx_simpler_rgbd'
else
  data_mix='bridge'
fi

n_gpu=1

"${python_bin}" -c "import rich, flash_attn, transformers, tensorflow, tensorflow_datasets, dlimp"
available_gpus=$("${python_bin}" -c "import torch; print(torch.cuda.device_count())")
if (( available_gpus < n_gpu )); then
  echo "Requested ${n_gpu} GPUs, but only ${available_gpus} are visible in this container." >&2
  echo "Expose ${n_gpu} GPUs or lower n_gpu and the batch sizes before launching." >&2
  exit 1
fi
bs=1
shuffle_buffer_size=1_024 # stream loader buffers decoded episodes, not individual frames

save_interval=2500
dp_step=4
future_action_window_size=15

image_aug=True
freeze_action_model="${FREEZE_ACTION_MODEL:-true}"
if [[ "${freeze_action_model}" != "true" && "${freeze_action_model}" != "false" ]]; then
  echo "FREEZE_ACTION_MODEL must be true or false, got: ${freeze_action_model}" >&2
  exit 1
fi
if [[ "${freeze_action_model}" == "true" && ( "${experiment_mode}" == "baseline" || "${experiment_mode}" == "query" ) ]]; then
  echo "${experiment_mode} is an evaluation-only ablation with the pretrained path frozen; no training is needed." >&2
  echo "Use script/eval/bridge/eval_bridge.sh with EXPERIMENT_MODE=${experiment_mode}." >&2
  exit 2
fi

run_root_dir='./log/bridge_generated'
run_id="${RUN_ID:-memvla_bridge_${experiment_mode}}"

is_resume=False
resume_step=0
resume_epoch=0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
"${python_bin}" -m torch.distributed.run --nproc_per_node=${n_gpu} train.py \
  --pretrained_checkpoint ${pretrained_ckpt} \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix ${data_mix} \
  --vla.expected_world_size ${n_gpu} \
  --vla.per_device_batch_size ${bs} \
  --vla.global_batch_size 8 \
  --vla.learning_rate 2e-5 \
  --vla.max_steps 50000 \
  --data_root_dir ${data_root_dir} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --image_aug ${image_aug} \
  --save_interval ${save_interval} \
  --repeated_diffusion_steps ${dp_step} \
  --future_action_window_size ${future_action_window_size} \
  --action_model_type 'DiT-L' \
  --dataloader_type 'stream' \
  --is_resume ${is_resume} \
  --resume_step ${resume_step} \
  --resume_epoch ${resume_epoch} \
  --trackers '[jsonl]' \
  --hf_token ${hf_token} \
  --vla.shuffle_buffer_size ${shuffle_buffer_size} \
  --experiment_mode "${experiment_mode}" \
  --query_retrieval_mode query \
  --query_retrieval_top_k "${QUERY_RETRIEVAL_TOP_K:-4}" \
  --episodic_max_steps "${EPISODIC_MAX_STEPS:-10}" \
  --episodic_top_k "${EPISODIC_TOP_K:-2}" \
  --freeze_vlm true \
  --freeze_action_model "${freeze_action_model}"
