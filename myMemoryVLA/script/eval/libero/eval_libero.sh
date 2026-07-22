#!/bin/bash
set -euo pipefail
export MKL_INTERFACE_LAYER=GNU

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${project_root}/script/setup/env.sh"
cd "${project_root}"
python_bin="${project_root}/.venv/bin/python"

ckpt_list=(
"${CKPT_PATH:-${MEMORYVLA_MODEL_ROOT}/memvla-libero-spatial/checkpoints/memvla-libero-spatial.pt}"
)

# libero_spatial, libero_object, libero_goal, libero_10, libero_90
task_suite_name="${TASK_SUITE_NAME:-libero_spatial}"
gpu_id="${GPU_ID:-0}"
action_chunking_window=8

num_trials_per_task="${NUM_TRIALS_PER_TASK:-50}"
spcial_task_id=None
run_id_note="ac${action_chunking_window}"
unnorm_key="${task_suite_name}_no_noops"

find_free_port() {
  local min=${1:-2000}
  local max=${2:-30000}
  local port
  local tries=1000  # max tries to find a free port

  for ((i=0; i<tries; i++)); do
    port=$(shuf -i"${min}"-"${max}" -n1)
    if ! lsof -iTCP:"${port}" -sTCP:LISTEN &>/dev/null; then
      echo "${port}"
      return 0
    fi
  done

  echo "ERROR: not found free port in range ${min}-${max}" >&2
  return 1
}

if [ "$spcial_task_id" != "None" ]; then
  spcial_task_id_arg="--spcial_task_id ${spcial_task_id}"
else
  spcial_task_id_arg=""
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}

for ckpt_path in "${ckpt_list[@]}"; do
  echo ">>> process ckpt：${ckpt_path}"
  port=$(find_free_port)
  echo "    port：${port}"

  local_log_dir="$(dirname "$(dirname "$ckpt_path")")/eval_libero/$(basename "$ckpt_path")"

  if [[ ! -f "${ckpt_path}" ]]; then
    echo "Missing checkpoint: ${ckpt_path}" >&2
    exit 1
  fi

  echo "Developing ..."
  "${python_bin}" deploy.py \
    --saved_model_path ${ckpt_path} \
    --unnorm_key ${unnorm_key} \
    --adaptive_ensemble_alpha 0.1 \
    --cfg_scale 1.5 \
    --port ${port} \
    --action_chunking \
    --action_chunking_window ${action_chunking_window} &

  DEPLOY_PID=$!

  cleanup() { kill "${DEPLOY_PID}" 2>/dev/null || true; }
  trap cleanup EXIT INT TERM
  echo "Waiting for model server readiness ..."
  until "${python_bin}" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${port}/health', timeout=2)" 2>/dev/null; do
    if ! kill -0 "${DEPLOY_PID}" 2>/dev/null; then
      echo "Model server exited before becoming ready." >&2
      wait "${DEPLOY_PID}"
    fi
    sleep 1
  done

  echo "Evaluating ..."
  "${python_bin}" evaluation/libero/eval_libero.py \
    --task_suite_name ${task_suite_name} \
    --num_trials_per_task ${num_trials_per_task} \
    --run_id_note ${run_id_note} \
    --local_log_dir ${local_log_dir} \
    --port ${port} \
    ${spcial_task_id_arg}

  echo "kill developed service PID ${DEPLOY_PID}"
  kill ${DEPLOY_PID}
  trap - EXIT INT TERM
  echo ">>> finish ${ckpt_path}"
  echo
done

echo "All done!"
