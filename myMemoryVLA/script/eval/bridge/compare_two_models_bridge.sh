#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash script/eval/bridge/compare_two_models_bridge.sh \
    --model-a /path/to/model_a/checkpoint.pt \
    --model-b /path/to/model_b/checkpoint.pt \
    [--gpu 0] [--episode-start 0] [--episode-end 24] [--use-bf16]

Runs both checkpoints on the same SimplerEnv Bridge task set and prints a
side-by-side summary from eval_simpler/<checkpoint-name> logs.
EOF
}

model_a=""
model_b=""
gpu_id=0
episode_start=0
episode_end=24
use_bf16=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-a)
      model_a="$2"
      shift 2
      ;;
    --model-b)
      model_b="$2"
      shift 2
      ;;
    --gpu)
      gpu_id="$2"
      shift 2
      ;;
    --episode-start)
      episode_start="$2"
      shift 2
      ;;
    --episode-end)
      episode_end="$2"
      shift 2
      ;;
    --use-bf16)
      use_bf16="--use_bf16"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${model_a}" || -z "${model_b}" ]]; then
  usage >&2
  exit 1
fi

ckpt_paths=("${model_a}" "${model_b}")
roots=()

add_root_once() {
  local candidate="$1"
  local existing
  if [[ ${#roots[@]} -gt 0 ]]; then
    for existing in "${roots[@]}"; do
      if [[ "${existing}" == "${candidate}" ]]; then
        return
      fi
    done
  fi
  roots+=("${candidate}")
}

run_bridge_task() {
  local ckpt_path="$1"
  local eval_dir="$2"
  local task_name="$3"
  local env_name="$4"
  local scene_name="$5"
  local robot="$6"
  local rgb_overlay_path="$7"
  local robot_init_x="$8"
  local robot_init_y="$9"

  CUDA_VISIBLE_DEVICES="${gpu_id}" python evaluation/simpler_env/simpler_env_inference.py ${use_bf16} \
    --ckpt-path "${ckpt_path}" \
    --robot "${robot}" --policy-setup widowx_bridge \
    --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
    --env-name "${env_name}" --scene-name "${scene_name}" \
    --rgb-overlay-path "${rgb_overlay_path}" \
    --robot-init-x "${robot_init_x}" "${robot_init_x}" 1 \
    --robot-init-y "${robot_init_y}" "${robot_init_y}" 1 \
    --obj-variation-mode episode --obj-episode-range "${episode_start}" "${episode_end}" \
    --robot-init-rot-quat-center 0 0 0 1 \
    --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 | tee "${eval_dir}/${task_name}.txt"
}

for ckpt_path in "${ckpt_paths[@]}"; do
  eval_root="$(dirname "$(dirname "${ckpt_path}")")"
  eval_dir="${eval_root}/eval_simpler/$(basename "${ckpt_path}")"
  mkdir -p "${eval_dir}"
  add_root_once "${eval_root}"

  run_bridge_task "${ckpt_path}" "${eval_dir}" "Cube" \
    "StackGreenCubeOnYellowCubeBakedTexInScene-v0" \
    "bridge_table_1_v1" \
    "widowx" \
    "./third_libs/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png" \
    "0.147" "0.028"

  run_bridge_task "${ckpt_path}" "${eval_dir}" "Carrot" \
    "PutCarrotOnPlateInScene-v0" \
    "bridge_table_1_v1" \
    "widowx" \
    "./third_libs/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png" \
    "0.147" "0.028"

  run_bridge_task "${ckpt_path}" "${eval_dir}" "Spoon" \
    "PutSpoonOnTableClothInScene-v0" \
    "bridge_table_1_v1" \
    "widowx" \
    "./third_libs/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png" \
    "0.147" "0.028"

  run_bridge_task "${ckpt_path}" "${eval_dir}" "Eggplant" \
    "PutEggplantInBasketScene-v0" \
    "bridge_table_1_v2" \
    "widowx_sink_camera_setup" \
    "./third_libs/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png" \
    "0.127" "0.06"
done

python script/eval/bridge/extract_bridge_results.py --style md "${roots[@]}"
