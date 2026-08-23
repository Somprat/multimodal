#!/bin/bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${project_root}"
source script/setup/env.sh

ckpt_paths=(
"${CKPT_PATH:-${MEMORYVLA_MODEL_ROOT}/model_b/checkpoints/memvla-bridge.pt}"
)

gpu_id="${GPU_ID:-0}"
use_bf16="${USE_BF16:-true}"
precision_args=()
if [[ "${use_bf16}" == "true" ]]; then
    precision_args+=(--use_bf16)
fi
unnorm_key="${UNNORM_KEY:-}"
unnorm_args=()
if [[ -n "${unnorm_key}" ]]; then
    unnorm_args+=(--unnorm-key "${unnorm_key}")
fi
experiment_mode="${EXPERIMENT_MODE:-full}"
episode_start="${EPISODE_START:-0}"
episode_end="${EPISODE_END:-24}"

if [[ "${experiment_mode}" != "baseline" && "${experiment_mode}" != "full" ]]; then
    echo "EXPERIMENT_MODE must be baseline or full, got: ${experiment_mode}" >&2
    exit 1
fi

echo "Bridge evaluation: mode=${experiment_mode}, unnorm_key=${unnorm_key:-policy default}, episodes=${episode_start}-${episode_end}"
for ckpt_path in "${ckpt_paths[@]}"; do
    eval_dir=$(dirname $(dirname ${ckpt_path}))/eval_simpler/$(basename ${ckpt_path})/${experiment_mode}
    mkdir -p ${eval_dir}

    scene_name=bridge_table_1_v1
    robot=widowx
    rgb_overlay_path=./third_libs/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
    robot_init_x=0.147
    robot_init_y=0.028

    CUDA_VISIBLE_DEVICES=${gpu_id} .venv/bin/python evaluation/simpler_env/simpler_env_inference.py --ckpt-path ${ckpt_path} --experiment-mode ${experiment_mode} "${precision_args[@]}" "${unnorm_args[@]}" \
      --robot ${robot} --policy-setup widowx_bridge \
      --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
      --env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0 --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range ${episode_start} ${episode_end} \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 | tee ${eval_dir}/Cube.txt;

    CUDA_VISIBLE_DEVICES=${gpu_id} .venv/bin/python evaluation/simpler_env/simpler_env_inference.py --ckpt-path ${ckpt_path} --experiment-mode ${experiment_mode} "${precision_args[@]}" "${unnorm_args[@]}" \
      --robot ${robot} --policy-setup widowx_bridge \
      --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
      --env-name PutCarrotOnPlateInScene-v0 --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range ${episode_start} ${episode_end} \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 | tee ${eval_dir}/Carrot.txt;

    CUDA_VISIBLE_DEVICES=${gpu_id} .venv/bin/python evaluation/simpler_env/simpler_env_inference.py --ckpt-path ${ckpt_path} --experiment-mode ${experiment_mode} "${precision_args[@]}" "${unnorm_args[@]}" \
      --robot ${robot} --policy-setup widowx_bridge \
      --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
      --env-name PutSpoonOnTableClothInScene-v0 --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range ${episode_start} ${episode_end} \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 | tee ${eval_dir}/Spoon.txt;

    scene_name=bridge_table_1_v2
    robot=widowx_sink_camera_setup
    rgb_overlay_path=./third_libs/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
    robot_init_x=0.127
    robot_init_y=0.06

    CUDA_VISIBLE_DEVICES=${gpu_id} .venv/bin/python evaluation/simpler_env/simpler_env_inference.py --ckpt-path ${ckpt_path} --experiment-mode ${experiment_mode} "${precision_args[@]}" "${unnorm_args[@]}" \
      --robot ${robot} --policy-setup widowx_bridge \
      --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
      --env-name PutEggplantInBasketScene-v0 --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range ${episode_start} ${episode_end} \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 | tee ${eval_dir}/Eggplant.txt;

done

echo "All done!"
