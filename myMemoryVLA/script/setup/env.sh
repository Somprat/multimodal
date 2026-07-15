#!/usr/bin/env bash

# Source this file from anywhere after cloning the repository.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
repo_root="$(cd "${project_root}/.." && pwd)"

if [[ -r /proc/1/environ ]]; then
  container_caps="$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^NVIDIA_DRIVER_CAPABILITIES=//p')"
  if [[ -n "${container_caps}" ]]; then
    export NVIDIA_DRIVER_CAPABILITIES="${container_caps}"
  fi
fi

local_vulkan_lib="${project_root}/.runtime/root/usr/lib/x86_64-linux-gnu"
if [[ -d "${local_vulkan_lib}" ]]; then
  export LD_LIBRARY_PATH="${local_vulkan_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export MS2_REAL2SIM_ASSET_DIR="${project_root}/third_libs/SimplerEnv/ManiSkill2_real2sim/data"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export HF_HOME="${repo_root}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export MEMORYVLA_MODEL_ROOT="${repo_root}/models"

local_llama2_7b="${MEMORYVLA_MODEL_ROOT}/llama2-7b-public"
if [[ -f "${local_llama2_7b}/config.json" && -f "${local_llama2_7b}/tokenizer.json" ]]; then
  export MEMORYVLA_LLAMA2_7B_PATH="${MEMORYVLA_LLAMA2_7B_PATH:-${local_llama2_7b}}"
fi
