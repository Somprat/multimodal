#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_root}"

python_cmd="${PYTHON_CMD:-python3.10}"
if ! command -v "${python_cmd}" >/dev/null 2>&1; then
  echo "Python 3.10 is required (set PYTHON_CMD if it has a different name)." >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  "${python_cmd}" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip wheel 'setuptools<81'
.venv/bin/python -m pip install \
  torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pip install -r script/setup/requirements-runpod-eval.txt
.venv/bin/python -m pip install -e third_libs/SimplerEnv/ManiSkill2_real2sim
.venv/bin/python -m pip install -e third_libs/SimplerEnv

asset_repo="${project_root}/.runtime/ManiSkill2_real2sim-assets"
asset_revision="ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682"
if [[ ! -d "${asset_repo}/.git" ]]; then
  mkdir -p "${project_root}/.runtime"
  git clone https://github.com/simpler-env/ManiSkill2_real2sim.git "${asset_repo}"
fi
git -C "${asset_repo}" fetch --depth 1 origin "${asset_revision}"
git -C "${asset_repo}" checkout --detach "${asset_revision}"
ln -sfn ../../../.runtime/ManiSkill2_real2sim-assets/data \
  third_libs/SimplerEnv/ManiSkill2_real2sim/data

if ! command -v vulkaninfo >/dev/null 2>&1 && [[ ! -x .runtime/root/usr/bin/vulkaninfo ]]; then
  echo "Warning: vulkaninfo is missing. Install the Vulkan/GLVND packages listed in the repository README." >&2
fi

echo "Bootstrap complete. Run: source script/setup/env.sh"
