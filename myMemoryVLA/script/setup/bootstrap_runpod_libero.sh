#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
libero_root="${project_root}/third_libs/LIBERO"
libero_revision="8f1084e3132a39270c3a13ebe37270a43ece2a01"
python_bin="${project_root}/.venv/bin/python"

if [[ ! -x "${python_bin}" ]]; then
  echo "Missing ${python_bin}; run script/setup/bootstrap_runpod_eval.sh first." >&2
  exit 1
fi

if [[ ! -d "${libero_root}/.git" ]]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${libero_root}"
fi
git -C "${libero_root}" checkout "${libero_revision}"

"${python_bin}" -m pip install -r "${project_root}/script/setup/requirements-runpod-libero.txt" --no-deps
"${python_bin}" -m pip install -e "${libero_root}" --no-deps

config_root="${project_root}/.runtime/libero"
mkdir -p "${config_root}"
config_file="${config_root}/config.yaml"
printf '%s\n' \
  "assets: ${libero_root}/libero/libero/assets" \
  "bddl_files: ${libero_root}/libero/libero/bddl_files" \
  "benchmark_root: ${libero_root}/libero/libero" \
  "datasets: ${libero_root}/libero/datasets" \
  "init_states: ${libero_root}/libero/libero/init_files" \
  > "${config_file}"

echo "LIBERO source and Python packages are ready."
echo "Install libosmesa6-dev libgl1-mesa-dev libglu1-mesa-dev on the host."
echo "Source script/setup/env.sh, then run script/setup/smoke_test_libero.py."
