# Fresh RunPod setup for MemoryVLA + SimplerEnv

This guide rebuilds the tested environment from a fresh RunPod and explains what each command does. The tested stack used an NVIDIA L40S, Python 3.10, PyTorch 2.2.0 with CUDA 12.1, SAPIEN 2.2.2, and NumPy 1.26.4.

## 1. Choose and inspect the Pod

Attach a persistent `/workspace` volume with at least 60 GB available. The MemoryVLA checkpoint alone is 33,507,496,130 bytes; the virtual environment, caches, assets, and evaluation output need more room.

```bash
nvidia-smi
python3.10 --version
df -h /workspace
```

- `nvidia-smi` proves that the container can see the GPU and NVIDIA driver.
- The Python command confirms the interpreter used by the tested dependency set.
- `df` confirms that the persistent volume is mounted and large enough.

## 2. Install system and graphics dependencies

On a Debian/Ubuntu RunPod image:

```bash
apt-get update
apt-get install -y \
  git git-lfs python3.10-venv build-essential \
  libvulkan1 vulkan-tools \
  libgl1 libegl1 libglx0 libopengl0 libwayland-client0
```

`apt-get update` refreshes the package index; it does not upgrade the whole machine. The install command adds:

- `git` for source and simulation assets, and `git-lfs` for repositories using large-file storage;
- `python3.10-venv` for an isolated Python environment;
- `build-essential` for Python packages containing native extensions;
- `libvulkan1` (the Vulkan loader) and `vulkan-tools` (including `vulkaninfo`);
- GLVND/OpenGL/EGL/Wayland libraries that SAPIEN's offscreen renderer may load.

SAPIEN is the Python simulator, while Vulkan is the lower-level graphics system used to render camera images. A successful `pip install sapien` does not prove that the container has a working NVIDIA Vulkan driver path. This distinction caused much of the earlier setup difficulty.

Use the pinned SAPIEN version. This SimplerEnv/ManiSkill code targets an older API, so installing the newest version can introduce incompatible APIs or binary dependencies.

## 3. Clone the repository

```bash
cd /workspace
git clone https://github.com/Somprat/multimodal.git
cd /workspace/multimodal
export REPO_ROOT="$PWD"
```

The repository contains MemoryVLA, vendored SimplerEnv/ManiSkill source, and reproducible setup scripts. `REPO_ROOT` gives later commands an unambiguous absolute base directory. Keep everything under `/workspace` so it survives Pod replacement when the same persistent volume is attached.

## 4. Install the Python stack, simulator packages, and assets

```bash
cd "$REPO_ROOT/myMemoryVLA"
bash script/setup/bootstrap_runpod_eval.sh
```

The bootstrap script does the following, in order:

1. Creates `.venv` with Python 3.10, isolating the tested packages from the Pod's global Python.
2. Updates `pip`, `setuptools`, and `wheel`, the Python installation/build tools.
3. Installs the matched CUDA 12.1 builds of PyTorch 2.2.0, torchvision 0.17.0, and torchaudio 2.2.0.
4. Installs MemoryVLA in editable mode with `--no-deps`. Editable mode imports this checkout directly; `--no-deps` stops broad project metadata from replacing pinned evaluation packages.
5. Installs `script/setup/requirements-runpod-eval.txt`. Important pins include SAPIEN 2.2.2, NumPy 1.26.4, Transformers 4.40.1, TensorFlow 2.15.0, and Ruckig 0.12.2.
6. Installs the vendored `ManiSkill2_real2sim` and `SimplerEnv` packages in editable mode. ManiSkill defines simulated robots/tasks; SimplerEnv provides evaluation wrappers.
7. Clones the official ManiSkill2-real2sim repository at the exact tested revision into `.runtime/ManiSkill2_real2sim-assets`, then symlinks its `data` directory into the vendored package. Python source alone is not enough: the simulator needs robot, scene, object, and task assets.

The script reuses an existing `.venv` and asset clone if rerun after an interruption. FlashAttention is intentionally omitted because this evaluation path does not use it, and compiling it adds CUDA compatibility risk.

## 5. Load runtime variables in every new shell

```bash
cd "$REPO_ROOT/myMemoryVLA"
source script/setup/env.sh
```

`source` executes the script in the current shell, so its exports remain active. It:

- restores `NVIDIA_DRIVER_CAPABILITIES` from the container's PID 1 environment when SSH omits it;
- uses repository-local fallback graphics libraries if `.runtime/root` exists;
- sets the ManiSkill asset directory and NVIDIA Vulkan ICD descriptor;
- keeps Hugging Face caches and model files on persistent storage;
- selects the local public Llama config/tokenizer when available.

Run this command after every SSH reconnect or new terminal. Exported variables do not carry into a new shell automatically.

## 6. Verify each layer independently

```bash
nvidia-smi
vulkaninfo --summary
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
.venv/bin/python -c 'import sapien, mani_skill2_real2sim, simpler_env; print("simulation imports OK")'
.venv/bin/python -c 'from evaluation.simpler_env import VLAInference; print("VLAInference import OK")'
```

Interpret failures by layer:

- If `nvidia-smi` fails, fix the Pod/container GPU configuration; Python packages cannot solve it.
- If NVIDIA works but `vulkaninfo` fails, inspect the Vulkan/GLVND libraries and `/etc/vulkan/icd.d/nvidia_icd.json`.
- If Vulkan works but SAPIEN fails to import, inspect the Python version and pinned package install.
- If SAPIEN imports but ManiSkill/Simulator fails, inspect editable installs and the asset symlink.
- The final command tests the MemoryVLA-specific inference import chain.

Some RunPod images expose CUDA but omit usable NVIDIA Vulkan libraries. Prefer an image with correct NVIDIA Vulkan support. The tested checkout can use previously extracted fallback libraries under `.runtime/root`, which `env.sh` detects, but copying driver libraries between unrelated hosts is fragile because they must match the host driver.

## 7. Prepare the Llama config and tokenizer

MemoryVLA needs Llama 2 architecture/tokenizer files here, not a second copy of the full Llama weights. Meta's gated repository returned HTTP 403 while access was pending, so this workspace uses equivalent public files from `NousResearch/Llama-2-7b-hf` in:

```text
$REPO_ROOT/models/llama2-7b-public
```

Verify the selection:

```bash
source "$REPO_ROOT/myMemoryVLA/script/setup/env.sh"
echo "$MEMORYVLA_LLAMA2_7B_PATH"
```

It should print the local directory. Required files include `config.json`, `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`, and `special_tokens_map.json`. At writing time this directory is untracked, so recreate it on a truly fresh clone:

```bash
mkdir -p "$REPO_ROOT/models/llama2-7b-public"
"$REPO_ROOT/myMemoryVLA/.venv/bin/huggingface-cli" download \
  NousResearch/Llama-2-7b-hf \
  config.json generation_config.json \
  tokenizer.json tokenizer.model tokenizer_config.json special_tokens_map.json \
  --local-dir "$REPO_ROOT/models/llama2-7b-public"
source "$REPO_ROOT/myMemoryVLA/script/setup/env.sh"
```

`huggingface-cli download` fetches named files only, avoiding the multi-gigabyte base-model weights. `--local-dir` materializes normal files where MemoryVLA expects them. Never print or commit a Hugging Face token.

## 8. Download and verify the MemoryVLA Bridge model

```bash
df -h /workspace
cd "$REPO_ROOT/myMemoryVLA"
source script/setup/env.sh
.venv/bin/python script/setup/download_memvla_bridge.py
```

The Python downloader fetches only the official Bridge checkpoint and its required `config.yaml`, `config.json`, and `dataset_statistics.json` from `shihao1895/memvla-bridge`. It stores them under `models/model_b`, reuses the persistent Hugging Face cache, and creates `step-2.pt` as a symlink to the real checkpoint.

Allow at least 40 GiB free before starting. Download tools may need cache or temporary space in addition to the final 31.2 GiB file.

```bash
stat -c '%s bytes  %n' "$REPO_ROOT/models/model_b/checkpoints/memvla-bridge.pt"
readlink "$REPO_ROOT/models/model_b/checkpoints/step-2.pt"
```

The expected size is `33507496130` bytes and the symlink target is `memvla-bridge.pt`. These checks validate the download without loading the large checkpoint into RAM.

## 9. State reached after these steps

You now have:

- the pinned Python/CUDA evaluation environment;
- working SAPIEN, ManiSkill2-real2sim, SimplerEnv, Vulkan, and simulation assets;
- the local public Llama config/tokenizer path;
- the official MemoryVLA Bridge checkpoint and metadata.

The next step is a one-model checkpoint load/inference smoke test and then an episode-0 Bridge evaluation.

Model A is still unresolved. CogACT is not a faithful baseline with the current loader because MemoryVLA-specific modules would be missing and randomly initialized. Using the same official checkpoint for A and B can test plumbing, but is not a scientific comparison.

## Troubleshooting order

Diagnose from the bottom layer upward instead of reinstalling everything:

1. Storage: `df -h /workspace`.
2. GPU/driver: `nvidia-smi`.
3. Vulkan: `vulkaninfo --summary`.
4. Runtime variables: source `script/setup/env.sh`.
5. PyTorch CUDA visibility.
6. SAPIEN, then ManiSkill, then SimplerEnv imports.
7. ManiSkill asset symlink.
8. Model metadata and checkpoint size.
9. `MEMORYVLA_LLAMA2_7B_PATH` and tokenizer/config files.

A package reinstall cannot repair insufficient storage, a missing GPU device, or a broken Vulkan driver mapping.

## Problems encountered during the original setup

This section records the actual blockers encountered while building the working Pod, how they were diagnosed, and the commands used to recover. Commands involving package names may vary slightly between RunPod base images.

### 1. CUDA worked, but Vulkan/SAPIEN rendering did not

**Symptom:** `nvidia-smi` and PyTorch could see the L40S, but `vulkaninfo` or SAPIEN offscreen rendering could not find a usable NVIDIA Vulkan device. CUDA compute and Vulkan rendering are separate driver paths, so one can work while the other fails.

We checked the GPU, Vulkan descriptors, and libraries separately:

```bash
nvidia-smi
ls -l /etc/vulkan/icd.d /usr/share/vulkan/icd.d 2>/dev/null
sed -n '1,120p' /etc/vulkan/icd.d/nvidia_icd.json
ldconfig -p | grep -E 'libvulkan|libGLX_nvidia'
command -v vulkaninfo
```

The RunPod image lacked some usable user-space Vulkan/GLVND tools. The normal first repair is:

```bash
apt-get update
apt-get install -y \
  libvulkan1 vulkan-tools \
  libgl1 libegl1 libglx0 libopengl0 libwayland-client0
```

On the original Pod, system libraries were instead extracted into the repository-local `.runtime/root` fallback. This avoids depending on writable system directories:

```bash
cd "$REPO_ROOT/myMemoryVLA"
mkdir -p .runtime/debs .runtime/root
cd .runtime/debs
apt-get download \
  libvulkan1 vulkan-tools \
  libglvnd0 libgl1 libegl1 libglx0 libopengl0 libwayland-client0
for package in ./*.deb; do
  dpkg-deb -x "$package" ../root
done
source "$REPO_ROOT/myMemoryVLA/script/setup/env.sh"
.runtime/root/usr/bin/vulkaninfo --summary
```

`apt-get download` retrieves package archives without installing them. `dpkg-deb -x` extracts their files under `.runtime/root`. `env.sh` prepends the extracted library directory to `LD_LIBRARY_PATH`. This fallback supplies generic Vulkan/GLVND loader components; it does not invent an NVIDIA driver. The Pod must still expose a compatible host NVIDIA driver and `libGLX_nvidia.so.0`.

### 2. SSH shells lost `NVIDIA_DRIVER_CAPABILITIES`

**Symptom:** the container was launched with NVIDIA graphics support, but an SSH shell did not contain the same `NVIDIA_DRIVER_CAPABILITIES` variable. Graphics discovery then behaved differently between the container startup process and the interactive shell.

We inspected PID 1 without printing unrelated environment values:

```bash
tr '\0' '\n' < /proc/1/environ \
  | sed -n 's/^NVIDIA_DRIVER_CAPABILITIES=//p'
```

The fix is built into `env.sh`; manually it is:

```bash
export NVIDIA_DRIVER_CAPABILITIES="$(
  tr '\0' '\n' < /proc/1/environ \
    | sed -n 's/^NVIDIA_DRIVER_CAPABILITIES=//p'
)"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
```

This copies only the capability setting from the container's main process and explicitly selects the NVIDIA Vulkan ICD descriptor.

### 3. SAPIEN, ManiSkill, and SimplerEnv needed a compatible version set

**Symptom:** installing packages independently risked incompatible simulator APIs, NumPy binaries, TensorFlow constraints, or a different ManiSkill package than SimplerEnv expects.

The solution was to stop upgrading packages individually and install the tested pins through the repository bootstrap:

```bash
cd "$REPO_ROOT/myMemoryVLA"
bash script/setup/bootstrap_runpod_eval.sh
```

The important choices are SAPIEN 2.2.2, NumPy 1.26.4, Python 3.10, PyTorch 2.2.0/CUDA 12.1, and editable installs of the vendored ManiSkill2-real2sim and SimplerEnv source. We intentionally did not install FlashAttention because this evaluation path does not require it.

To see exactly what was installed rather than guessing:

```bash
.venv/bin/python -m pip show \
  torch sapien numpy mani-skill2-real2sim simpler-env transformers
.venv/bin/python -m pip check
```

`pip show` reports selected versions and locations. `pip check` reports broken or incompatible declared dependencies.

### 4. Simulator source installed, but task assets were missing

**Symptom:** Python imports could succeed while environment construction failed because robot models, objects, scenes, or task assets were not in the package's `data` directory.

The bootstrap clones the official assets at the tested revision and creates the expected symlink. The equivalent commands are:

```bash
cd "$REPO_ROOT/myMemoryVLA"
mkdir -p .runtime
git clone https://github.com/simpler-env/ManiSkill2_real2sim.git \
  .runtime/ManiSkill2_real2sim-assets
git -C .runtime/ManiSkill2_real2sim-assets fetch --depth 1 origin \
  ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682
git -C .runtime/ManiSkill2_real2sim-assets checkout --detach \
  ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682
ln -sfn ../../../.runtime/ManiSkill2_real2sim-assets/data \
  third_libs/SimplerEnv/ManiSkill2_real2sim/data
source script/setup/env.sh
```

The pinned revision prevents asset/source drift. The symlink places the assets where the vendored package expects them without duplicating a large directory. `env.sh` also exports `MS2_REAL2SIM_ASSET_DIR`.

Verify the link and target:

```bash
ls -ld third_libs/SimplerEnv/ManiSkill2_real2sim/data
readlink -f third_libs/SimplerEnv/ManiSkill2_real2sim/data
```

### 5. The checkpoint download failed with `Disk quota exceeded`

**Symptom:** Hugging Face attempted to preallocate the 33,507,496,130-byte checkpoint, but the original persistent volume quota was too small. A filesystem may report apparent free capacity while a per-volume quota still blocks the write.

We inspected both filesystem space and the largest workspace directories:

```bash
df -h /workspace
du -sh \
  "$REPO_ROOT/myMemoryVLA/.venv" \
  "$REPO_ROOT/myMemoryVLA/.runtime" \
  "$REPO_ROOT/models" \
  "$REPO_ROOT/.cache" 2>/dev/null
find "$REPO_ROOT/models/model_b" -name '*.incomplete' -type f -ls
```

The actual fix was to expand or replace the persistent volume, reattach it at `/workspace`, and rerun the repository downloader:

```bash
cd "$REPO_ROOT/myMemoryVLA"
source script/setup/env.sh
.venv/bin/python script/setup/download_memvla_bridge.py
```

The final validation was:

```bash
stat -c '%s bytes  %n' \
  "$REPO_ROOT/models/model_b/checkpoints/memvla-bridge.pt"
```

It now reports exactly `33507496130` bytes. Do not assume a `.pt` filename proves completion; always check its size and look for `.incomplete` files.

### 6. Meta Llama 2 access returned HTTP 403

**Symptom:** the Hugging Face token was syntactically valid, but `meta-llama/Llama-2-7b-hf` access was still awaiting Meta approval. Retrying authentication could not grant repository permission.

The evaluator only needs Llama's architecture config and tokenizer at this stage, so we downloaded those small files from the compatible public `NousResearch/Llama-2-7b-hf` mirror:

```bash
mkdir -p "$REPO_ROOT/models/llama2-7b-public"
"$REPO_ROOT/myMemoryVLA/.venv/bin/huggingface-cli" download \
  NousResearch/Llama-2-7b-hf \
  config.json generation_config.json \
  tokenizer.json tokenizer.model tokenizer_config.json special_tokens_map.json \
  --local-dir "$REPO_ROOT/models/llama2-7b-public"
source "$REPO_ROOT/myMemoryVLA/script/setup/env.sh"
echo "$MEMORYVLA_LLAMA2_7B_PATH"
```

`env.sh` exposes the local directory through `MEMORYVLA_LLAMA2_7B_PATH`, and the Llama backbone uses that override. We did not globally force Hugging Face offline mode because uncached vision/TIMM weights may still require network access.

### 7. Original model paths were not portable

**Symptom:** example commands referred to `/model/...`, which did not exist on a new Pod and would place state outside the persistent repository layout.

The solution was to resolve all paths from the checkout:

```bash
cd /workspace/multimodal
export REPO_ROOT="$PWD"
export MEMORYVLA_MODEL_ROOT="$REPO_ROOT/models"
```

Use paths such as:

```text
$REPO_ROOT/models/model_b/checkpoints/step-2.pt
```

This keeps checkpoints, caches, assets, and outputs under the mounted `/workspace` volume.

### 8. CogACT was not a valid Model A for this loader

**Symptom:** CogACT looked like a convenient public baseline, but the current evaluator constructs MemoryVLA-specific memory modules. Loading a checkpoint without those weights would leave randomly initialized modules active, producing a misleading comparison rather than a faithful CogACT baseline.

There is no installation command that fixes this conceptual incompatibility. The safe choices are:

- obtain a second compatible MemoryVLA-style checkpoint and its metadata;
- write a genuine CogACT inference adapter that does not construct/use MemoryVLA modules; or
- point both slots at the same checkpoint only for a clearly labeled plumbing test.

This remains unresolved and is intentionally recorded as a blocker for a scientific two-model comparison.
