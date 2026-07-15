# MemoryVLA / SimplerEnv RunPod handoff

Updated: 2026-07-15 UTC

All resumable source, metadata, setup scripts, and small test artifacts now live
inside this repository. After cloning, set the repository root:

```bash
export REPO_ROOT="$(pwd)"
```
Machine-local environments, caches, assets, and checkpoints are recreated by
`myMemoryVLA/script/setup/`; the 31.2 GiB checkpoint cannot live in normal Git.

## Goal

Run a Bridge evaluation comparing two model checkpoints:

```bash
bash "$REPO_ROOT/myMemoryVLA/script/eval/bridge/compare_two_models_bridge.sh" \
  --model-a $REPO_ROOT/models/model_a/checkpoints/step-1.pt \
  --model-b $REPO_ROOT/models/model_b/checkpoints/step-2.pt \
  --gpu 0 \
  --episode-start 0 \
  --episode-end 1 \
  --use-bf16
```

Keep generated state inside the repository checkout. The original `/model/...` paths were examples and are not portable.

Repository:

```text
$REPO_ROOT/myMemoryVLA
```

## What already works

- RunPod has an NVIDIA L40S and the NVIDIA driver/CUDA are visible.
- PyTorch is `2.2.0+cu121`.
- `NVIDIA_DRIVER_CAPABILITIES=all` is present in PID 1's environment, although SSH shells may need to import it explicitly.
- Vulkan was repaired using GLVND/Vulkan libraries extracted under `.runtime/root`.
- `vulkaninfo` detects the L40S when the environment below is set.
- SAPIEN 2.2.2 offscreen rendering passed (`64x64x4`, float32, finite output).
- Official SimplerEnv/ManiSkill assets were cloned under `.runtime/ManiSkill2_real2sim-assets`.
- A Bridge environment smoke test passed for `widowx_spoon_on_towel`; reset produced a `(480, 640, 3)` uint8 observation and one random step completed.
- `from evaluation.simpler_env import VLAInference` imports successfully.

The repo virtual environment is:

```text
$REPO_ROOT/myMemoryVLA/.venv
```

Important installed versions include:

```text
torch 2.2.0
sapien 2.2.2
mani-skill2-real2sim 0.5.3
simpler-env 0.0.1
tensorflow 2.15.0
tensorflow-metadata 1.14.0
transformers 4.40.1
ruckig 0.12.2
numpy 1.26.4
opencv-python 4.11.0.86
scipy 1.12.0
imageio 2.37.0
```

FlashAttention was intentionally not installed because this evaluation path does not require it.

## Environment to restore after reconnecting

```bash
cd $REPO_ROOT/myMemoryVLA

export NVIDIA_DRIVER_CAPABILITIES="$(tr '\0' '\n' < /proc/1/environ \
  | sed -n 's/^NVIDIA_DRIVER_CAPABILITIES=//p')"
export LD_LIBRARY_PATH="$PWD/.runtime/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export MS2_REAL2SIM_ASSET_DIR="$PWD/third_libs/SimplerEnv/ManiSkill2_real2sim/data"
export HF_HUB_CACHE=$REPO_ROOT/.cache/huggingface/hub
export TRANSFORMERS_CACHE=$REPO_ROOT/.cache/huggingface/hub
```

Basic checks:

```bash
echo "$NVIDIA_DRIVER_CAPABILITIES"
nvidia-smi
.runtime/root/usr/bin/vulkaninfo --summary
.venv/bin/python -c 'from evaluation.simpler_env import VLAInference; print("VLAInference import OK")'
```

## Storage blocker and checkpoint state

The public MemoryVLA Bridge model is:

```text
shihao1895/memvla-bridge
https://huggingface.co/shihao1895/memvla-bridge
```

Its checkpoint is approximately 31.2 GiB (`33,507,496,130` bytes). The earlier download failed with `Disk quota exceeded`, but the download completed after moving to the larger persistent volume. The verified file is `$REPO_ROOT/models/model_b/checkpoints/memvla-bridge.pt`, with exactly `33,507,496,130` bytes. The alias `step-2.pt -> memvla-bridge.pt` and all three metadata files are present. No download process is running because the download is complete.

Use a persistent `/workspace` volume of at least 50 GB; 60 GB or more is recommended because the environment and assets already use roughly 11 GB and evaluation outputs need headroom. Confirm that the same persistent volume is mounted after restarting.

On every fresh or replaced Pod, verify storage before downloading:

```bash
df -h /workspace
du -sh /workspace
du -sh $REPO_ROOT/myMemoryVLA/.venv \
  $REPO_ROOT/myMemoryVLA/.runtime \
  $REPO_ROOT/models 2>/dev/null
```

Restore/download the official MemoryVLA checkpoint on a fresh Pod:

```bash
cd "$REPO_ROOT/myMemoryVLA"
source script/setup/env.sh
.venv/bin/python script/setup/download_memvla_bridge.py
```

The downloader creates the requested alias automatically. To recreate it manually:

```bash
ln -sfn memvla-bridge.pt $REPO_ROOT/models/model_b/checkpoints/step-2.pt
```

Each model directory needs its associated `config.yaml`, `config.json`, and `dataset_statistics.json`, not only the `.pt` file.

## Hugging Face / Llama status

- `/workspace/.hf_token` was created and appears to contain a syntactically valid token, but Meta's `meta-llama/Llama-2-7b-hf` access returned HTTP 403 because access is awaiting review.
- Do not print or copy the token into logs or this handoff.
- A matching public config/tokenizer from `NousResearch/Llama-2-7b-hf` is materialized at `$REPO_ROOT/models/llama2-7b-public`; no separate Llama weights are required here.
- `script/setup/env.sh` exports `MEMORYVLA_LLAMA2_7B_PATH` for this directory, and the Llama backbone reads the override. The directory is currently untracked and must be committed or recreated for a fresh clone.
- Do not globally enable offline mode yet: uncached vision/TIMM weights may still need network access.
- The next step is to smoke-test model loading with the local Llama path while retaining online access for other weights.

## Model A is still unresolved

There is no genuine model A checkpoint yet. The published CogACT/CogACT-Large checkpoint is not a faithful baseline in the current evaluator: this loader constructs MemoryVLA memory modules, so missing MemoryVLA weights would be randomly initialized and used.

For a meaningful comparison, provide a second compatible MemoryVLA-style checkpoint and its metadata. To validate plumbing only, model A and model B may temporarily point at the same official checkpoint, but that is not a scientific comparison.

## Recommended resume sequence

1. Follow `FRESH_RUNPOD_SETUP.md` for a new Pod.
2. Source `myMemoryVLA/script/setup/env.sh` and rerun Vulkan and import checks.
3. Ensure `$REPO_ROOT/models/llama2-7b-public` is present.
4. Run a one-model load/inference smoke test with the completed official checkpoint.
5. Resolve any model-loading issue before launching an episode.
6. Decide what valid compatible checkpoint should be model A.
7. Run `compare_two_models_bridge.sh` for episode 0 only.

## Current exact stopping point

- SAPIEN, ManiSkill2-real2sim, Vulkan rendering, environment reset, and a random step have passed.
- The official MemoryVLA Bridge checkpoint and metadata are complete.
- The local public Llama config/tokenizer is present and wired into the loader.
- Next: one-model checkpoint load/inference smoke test.
- Model A remains unresolved; a same-checkpoint run tests plumbing, not scientific performance.

## Prompt for the next Codex session

Paste this:

> Continue my MemoryVLA/SimplerEnv RunPod setup. Read `$REPO_ROOT/MEMORYVLA_HANDOFF.md` and `$REPO_ROOT/FRESH_RUNPOD_SETUP.md` completely and inspect existing state; do not reinstall or redownload working components. The official Bridge checkpoint and local public Llama config/tokenizer are present. Source `myMemoryVLA/script/setup/env.sh`, verify Vulkan/imports, then perform a one-model checkpoint load/inference smoke test and work toward episode 0. Model A is undefined, so do not treat CogACT as a valid MemoryVLA baseline.
