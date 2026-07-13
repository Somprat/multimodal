# MemoryVLA / SimplerEnv RunPod handoff

Updated: 2026-07-13 UTC

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

Its checkpoint is approximately 31.2 GiB (`33,507,496,130` bytes). The optimized Hugging Face transfer failed with `Disk quota exceeded` while preallocating the checkpoint. The partial checkpoint was reset/truncated, so no meaningful checkpoint payload remains. Only small metadata files may exist under `$REPO_ROOT/models/model_b`.

Use a persistent `/workspace` volume of at least 50 GB; 60 GB or more is recommended because the environment and assets already use roughly 11 GB and evaluation outputs need headroom. Confirm that the same persistent volume is mounted after restarting.

After enlarging storage:

```bash
df -h /workspace
du -sh /workspace
du -sh $REPO_ROOT/myMemoryVLA/.venv \
  $REPO_ROOT/myMemoryVLA/.runtime \
  $REPO_ROOT/models 2>/dev/null
```

Resume/download the official MemoryVLA checkpoint:

```bash
cd "$REPO_ROOT/myMemoryVLA"
source script/setup/env.sh
.venv/bin/python script/setup/download_memvla_bridge.py
```

For the requested filename, either pass the real checkpoint path to the script or create a symlink:

```bash
ln -sfn memvla-bridge.pt $REPO_ROOT/models/model_b/checkpoints/step-2.pt
```

Each model directory needs its associated `config.yaml`, `config.json`, and `dataset_statistics.json`, not only the `.pt` file.

## Hugging Face / Llama status

- `/workspace/.hf_token` was created and appears to contain a syntactically valid token, but Meta's `meta-llama/Llama-2-7b-hf` access returned HTTP 403 because access is awaiting review.
- Do not print or copy the token into logs or this handoff.
- A matching public config/tokenizer from `NousResearch/Llama-2-7b-hf` was cached in the original environment; a fresh clone may need to fetch it again into `$REPO_ROOT/.cache/huggingface/hub`.
- A cache alias for the canonical Meta repository exists and was verified with `HF_HUB_OFFLINE=1`.
- Do not globally enable offline mode yet: uncached vision/TIMM weights may still need network access.
- The clean next step is to make the evaluator use the cached public Llama config/tokenizer path (or an explicit mirror) while retaining online access for other weights.

## Model A is still unresolved

There is no genuine model A checkpoint yet. The published CogACT/CogACT-Large checkpoint is not a faithful baseline in the current evaluator: this loader constructs MemoryVLA memory modules, so missing MemoryVLA weights would be randomly initialized and used.

For a meaningful comparison, provide a second compatible MemoryVLA-style checkpoint and its metadata. To validate plumbing only, model A and model B may temporarily point at the same official checkpoint, but that is not a scientific comparison.

## Recommended resume sequence

1. Clone/pull this repository and run `myMemoryVLA/script/setup/bootstrap_runpod_eval.sh` if `.venv`, `.runtime`, or assets are absent.
2. Restore the environment variables shown above.
3. Re-run the Vulkan and import checks.
4. Confirm at least 40 GiB of usable quota remains, then download `shihao1895/memvla-bridge`.
5. Fix the Llama tokenizer/config lookup to use the cached public mirror without forcing all Hugging Face access offline.
6. Run a one-model load/inference smoke test with the official checkpoint.
7. Decide what valid compatible checkpoint should be model A.
8. Run `compare_two_models_bridge.sh` for episode 0 only.

## Prompt for the next Codex session

Paste this:

> Continue my MemoryVLA/SimplerEnv RunPod setup. First read `$REPO_ROOT/MEMORYVLA_HANDOFF.md` completely and inspect the existing state; do not reinstall or redownload working dependencies unless they are missing. Stay inside `/workspace`. I expanded/replaced the persistent volume. Resume the official MemoryVLA Bridge checkpoint download, solve the remaining Llama config/tokenizer lookup cleanly, smoke-test model loading, and then work toward the episode-0 two-model Bridge evaluation. Model A is not yet defined, so do not treat CogACT as a valid MemoryVLA baseline.
