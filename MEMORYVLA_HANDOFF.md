# MemoryVLA / SimplerEnv + LIBERO RunPod handoff

Updated: 2026-07-29 UTC

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

## Completed official Bridge evaluation (2026-07-18)

The official `shihao1895/memvla-bridge` checkpoint completed the full SimplerEnv-Bridge evaluation: four tasks with 24 object episodes each, 96 episodes total, full precision on one NVIDIA L40S.

| Task | Local successes | Local rate | Published rate | Difference |
| --- | ---: | ---: | ---: | ---: |
| Spoon | 21/24 | 87.50% | 75.0% | +12.50 points |
| Carrot | 19/24 | 79.17% | 75.0% | +4.17 points |
| Cube | 9/24 | 37.50% | 37.5% | 0 points |
| Eggplant | 24/24 | 100.00% | 100.0% | 0 points |
| Overall | 73/96 | 76.04% | 71.9% | +4.17 points |

Persistent logs are under `$REPO_ROOT/models/model_b/eval_simpler/memvla-bridge.pt/`. The dated report is `$REPO_ROOT/MEMORYVLA_BRIDGE_EVAL_REPORT_2026-07-18.md`.

Important fixes learned during the run:

- Malformed and interrupted Hugging Face partials exhausted a hidden network-volume quota even though `df` showed free cluster capacity.
- Staging the download on `/tmp`, exact-size verifying it, and copying it once to network storage was reliable.
- SAPIEN 2.2.2 needs `pkg_resources`; the bootstrap now pins `setuptools<81`.
- Video writing needs an executable named `ffmpeg`; install it or expose the `imageio_ffmpeg` binary through a symlink.
- A failed simulation created a zero-byte generated `.nonconvex.stl`; deleting only that derivative allowed regeneration from the valid GLB.
- Interrupted pipelines can leave evaluator children holding GPU memory. Check `nvidia-smi` before retrying.
- Headless GLFW warnings are expected when Vulkan offscreen rendering continues.
- The Bridge script now uses the repository environment/checkpoint, accepts `CKPT_PATH`, and fails fast.

## LIBERO setup status (2026-07-22)

Goal: reproduce the MemoryVLA LIBERO evaluation path, starting with the
published Spatial checkpoint and eventually running the official benchmark.

### Completed and verified

- Cloned official LIBERO under `myMemoryVLA/third_libs/LIBERO` at commit
  `8f1084e3132a39270c3a13ebe37270a43ece2a01`. This clone is machine-local
  and ignored by Git; the bootstrap script recreates it.
- Installed the required Mesa/OSMesa host libraries and pinned Python
  dependencies in the existing `myMemoryVLA/.venv`.
- Added a reproducible bootstrap, pinned requirements, checkpoint downloader,
  and simulator smoke test under `myMemoryVLA/script/setup/`.
- `script/setup/env.sh` now configures LIBERO paths and defaults
  `MUJOCO_GL=osmesa`. EGL did not initialize on this Pod; OSMesa works.
- The real LIBERO simulator smoke test passed twice. Reset returned a
  `(64, 64, 3)` uint8 agent-view image and one environment step completed
  with `reward=0`, `done=False`.
- `deploy.py` now provides `/health` after the model is loaded. The LIBERO
  launcher uses that health check, repository defaults, environment
  overrides, checkpoint validation, and cleanup traps.
- Shell and Python syntax checks passed for the new and modified LIBERO files.

Recreate and test the simulator on a new Pod:

```bash
cd "$REPO_ROOT/myMemoryVLA"
bash script/setup/bootstrap_runpod_libero.sh
source script/setup/env.sh
.venv/bin/python script/setup/smoke_test_libero.py
```

Nonfatal warnings include LIBERO's absent dataset directory, robosuite's
missing private macro file, and duplicate TensorFlow CUDA factory
registration. The demonstration dataset is not required for benchmark
rollouts.

### Checkpoint and completed evaluation

The official Spatial model is `shihao1895/memvla-libero-spatial`. Its checkpoint is complete and exact-size verified at `33,507,487,606` bytes:

```text
$REPO_ROOT/models/memvla-libero-spatial/checkpoints/memvla-libero-spatial.pt
```

Its `config.yaml`, `config.json`, and `dataset_statistics.json` are present. On a fresh Pod, download it with:

```bash
cd "$REPO_ROOT/myMemoryVLA"
source script/setup/env.sh
.venv/bin/python script/setup/download_memvla_libero.py spatial \
  --output /tmp/memvla-libero-spatial
```

The 10-task/one-trial policy smoke completed at 10/10. The official protocol then completed all 10 tasks x 50 trials, 500 episodes total, seed 7, with action-chunk window 8.

| Result | Local | Published | Difference |
| --- | ---: | ---: | ---: |
| LIBERO Spatial | **494/500 (98.8%)** | **98.4%** | **+0.4 points** |

The complete result log is:

```text
$REPO_ROOT/models/memvla-libero-spatial/eval_libero/memvla-libero-spatial.pt/libero_spatial-50trials-seed7-ac8-2026_07_25-00_00_15.txt
```

All 500 rollout MP4s are under the matching `_videos/` directory. The run took about 3.5 hours on one NVIDIA L40S in full precision.

The first attempt on the replacement Pod failed before episode 1 because OSMesa host libraries were absent. This fixed it:

```bash
apt-get update
apt-get install -y libosmesa6-dev libgl1-mesa-dev libglu1-mesa-dev
```

The simulator smoke passed after the repair. For future unattended runs, use `tmux` so closing SSH cannot terminate the evaluation shell.

Policy smoke:

```bash
cd "$REPO_ROOT/myMemoryVLA"
source script/setup/env.sh
NUM_TRIALS_PER_TASK=1 TASK_SUITE_NAME=libero_spatial \
  bash script/eval/libero/eval_libero.sh
```

Official 500-episode benchmark:

```bash
NUM_TRIALS_PER_TASK=50 TASK_SUITE_NAME=libero_spatial \
  bash script/eval/libero/eval_libero.sh
```

### Docker decision

Docker is not installed on this RunPod. OpenPI's existing LIBERO Compose image
evaluates OpenPI pi0.5, not MemoryVLA, so it cannot be used unchanged. A
MemoryVLA image would still need NVIDIA runtime/GPU passthrough, checkpoint
mounts, model server/client orchestration, and Mesa/OSMesa libraries. Reusing
the working repository `.venv` is currently the shortest reproducible path.

### Files from this LIBERO pass

Modified: `deploy.py`, `evaluation/libero/eval_libero.py`,
`script/eval/libero/eval_libero.sh`, `script/setup/env.sh`, and both
relevant `.gitignore` files.

Added under `myMemoryVLA/script/setup/`:
`bootstrap_runpod_libero.sh`, `download_memvla_libero.py`,
`requirements-runpod-libero.txt`, and `smoke_test_libero.py`.

The tracked modification to
`myMemoryVLA/evaluation/simpler_env/simpler_env_inference.py` predates this
LIBERO pass and was deliberately left untouched.

## Spatial dataset plumbing progress (2026-07-29)

Goal: validate the new depth/intrinsics -> point cloud -> spatial encoder ->
spatial memory/retrieval path without launching full VLA training.

### Key contract confirmed

- Raw RLDS configuration uses `depth_obs_keys["primary"]` for metric depth and
  either `camera_intrinsics_key` for a per-step `[3, 3]` matrix or
  `camera_intrinsics` for a static matrix.
- The normalized observation names are `depth_primary` and
  `camera_intrinsics`. `RLDSBatchTransform` emits model fields `depth` and
  `intrinsic`; inference passes the latter to the model as `intrinsics`.
- SimplerEnv runtime observations use
  `obs["image"][camera_name]["depth"]` and
  `obs["camera_param"][camera_name]["intrinsic_cv"]`, with
  `sensor_param`/sensor lookup fallbacks. WidowX/Bridge uses
  `camera_name="3rd_view_camera"`.
- The bundled Bridge calibration is:

```python
[[623.588, 0.0, 319.501],
 [0.0, 623.588, 239.545],
 [0.0, 0.0, 1.0]]
```

### Tiny RLDS spatial fixture

Added a deterministic, renderer-independent ManiSkill-shaped RLDS fixture:

```text
myMemoryVLA/artifacts/tiny_maniskill_rlds/tiny_maniskill_spatial/1.0.0
```

It contains two episodes with four steps each and raw observation keys
`image`, `depth`, `camera_intrinsics`, and `proprio`. Added:

```text
myMemoryVLA/script/create_tiny_maniskill_rlds.py
myMemoryVLA/script/inspect_rlds_observation.py
myMemoryVLA/script/smoke_test_tiny_spatial_rlds.py
```

Registered `tiny_maniskill_spatial` in the OXE config with:

```python
"depth_obs_keys": {"primary": "depth", "secondary": None, "wrist": None},
"camera_intrinsics_key": "camera_intrinsics",
```

The lightweight workflow passed:

```text
depth=(1, 64, 64, 1)
intrinsics=(1, 3, 3)
points=(1, 4096, 3), valid_points=4096
spatial_tokens=(1, 4, 32)
memory=(1, 1, 4, 32)
retrieval=step_0, score=0.9750
Tiny spatial RLDS workflow: OK
```

Regenerate and retest:

```bash
cd "$REPO_ROOT/myMemoryVLA"
.venv/bin/python -m script.create_tiny_maniskill_rlds
PYTHONPATH=. .venv/bin/python script/smoke_test_tiny_spatial_rlds.py \
  artifacts/tiny_maniskill_rlds/tiny_maniskill_spatial/1.0.0
```

### RoboCasa investigation

Downloaded the 100 smallest episodes (511 MiB locally) from
`daixianjie/robocasa_human_lerobot`:

```text
myMemoryVLA/artifacts/robocasa_lerobot_100
```

Both `meta/info.json` and raw Parquet metadata were inspected. Actual stored
keys are:

```text
image_left, image_right, wrist_image
state, actions
timestamp, frame_index, episode_index, index, task_index
```

There is no stored metric depth, camera intrinsic matrix, calibration field,
MuJoCo state, or model XML in this LeRobot redistribution. Therefore there is
no truthful `ACTUAL_DEPTH_KEY` or `ACTUAL_INTRINSICS_KEY`, and it cannot enter
the point-cloud workflow as downloaded. Do not configure invented keys or use
RGB intensity as depth.

The official RoboCasa365 LeRobot format likewise documents RGB streams,
low-dimensional state/action data, and language, but not depth/intrinsics.
Creating usable spatial training data requires raw demonstrations retaining
MuJoCo state/model metadata, installing RoboCasa assets, replaying states, and
exporting rendered metric depth plus the matching camera matrix into a new
RLDS conversion. Recommended explicit conversion keys are `depth` and
`camera_intrinsics`, after which the tiny-fixture config applies.

### Renderer note

An attempted local SimplerEnv render during this pass exited with Vulkan
incompatibility/code 139 even though CUDA sees the L40S and
`NVIDIA_DRIVER_CAPABILITIES=all`. This conflicts with the earlier known-good
renderer state documented above, so restore `script/setup/env.sh` and verify
`vulkaninfo --summary` before attempting replay-based dataset generation.

## Recommended resume sequence

1. Follow `FRESH_RUNPOD_SETUP.md` for a new Pod.
2. Source `myMemoryVLA/script/setup/env.sh` and rerun Vulkan and import checks.
3. Ensure `$REPO_ROOT/models/llama2-7b-public` is present.
4. Verify the checkpoint size is exactly `33507496130` bytes and metadata is present.
5. Check `nvidia-smi` for stale evaluator processes.
6. Run the one-episode smoke command in steps 10-11 of `FRESH_RUNPOD_SETUP.md`.
7. Run `bash script/eval/bridge/eval_bridge.sh` only after a clean smoke test.
8. Extract and preserve all four task logs.
9. Resolve a genuinely compatible Model A only if a two-model comparison is needed.

## Current exact stopping point

- SAPIEN, Vulkan rendering, checkpoint loading, diffusion inference, video writing, and extraction have passed.
- The official Bridge checkpoint and metadata are persistently installed and exact-size verified.
- The local Llama config/tokenizer and TIMM vision weights are present.
- The full 96-episode Bridge evaluation completed with 76.04% overall success.
- The full 500-episode LIBERO Spatial evaluation completed with 98.8% success (494/500), compared with 98.4% published.
- Do not rerun it on a new Pod unless another stochastic sample is desired.
- Model A remains unresolved; a same-checkpoint run tests plumbing, not scientific performance.

## Prompt for the next Codex session

Paste this:

> Continue my MemoryVLA RunPod work. Read `$REPO_ROOT/MEMORYVLA_HANDOFF.md`, `$REPO_ROOT/FRESH_RUNPOD_SETUP.md`, and `$REPO_ROOT/MEMORYVLA_BRIDGE_EVAL_REPORT_2026-07-18.md` completely; inspect existing state and do not reinstall, redownload, or rerun working components unnecessarily. The full Bridge evaluation completed at 76.04% (73/96), and the full LIBERO Spatial evaluation completed at 98.8% (494/500). Source `myMemoryVLA/script/setup/env.sh`, verify persistent paths and GPU/renderer health, then help with the next requested experiment. Model A remains undefined, so do not treat CogACT as a valid MemoryVLA baseline.
