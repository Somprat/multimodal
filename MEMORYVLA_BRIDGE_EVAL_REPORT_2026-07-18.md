# MemoryVLA SimplerEnv-Bridge evaluation report

Date: 2026-07-18 UTC

## Objective

Run the official MemoryVLA Bridge checkpoint locally on the complete SimplerEnv-Bridge benchmark, calculate its success rate, and compare it with the result published by the MemoryVLA authors.

## Model and runtime

- Model: `shihao1895/memvla-bridge`
- Checkpoint: `models/model_b/checkpoints/memvla-bridge.pt`
- Verified checkpoint size: `33,507,496,130` bytes
- Alias: `step-2.pt -> memvla-bridge.pt`
- GPU: NVIDIA L40S, 46 GB
- Driver: 580.159.04
- Python: 3.10.12
- PyTorch: 2.2.0+cu121
- CUDA runtime used by PyTorch: 12.1
- SAPIEN: 2.2.2
- NumPy: 1.26.4
- Evaluation mode: full precision (`torch.float32`)
- Simulator: SimplerEnv with ManiSkill2-real2sim assets at revision `ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682`

MemoryVLA uses the local public Llama 2 config/tokenizer files from `models/llama2-7b-public`; a second copy of the base Llama weights was not required. TIMM DINOv2 and SigLIP vision weights were downloaded and cached during the first model load.

## Protocol

The repository's official Bridge task configuration was used:

- Cube: `StackGreenCubeOnYellowCubeBakedTexInScene-v0`
- Carrot: `PutCarrotOnPlateInScene-v0`
- Spoon: `PutSpoonOnTableClothInScene-v0`
- Eggplant: `PutEggplantInBasketScene-v0`
- 24 object episode variations per task
- 96 total episodes
- Maximum 120 control steps per episode
- Control frequency 5 Hz; simulation frequency 500 Hz
- One NVIDIA L40S, GPU 0

The full run took approximately two hours, including four checkpoint loads. Each simulated episode took roughly one minute and produced a video plus an action visualization.

## Results

| Task | Local successes | Local rate | Published rate | Difference |
| --- | ---: | ---: | ---: | ---: |
| Spoon | 21/24 | 87.50% | 75.0% | +12.50 points |
| Carrot | 19/24 | 79.17% | 75.0% | +4.17 points |
| Cube | 9/24 | 37.50% | 37.5% | 0 points |
| Eggplant | 24/24 | 100.00% | 100.0% | 0 points |
| **Overall** | **73/96** | **76.04%** | **71.9%** | **+4.17 points** |

The official average is the rounded form of 71.875%. The local average is `73 / 96 = 76.0417%`.

## Interpretation

The local run exceeded the published overall result by 4.17 percentage points. Cube and Eggplant matched the published scores exactly, Carrot was one success higher, and Spoon was three successes higher.

This is not evidence that the downloaded model differs from the official checkpoint. The project explicitly warns that SimplerEnv and diffusion sampling can vary significantly between runs. A stronger reproducibility study would repeat all 96 episodes several times and report a mean, spread, and seeds/environment details.

## Persistent artifacts

Checkpoint and metadata:

```text
models/model_b/checkpoints/memvla-bridge.pt
models/model_b/checkpoints/step-2.pt
models/model_b/config.yaml
models/model_b/config.json
models/model_b/dataset_statistics.json
```

Task logs:

```text
models/model_b/eval_simpler/memvla-bridge.pt/Cube.txt
models/model_b/eval_simpler/memvla-bridge.pt/Carrot.txt
models/model_b/eval_simpler/memvla-bridge.pt/Spoon.txt
models/model_b/eval_simpler/memvla-bridge.pt/Eggplant.txt
```

Videos and action plots are stored in nested task directories below `models/model_b/eval_simpler/`.

## Problems found and resolved

1. Hugging Face left an oversized malformed `.incomplete` checkpoint. A direct retry also produced an abandoned partial, and both files consumed a hidden network-volume quota.
2. The model was downloaded cleanly to fast local `/tmp` storage with Xet disabled, exact-size verified, then copied once to persistent network storage. Only after verification were unusable partials removed.
3. The bootstrap had installed setuptools 83, while SAPIEN 2.2.2 still imports `pkg_resources`. Pinning `setuptools<81` restored imports.
4. `mediapy` could not find ffmpeg. The installed `imageio_ffmpeg` executable had a versioned filename, so a temporary `ffmpeg` symlink was added to `PATH`. Fresh Pods should install the system `ffmpeg` package.
5. A failed scene initialization created a zero-byte generated collision cache, `bridge_table_1_v1.glb.nonconvex.stl`. The source GLB was valid. Removing only the generated zero-byte derivative allowed SAPIEN to regenerate it.
6. Interrupting an early piped evaluation left child evaluator processes alive. Their overlapping checkpoint loads exhausted the L40S. Stale PIDs were identified through `nvidia-smi` and terminated before the clean run.
7. Headless GLFW warnings were benign because SAPIEN continued with Vulkan offscreen rendering.
8. The supplied Bridge shell script contained checkpoint placeholders and an extra `done`. It was corrected to source the runtime environment, use the repository virtual environment, default to the official checkpoint, accept `CKPT_PATH`, and fail fast.

## Reproduction

For a new Pod, follow `FRESH_RUNPOD_SETUP.md` from step 1 through step 14. After a clean smoke episode, the full evaluation is:

```bash
cd "$REPO_ROOT/myMemoryVLA"
source script/setup/env.sh
bash script/eval/bridge/eval_bridge.sh
```

Summarize the result with:

```bash
.venv/bin/python script/eval/bridge/extract_bridge_results.py \
  "$MEMORYVLA_MODEL_ROOT/model_b"
```

## Published comparison source

The MemoryVLA repository's SimplerEnv-Bridge table reports Spoon 75.0, Carrot 75.0, Cube 37.5, Eggplant 100.0, and 71.9 average:

<https://github.com/shihao1895/MemoryVLA#simplerenv-bridge>

The official checkpoint is hosted at:

<https://huggingface.co/shihao1895/memvla-bridge>
