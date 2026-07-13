# Multimodal robotics workspace

This repository contains the source, tested setup metadata, and resume notes for the MemoryVLA/SimplerEnv Bridge evaluation. It is intentionally self-contained at the repository level: scripts resolve paths relative to this checkout, model metadata lives under `models/`, and small smoke-test artifacts live under `artifacts/`.

Machine-specific virtual environments, caches, extracted system libraries, and model checkpoints are not committed. The official MemoryVLA Bridge checkpoint is about 31.2 GiB, which is too large for a normal GitHub repository. It can be restored into the expected repo-local location with the included downloader.

## Fresh clone

On the tested RunPod image (Python 3.10, CUDA 12.1, NVIDIA L40S):

```bash
git clone https://github.com/Somprat/multimodal.git
cd multimodal/myMemoryVLA
bash script/setup/bootstrap_runpod_eval.sh
source script/setup/env.sh
python script/setup/download_memvla_bridge.py
```

The bootstrap creates `myMemoryVLA/.venv`, installs the evaluation dependencies, clones the official ManiSkill2 Real2Sim assets at the tested revision, and links them into the vendored SimplerEnv tree. If the host image does not already provide Vulkan/GLVND, install `libvulkan1 vulkan-tools libgl1 libegl1 libglx0 libopengl0 libwayland-client0` with its system package manager. It does not install FlashAttention because the evaluation path does not use it.

The model downloader writes to `multimodal/models/model_b` by default. It requires roughly 35 GiB free and downloads the official `shihao1895/memvla-bridge` checkpoint plus its metadata. Hugging Face credentials should be supplied through the normal Hugging Face CLI/environment; never commit a token.

## Layout

- `myMemoryVLA/`: MemoryVLA source and vendored SimplerEnv source.
- `openpi/`: OpenPI source retained from the original workspace.
- `models/model_b/`: small, committed MemoryVLA Bridge metadata; downloaded checkpoints are ignored.
- `artifacts/simplerenv_rollout/`: small renderer/environment smoke-test outputs.
- `MEMORYVLA_HANDOFF.md`: detailed current state, blockers, and resume sequence.

Model A is still unresolved. Do not use CogACT as a scientific baseline with the current MemoryVLA loader; see the handoff for details.
