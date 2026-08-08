# RoboMME isolated environment setup

This guide installs RoboMME without changing the working MemoryVLA environment.
The two projects require incompatible Python and PyTorch versions, so they must
not share one virtual environment.

| Project | Environment | Python | PyTorch |
| --- | --- | --- | --- |
| MemoryVLA | `myMemoryVLA/.venv` | 3.10 | 2.2 + CUDA 12.1 |
| RoboMME | `robomme_benchmark/.venv-robomme` | 3.11 | 2.9.1 + CUDA 12.8 |

Separate environments are compatible with evaluation because the model and
simulator can run as separate processes and exchange observations/actions
through a server/client interface.

## 1. Choose the workspace

```bash
export MEMORYVLA_ROOT=/workspace/multimodal/myMemoryVLA
cd "$MEMORYVLA_ROOT"
```

`MEMORYVLA_ROOT` is a convenience variable used by the commands below. It
does not change either Python environment.

## 2. Record the working MemoryVLA environment

These files are a reproducibility record and recovery reference. The commands
only read the environment; they do not install or uninstall anything.

```bash
cd "$MEMORYVLA_ROOT"

.venv/bin/python -m pip freeze > memoryvla-before-robomme-pip.txt

.venv/bin/python -c 'import json, platform, sys, torch; print(json.dumps({"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda}, indent=2))' > memoryvla-before-robomme-runtime.json

git rev-parse HEAD > memoryvla-before-robomme-git-revision.txt
```

- `pip freeze` records exact Python package versions.
- The runtime JSON records Python, operating-system, Torch, and CUDA versions.
- `git rev-parse HEAD` records the exact MemoryVLA source revision.

This checkout already contains these snapshot files, but rerunning the commands
is useful after intentional MemoryVLA environment changes.

## 3. Clone RoboMME

Skip this step when `robomme_benchmark` already exists.

```bash
cd "$MEMORYVLA_ROOT"
git clone https://github.com/RoboMME/robomme_benchmark.git
cd robomme_benchmark
```

Do not clone RoboMME over the MemoryVLA repository and do not install its
dependencies with `myMemoryVLA/.venv/bin/pip`.

## 4. Bootstrap uv privately

The following creates a small, separate environment used only to run `uv`:

```bash
cd "$MEMORYVLA_ROOT/robomme_benchmark"

python3 -m venv .uv-bootstrap
.uv-bootstrap/bin/python -m pip install uv
```

- `.uv-bootstrap` does not contain RoboMME itself.
- Installing `uv` here avoids changing the system Python or MemoryVLA.
- All paths beginning with a dot stay inside the RoboMME directory.

## 5. Install the locked RoboMME environment

```bash
cd "$MEMORYVLA_ROOT/robomme_benchmark"

UV_PROJECT_ENVIRONMENT=.venv-robomme \
UV_CACHE_DIR=.uv-cache \
UV_PYTHON_INSTALL_DIR=.uv-python \
.uv-bootstrap/bin/uv sync --frozen --python 3.11
```

Explanation:

- `UV_PROJECT_ENVIRONMENT=.venv-robomme` places RoboMME packages in a
  dedicated virtual environment instead of MemoryVLA's `.venv`.
- `UV_CACHE_DIR=.uv-cache` keeps downloaded wheels and build artifacts local
  to RoboMME.
- `UV_PYTHON_INSTALL_DIR=.uv-python` keeps uv's managed Python 3.11 local to
  RoboMME.
- `sync --frozen` installs exactly the versions in `uv.lock` and fails
  rather than silently changing the lockfile.
- `--python 3.11` selects RoboMME's required Python series.

Do not subsequently run commands such as `pip install -U torch`,
`pip install -U sapien`, or `pip install -U mani-skill`. RoboMME pins a
custom ManiSkill source and depends on its locked package set.

## 6. Verify the correct environment

```bash
cd "$MEMORYVLA_ROOT/robomme_benchmark"

.venv-robomme/bin/python --version

.venv-robomme/bin/python -c 'import sys, torch; print("Python:", sys.version.split()[0]); print("Torch:", torch.__version__); print("Torch CUDA:", torch.version.cuda)'

UV_CACHE_DIR=.uv-cache \
.uv-bootstrap/bin/uv pip check \
  --python .venv-robomme/bin/python

.venv-robomme/bin/python -c 'import importlib.metadata as m; print("RoboMME:", m.version("robomme")); print("ManiSkill:", m.version("mani-skill")); print("SAPIEN:", m.version("sapien")); print("OpenCV:", m.version("opencv-python"))'
```

On the tested installation, the important versions are:

```text
Python 3.11.15
Torch 2.9.1+cu128
ManiSkill 3.0.0b21
SAPIEN 3.0.2
OpenCV 4.11.0.86
```

`uv pip check` should finish with:

```text
All installed packages are compatible
```

Always include `--python .venv-robomme/bin/python` in the package-check
command. Without it, uv may inspect an already activated MemoryVLA environment
instead.

## 7. Run commands in the RoboMME environment

The most explicit method is to invoke its Python directly:

```bash
cd "$MEMORYVLA_ROOT/robomme_benchmark"
.venv-robomme/bin/python scripts/run_example.py
```

Alternatively, run through uv while preserving the isolated paths:

```bash
cd "$MEMORYVLA_ROOT/robomme_benchmark"

UV_PROJECT_ENVIRONMENT=.venv-robomme \
UV_CACHE_DIR=.uv-cache \
UV_PYTHON_INSTALL_DIR=.uv-python \
.uv-bootstrap/bin/uv run scripts/run_example.py
```

Or activate it in a dedicated terminal:

```bash
cd "$MEMORYVLA_ROOT/robomme_benchmark"
source .venv-robomme/bin/activate
python scripts/run_example.py
deactivate
```

Do not activate MemoryVLA's `.venv` and RoboMME's `.venv-robomme` in the
same terminal session.

## 8. GPU and Vulkan smoke test

First confirm the GPU is visible:

```bash
nvidia-smi
```

Then run RoboMME's example:

```bash
cd "$MEMORYVLA_ROOT/robomme_benchmark"

UV_PROJECT_ENVIRONMENT=.venv-robomme \
UV_CACHE_DIR=.uv-cache \
UV_PYTHON_INSTALL_DIR=.uv-python \
.uv-bootstrap/bin/uv run scripts/run_example.py
```

This step initializes SAPIEN/ManiSkill rendering and uses the GPU. Do not run it
while another important evaluation occupies the same GPU unless sufficient
VRAM is available and slower execution is acceptable.

A successful smoke test should finish a rollout and write a video under
`sample_run_videos`.

## 9. MemoryVLA-to-RoboMME integration

Installing both environments is only the dependency setup. It does not by
itself connect the MemoryVLA policy to RoboMME.

The current MemoryVLA checkout does not contain
`script/eval/robomme/server.sh` or `client.sh`. The official RoboMME project
links to a separately adapted MemoryVLA implementation. Use that adapter as the
reference, or add an equivalent server/client bridge here.

The intended runtime separation is:

```text
MemoryVLA process (.venv)
    model and checkpoint
            |
            | HTTP/WebSocket observations and actions
            v
RoboMME process (.venv-robomme)
    ManiSkill, SAPIEN, and benchmark environment
```

The two processes may use different Python, Torch, and CUDA runtime package
versions because they communicate over a network protocol rather than importing
one another.

## 10. Useful maintenance commands

Check the RoboMME environment:

```bash
cd "$MEMORYVLA_ROOT/robomme_benchmark"
UV_CACHE_DIR=.uv-cache .uv-bootstrap/bin/uv pip check \
  --python .venv-robomme/bin/python
```

Reproduce the locked environment after cloning on another machine:

```bash
python3 -m venv .uv-bootstrap
.uv-bootstrap/bin/python -m pip install uv

UV_PROJECT_ENVIRONMENT=.venv-robomme \
UV_CACHE_DIR=.uv-cache \
UV_PYTHON_INSTALL_DIR=.uv-python \
.uv-bootstrap/bin/uv sync --frozen --python 3.11
```

Inspect disk usage:

```bash
du -sh .venv-robomme .uv-cache .uv-python .uv-bootstrap
```

The cache can be recreated from `uv.lock`, but do not delete it during an
active installation. The `.venv-robomme` directory is the actual installed
RoboMME environment.
