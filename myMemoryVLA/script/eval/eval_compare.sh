set -euo pipefail

BASELINE_CKPT
FULL_CKPT

COMMON_ENV = (
    GPU_ID=0
    EPISODE_START=0
    EPISODE_END=24
    USE_BF16=True
)

echo "Evaluating Baseline"

env "${COMMON_ENV[@]}" \
    CKPT_PATH="$BASELINE_CKPT" \
    EXPERIMENT_MODE=baseline \
    bash script/eval/bridge/eval_bridge.sh

echo "Evaluating the full model"

env "${COMMON_ENV[@]}" \
    CKPT_PATH="$FULL_CKPT" \
    EXPERIMENT_MODE=full \
    bash script/eval/bridge/eval_bridge.sh