#!/bin/bash
# Task1 plain SERL-SAC (online-only by default). Does not kill other GPU jobs.
set -eo pipefail
ES=/mnt/data2/wjy_SERL/Embodied_Safety
ISAACLAB=/mnt/data2/workspace_pyx/LNP_Locomotion-main/IsaacLab
MAX_STEPS=${MAX_STEPS:-35000}
SEED=${SEED:-0}
OFFLINE_DEMO=${OFFLINE_DEMO:-none}
CONTROL_MODE=${CONTROL_MODE:-diff_ik}
START_STEPS=${START_STEPS:-1000}
LOGDIR=${LOGDIR:-$ES/outputs/runs/task1_dimm_insert/serl/task1_serl_sac_plain/seed${SEED}_$(date +%Y%m%d_%H%M%S)}
DEMO_PKL=${DEMO_PKL:-}
EXTRA=()
if [[ -n "${DEMO_PKL}" ]]; then
  EXTRA+=(--demo-pkl "$DEMO_PKL")
fi
if [[ "$OFFLINE_DEMO" == "required" && -z "${DEMO_PKL}" ]]; then
  echo "OFFLINE_DEMO=required but DEMO_PKL empty. See DEMO_REQUIREMENTS.md"
  exit 2
fi
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d " ")
if [[ "${USED:-0}" -gt 2000 ]]; then
  echo "GPU memory used ${USED} MiB > 2000. Refusing to start SERL (do not grab busy GPU)."
  nvidia-smi
  exit 3
fi
# Match kit_train_plain: conda env_isaaclab + isaaclab.sh
# Avoid set -u: conda activate / setup_conda_env.sh reference unbound ZSH_VERSION.
source /home/ufdwaa/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export TERM=xterm
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda}
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
export PYTHONPATH="$ES:${PYTHONPATH:-}"
mkdir -p "$LOGDIR"
cd "$ISAACLAB"
echo "[run_sac_online] python=$(which python)"
echo "[run_sac_online] LOGDIR=$LOGDIR"
exec ./isaaclab.sh -p "$ES/algos/serl/train_task1_serl.py" \
  --headless \
  --seed "$SEED" \
  --max_steps "$MAX_STEPS" \
  --start_steps "$START_STEPS" \
  --offline-demo "$OFFLINE_DEMO" \
  --control-mode "$CONTROL_MODE" \
  --logdir "$LOGDIR" \
  "${EXTRA[@]}" \
  "$@"
