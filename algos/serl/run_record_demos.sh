#!/bin/bash
# GUI teleop demo recorder for Task1 → SERL .pkl
#
# Uses Isaac Lab → Isaac Sim 4.5 (same kit as plain PPO via isaaclab.sh).
# DirectRLEnv does NOT open_stage(manual_franka_scene.usda); it spawns Lab
# Franka/DIMM/Socket under env_0 and composes Task1 Floor+Table backdrop.
# Isaac Sim 5.0 Script-Editor teleop of the USDA is a separate workflow
# (tasks/task1_dimm_insert/scene/README.md) and does not emit 30-D Lab obs.
#
# Do NOT use --headless. Click the Isaac viewport after launch for keyboard focus.
# Keep IsaacLab Environment Index = 0 (num_envs=1).
set -eo pipefail
ES=/mnt/data2/wjy_SERL/Embodied_Safety
ISAACLAB=/mnt/data2/workspace_pyx/LNP_Locomotion-main/IsaacLab
SUCCESS_NEEDED=${SUCCESS_NEEDED:-10}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-80}
CONTROL_MODE=${CONTROL_MODE:-diff_ik}
OUT_DIR=${OUT_DIR:-$ES/outputs/demos/task1_dimm_insert}
SCENE_PATH=${SCENE_PATH:-$ES/tasks/task1_dimm_insert/scene}
SCENE_USDA="$SCENE_PATH/manual_franka_scene.usda"
BACKDROP_USD="$ES/tasks/task1_dimm_insert/env/assets/lab_task1_backdrop.usda"
if [[ ! -f "$SCENE_USDA" ]]; then
  echo "Missing confirmed Task1 scene: $SCENE_USDA"
  exit 4
fi
if [[ ! -f "$BACKDROP_USD" ]]; then
  echo "Missing Task1 Lab backdrop (Floor+Table): $BACKDROP_USD"
  exit 4
fi
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d " ")
if [[ "${USED:-0}" -gt 12000 ]]; then
  echo "GPU memory used ${USED} MiB looks busy. Free the card or set IGNORE_GPU_BUSY=1."
  if [[ "${IGNORE_GPU_BUSY:-0}" != "1" ]]; then
    nvidia-smi
    exit 3
  fi
fi
source /home/ufdwaa/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export TERM=xterm
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ES:${PYTHONPATH:-}"
mkdir -p "$OUT_DIR"
cd "$ISAACLAB"
echo "[run_record_demos] KIT=IsaacLab->_isaac_sim 4.5.0 (plain PPO same; NOT Isaac Sim 5.0 teleop entry)"
echo "[run_record_demos] SCENE_PATH=$SCENE_PATH"
echo "[run_record_demos] SCENE_USDA=$SCENE_USDA (authority; safety JSON — not open_stage)"
echo "[run_record_demos] BACKDROP_USD=$BACKDROP_USD (Floor+Table composed into Lab stage)"
echo "[run_record_demos] SUCCESS_NEEDED=$SUCCESS_NEEDED OUT_DIR=$OUT_DIR"
echo "[run_record_demos] After Isaac opens: click viewport, keep Environment Index=0, drive with arrow keys."
echo "[run_record_demos] Stage should show /World/Task1Backdrop (Table) + /World/envs/env_0."
exec ./isaaclab.sh -p "$ES/algos/serl/record_demos.py" \
  --success-needed "$SUCCESS_NEEDED" \
  --max-attempts "$MAX_ATTEMPTS" \
  --control-mode "$CONTROL_MODE" \
  --out-dir "$OUT_DIR" \
  --scene-path "$SCENE_PATH" \
  "$@"
