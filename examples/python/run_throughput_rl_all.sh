#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p artifacts

ENV_COUNTS="${ENV_COUNTS:-1,8,32,64}"
PPO_WARMUP_UPDATES="${PPO_WARMUP_UPDATES:-5}"
PPO_UPDATE_COUNT="${PPO_UPDATE_COUNT:-15}"
PPO_ROLLOUT_STEPS="${PPO_ROLLOUT_STEPS:-128}"
DEVICE="${DEVICE:-cuda}"

TASKS=(
  cartpole
  soft_body_push
  fluid_pour
  target_center
  tissue_retract
  blood_suction
  ultrasound_scan
)

declare -A TASK_ENV_COUNTS=(
  [cartpole]="1,8,32,64,128,256,512,1024,2048,4096,8192"
  [soft_body_push]="1,8,32,64,128,256,512,1024"
  [fluid_pour]="1,8,32,64,128,256"
  [target_center]="1,8,32,64,128,256"
  [tissue_retract]="1,8,32,64,128,256"
  [blood_suction]="1,8,32,64,128,256"
  [ultrasound_scan]="1,8,32,64"
)

for task in "${TASKS[@]}"; do
  log_path="artifacts/throughput_rl_${task}.log"
  task_env_counts="${TASK_ENV_COUNTS[$task]}"
  if [[ -n "${ENV_COUNTS_OVERRIDE:-}" ]]; then
    task_env_counts="${ENV_COUNTS_OVERRIDE}"
  fi
  echo "Running PPO throughput benchmark for ${task}..."
  python examples/python/throughput_benchmark.py \
    --mode ppo \
    --tasks "${task}" \
    --env-counts "${task_env_counts}" \
    --ppo-warmup-updates "${PPO_WARMUP_UPDATES}" \
    --ppo-update-count "${PPO_UPDATE_COUNT}" \
    --ppo-rollout-steps "${PPO_ROLLOUT_STEPS}" \
    --device "${DEVICE}" \
    > "${log_path}" 2>&1
  echo "Saved log to ${log_path}"
done

echo "All PPO throughput benchmarks completed."
