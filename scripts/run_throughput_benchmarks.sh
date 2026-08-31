#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BENCHMARK_DIR="artifacts/benchmarks"
mkdir -p "${BENCHMARK_DIR}"

BENCHMARK_MODE="${BENCHMARK_MODE:-both}"
ENV_COUNTS="${ENV_COUNTS:-1,8,32,64}"
STEP_WARMUP_STEPS="${STEP_WARMUP_STEPS:-64}"
STEP_MEASURED_STEPS="${STEP_MEASURED_STEPS:-256}"
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

case "${BENCHMARK_MODE}" in
  step)
    LOG_PREFIX="throughput_step"
    MODE_LABEL="step-throughput"
    ;;
  ppo)
    LOG_PREFIX="throughput_ppo"
    MODE_LABEL="ppo-throughput"
    ;;
  both)
    LOG_PREFIX="throughput_both"
    MODE_LABEL="combined-throughput"
    ;;
  *)
    echo "Unsupported BENCHMARK_MODE: ${BENCHMARK_MODE}" >&2
    echo "Expected one of: step, ppo, both" >&2
    exit 1
    ;;
esac

for task in "${TASKS[@]}"; do
  log_path="${BENCHMARK_DIR}/${LOG_PREFIX}_${task}.log"
  task_env_counts="${TASK_ENV_COUNTS[$task]}"
  if [[ -n "${ENV_COUNTS_OVERRIDE:-}" ]]; then
    task_env_counts="${ENV_COUNTS_OVERRIDE}"
  fi
  echo "Running ${MODE_LABEL} benchmark for ${task}..."
  python examples/python/throughput_benchmark.py \
    --mode "${BENCHMARK_MODE}" \
    --tasks "${task}" \
    --env-counts "${task_env_counts}" \
    --step-warmup-steps "${STEP_WARMUP_STEPS}" \
    --step-measured-steps "${STEP_MEASURED_STEPS}" \
    --ppo-warmup-updates "${PPO_WARMUP_UPDATES}" \
    --ppo-update-count "${PPO_UPDATE_COUNT}" \
    --ppo-rollout-steps "${PPO_ROLLOUT_STEPS}" \
    --device "${DEVICE}" \
    > "${log_path}" 2>&1
  echo "Saved log to ${log_path}"
done

echo "All ${MODE_LABEL} benchmarks completed."
