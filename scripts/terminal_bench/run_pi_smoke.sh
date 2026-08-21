#!/usr/bin/env bash
set -euo pipefail

SUBSTRATE_ROOT="${SUBSTRATE_ROOT:?set SUBSTRATE_ROOT to the tb-with-pi checkout cloned as substrate}"
TB21_ROOT="${TB21_ROOT:?set TB21_ROOT to the terminal-bench-2-1 checkout}"
HARBOR_BIN="${HARBOR_BIN:-harbor}"
SMC_BASE_URL="${SMC_BASE_URL:-http://172.17.0.1:30000/v1}"
SMC_MODEL="${SMC_MODEL:-Qwen/Qwen3.5-9B}"
TASK_ID="${TASK_ID:-fix-git}"
TASK_IDS="${TASK_IDS:-$TASK_ID}"
PI_VERSION="${PI_VERSION:-0.84.2}"
CONCURRENCY="${CONCURRENCY:-1}"
TRIALS="${TRIALS:-1}"
FORCE_BUILD="${FORCE_BUILD:-false}"
JOBS_DIR="${JOBS_DIR:-$(dirname "${SUBSTRATE_ROOT}")/jobs}"
JOB_NAME="${JOB_NAME:-}"

if [[ "$(basename "${SUBSTRATE_ROOT}")" != "substrate" ]]; then
  echo "SUBSTRATE_ROOT must be cloned under the module name 'substrate'" >&2
  exit 2
fi

extra=()
if [[ "${FORCE_BUILD}" == "true" ]]; then
  extra+=(--force-build)
fi

include_args=()
IFS=',' read -r -a requested_tasks <<< "${TASK_IDS}"
for task in "${requested_tasks[@]}"; do
  task="${task//[[:space:]]/}"
  if [[ -n "${task}" ]]; then
    include_args+=(-i "${task}")
  fi
done
if [[ "${#include_args[@]}" -eq 0 ]]; then
  echo "TASK_IDS must contain at least one task name" >&2
  exit 2
fi
if [[ -n "${JOB_NAME}" ]]; then
  extra+=(--job-name "${JOB_NAME}")
fi

PYTHONPATH="$(dirname "${SUBSTRATE_ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" \
  "${HARBOR_BIN}" run \
    --yes \
    --jobs-dir "${JOBS_DIR}" \
    -p "${TB21_ROOT}/tasks" \
    "${include_args[@]}" \
    -a substrate.harbor.pi_local:PiLocal \
    -m "local-vllm/${SMC_MODEL}" \
    --ak "base_url=${SMC_BASE_URL}" \
    --ak thinking=off \
    --ak "version=${PI_VERSION}" \
    --ak capture=true \
    --allow-agent-host 172.17.0.1 \
    -e docker \
    -n "${CONCURRENCY}" \
    -k "${TRIALS}" \
    "${extra[@]}"
