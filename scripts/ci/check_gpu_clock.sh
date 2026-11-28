#!/usr/bin/env bash
# Fail unless every visible GPU reports the locked graphics clock.
set -euo pipefail

TARGET_CLOCK_MHZ=1500
RETRIES=5

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "::error::nvidia-smi is not available on this runner"
  exit 1
fi

for attempt in $(seq 1 "${RETRIES}"); do
  if ! rows=$(nvidia-smi --query-gpu=index,clocks.current.graphics --format=csv,noheader,nounits); then
    echo "::error::nvidia-smi failed to report GPU clocks"
    exit 1
  fi
  gpus=0
  locked=0
  while IFS=',' read -r gpu_index gpu_clock; do
    [ -n "${gpu_index//[[:space:]]/}" ] || continue
    gpu_index="$(echo "${gpu_index}" | xargs)"
    gpu_clock="$(echo "${gpu_clock}" | xargs)"
    echo "GPU ${gpu_index} clock: ${gpu_clock} MHz (attempt ${attempt}/${RETRIES})"
    gpus=$((gpus + 1))
    if [ "${gpu_clock}" = "${TARGET_CLOCK_MHZ}" ]; then
      locked=$((locked + 1))
    fi
  done <<< "${rows}"
  # An empty query reads zero rows, which must fail rather than pass.
  if [ "${gpus}" -gt 0 ] && [ "${locked}" -eq "${gpus}" ]; then
    exit 0
  fi
  sleep 2
done
echo "::error::GPU frequency validation failed. Expected ${TARGET_CLOCK_MHZ} MHz on every GPU."
exit 1
