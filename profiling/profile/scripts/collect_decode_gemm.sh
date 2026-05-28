#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-gemm_full}"
if [[ "$MODE" == "smoke" ]]; then
  MODE="gemm_smoke"
elif [[ "$MODE" == "full" ]]; then
  MODE="gemm_full"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/collect_decode_kernels.sh" "$MODE"
