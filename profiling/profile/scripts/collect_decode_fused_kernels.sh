#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-fused_full}"
if [[ "$MODE" == "smoke" ]]; then
  MODE="fused_smoke"
elif [[ "$MODE" == "full" ]]; then
  MODE="fused_full"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/collect_decode_kernels.sh" "$MODE"
