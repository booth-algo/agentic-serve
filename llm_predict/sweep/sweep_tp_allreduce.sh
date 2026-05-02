#!/usr/bin/env bash
# NCCL all-reduce latency microbenchmark for TP communication calibration.
#
# Measures all-reduce latency at tensor-parallel message sizes (hidden_dim
# × dtype bytes) for each model architecture. Run on each GPU host with
# the desired tensor-parallel size.
#
# Usage (run directly on GPU host):
#   bash sweep_tp_allreduce.sh H100 8    # TP=8 sweep
#   bash sweep_tp_allreduce.sh A100 4    # TP=4 sweep
set -euo pipefail

GPU="${1:?usage: $0 GPU TP_SIZE}"
TP="${2:?usage: $0 GPU TP_SIZE}"

PY="${PY:-python3}"
OUT="/tmp/tp_allreduce_${GPU}_tp${TP}.csv"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_PY="$SCRIPT_DIR/sweep_tp_allreduce.py"

echo "[*] GPU=$GPU  TP=$TP  out=$OUT"

"$PY" "$SWEEP_PY" \
    --tp "$TP" \
    --out "$OUT"

echo "[+] done: $OUT"
