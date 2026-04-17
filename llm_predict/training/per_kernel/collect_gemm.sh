#!/usr/bin/env bash
# Run the dense-model GEMM sweep under ncu on the current host. Emits a
# .ncu-rep + .csv + .manifest.json under /tmp keyed by a GPU short name.
#
# Usage on each GPU host:
#   bash collect_gemm.sh A100      # bf16 sweep
#   bash collect_gemm.sh RTX3090   # bf16
#   bash collect_gemm.sh RTX2080Ti fp16
#
# Env overrides:
#   NCU=/usr/local/cuda-12.6/bin/ncu                 (gpu-4)
#   NCU=/opt/nvidia/nsight-compute/2024.3.2/ncu      (3090/2080Ti)
#   PY=/path/to/python-with-torch
set -euo pipefail

GPU="${1:?usage: $0 GPU [dtype]}"
DTYPE="${2:-bf16}"

NCU="${NCU:-ncu}"
PY="${PY:-python3}"

OUT_BASE="/tmp/ncu_gemm_sweep_${GPU}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[*] GPU=$GPU  dtype=$DTYPE  ncu=$NCU  py=$PY"
echo "[*] sweep_gemm.py=$SCRIPT_DIR/sweep_gemm.py"
echo "[*] out_base=$OUT_BASE"

"$NCU" \
    --target-processes all \
    --metrics "gpu__time_duration.sum,dram__bytes.sum,launch__grid_size,launch__block_size,launch__registers_per_thread" \
    --kernel-name-base "demangled" \
    --kernel-name "regex:gemm|GEMM|sgemm|hgemm|fmha|flash" \
    -o "$OUT_BASE" --force-overwrite \
    "$PY" "$SCRIPT_DIR/sweep_gemm.py" \
        --dtype "$DTYPE" \
        --out-manifest "${OUT_BASE}.manifest.json"

echo "[*] exporting csv..."
"$NCU" --import "${OUT_BASE}.ncu-rep" --csv > "${OUT_BASE}.csv"

echo "[+] done:"
echo "    ${OUT_BASE}.ncu-rep"
echo "    ${OUT_BASE}.csv ($(wc -l < "${OUT_BASE}.csv") lines)"
echo "    ${OUT_BASE}.manifest.json"
