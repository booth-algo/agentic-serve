#!/usr/bin/env bash
# FlashAttention NCU sweep — serving shapes. FA2 requires sm_80+ (A100, H100, RTX3090).
# NOT compatible with RTX 2080Ti (sm_75).
#
# Usage (run directly on each GPU host):
#   bash collect_flash_serving.sh H100
#   bash collect_flash_serving.sh A100
#   bash collect_flash_serving.sh RTX3090
#
# NCU path auto-detected per host; override with NCU=/path/to/ncu.
set -euo pipefail

GPU="${1:?usage: $0 GPU [dtype]}"
DTYPE="${2:-bf16}"

# Auto-detect NCU path per host
if [[ -z "${NCU:-}" ]]; then
    if [[ "$GPU" == "H100" ]]; then
        NCU="/usr/local/cuda-12.6/bin/ncu"
    elif [[ "$GPU" == "A100" ]]; then
        NCU="/usr/local/cuda-12.6/bin/ncu"
    elif [[ "$GPU" == "RTX3090" ]]; then
        NCU="/opt/nvidia/nsight-compute/2024.3.2/ncu"
    else
        NCU="ncu"  # fallback to PATH
    fi
fi

PY="${PY:-python3}"
OUT_BASE="/tmp/ncu_flash_serving_${GPU}"

# Locate the sweep script relative to the agentic-serve repo
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SWEEP_PY="$REPO_ROOT/llm_predict/sweep/sweep_flash_serving.py"

echo "[*] GPU=$GPU  dtype=$DTYPE  ncu=$NCU  py=$PY"
echo "[*] repo=$REPO_ROOT"
echo "[*] sweep=$SWEEP_PY"
echo "[*] out=$OUT_BASE"

"$NCU" \
    --target-processes all \
    --metrics "gpu__time_duration.sum,dram__bytes.sum,launch__grid_size,launch__block_size,launch__registers_per_thread" \
    --kernel-name-base "demangled" \
    --kernel-name "regex:flash_fwd|flash_bwd|fmha|pytorch_flash" \
    -o "$OUT_BASE" --force-overwrite \
    "$PY" "$SWEEP_PY" \
        --gpu "$GPU" \
        --dtype "$DTYPE" \
        --out-manifest "${OUT_BASE}.manifest.json"

echo "[*] exporting csv..."
"$NCU" --import "${OUT_BASE}.ncu-rep" --csv > "${OUT_BASE}.csv"

echo "[+] done:"
echo "    ${OUT_BASE}.ncu-rep"
echo "    ${OUT_BASE}.csv ($(wc -l < "${OUT_BASE}.csv") lines)"
echo "    ${OUT_BASE}.manifest.json"
