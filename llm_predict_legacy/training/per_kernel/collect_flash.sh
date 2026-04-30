#!/usr/bin/env bash
# Run the FlashAttention sweep under ncu on a single host. FA2 requires
# sm_80+ — do NOT run on RTX 2080Ti (sm_75).
#
# Usage:
#   bash collect_flash.sh A100      # bf16
#   bash collect_flash.sh RTX3090   # bf16
set -euo pipefail

GPU="${1:?usage: $0 GPU [dtype]}"
DTYPE="${2:-bf16}"

NCU="${NCU:-ncu}"
PY="${PY:-python3}"

OUT_BASE="/tmp/ncu_flash_sweep_${GPU}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[*] GPU=$GPU  dtype=$DTYPE  ncu=$NCU  py=$PY"
echo "[*] sweep_flash.py=$SCRIPT_DIR/sweep_flash.py"
echo "[*] out_base=$OUT_BASE"

"$NCU" \
    --target-processes all \
    --metrics "gpu__time_duration.sum,dram__bytes.sum,launch__grid_size,launch__block_size,launch__registers_per_thread" \
    --kernel-name-base "demangled" \
    --kernel-name "regex:flash_fwd|flash_bwd|fmha|pytorch_flash" \
    -o "$OUT_BASE" --force-overwrite \
    "$PY" "$SCRIPT_DIR/sweep_flash.py" \
        --dtype "$DTYPE" \
        --out-manifest "${OUT_BASE}.manifest.json"

echo "[*] exporting csv..."
"$NCU" --import "${OUT_BASE}.ncu-rep" --csv > "${OUT_BASE}.csv"

echo "[+] done:"
echo "    ${OUT_BASE}.ncu-rep"
echo "    ${OUT_BASE}.csv ($(wc -l < "${OUT_BASE}.csv") lines)"
echo "    ${OUT_BASE}.manifest.json"
