#!/usr/bin/env bash
# =============================================================================
# Pod setup — clone inference-benchmark and install deps.
#
# Run once after a RunPod pod starts. Works with any pre-built engine image:
#   vLLM:    vllm/vllm-openai:latest
#   SGLang:  lmsysorg/sglang:latest
#   TRT-LLM: nvcr.io/nvidia/tritonserver:latest (with TRT-LLM backend)
#
# Usage:
#   bash <(curl -s https://raw.githubusercontent.com/<org>/inference-benchmark/main/scripts/pod_setup.sh)
#   # or after SSH:
#   ./scripts/pod_setup.sh [BRANCH]
#
# Examples:
#   ./scripts/pod_setup.sh            # clone main branch
#   ./scripts/pod_setup.sh dev        # clone dev branch
#   ./scripts/pod_setup.sh feature/x  # clone feature branch
#
# =============================================================================
set -euo pipefail

BRANCH="${1:-main}"
REPO="https://github.com/sequrity-ai/inference-benchmark.git"  # TODO: update org/repo
DEST="/workspace/inference-benchmark"

echo "=== inference-benchmark pod setup ==="
echo "Branch: $BRANCH"
echo "Dest:   $DEST"
echo ""

# Clone or update
if [[ -d "$DEST/.git" ]]; then
    echo "Repo exists — pulling latest..."
    cd "$DEST"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    echo "Cloning..."
    git clone -b "$BRANCH" "$REPO" "$DEST"
    cd "$DEST"
fi

# Install benchmark deps (engine is already installed in the base image)
echo ""
echo "Installing benchmark dependencies..."
pip install -q -r requirements.txt

# Create results dir
mkdir -p "$DEST/results"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Start the inference server for your engine:"
echo "       vLLM:   $DEST/scripts/launch_server.sh single-turn --model <MODEL>"
echo "       SGLang: python -m sglang.launch_server --model <MODEL> --port 8000"
echo ""
echo "  2. Run benchmarks:"
echo "       cd $DEST && ./benchmark.sh chatbot"
echo "       cd $DEST && ./benchmark.sh cross_validate"
echo ""
echo "  3. Results saved to: $DEST/results/"
