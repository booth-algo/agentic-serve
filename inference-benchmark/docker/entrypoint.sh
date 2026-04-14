#!/usr/bin/env bash
# =============================================================================
# Entrypoint — clone/pull latest inference-benchmark code at container start.
#
# Environment variables:
#   BENCH_REPO    git repo URL (default: set at build time)
#   BENCH_BRANCH  branch to clone (default: main)
#   BENCH_DIR     where to clone (default: /workspace/inference-benchmark)
#
# Usage (docker run):
#   docker run -e BENCH_BRANCH=dev <image>
#   docker run -e BENCH_BRANCH=feature/sglang <image> ./benchmark.sh chatbot
# =============================================================================
set -euo pipefail

BENCH_REPO="${BENCH_REPO:-https://github.com/sequrity-ai/inference-benchmark.git}"
BENCH_BRANCH="${BENCH_BRANCH:-main}"
BENCH_DIR="${BENCH_DIR:-/workspace/inference-benchmark}"

echo "[entrypoint] Branch: $BENCH_BRANCH"

if [[ -d "$BENCH_DIR/.git" ]]; then
    echo "[entrypoint] Updating existing clone..."
    cd "$BENCH_DIR"
    git fetch --all
    git checkout -B "$BENCH_BRANCH" "origin/$BENCH_BRANCH"
else
    echo "[entrypoint] Cloning $BENCH_REPO..."
    git clone -b "$BENCH_BRANCH" "$BENCH_REPO" "$BENCH_DIR"
    cd "$BENCH_DIR"
    pip install -q -r requirements.txt
fi

mkdir -p "$BENCH_DIR/results"
cd "$BENCH_DIR"

# SSH setup — RunPod injects public key via $PUBLIC_KEY env var
if command -v sshd &>/dev/null; then
    mkdir -p /root/.ssh /run/sshd
    chmod 700 /root/.ssh
    if [[ -n "${PUBLIC_KEY:-}" ]]; then
        echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
        chmod 600 /root/.ssh/authorized_keys
    fi
    service ssh start 2>/dev/null || /usr/sbin/sshd
    echo "[entrypoint] sshd started"
fi

# If a command was passed, run it. Otherwise sleep forever (keeps container alive for SSH).
if [[ $# -gt 0 ]]; then
    exec "$@"
else
    echo "[entrypoint] Ready. Working dir: $BENCH_DIR"
    exec sleep infinity
fi
