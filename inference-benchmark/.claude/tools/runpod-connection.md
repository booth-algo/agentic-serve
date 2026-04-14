# RunPod — Connection & Pod Management

## SSH Access

Two methods to connect to RunPod pods:

### Gateway SSH (interactive only)
```bash
ssh <POD_ID>-<SUFFIX>@ssh.runpod.io -i ~/.ssh/id_ed25519_runpod
```
- Works for interactive sessions
- Does NOT support non-interactive commands (returns "Your SSH client doesn't support PTY")

### Direct TCP SSH (full access)
```bash
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519_runpod
```
- Full non-interactive access (needed for scp, remote commands)
- Requires sshd running in the container
- IP and port visible in RunPod dashboard under "TCP Port Mappings"

### SSH Key
Always use `~/.ssh/id_ed25519_runpod` for RunPod connections.

## Pod Setup

After SSH into a new pod:
```bash
# Option 1: Run setup script
bash <(curl -s https://raw.githubusercontent.com/sequrity-ai/inference-benchmark/main/scripts/pod_setup.sh)

# Option 2: Manual
git clone https://github.com/sequrity-ai/inference-benchmark.git /workspace/inference-benchmark
cd /workspace/inference-benchmark
pip install -r requirements.txt
mkdir -p results
```

The setup script (`scripts/pod_setup.sh`) handles:
- Cloning/updating the repo to `/workspace/inference-benchmark`
- Installing Python dependencies
- Creating results directory
- Supports branch selection: `./scripts/pod_setup.sh feature/x`

## Docker Images

| Image | Engine | Status |
|-------|--------|--------|
| `boothalgo01/bench-vllm:latest` | vLLM | Built, sshd included |
| `boothalgo01/bench-sglang:latest` | SGLang | TODO |
| `boothalgo01/bench-trtllm:latest` | TRT-LLM | TODO |

Custom images include `openssh-server` and an entrypoint that:
- Writes `$PUBLIC_KEY` env var to `/root/.ssh/authorized_keys`
- Starts sshd automatically
- Set `$PUBLIC_KEY` in RunPod template for auto SSH access

## Pod Limitations

- **Docker-in-Docker blocked:** RunPod containers have seccomp restrictions (`unshare` syscall denied). Install engines natively via pip instead.
- **GPU visibility:** Some 2xGPU pods only expose 1 GPU to the container. Check with `nvidia-smi`. Use `--tensor-parallel-size 2` with Ray if both GPUs are visible.
- **Network volume:** Persistent storage at `/workspace/` survives pod restarts. Models stored at `/workspace/models/`.

## vLLM Proxy URL

RunPod exposes HTTP ports via proxy:
```
https://<POD_ID>-<PORT>.proxy.runpod.net
```
Example: `https://szfphnq91bba5v-8000.proxy.runpod.net`

## Model Storage

Models are downloaded to `/workspace/models/` on the network volume.
See `.claude/tools/model-registry.md` for the full list of downloaded models and access requirements.
