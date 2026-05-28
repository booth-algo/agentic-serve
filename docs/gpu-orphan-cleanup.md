# GPU Orphan Cleanup

Created: 2026-05-12 12:14:54 UTC

The cleanup loop is separate from the dashboard and reporter. The reporter stays
read-only and classifies orphaned same-user GPU holders as `same-user-orphan`.
The cleaner then takes a fresh live snapshot, applies extra gates, and records
an audit event before any cleanup action.

## Default Mode

The default config is:

`inference-benchmark/scripts/gpu_cleanup.json`

Important defaults:

- `dry_run: true`
- `min_age_seconds: 900`
- `required_observations: 2`
- `allowed_statuses: ["same-user-orphan"]`
- `allowed_parent_commands: ["VLLM::EngineCore"]`
- `allow_sigkill: false`
- `hosts: ["a100", "3090", "2080ti", "h100", "h100-2"]`

This means the system will not kill anything by default. It only logs what it
would clean after seeing the same candidate twice and after the process is at
least 15 minutes old.

## Audit Files

The cleaner writes:

- `/mnt/100g/agent-bench/state/gpu-orphan-observations.json`
- `/mnt/100g/agent-bench/state/gpu-cleanup-events.jsonl`

The observation store is the gating state. If a process disappears, it drops out
of the next observation store write.

## Manual Dry Run

```bash
python3 inference-benchmark/scripts/clean_orphan_gpus.py \
  --config inference-benchmark/scripts/gpu_cleanup.json \
  --jobs-config inference-benchmark/scripts/sweep.yaml \
  --scope synthetic_distributional \
  --state-dir /mnt/100g/agent-bench/state \
  --dry-run
```

For a one-off check without waiting for the second observation:

```bash
python3 inference-benchmark/scripts/clean_orphan_gpus.py \
  --config inference-benchmark/scripts/gpu_cleanup.json \
  --jobs-config inference-benchmark/scripts/sweep.yaml \
  --scope synthetic_distributional \
  --state-dir /mnt/100g/agent-bench/state \
  --required-observations 1 \
  --dry-run
```

## Actual Cleanup

Actual cleanup should only be enabled after the dry-run audit log shows the
candidates are correct.

```bash
python3 inference-benchmark/scripts/clean_orphan_gpus.py \
  --config inference-benchmark/scripts/gpu_cleanup.json \
  --jobs-config inference-benchmark/scripts/sweep.yaml \
  --scope synthetic_distributional \
  --state-dir /mnt/100g/agent-bench/state \
  --execute
```

The execute path sends `SIGTERM` to exact observed PIDs only. For the current
vLLM orphan shape, that means the orphaned `VLLM::EngineCore` parent and the GPU
worker PID. `SIGKILL` is disabled unless `allow_sigkill` is explicitly set to
`true`.

## Systemd

Draft units are provided but not automatically installed:

- `deploy/systemd/agentic-serve-gpu-orphan-cleaner.service`
- `deploy/systemd/agentic-serve-gpu-orphan-cleaner.timer`

The service uses the default dry-run config. Enabling the timer will create
observation and audit logs, but it will not kill processes unless the config is
changed or the service is edited to pass `--execute`.
