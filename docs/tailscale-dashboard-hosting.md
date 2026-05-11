# Tailscale Dashboard Hosting

Verified on 2026-05-11 03:03 UTC.

## Live Setup

The private dashboard is served by two layers:

1. `agentic-serve-dashboard.service` runs Vite preview on localhost.
2. Tailscale Serve proxies the tailnet HTTPS URL to that localhost port.

Current host state:

```text
tailscaled.service: enabled, active
agentic-serve-dashboard.service: enabled, active
agentic-serve-dashboard-refresh.timer: enabled, active
```

Current Tailscale Serve route:

```text
https://agenticserve.tail2bcc6a.ts.net (tailnet only)
|-- / proxy http://127.0.0.1:4180
```

Equivalent `tailscale serve status --json` shape:

```json
{
  "TCP": {
    "443": {
      "HTTPS": true
    }
  },
  "Web": {
    "agenticserve.tail2bcc6a.ts.net:443": {
      "Handlers": {
        "/": {
          "Proxy": "http://127.0.0.1:4180"
        }
      }
    }
  }
}
```

## Dashboard Service

The live service definition is checked in at:

```text
deploy/systemd/agentic-serve-dashboard.service
```

It runs:

```bash
cd /root/agentic-serve/inference-benchmark/dashboard
npm run preview -- --host 127.0.0.1 --port 4180
```

Vite serves the app at:

```text
http://127.0.0.1:4180/agentic-serve/
```

The dashboard uses `base: "/agentic-serve/"` and allows the tailnet hostname in
`inference-benchmark/dashboard/vite.config.ts`.

## Recovery Commands

From a fresh server that already has Tailscale installed and authenticated:

```bash
cd /root/agentic-serve/inference-benchmark/dashboard
npm ci
```

Build the local dashboard artifacts:

```bash
cd /root/agentic-serve/inference-benchmark
BENCHMARK_RESULTS_DIR=/mnt/100g/agent-bench/results \
BENCH_STATE_ROOT=/mnt/100g/agent-bench/state \
DASHBOARD_JSON_BASE=/agentic-serve \
MIRROR_R2=1 \
bash scripts/rebuild-local-dashboard.sh
```

Install and start the dashboard service:

```bash
cp /root/agentic-serve/deploy/systemd/agentic-serve-dashboard.service /etc/systemd/system/
cp /root/agentic-serve/deploy/systemd/agentic-serve-dashboard-refresh.service /etc/systemd/system/
cp /root/agentic-serve/deploy/systemd/agentic-serve-dashboard-refresh.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now agentic-serve-dashboard.service
systemctl enable --now agentic-serve-dashboard-refresh.timer
```

Recreate the Tailscale Serve route:

```bash
tailscale serve --yes --bg --https=443 http://127.0.0.1:4180
```

Verify:

```bash
systemctl status --no-pager agentic-serve-dashboard.service
systemctl status --no-pager agentic-serve-dashboard-refresh.timer
tailscale serve status
```

## Notes

Tailscale Serve config is persisted by `tailscaled`; there is no separate repo
script currently launching it on every boot. The boot-critical service is the
standard OS `tailscaled.service`, and the repo-owned part is the localhost
dashboard service plus refresh timer.

The legacy `/etc/systemd/system/agentperfbench-dashboard.service` unit was part
of the deleted AgenticServeNew setup. It should not exist on current installs.
If it is still present, disable and remove it:

```bash
systemctl disable --now agentperfbench-dashboard.service
rm -f /etc/systemd/system/agentperfbench-dashboard.service
systemctl daemon-reload
```

The only dashboard service that should remain is
`agentic-serve-dashboard.service`.
