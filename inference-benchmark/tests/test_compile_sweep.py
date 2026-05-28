import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import compile_sweep  # noqa: E402


def base_manifest():
    return {
        "hosts": {
            "a100": {
                "hardware_label": "A100-40GB",
                "model_root": "/models",
                "python": "/python",
                "python_sglang": "/python-sglang",
                "total_gpus": 8,
                "vram_gb_per_gpu": 40,
            }
        },
        "models": {"Qwen3.5-27B": {"dir": "Qwen3.5-27B", "weights_gb": 54}},
        "presets": {
            "single": {
                "max_len": 8192,
                "gpu_mem": 0.9,
                "concurrencies": [1],
                "profiles": ["chat-singleturn"],
            }
        },
        "feasibility_ratio": 0.9,
        "known_oom": [
            {
                "host": "a100",
                "model": "Qwen3.5-27B",
                "tp": 2,
                "backend": "vllm",
                "reason": "vllm-only failure",
            }
        ],
        "cells": [
            {
                "host": "a100",
                "model": "Qwen3.5-27B",
                "tp": 2,
                "mode": "single",
                "preset": "single",
            },
            {
                "host": "a100",
                "model": "Qwen3.5-27B",
                "tp": 2,
                "mode": "single",
                "backend": "sglang",
                "preset": "single",
            },
        ],
    }


class CompileSweepTests(unittest.TestCase):
    def test_known_oom_can_be_backend_specific(self):
        emitted, skipped = compile_sweep.compile_jobs(base_manifest(), "current")

        self.assertEqual([cell.get("backend", "vllm") for cell, _row in emitted], ["sglang"])
        self.assertEqual(len(skipped), 1)
        skipped_cell, status, reason = skipped[0]
        self.assertEqual(skipped_cell.get("backend", "vllm"), "vllm")
        self.assertEqual(status, "known_oom")
        self.assertEqual(reason, "vllm-only failure")

    def test_synthetic_concurrencies_override_trace_replay_default(self):
        manifest = base_manifest()
        manifest["presets"]["fixed_single"] = dict(manifest["presets"]["single"])
        manifest["cells"] = [
            {
                "host": "a100",
                "model": "Qwen3.5-27B",
                "tp": 2,
                "mode": "single",
                "backend": "sglang",
                "preset": "fixed_single",
                "synthetic_concurrencies": [1, 10, 20],
            }
        ]

        emitted, skipped = compile_sweep.compile_jobs(manifest, "synthetic_distributional")

        self.assertEqual(skipped, [])
        self.assertEqual(len(emitted), 1)
        cell, _row = emitted[0]
        record = compile_sweep.job_record(cell, manifest)
        self.assertEqual(record["data_scope"], "synthetic_distributional")
        self.assertEqual(record["concurrencies"], [1, 10, 20])
        self.assertEqual(record["profiles"], ["chat-singleturn-synth"])
        self.assertNotIn("synthetic_concurrencies", cell)

    def test_real_sweep_has_no_missing_feasible_fixed_or_current_cells(self):
        manifest = compile_sweep.load_manifest(SCRIPT_DIR / "sweep.yaml")
        hosts = {str(host): config for host, config in manifest["hosts"].items()}
        models = {str(model): config for model, config in manifest["models"].items()}
        host_order = ("a100", "3090", "2080ti", "h100")
        tp_candidates = (1, 2, 4)
        modes = ("single", "multi")
        backends = ("vllm", "sglang")

        def cell_scope(cell):
            scope = (
                cell.get("data_scope")
                or cell.get("dashboard_scope")
                or cell.get("scope")
                or ("fixed" if str(cell.get("preset", "")).startswith("fixed_") else "current")
            )
            return str(scope)

        existing = {"current": set(), "fixed": set()}
        for cell in manifest["cells"]:
            scope = cell_scope(cell)
            if scope not in existing:
                continue
            existing[scope].add(
                (
                    str(cell["host"]),
                    str(cell["model"]),
                    int(cell["tp"]),
                    str(cell["mode"]),
                    str(cell.get("backend", "vllm")),
                )
            )

        missing = []
        for host in host_order:
            for model in models:
                for tp in tp_candidates:
                    budget_gb = hosts[host]["vram_gb_per_gpu"] * tp * manifest["feasibility_ratio"]
                    if models[model]["weights_gb"] > budget_gb:
                        continue
                    for mode in modes:
                        for backend in backends:
                            cell = {
                                "host": host,
                                "model": model,
                                "tp": tp,
                                "mode": mode,
                                "backend": backend,
                                "preset": "fixed_single" if mode == "single" else "fixed_multi",
                            }
                            if compile_sweep.is_known_oom(cell, manifest):
                                continue
                            profile_reasons = compile_sweep.profile_infeasible_reasons(
                                cell,
                                manifest,
                                ignore_max_len_rules=True,
                            )
                            if len(profile_reasons) >= len(compile_sweep.resolve(cell, manifest)["profiles"]):
                                continue
                            key = (host, model, tp, mode, backend)
                            for scope in ("current", "fixed"):
                                if key not in existing[scope]:
                                    missing.append((scope, key))

        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
