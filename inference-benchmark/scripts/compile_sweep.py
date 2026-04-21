#!/usr/bin/env python3
"""Compile sweep.yaml → bench_jobs.txt.

Reads the authoritative sweep matrix in sweep.yaml, applies the feasibility
rule and known_oom skiplist, and emits bench_jobs.txt rows for every
runnable cell. The orchestrator (bench_orchestrator.sh) consumes that file.

Run:  python scripts/compile_sweep.py
      python scripts/compile_sweep.py --dry-run   # print to stdout, don't write
      python scripts/compile_sweep.py --verbose   # show skip reasons
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SWEEP_YAML = HERE / "sweep.yaml"
BENCH_JOBS_TXT = HERE / "bench_jobs.txt"

PRESET_KEYS = ("max_len", "gpu_mem", "concurrencies", "profiles")
CELL_REQUIRED = ("host", "model", "tp", "mode", "preset")


def load_manifest(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def validate(m: dict) -> None:
    for key in ("hosts", "models", "presets", "feasibility_ratio", "cells"):
        if key not in m:
            raise ValueError(f"sweep.yaml missing top-level key: {key}")
    for name, preset in m["presets"].items():
        missing = [k for k in PRESET_KEYS if k not in preset]
        if missing:
            raise ValueError(f"preset {name!r} missing keys: {missing}")
    for i, cell in enumerate(m["cells"]):
        missing = [k for k in CELL_REQUIRED if k not in cell]
        if missing:
            raise ValueError(f"cell #{i} missing keys: {missing}; cell={cell}")
        if cell["host"] not in m["hosts"]:
            raise ValueError(f"cell #{i}: unknown host {cell['host']!r}")
        if cell["model"] not in m["models"]:
            raise ValueError(f"cell #{i}: unknown model {cell['model']!r}")
        if cell["preset"] not in m["presets"]:
            raise ValueError(f"cell #{i}: unknown preset {cell['preset']!r}")
        if cell["mode"] not in ("single", "multi"):
            raise ValueError(f"cell #{i}: mode must be single|multi, got {cell['mode']!r}")


def resolve(cell: dict, manifest: dict) -> dict:
    """Merge preset defaults with cell overrides; return concrete launch params."""
    preset = manifest["presets"][cell["preset"]]
    out = {k: preset[k] for k in PRESET_KEYS}
    for k in PRESET_KEYS + ("extra_env",):
        if k in cell:
            out[k] = cell[k]
    return out


def is_known_oom(cell: dict, manifest: dict) -> str | None:
    for entry in manifest.get("known_oom", []):
        if entry["host"] == cell["host"] and entry["model"] == cell["model"] and entry["tp"] == cell["tp"]:
            return entry["reason"]
    return None


def feasibility_reason(cell: dict, manifest: dict) -> str | None:
    host = manifest["hosts"][cell["host"]]
    model = manifest["models"][cell["model"]]
    ratio = manifest["feasibility_ratio"]
    budget_gb = host["vram_gb_per_gpu"] * cell["tp"] * ratio
    if model["weights_gb"] > budget_gb:
        min_gb = math.ceil(model["weights_gb"] / ratio)
        have_gb = host["vram_gb_per_gpu"] * cell["tp"]
        return f"needs >={min_gb} GB VRAM (weights {model['weights_gb']} GB); this config has {have_gb} GB"
    return None


def render_row(cell: dict, manifest: dict) -> str:
    host = manifest["hosts"][cell["host"]]
    model = manifest["models"][cell["model"]]
    resolved = resolve(cell, manifest)
    model_path = f"{host['model_root']}/{model['dir']}"
    concs = " ".join(str(c) for c in resolved["concurrencies"])
    profiles = " ".join(resolved["profiles"])
    extra_env = resolved.get("extra_env", "")
    backend = str(cell.get("backend", "vllm"))
    fields = [
        str(cell["host"]),
        model_path,
        str(cell["tp"]),
        str(cell["model"]),
        str(cell["mode"]),
        backend,
        str(resolved["max_len"]),
        str(resolved["gpu_mem"]),
        concs,
        profiles,
        str(extra_env),
    ]
    return "|".join(fields)


def compile_jobs(manifest: dict):
    emitted: list[tuple[dict, str]] = []
    skipped: list[tuple[dict, str, str]] = []  # (cell, status, reason)

    for cell in manifest["cells"]:
        reason = is_known_oom(cell, manifest)
        if reason:
            skipped.append((cell, "known_oom", reason))
            continue
        reason = feasibility_reason(cell, manifest)
        if reason:
            skipped.append((cell, "infeasible", reason))
            continue
        emitted.append((cell, render_row(cell, manifest)))
    return emitted, skipped


def render_file(emitted: list[tuple[dict, str]]) -> str:
    lines = [
        "# Benchmark job matrix consumed by bench_orchestrator.sh.",
        "# GENERATED from scripts/sweep.yaml by scripts/compile_sweep.py — DO NOT EDIT DIRECTLY.",
        "# Format: HOST|MODEL_PATH|TP|SHORT|MODE|BACKEND|MAX_LEN|GPU_MEM|CONCS|PROFILES|EXTRA_ENV",
        "# MODE: single | multi",
        "# BACKEND: vllm | sglang",
        "# EXTRA_ENV: optional `KEY=VAL KEY=VAL`.",
        "",
    ]
    current_host: str | None = None
    for cell, row in emitted:
        if cell["host"] != current_host:
            if current_host is not None:
                lines.append("")
            current_host = cell["host"]
            lines.append(f"# === {current_host} ===")
        lines.append(row)
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", type=Path, default=SWEEP_YAML)
    ap.add_argument("--out", type=Path, default=BENCH_JOBS_TXT)
    ap.add_argument("--dry-run", action="store_true", help="print to stdout, don't write")
    ap.add_argument("--verbose", "-v", action="store_true", help="show skip reasons")
    args = ap.parse_args()

    manifest = load_manifest(args.yaml)
    validate(manifest)
    emitted, skipped = compile_jobs(manifest)
    output = render_file(emitted)

    if args.dry_run:
        sys.stdout.write(output)
    else:
        args.out.write_text(output)
        print(f"wrote {args.out} ({len(emitted)} rows)", file=sys.stderr)

    print(f"\nsummary: {len(emitted)} emitted, {len(skipped)} skipped", file=sys.stderr)
    if args.verbose or skipped:
        by_status: dict[str, list] = {}
        for cell, status, reason in skipped:
            by_status.setdefault(status, []).append((cell, reason))
        for status, items in sorted(by_status.items()):
            print(f"  {status} ({len(items)}):", file=sys.stderr)
            for cell, reason in items:
                print(
                    f"    {cell['host']} / {cell['model']} / tp{cell['tp']} / {cell['mode']}"
                    f"  -- {reason}",
                    file=sys.stderr,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
