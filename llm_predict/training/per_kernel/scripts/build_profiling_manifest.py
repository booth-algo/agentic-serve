"""build_profiling_manifest.py — scan local disk + R2 and emit profiling manifest YAML.

Usage:
    python -m llm_predict.training.per_kernel.scripts.build_profiling_manifest \
        --ncu-root /root/llm/profiling-data \
        --r2-bucket agent-bench \
        --r2-endpoint https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com \
        --r2-profile r2 \
        --out llm_predict/training/per_kernel/profiling_manifest.yaml

Idempotent: re-running yields identical output modulo `generated_at`.
"""
from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# ── canonical sets ────────────────────────────────────────────────────────────
GPUS = ["A100", "RTX3090", "RTX2080Ti", "H100"]

# model_short values as used in the manifest (spec names)
MODELS = [
    "Llama-8B",
    "Llama-70B",
    "Llama-3.3-70B",
    "Mixtral-8x7B",
    "Qwen-72B",
    "Qwen3.5-9B",
    "Qwen3.5-27B",
    "gpt-oss-20b",
]

VRAM_GB: dict[str, float] = {
    "A100": 40,
    "RTX3090": 24,
    "RTX2080Ti": 22,
    "H100": 80,
}

WEIGHTS_GB: dict[str, float] = {
    "Llama-8B": 16,
    "Llama-70B": 140,
    "Llama-3.3-70B": 140,
    "Mixtral-8x7B": 94,
    "Qwen-72B": 144,
    "Qwen3.5-9B": 18,
    "Qwen3.5-27B": 54,
    "gpt-oss-20b": 40,
}

# ncu directory name -> model_short (spec canonical name)
_DIR_TO_SHORT: dict[str, str] = {
    "Llama-3.1-8B-Instruct":  "Llama-8B",
    "Llama-3.1-70B-Instruct": "Llama-70B",
    "Llama-3.3-70B-Instruct": "Llama-3.3-70B",
    "Qwen2.5-72B-Instruct":   "Qwen-72B",
    "Qwen3.5-9B":             "Qwen3.5-9B",
    "Qwen3.5-27B":            "Qwen3.5-27B",
    "gpt-oss-20b":            "gpt-oss-20b",
    "Mixtral-8x7B-Instruct":  "Mixtral-8x7B",
}

# Reverse: model_short -> ncu dir name
_SHORT_TO_DIR: dict[str, str] = {v: k for k, v in _DIR_TO_SHORT.items()}

# ── R2 listing ────────────────────────────────────────────────────────────────

def build_r2_index(bucket: str, endpoint: str, profile: str) -> list[str]:
    """One full recursive listing of the profiling-data prefix."""
    cmd = [
        "aws", "--profile", profile,
        "--endpoint-url", endpoint,
        "s3", "ls", f"s3://{bucket}/profiling-data/",
        "--recursive",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"R2 listing failed: {result.stderr[:200]}", file=sys.stderr)
            return []
        keys: list[str] = []
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 4:
                keys.append(parts[3])
        return keys
    except Exception as exc:
        print(f"R2 listing exception: {exc}", file=sys.stderr)
        return []


# ── helpers ───────────────────────────────────────────────────────────────────

def _iso_mtime(path: Path) -> str | None:
    try:
        ts = os.stat(path).st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
    except OSError:
        return None


def _row_count(path: Path) -> int | None:
    """Count data rows (lines - 1 header) in a CSV."""
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f) - 1
    except OSError:
        return None


def _size(path: Path) -> int | None:
    try:
        return os.stat(path).st_size
    except OSError:
        return None


# ── per_kernel.prefill_seq128_bs1 ─────────────────────────────────────────────

def prefill_status(
    gpu: str,
    model: str,
    ncu_root: Path,
    r2_set: set[str],
) -> dict[str, Any]:
    # H100 -> pending_infra_access regardless
    if gpu == "H100":
        return {
            "status": "pending_infra_access",
            "local_path": None,
            "r2_key": None,
            "rows": None,
            "size_bytes": None,
            "mtime": None,
            "reason": "runpod container perms",
        }

    dir_name = _SHORT_TO_DIR.get(model)

    # check local disk first — done overrides infeasible if data actually exists
    local_path: Path | None = None
    if dir_name:
        candidate = ncu_root / gpu / "ncu" / dir_name / "prefill_seq128_bs1.csv"
        if candidate.is_file():
            local_path = candidate

    # check R2
    r2_key: str | None = None
    if dir_name:
        expected_key = f"profiling-data/{gpu}/ncu/{dir_name}/prefill_seq128_bs1.csv"
        if expected_key in r2_set:
            r2_key = expected_key

    if local_path is not None:
        return {
            "status": "done",
            "local_path": str(local_path),
            "r2_key": r2_key,
            "rows": _row_count(local_path),
            "size_bytes": _size(local_path),
            "mtime": _iso_mtime(local_path),
            "reason": None,
        }

    if r2_key is not None:
        return {
            "status": "done",
            "local_path": None,
            "r2_key": r2_key,
            "rows": None,
            "size_bytes": None,
            "mtime": None,
            "reason": None,
        }

    # infeasible check (only when no data found)
    vram = VRAM_GB[gpu]
    weights = WEIGHTS_GB[model]
    budget = vram * 0.85
    if weights > budget:
        return {
            "status": "infeasible",
            "local_path": None,
            "r2_key": None,
            "rows": None,
            "size_bytes": None,
            "mtime": None,
            "reason": f"weights {weights:.0f} GB > single-GPU budget {budget:.1f} GB",
        }

    return {
        "status": "missing",
        "local_path": None,
        "r2_key": None,
        "rows": None,
        "size_bytes": None,
        "mtime": None,
        "reason": "no local or R2 data found",
    }


# ── per_kernel.roofline_sweep ─────────────────────────────────────────────────

def roofline_status(gpu: str, r2_set: set[str]) -> dict[str, Any]:
    # check local /tmp file
    local_csv = Path(f"/tmp/ncu_gemm_sweep_{gpu}.csv")
    if local_csv.is_file():
        return {
            "status": "done",
            "local_path": str(local_csv),
            "rows": _row_count(local_csv),
        }

    # check R2
    prefix = f"profiling-data/{gpu}/roofline/"
    has_r2 = any(k.startswith(prefix) for k in r2_set)
    if has_r2:
        return {
            "status": "done",
            "local_path": None,
            "rows": None,
        }

    return {
        "status": "missing",
        "local_path": None,
        "rows": None,
    }


# ── per_op.trained_pkl ────────────────────────────────────────────────────────

def _read_pkl_meta(pkl_path: Path) -> dict[str, Any]:
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            version = data.get("version")
            n_training = data.get("n_training")
            models_in_training = data.get("models", [])
        else:
            version = getattr(data, "version", None)
            n_training = getattr(data, "n_training", None)
            models_in_training = getattr(data, "models", [])
        return {
            "version": version,
            "n_training": n_training,
            "models_in_training": list(models_in_training) if models_in_training else [],
        }
    except Exception:
        return {"version": None, "n_training": None, "models_in_training": []}


def trained_pkl_status(
    gpu: str,
    repo_root: Path,
    r2_set: set[str],
) -> dict[str, Any]:
    if gpu == "H100":
        return {
            "status": "pending_infra_access",
            "pkl": None,
            "r2_key": None,
            "size_bytes": None,
            "version": None,
            "n_training": None,
            "models_in_training": [],
            "reason": "runpod container perms",
        }

    # check local
    trained_dir = repo_root / "llm_predict" / "profiling" / "data" / gpu / "trained"
    local_pkl: Path | None = None
    if trained_dir.is_dir():
        candidates = sorted(trained_dir.glob("perop_*.pkl"))
        if candidates:
            local_pkl = candidates[-1]  # most recent alphabetically

    if local_pkl is not None:
        meta = _read_pkl_meta(local_pkl)
        r2_key_match: str | None = None
        expected = f"profiling-data/{gpu}/trained/{local_pkl.name}"
        if expected in r2_set:
            r2_key_match = expected
        return {
            "status": "done",
            "pkl": str(local_pkl),
            "r2_key": r2_key_match,
            "size_bytes": _size(local_pkl),
            **meta,
        }

    # fall back to R2
    r2_prefix = f"profiling-data/{gpu}/trained/"
    r2_pkls = [
        k for k in r2_set
        if k.startswith(r2_prefix)
        and k.endswith(".pkl")
        and "/per_kernel/" not in k
        and "perop_" in k.split("/")[-1]
    ]
    if r2_pkls:
        r2_key = sorted(r2_pkls)[-1]
        return {
            "status": "done",
            "pkl": None,
            "r2_key": r2_key,
            "size_bytes": None,
            "version": None,
            "n_training": None,
            "models_in_training": [],
        }

    return {
        "status": "missing",
        "pkl": None,
        "r2_key": None,
        "size_bytes": None,
        "version": None,
        "n_training": None,
        "models_in_training": [],
    }


# ── per_op.cuda_events ────────────────────────────────────────────────────────

# Models with only partial (MoE-only) H100 perop data
_H100_PARTIAL_MODELS = {"Mixtral-8x7B", "gpt-oss-20b"}

# Experiment CSV name -> model_short mappings (historical A100)
_EXPERIMENT_SLUGS: dict[str, str] = {
    "ncu_llama8b_model.csv": "Llama-8B",
    "ncu_mixtral_model.csv": "Mixtral-8x7B",
}


def _model_r2_slugs(model: str) -> list[str]:
    """Return lowercase substrings to match against R2 perop key filenames."""
    mapping: dict[str, list[str]] = {
        "Llama-8B":      ["llama8b", "llama_8b", "llama-8b", "llama3.1_8b", "llama_3.1_8b"],
        "Llama-70B":     ["llama70b", "llama_70b", "llama-70b", "llama3.1_70b"],
        "Llama-3.3-70B": ["llama3.3", "llama_3.3"],
        "Mixtral-8x7B":  ["mixtral"],
        "Qwen-72B":      ["qwen2.5", "qwen_72b", "qwen-72b"],
        "Qwen3.5-9B":    ["qwen3.5_9b", "qwen3.5-9b"],
        "Qwen3.5-27B":   ["qwen3.5_27b", "qwen3.5-27b"],
        "gpt-oss-20b":   ["gpt_oss_20b", "gpt-oss-20b"],
    }
    return mapping.get(model, [model.lower()])


def cuda_events_status(
    gpu: str,
    model: str,
    r2_set: set[str],
    repo_root: Path,
    is_prefill_infeasible: bool,
) -> dict[str, Any]:
    sources: list[str] = []

    if gpu == "H100":
        if model in _H100_PARTIAL_MODELS:
            prefix = "profiling-data/H100/perop/"
            for k in sorted(r2_set):
                if k.startswith(prefix):
                    slugs = _model_r2_slugs(model)
                    fname = k.split("/")[-1].lower()
                    if any(s in fname for s in slugs):
                        sources.append(k)
            return {
                "status": "partial",
                "sources": sources,
                "reason": "only MoE profile exists (decode OR prefill phase only)",
            }
        return {
            "status": "pending_infra_access",
            "sources": [],
            "reason": "runpod container perms",
        }

    if gpu == "RTX2080Ti" and is_prefill_infeasible:
        return {
            "status": "infeasible",
            "sources": [],
            "reason": "propagated from per_kernel.prefill infeasibility",
        }

    # check experiment/ directory (A100 historical)
    experiment_dir = repo_root / "experiment"
    for csv_name, csv_model in _EXPERIMENT_SLUGS.items():
        if csv_model == model:
            p = experiment_dir / csv_name
            if p.is_file():
                sources.append(str(p))

    # check R2 perop prefix
    r2_prefix = f"profiling-data/{gpu}/perop/"
    slugs = _model_r2_slugs(model)
    for k in sorted(r2_set):
        if k.startswith(r2_prefix):
            fname = k.split("/")[-1].lower()
            if any(s in fname for s in slugs):
                sources.append(k)

    if sources:
        return {"status": "done", "sources": sources}

    return {"status": "missing", "sources": []}


# ── manifest builder ──────────────────────────────────────────────────────────

def build_manifest(
    ncu_root: Path,
    repo_root: Path,
    r2_keys: list[str],
) -> dict[str, Any]:
    r2_set = set(r2_keys)

    gpus_data: dict[str, Any] = {}
    for gpu in GPUS:
        models_data: dict[str, Any] = {}
        for model in MODELS:
            pf = prefill_status(gpu, model, ncu_root, r2_set)
            is_infeasible = pf["status"] == "infeasible"

            rf = roofline_status(gpu, r2_set)
            pkl = trained_pkl_status(gpu, repo_root, r2_set)
            ce = cuda_events_status(gpu, model, r2_set, repo_root, is_infeasible)

            models_data[model] = {
                "per_kernel": {
                    "prefill_seq128_bs1": pf,
                    "roofline_sweep": rf,
                },
                "per_op": {
                    "cuda_events": ce,
                    "trained_pkl": pkl,
                },
            }
        gpus_data[gpu] = models_data

    return {
        "generated_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gpus": gpus_data,
    }


# ── YAML serialisation ────────────────────────────────────────────────────────

def _none_representer(dumper: yaml.Dumper, _: None) -> yaml.Node:
    return dumper.represent_scalar("tag:yaml.org,2002:null", "null")


def dump_yaml(manifest: dict[str, Any]) -> str:
    class _Dumper(yaml.Dumper):
        pass

    _Dumper.add_representer(type(None), _none_representer)
    return yaml.dump(
        manifest,
        Dumper=_Dumper,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=True,
        width=120,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build profiling manifest YAML.")
    parser.add_argument("--ncu-root", default="/root/llm/profiling-data",
                        help="Root of per-GPU profiling data")
    parser.add_argument("--r2-bucket", default="agent-bench")
    parser.add_argument("--r2-endpoint",
                        default="https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com")
    parser.add_argument("--r2-profile", default="r2")
    parser.add_argument("--out", default="llm_predict/training/per_kernel/profiling_manifest.yaml")
    args = parser.parse_args(argv)

    ncu_root = Path(args.ncu_root)
    out_path = Path(args.out)
    # repo_root: scripts/ -> per_kernel/ -> training/ -> llm_predict/ -> repo/
    repo_root = Path(__file__).resolve().parents[4]

    print(f"Listing R2 bucket {args.r2_bucket!r} (one recursive call)...", file=sys.stderr)
    r2_keys = build_r2_index(args.r2_bucket, args.r2_endpoint, args.r2_profile)
    print(f"  {len(r2_keys)} R2 objects found.", file=sys.stderr)

    manifest = build_manifest(ncu_root, repo_root, r2_keys)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = dump_yaml(manifest)
    out_path.write_text(text, encoding="utf-8")
    print(f"Manifest written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
