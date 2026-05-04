"""Build HuggingFace dataset parquets from dashboard data.json."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from sanitize import (
    detect_model_family,
    generate_run_id,
    parse_tensor_parallelism,
    resolve_model_name,
)

SUMMARY_COLUMNS = [
    "run_id", "model", "model_family", "hardware", "engine", "tensor_parallelism",
    "profile", "concurrency", "num_requests", "duration_s", "successful_requests",
    "failed_requests", "request_throughput", "input_token_throughput",
    "output_token_throughput", "total_token_throughput",
    "mean_ttft_ms", "median_ttft_ms", "p90_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "median_tpot_ms", "p90_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "median_itl_ms", "p90_itl_ms", "p99_itl_ms",
    "mean_e2el_ms", "median_e2el_ms", "p90_e2el_ms", "p99_e2el_ms",
]

KNOWN_DATA_JSON_PATHS = [
    Path(__file__).resolve().parent.parent.parent / "dashboard" / "public" / "data.json",
    Path("/tmp/agentic-serve/inference-benchmark/dashboard/public/data.json"),
]


def find_data_json() -> Path:
    for p in KNOWN_DATA_JSON_PATHS:
        if p.exists():
            return p
    print("Error: data.json not found. Use --data-json to specify path.", file=sys.stderr)
    sys.exit(1)


def load_json(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def classify_scope(data_scope: str) -> str:
    if data_scope == "archive":
        return "trace_replay"
    if data_scope in ("current", "fixed"):
        return "distributional"
    return "unknown"


def should_filter(entry: dict, config_name: str) -> bool:
    concurrency = entry["config"]["concurrency"]
    data_scope = entry.get("dataScope", "")
    if config_name == "trace_replay" and concurrency > 100:
        return True
    if config_name == "distributional" and data_scope == "current" and concurrency > 10:
        return True
    return False


def entry_to_row(entry: dict) -> dict:
    config = entry["config"]
    summary = entry["summary"]
    hardware = entry["hardware"]
    model = entry.get("modelShort") or resolve_model_name(config["model"])
    engine = config["backend"]
    tp = parse_tensor_parallelism(hardware, config)
    profile = config["profile"]
    concurrency = config["concurrency"]

    run_id = generate_run_id(model, hardware, engine, tp, profile, concurrency)

    row = {
        "run_id": run_id,
        "model": model,
        "model_family": detect_model_family(model),
        "hardware": hardware,
        "engine": engine,
        "tensor_parallelism": tp,
        "profile": profile,
        "concurrency": concurrency,
    }

    metric_keys = [
        "num_requests", "duration_s", "successful_requests", "failed_requests",
        "request_throughput", "input_token_throughput", "output_token_throughput",
        "total_token_throughput", "mean_ttft_ms", "median_ttft_ms", "p90_ttft_ms",
        "p99_ttft_ms", "mean_tpot_ms", "median_tpot_ms", "p90_tpot_ms", "p99_tpot_ms",
        "mean_itl_ms", "median_itl_ms", "p90_itl_ms", "p99_itl_ms",
        "mean_e2el_ms", "median_e2el_ms", "p90_e2el_ms", "p99_e2el_ms",
    ]
    for k in metric_keys:
        row[k] = summary.get(k)

    return row


def merge_r2_data(data: list[dict], r2_path: Path) -> list[dict]:
    """Merge R2 data, deduplicating by run parameters."""
    r2_data = load_json(r2_path)
    existing_keys = set()
    for entry in data:
        config = entry["config"]
        model = entry.get("modelShort") or resolve_model_name(config["model"])
        key = f"{model}|{entry['hardware']}|{config['backend']}|{config['profile']}|{config['concurrency']}"
        existing_keys.add(key)

    added = 0
    for entry in r2_data:
        config = entry.get("config", {})
        if not config.get("backend") or not config.get("profile"):
            continue
        model = entry.get("modelShort") or resolve_model_name(config.get("model", ""))
        hw = entry.get("hardware", "")
        key = f"{model}|{hw}|{config['backend']}|{config['profile']}|{config.get('concurrency', 0)}"
        if key not in existing_keys:
            if "dataScope" not in entry:
                entry["dataScope"] = "archive"
            data.append(entry)
            existing_keys.add(key)
            added += 1

    if added:
        print(f"  Merged {added} unique rows from R2 data")
    return data


def build_parquet(rows: list[dict], output_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    df["tensor_parallelism"] = df["tensor_parallelism"].astype("int64")
    df["concurrency"] = df["concurrency"].astype("int64")
    df["num_requests"] = df["num_requests"].astype("Int64")
    df["successful_requests"] = df["successful_requests"].astype("Int64")
    df["failed_requests"] = df["failed_requests"].astype("Int64")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def update_croissant(output_dir: Path, hashes: dict[str, str]):
    croissant_path = output_dir / "croissant.json"
    if not croissant_path.exists():
        return
    with open(croissant_path) as f:
        croissant = json.load(f)

    id_to_hash = {
        "trace-replay-summary-parquet": hashes.get("trace_replay"),
        "distributional-summary-parquet": hashes.get("distributional"),
        "kernels-labeled-parquet": hashes.get("kernels_labeled"),
    }

    for dist in croissant.get("distribution", []):
        file_id = dist.get("@id", "")
        if file_id in id_to_hash and id_to_hash[file_id]:
            dist["sha256"] = id_to_hash[file_id]

    with open(croissant_path, "w") as f:
        json.dump(croissant, f, indent=2)
        f.write("\n")
    print(f"  Updated croissant.json hashes")


def print_retention_report(stats: dict):
    print("\n--- Retention Report ---")
    for config_name, s in stats.items():
        pct = (s["after"] / s["before"] * 100) if s["before"] else 0
        print(f"  {config_name}: {s['before']} -> {s['after']} rows ({pct:.1f}% retained, {s['filtered']} filtered)")
    total_before = sum(s["before"] for s in stats.values())
    total_after = sum(s["after"] for s in stats.values())
    total_filtered = sum(s["filtered"] for s in stats.values())
    pct = (total_after / total_before * 100) if total_before else 0
    print(f"  total: {total_before} -> {total_after} rows ({pct:.1f}% retained, {total_filtered} filtered)")


def main():
    parser = argparse.ArgumentParser(description="Build HuggingFace dataset parquets")
    parser.add_argument("--data-json", type=Path, default=None, help="Path to data.json")
    parser.add_argument("--r2-json", type=Path, default=None, help="Path to R2 data JSON (optional)")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent.parent / "hf_dataset",
                        help="Output directory for parquets")
    args = parser.parse_args()

    data_json_path = args.data_json or find_data_json()
    print(f"Loading {data_json_path}")
    data = load_json(data_json_path)
    print(f"  {len(data)} entries loaded")

    if args.r2_json and args.r2_json.exists():
        print(f"Loading R2 supplement: {args.r2_json}")
        data = merge_r2_data(data, args.r2_json)

    configs = {"trace_replay": [], "distributional": []}
    stats = {}

    for config_name in ("trace_replay", "distributional"):
        entries = [e for e in data if classify_scope(e.get("dataScope", "")) == config_name]
        before_count = len(entries)
        filtered = [e for e in entries if not should_filter(e, config_name)]
        after_count = len(filtered)

        stats[config_name] = {
            "before": before_count,
            "after": after_count,
            "filtered": before_count - after_count,
        }

        rows = [entry_to_row(e) for e in filtered]
        configs[config_name] = rows

    hashes = {}
    for config_name, rows in configs.items():
        if not rows:
            print(f"  {config_name}: no rows, skipping parquet")
            continue
        output_path = args.output_dir / config_name / "summary.parquet"
        df = build_parquet(rows, output_path)
        hashes[config_name] = compute_sha256(output_path)
        print(f"  {config_name}/summary.parquet: {len(df)} rows, {len(df.columns)} columns")
        print(f"    models: {sorted(df['model'].unique())}")
        print(f"    hardware: {sorted(df['hardware'].unique())}")
        print(f"    profiles: {sorted(df['profile'].unique())}")

    supplementary = {
        "kernels_labeled": "kernel_profiles/kernels_labeled.parquet",
    }
    for key, rel_path in supplementary.items():
        path = args.output_dir / rel_path
        if path.exists():
            hashes[key] = compute_sha256(path)

    update_croissant(args.output_dir, hashes)
    print_retention_report(stats)


if __name__ == "__main__":
    main()
