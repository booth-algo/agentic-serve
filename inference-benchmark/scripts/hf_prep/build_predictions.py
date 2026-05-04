"""Build serving_predictions.parquet from R2 serving-predictions JSON."""

import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd

R2_BASE = "https://get-data.kevinlauofficial01.workers.dev/"

SCHEMA = {
    "hardware_config": "str",
    "model": "str",
    "backend": "str",
    "profile": "str",
    "data_scope": "str",
    "concurrency": "int64",
    "isl": "int64",
    "osl": "int64",
    "calibration_status": "str",
    "ttft_validation_scope": "str",
    "ttft_kernel_ms": "float64",
    "ttft_base_ms": "float64",
    "ttft_floor_ms": "float64",
    "ttft_first_decode_ms": "float64",
    "ttft_queue_ms": "float64",
    "itl_meas": "float64",
    "total_context_tokens": "int64",
    "new_prefill_tokens": "int64",
    "cached_context_tokens": "int64",
    "cache_hit_rate": "float64",
    "cache_aware_applied": "bool",
    "cache_feature_source": "str",
    "cache_prediction_regime": "str",
    "ttft_prediction_supported": "bool",
    "multiturn_prediction_mode": "str",
    "predicted_turn_count": "float64",
    "total_successful_turn_requests": "float64",
    "mean_predicted_turn_ttft_ms": "float64",
    "mean_predicted_turn_tpot_ms": "float64",
    "multiturn_turn_predictions": "str",
    "ttft_pred": "float64",
    "ttft_meas": "float64",
    "ttft_err": "float64",
    "tpot_pred": "float64",
    "tpot_meas": "float64",
    "tpot_err": "float64",
    "e2el_pred": "float64",
    "e2el_meas": "float64",
    "e2el_err": "float64",
    "measurement_semantics_warning": "str",
}

COLUMNS = list(SCHEMA.keys())

BOOL_COLUMNS = [k for k, v in SCHEMA.items() if v == "bool"]
INT_COLUMNS = [k for k, v in SCHEMA.items() if v == "int64"]
FLOAT_COLUMNS = [k for k, v in SCHEMA.items() if v == "float64"]
STR_COLUMNS = [k for k, v in SCHEMA.items() if v == "str"]


def flatten_predictions(raw) -> list[dict]:
    if isinstance(raw, list):
        return raw
    rows = []
    for hw_config, records in raw.items():
        for record in records:
            record["hardware_config"] = hw_config
            rows.append(record)
    return rows


def download_json(r2_base: str) -> list[dict]:
    url = r2_base.rstrip("/") + "/serving-predictions.json"
    req = urllib.request.Request(url, headers={"User-Agent": "AgentPerfBench/1.0"})
    with urllib.request.urlopen(req) as resp:
        chunks = []
        while chunk := resp.read(8192):
            chunks.append(chunk)
        return flatten_predictions(json.loads(b"".join(chunks)))


def load_json(path: Path) -> list[dict]:
    with open(path) as f:
        return flatten_predictions(json.load(f))


def coerce_bool(series: pd.Series) -> pd.Series:
    return series.map(
        lambda v: True if v in (True, "true", "True", "1", 1)
        else (False if v in (False, "false", "False", "0", 0, None) else bool(v))
    )


def build_dataframe(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df = df.reindex(columns=COLUMNS)

    for col in STR_COLUMNS:
        df[col] = df[col].fillna("").astype("str")

    for col in FLOAT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    for col in INT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")

    for col in BOOL_COLUMNS:
        df[col] = coerce_bool(df[col]).astype("bool")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent.parent / "hf_dataset")
    parser.add_argument("--r2-base", type=str, default=R2_BASE)
    args = parser.parse_args()

    if args.predictions_json:
        print(f"Loading {args.predictions_json}")
        records = load_json(args.predictions_json)
    else:
        print(f"Downloading serving-predictions.json from {args.r2_base}")
        records = download_json(args.r2_base)

    print(f"  {len(records)} records loaded")

    df = build_dataframe(records)

    output_path = args.output_dir / "predictions" / "serving_predictions.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"Written: {output_path}")
    print(f"  Rows:    {len(df)}")
    print(f"  Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
