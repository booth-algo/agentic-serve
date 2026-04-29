"""Per-op serving_e2e validation — ablation comparison vs per-kernel.

Compares `predict_serving_e2e_perop()` against real benchmark results
from `data.json`, producing markdown reports in the same format as
the per-kernel `validate.py --mode serving_e2e`.

CLI
---
    python -m llm_predict.training.per_op.validate \
        --gpu RTX2080Ti --profile chat-singleturn
    python -m llm_predict.training.per_op.validate \
        --gpu RTX2080Ti
"""
from __future__ import annotations

import argparse
from pathlib import Path

from llm_predict.predictors.per_op.predictor import PerOpPredictor
from llm_predict.training.per_kernel import model_specs
from llm_predict.training.per_kernel.validate import (
    _load_measured_rows, _MODELSHORT_TO_DIR, _SHORT_TO_DIR,
    normalize_profile_name,
)

from .serving_e2e import predict_serving_e2e_perop


_HYBRID_ATTN_MODELS: set[str] = {"Qwen3.5-9B", "Qwen3.5-27B"}


def _arch_class(short: str, cfg) -> str:
    if short in _HYBRID_ATTN_MODELS:
        return "hybrid_attn"
    if getattr(cfg, "is_moe", False):
        return "moe"
    return "supported"


def validate_serving_e2e_perop_gpu(gpu: str,
                                    data_json_path: Path,
                                    report_path: Path,
                                    profile_name: str,
                                    concurrency: int = 1) -> None:
    pred = PerOpPredictor(gpu=gpu)
    if not pred.load():
        print(f"[{gpu}][perop_serving_e2e] no pkl — skipping")
        return

    measured_rows = _load_measured_rows(
        data_json_path, gpu, concurrency=concurrency,
        profile_filter=profile_name,
    )
    if not measured_rows:
        print(f"[{gpu}][perop_serving_e2e] no rows for profile={profile_name} — skipping")
        return

    out_rows: list[dict] = []
    for row in measured_rows:
        dir_name = _MODELSHORT_TO_DIR[row["modelShort"]]
        cfg = model_specs.get_model_config(
            dir_name, held_out=model_specs.is_held_out(dir_name, gpu))
        if cfg is None:
            continue
        isl = max(1, int(round(row["avg_isl"])))
        osl = max(0, int(round(row["avg_osl"])))
        bs = row["concurrency"]

        short = next((s for s, d in _SHORT_TO_DIR.items() if d == dir_name), row["modelShort"])
        arch = _arch_class(short, cfg)

        result = predict_serving_e2e_perop(pred, cfg, isl=isl, osl=osl, bs=bs, gpu=gpu)

        meas_ttft = row["measured_ttft_ms"]
        meas_tpot = row.get("median_tpot_ms")
        meas_e2el = row.get("median_e2el_ms")

        ttft_err = abs(result["ttft_ms"] - meas_ttft) / max(meas_ttft, 1e-9) * 100.0
        tpot_err = (abs(result["tpot_ms"] - meas_tpot) / max(meas_tpot, 1e-9) * 100.0
                    if meas_tpot is not None and meas_tpot > 0 else None)
        e2el_err = (abs(result["e2el_ms"] - meas_e2el) / max(meas_e2el, 1e-9) * 100.0
                    if meas_e2el is not None and meas_e2el > 0 else None)

        out_rows.append({
            "short": short, "arch": arch, "backend": row["backend"],
            "isl": isl, "osl": osl, "bs": bs,
            "pred_ttft": result["ttft_ms"],
            "pred_tpot": result["tpot_ms"],
            "pred_e2el": result["e2el_ms"],
            "pred_decode": result["decode_ms"],
            "meas_ttft": meas_ttft,
            "meas_tpot": meas_tpot,
            "meas_e2el": meas_e2el,
            "ttft_err": ttft_err,
            "tpot_err": tpot_err,
            "e2el_err": e2el_err,
            "held_out": model_specs.is_held_out(dir_name, gpu),
        })

    lines: list[str] = [
        f"# {gpu} — per-op serving_e2e Validation: {profile_name}",
        "",
        f"- Predictor track: **per-op serving_e2e** (ablation vs per-kernel)",
        f"- Predictor: {gpu} perop_v5_shape.pkl",
        f"- Profile: `{profile_name}` (concurrency={concurrency})",
        f"- Ground truth: `summary.{{median_ttft_ms, median_tpot_ms, median_e2el_ms}}`",
        "- Headline MAPE = supported architectures only.",
        "",
        "## Per-row",
        "",
        "| Model | arch | backend | ISL | OSL | bs "
        "| pred TTFT | meas TTFT | TTFT err "
        "| pred TPOT | meas TPOT | TPOT err "
        "| pred E2EL | meas E2EL | E2EL err |",
        "|---|---|---|---:|---:|---:"
        "|---:|---:|---:"
        "|---:|---:|---:"
        "|---:|---:|---:|",
    ]

    supported_ttft_errs: list[float] = []
    supported_tpot_errs: list[float] = []
    supported_e2el_errs: list[float] = []
    oos_counts = {"moe": 0, "hybrid_attn": 0}

    for r in out_rows:
        marker = " _(held-out)_" if r["held_out"] else ""
        tpot_pred = f"{r['pred_tpot']:.2f}" if r["osl"] > 0 else "—"
        tpot_meas = f"{r['meas_tpot']:.2f}" if r["meas_tpot"] is not None else "—"
        tpot_err_s = f"{r['tpot_err']:.1f}%" if r["tpot_err"] is not None else "—"
        e2el_meas = f"{r['meas_e2el']:.2f}" if r["meas_e2el"] is not None else "—"
        e2el_err_s = f"{r['e2el_err']:.1f}%" if r["e2el_err"] is not None else "—"

        lines.append(
            f"| {r['short']}{marker} | {r['arch']} | {r['backend']} "
            f"| {r['isl']} | {r['osl']} | {r['bs']} "
            f"| {r['pred_ttft']:.2f} | {r['meas_ttft']:.2f} | {r['ttft_err']:.1f}% "
            f"| {tpot_pred} | {tpot_meas} | {tpot_err_s} "
            f"| {r['pred_e2el']:.2f} | {e2el_meas} | {e2el_err_s} |"
        )

        if r["arch"] == "supported":
            supported_ttft_errs.append(r["ttft_err"])
            if r["tpot_err"] is not None:
                supported_tpot_errs.append(r["tpot_err"])
            if r["e2el_err"] is not None:
                supported_e2el_errs.append(r["e2el_err"])
        else:
            oos_counts[r["arch"]] += 1

    lines.append("")
    lines.append("## Summary (supported architectures only)")
    lines.append("")

    def _mape(errs: list[float]) -> str:
        return f"{sum(errs)/len(errs):.2f}%" if errs else "n/a"

    lines.append(f"| Metric | MAPE | n rows |")
    lines.append(f"|---|---:|---:|")
    lines.append(f"| TTFT | {_mape(supported_ttft_errs)} | {len(supported_ttft_errs)} |")
    lines.append(f"| TPOT | {_mape(supported_tpot_errs)} | {len(supported_tpot_errs)} |")
    lines.append(f"| E2EL | {_mape(supported_e2el_errs)} | {len(supported_e2el_errs)} |")

    if any(oos_counts.values()):
        oos_desc = ", ".join(f"{v} {k}" for k, v in oos_counts.items() if v)
        lines.append(f"\n_Out-of-scope: {oos_desc} — excluded from headline._")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))

    ttft_s = _mape(supported_ttft_errs)
    tpot_s = _mape(supported_tpot_errs)
    e2el_s = _mape(supported_e2el_errs)
    print(
        f"[{gpu}][perop_serving_e2e][{profile_name}] "
        f"TTFT MAPE={ttft_s}  TPOT MAPE={tpot_s}  E2EL MAPE={e2el_s} "
        f"({len(supported_ttft_errs)}/{len(out_rows)} in scope) — wrote {report_path}"
    )


def run(report_dir: Path, gpus: list[str], data_json: Path,
        profile: str | None = None, concurrency: int = 1) -> None:
    profile = normalize_profile_name(profile)
    profiles = [profile] if profile else ["chat-singleturn"]
    for gpu in gpus:
        for p in profiles:
            validate_serving_e2e_perop_gpu(
                gpu, data_json,
                report_dir / f"{gpu}_serving_e2e_perop_{p}.md",
                profile_name=p,
                concurrency=concurrency,
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--gpus", nargs="+", default=["RTX2080Ti"])
    ap.add_argument("--data-json", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--report-dir", default=None)
    args = ap.parse_args()

    gpus = [args.gpu] if args.gpu else args.gpus

    pkg_dir = Path(__file__).resolve().parent
    report_dir = Path(args.report_dir) if args.report_dir else pkg_dir / "reports"

    if args.data_json:
        data_json = Path(args.data_json)
    else:
        repo_root = pkg_dir.parent.parent.parent
        data_json = repo_root / "inference-benchmark" / "dashboard" / "public" / "data.json"
    if not data_json.exists():
        raise SystemExit(f"data.json not found: {data_json}")

    run(report_dir, gpus, data_json, profile=args.profile, concurrency=args.concurrency)


if __name__ == "__main__":
    main()
