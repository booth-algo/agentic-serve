"""Phase A1 labeler — ncu → per-kernel shape-annotated CSV.

Port of `/root/per-kernel-rebuild/label_all_v3.py` with three material changes:

1. Reads long-format ncu CSVs via `ncu_loader` (new schema: `.sum` target,
   combined `dram_bytes_sum`, launch metrics as metric rows).
2. Walks nested layout `{ncu_root}/{gpu}/ncu/{model_dir}/prefill_seq128_bs1*.csv`
   and concatenates `_ext` variants (main is loaded first; `_ext` contributes
   extra columns via left-join key). Emits a `gpu` column in the output.
3. Adds a 2080Ti-style fallback path for CSVs containing zero flash_attn
   kernels (sm_75 pre-FA2 and fp32 fallback paths): shape-solve every GEMM
   against the per-model candidate set, no layer-walker.

Gate-vs-Down disambiguation via `dram_bytes_write` is no longer possible
(new collection only has combined `dram__bytes.sum`). The shape scorer uses
`bytes_total ≈ 2·(MK + NK + MN)` with looser tolerance; grid + tile remain
the dominant signals.

CLI
---
    python -m llm_predict.training.per_kernel.labeler \\
        --ncu-root /root/llm/profiling-data \\
        --out      llm_predict/training/per_kernel/data/kernels_labeled.csv \\
        --gpus     A100 RTX3090 RTX2080Ti
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import ncu_loader
from . import gpu_kernel_regex as gkr
from . import model_specs


DTYPE_DEFAULT = "bf16"
BYTES_PER_ELT = 2


OUT_COLS = [
    "source", "gpu", "model", "kernel_family", "kernel_name", "dtype", "held_out",
    "M", "N", "K", "bs", "seq", "n_heads", "head_dim", "kv_heads",
    "numel", "op_type", "gpu_time_duration_ms",
    "launch_block_size", "launch_grid_size",
    "dram_bytes_sum", "launch_registers_per_thread",
]


def _base_row(row: pd.Series, source: str, gpu: str, model: str, family: str) -> dict:
    return {
        "source": source, "gpu": gpu, "model": model, "kernel_family": family,
        "kernel_name": row.get("Kernel Name"), "dtype": DTYPE_DEFAULT, "held_out": False,
        "M": np.nan, "N": np.nan, "K": np.nan,
        "bs": np.nan, "seq": np.nan,
        "n_heads": np.nan, "head_dim": np.nan, "kv_heads": np.nan,
        "numel": np.nan, "op_type": "",
        "gpu_time_duration_ms": row.get("gpu_time_duration_ms", np.nan),
        "launch_block_size": row.get("launch_block_size", np.nan),
        "launch_grid_size": row.get("launch_grid_size", np.nan),
        "dram_bytes_sum": row.get("dram_bytes_sum", np.nan),
        "launch_registers_per_thread": row.get("launch_registers_per_thread", np.nan),
    }


def _expected_bytes_sum(M: int, N: int, K: int) -> float:
    return BYTES_PER_ELT * (M * K + N * K + M * N)


def score_shape(M: int, N: int, K: int,
                observed_bytes_sum: float, observed_grid: float,
                tile_m: int | None, tile_n: int | None) -> float:
    """Lower is better. Combines bytes fit with tile/grid fit."""
    score = 0.0
    pred = _expected_bytes_sum(M, N, K)
    if not pd.isna(observed_bytes_sum) and observed_bytes_sum > 0:
        score += abs(math.log2(max(pred, 1)) - math.log2(max(observed_bytes_sum, 1)))
    else:
        score += 5.0
    if tile_m is not None and tile_n is not None and not pd.isna(observed_grid) and observed_grid > 0:
        base = math.ceil(M / tile_m) * math.ceil(N / tile_n)
        if base > 0:
            ratio = observed_grid / base
            nearest_int = round(ratio)
            if nearest_int < 1:
                score += 3.0
            elif 0.9 <= ratio / nearest_int <= 1.1 and 1 <= nearest_int <= 12:
                if nearest_int <= 3:
                    score += 0.0
                elif nearest_int <= 6:
                    score += 0.2
                else:
                    score += 0.5
            else:
                score += 1.5
    return score


def _dense_layer_expected(cfg: model_specs.ModelConfig):
    bs_seq = cfg.bs * cfg.seq
    Q_OR_O  = (bs_seq, cfg.h * cfg.head_dim, cfg.d)
    K_OR_V  = (bs_seq, cfg.kv * cfg.head_dim, cfg.d)
    GATE_UP = (bs_seq, cfg.ffn, cfg.d)
    DOWN    = (bs_seq, cfg.d, cfg.ffn)
    return [
        (*Q_OR_O, "o_proj"),
        (*GATE_UP, "gate"),
        (*GATE_UP, "up"),
        (*DOWN,    "down_proj"),
        (*Q_OR_O,  "q_proj"),
        (*K_OR_V,  "k_proj"),
        (*K_OR_V,  "v_proj"),
    ]


def _first_window_expected(cfg: model_specs.ModelConfig):
    bs_seq = cfg.bs * cfg.seq
    Q_OR_O = (bs_seq, cfg.h * cfg.head_dim, cfg.d)
    K_OR_V = (bs_seq, cfg.kv * cfg.head_dim, cfg.d)
    return [(*Q_OR_O, "q_proj"), (*K_OR_V, "k_proj"), (*K_OR_V, "v_proj")]


def annotate_gemm_dense_model(df: pd.DataFrame, source: str, gpu: str,
                                cfg: model_specs.ModelConfig):
    stats = {"total_gemm": 0, "labeled_by_position": 0, "labeled_by_shape": 0, "dropped": 0}
    flash_idx = df.index[df["kernel_family"] == "flash_attn"].tolist()
    bs_seq = cfg.bs * cfg.seq
    LM_HEAD = (bs_seq, cfg.vocab, cfg.d)
    out: list[dict] = []
    if not flash_idx:
        return out, stats

    windows = []
    prev = 0
    for fi in flash_idx:
        windows.append((prev, fi))
        prev = fi
    windows.append((prev, len(df)))

    for w_i, (lo, hi) in enumerate(windows):
        gemms_in_window = []
        for idx in range(lo, hi):
            r = df.iloc[idx]
            if r["kernel_family"] != "gemm":
                continue
            sub = r["kernel_subtype"]
            if sub == "gemv":
                r2 = _base_row(r, source, gpu, cfg.name, "gemm")
                r2["op_type"] = "gemv"
                r2["held_out"] = cfg.held_out
                r2["bs"], r2["seq"] = cfg.bs, cfg.seq
                r2["_idx"] = idx
                out.append(r2)
                stats["labeled_by_position"] += 1
                stats["total_gemm"] += 1
                continue
            gemms_in_window.append((idx, r))
            stats["total_gemm"] += 1

        if not gemms_in_window:
            continue

        lm_head_pos: int | None = None
        for pos, (_, row) in enumerate(gemms_in_window):
            gs = row.get("launch_grid_size", np.nan)
            if not pd.isna(gs) and gs > 500:
                lm_head_pos = pos
                break

        expected = _first_window_expected(cfg) if w_i == 0 else _dense_layer_expected(cfg)
        expected_full = list(expected)
        if lm_head_pos is not None:
            expected_full.insert(lm_head_pos, (*LM_HEAD, "lm_head"))

        for pos, (idx, row) in enumerate(gemms_in_window):
            if pos < len(expected_full):
                M, N, K, label = expected_full[pos]
                r2 = _base_row(row, source, gpu, cfg.name, "gemm")
                r2["M"], r2["N"], r2["K"] = M, N, K
                r2["bs"], r2["seq"] = cfg.bs, cfg.seq
                r2["op_type"] = label
                r2["held_out"] = cfg.held_out
                r2["_idx"] = idx
                out.append(r2)
                stats["labeled_by_position"] += 1
            else:
                tile = gkr.parse_gemm_tile(row.get("Kernel Name") or "", gpu=gpu)
                tm, tn = tile if tile else (None, None)
                candidates = expected + [(*LM_HEAD, "lm_head")]
                best = min(candidates,
                           key=lambda c: score_shape(c[0], c[1], c[2],
                                                      row.get("dram_bytes_sum", np.nan),
                                                      row.get("launch_grid_size", np.nan),
                                                      tm, tn))
                M, N, K, label = best
                r2 = _base_row(row, source, gpu, cfg.name, "gemm")
                r2["M"], r2["N"], r2["K"] = M, N, K
                r2["bs"], r2["seq"] = cfg.bs, cfg.seq
                r2["op_type"] = label
                r2["held_out"] = cfg.held_out
                r2["_idx"] = idx
                out.append(r2)
                stats["labeled_by_shape"] += 1
    return out, stats


def annotate_gemm_moe_model(df: pd.DataFrame, source: str, gpu: str,
                              cfg: model_specs.ModelConfig):
    stats = {"total_gemm": 0, "labeled_by_position": 0, "labeled_by_shape": 0, "dropped": 0}
    flash_idx = df.index[df["kernel_family"] == "flash_attn"].tolist()
    bs_seq = cfg.bs * cfg.seq
    Q_OR_O  = (bs_seq, cfg.h * cfg.head_dim, cfg.d, "q_proj")
    O_PROJ  = (bs_seq, cfg.h * cfg.head_dim, cfg.d, "o_proj")
    K_OR_V  = (bs_seq, cfg.kv * cfg.head_dim, cfg.d, "k_proj")
    V_OR_K  = (bs_seq, cfg.kv * cfg.head_dim, cfg.d, "v_proj")
    LM_HEAD = (bs_seq, cfg.vocab, cfg.d, "lm_head")
    ROUTER  = (bs_seq, cfg.E, cfg.d, "moe_router")

    expert_candidates = []
    for tok in range(1, bs_seq + 1):
        expert_candidates.append((tok, cfg.ffn, cfg.d, f"expert_gate_up_tok{tok}"))
        expert_candidates.append((tok, cfg.d, cfg.ffn, f"expert_down_tok{tok}"))

    out: list[dict] = []
    if not flash_idx:
        return out, stats

    windows = []
    prev = 0
    for fi in flash_idx:
        windows.append((prev, fi))
        prev = fi
    windows.append((prev, len(df)))

    for w_i, (lo, hi) in enumerate(windows):
        gemms_in_window = []
        for idx in range(lo, hi):
            r = df.iloc[idx]
            if r["kernel_family"] != "gemm":
                continue
            sub = r["kernel_subtype"]
            if sub == "gemv":
                r2 = _base_row(r, source, gpu, cfg.name, "gemm")
                r2["op_type"] = "gemv"
                r2["held_out"] = cfg.held_out
                r2["bs"], r2["seq"] = cfg.bs, cfg.seq
                r2["_idx"] = idx
                out.append(r2)
                stats["labeled_by_position"] += 1
                stats["total_gemm"] += 1
                continue
            gemms_in_window.append((idx, r))
            stats["total_gemm"] += 1
        if not gemms_in_window:
            continue

        lm_head_positions: set[int] = set()
        for pos, (_, row) in enumerate(gemms_in_window):
            ds = row.get("dram_bytes_sum", np.nan)
            if not pd.isna(ds) and ds > 5e8:
                lm_head_positions.add(pos)

        n = len(gemms_in_window)
        if w_i == 0:
            fixed = {0: Q_OR_O, 1: K_OR_V, 2: V_OR_K}
        else:
            fixed = {0: O_PROJ}
            if n >= 3:
                fixed[n - 3] = Q_OR_O
                fixed[n - 2] = K_OR_V
                fixed[n - 1] = V_OR_K

        for pos, (idx, row) in enumerate(gemms_in_window):
            if pos in fixed:
                M, N, K, label = fixed[pos]
                r2 = _base_row(row, source, gpu, cfg.name, "gemm")
                r2["M"], r2["N"], r2["K"] = M, N, K
                r2["bs"], r2["seq"] = cfg.bs, cfg.seq
                r2["op_type"] = label
                r2["held_out"] = cfg.held_out
                r2["_idx"] = idx
                out.append(r2)
                stats["labeled_by_position"] += 1
            elif pos in lm_head_positions:
                M, N, K, label = LM_HEAD
                r2 = _base_row(row, source, gpu, cfg.name, "gemm")
                r2["M"], r2["N"], r2["K"] = M, N, K
                r2["bs"], r2["seq"] = cfg.bs, cfg.seq
                r2["op_type"] = label
                r2["held_out"] = cfg.held_out
                r2["_idx"] = idx
                out.append(r2)
                stats["labeled_by_position"] += 1
            else:
                tile = gkr.parse_gemm_tile(row.get("Kernel Name") or "", gpu=gpu)
                tm, tn = tile if tile else (None, None)
                cands = [ROUTER] + expert_candidates
                best = min(cands,
                           key=lambda c: score_shape(c[0], c[1], c[2],
                                                      row.get("dram_bytes_sum", np.nan),
                                                      row.get("launch_grid_size", np.nan),
                                                      tm, tn))
                M, N, K, label = best
                r2 = _base_row(row, source, gpu, cfg.name, "gemm")
                r2["M"], r2["N"], r2["K"] = M, N, K
                r2["bs"], r2["seq"] = cfg.bs, cfg.seq
                r2["op_type"] = label
                r2["held_out"] = cfg.held_out
                r2["_idx"] = idx
                out.append(r2)
                stats["labeled_by_shape"] += 1
    return out, stats


def annotate_gemm_noflash_fallback(df: pd.DataFrame, source: str, gpu: str,
                                      cfg: model_specs.ModelConfig):
    """Shape-solve every GEMM against the per-model candidate set.

    Used on RTX 2080Ti (no FA2, sm_75 pre-FA2) and on any fp32-fallback
    collection where flash_attn kernels are absent. No positional labeling.
    """
    stats = {"total_gemm": 0, "labeled_by_position": 0, "labeled_by_shape": 0, "dropped": 0}
    bs_seq = cfg.bs * cfg.seq
    cands: list[tuple[int, int, int, str]] = [
        (bs_seq, cfg.h * cfg.head_dim, cfg.d, "q_or_o"),
        (bs_seq, cfg.kv * cfg.head_dim, cfg.d, "k_or_v"),
        (bs_seq, (cfg.h + 2 * cfg.kv) * cfg.head_dim, cfg.d, "qkv_fused"),
        (bs_seq, cfg.ffn, cfg.d, "gate_up"),
        (bs_seq, cfg.d, cfg.ffn, "down_proj"),
        (bs_seq, cfg.vocab, cfg.d, "lm_head"),
    ]
    if cfg.is_moe:
        cands.append((bs_seq, cfg.E, cfg.d, "moe_router"))
        for tok in range(1, bs_seq + 1):
            cands.append((tok, cfg.ffn, cfg.d, f"expert_gate_up_tok{tok}"))
            cands.append((tok, cfg.d, cfg.ffn, f"expert_down_tok{tok}"))

    out: list[dict] = []
    for idx, row in df.iterrows():
        if row["kernel_family"] != "gemm":
            continue
        stats["total_gemm"] += 1
        sub = row["kernel_subtype"]
        if sub == "gemv":
            r2 = _base_row(row, source, gpu, cfg.name, "gemm")
            r2["op_type"] = "gemv"
            r2["bs"], r2["seq"] = cfg.bs, cfg.seq
            r2["held_out"] = cfg.held_out
            r2["_idx"] = idx
            out.append(r2)
            stats["labeled_by_position"] += 1
            continue
        tile = gkr.parse_gemm_tile(row.get("Kernel Name") or "", gpu=gpu)
        tm, tn = tile if tile else (None, None)
        best = min(cands,
                   key=lambda c: score_shape(c[0], c[1], c[2],
                                              row.get("dram_bytes_sum", np.nan),
                                              row.get("launch_grid_size", np.nan),
                                              tm, tn))
        M, N, K, label = best
        r2 = _base_row(row, source, gpu, cfg.name, "gemm")
        r2["M"], r2["N"], r2["K"] = M, N, K
        r2["bs"], r2["seq"] = cfg.bs, cfg.seq
        r2["op_type"] = label
        r2["held_out"] = cfg.held_out
        r2["_idx"] = idx
        out.append(r2)
        stats["labeled_by_shape"] += 1
    return out, stats


def annotate_splitk(df: pd.DataFrame, source: str, gpu: str,
                      cfg: model_specs.ModelConfig, gemm_labels: list[dict]):
    idx_to_shape = {l["_idx"]: (l["M"], l["N"], l["K"])
                    for l in gemm_labels
                    if "_idx" in l and not pd.isna(l.get("M"))}
    sorted_gemm = sorted(idx_to_shape.items())
    out: list[dict] = []
    for idx, row in df.iterrows():
        if row["kernel_family"] != "splitk_reduce":
            continue
        preceding = None
        for g_idx, shape in sorted_gemm:
            if g_idx < idx:
                preceding = shape
            else:
                break
        if preceding is None:
            continue
        M, N, K = preceding
        r = _base_row(row, source, gpu, cfg.name, "splitk_reduce")
        r["M"], r["N"], r["K"] = M, N, K
        r["held_out"] = cfg.held_out
        out.append(r)
    return out


def annotate_flash(df: pd.DataFrame, source: str, gpu: str, cfg: model_specs.ModelConfig):
    out: list[dict] = []
    for _, row in df.iterrows():
        if row["kernel_family"] != "flash_attn":
            continue
        r = _base_row(row, source, gpu, cfg.name, "flash_attn")
        r["bs"] = cfg.bs
        r["seq"] = cfg.seq
        r["n_heads"] = cfg.h
        r["head_dim"] = cfg.head_dim
        r["kv_heads"] = cfg.kv
        r["op_type"] = "flash_attn"
        r["held_out"] = cfg.held_out
        out.append(r)
    return out


def annotate_elementwise(df: pd.DataFrame, source: str, gpu: str,
                           cfg: model_specs.ModelConfig):
    out: list[dict] = []
    numel_d    = cfg.seq * cfg.d
    numel_ffn  = cfg.seq * cfg.ffn
    numel_kv   = cfg.seq * (2 * cfg.kv * cfg.head_dim)
    numel_q    = cfg.seq * (cfg.h * cfg.head_dim)
    numel_vocab = cfg.seq * cfg.vocab
    vec = 4
    cands = [c for c in [numel_d, numel_ffn, numel_kv, numel_q, numel_vocab,
                         cfg.seq, cfg.seq * 2, cfg.seq * 4, cfg.seq * cfg.d * 2,
                         cfg.seq * cfg.E] if c > 0]

    def snap(x):
        if pd.isna(x):
            return x
        best = min(cands, key=lambda c: abs(math.log2(max(x, 1)) - math.log2(c)))
        if abs(math.log2(max(x, 1)) - math.log2(best)) < 1:
            return best
        return x

    for _, row in df.iterrows():
        fam = row["kernel_family"]
        sub = row["kernel_subtype"]
        if fam not in ("elementwise", "reduce", "copy", "cast"):
            continue
        bs_v = int(row.get("launch_block_size", 0) or 0)
        gs_v = int(row.get("launch_grid_size", 0) or 0)
        threads = gs_v * bs_v if (bs_v and gs_v) else np.nan
        op = gkr.op_type_for(sub)
        numel_guess = threads * vec if not pd.isna(threads) else np.nan
        numel_snapped = snap(numel_guess) if not pd.isna(numel_guess) else np.nan
        r = _base_row(row, source, gpu, cfg.name, fam)
        r["op_type"] = op
        r["numel"] = numel_snapped
        r["bs"] = cfg.bs
        r["seq"] = cfg.seq
        r["held_out"] = cfg.held_out
        out.append(r)
    return out


def _classify(df: pd.DataFrame) -> pd.DataFrame:
    pairs = df["Kernel Name"].fillna("").map(gkr.classify_kernel)
    df = df.copy()
    df["kernel_subtype"] = pairs.map(lambda p: p[0])
    df["kernel_family"]  = pairs.map(lambda p: p[1])
    return df


def label_one_csv(ncu_paths: list[Path], gpu: str, dir_name: str,
                    held_out: bool) -> tuple[list[dict], dict]:
    cfg = model_specs.get_model_config(dir_name, held_out=held_out)
    if cfg is None:
        return [], {"error": f"no ModelConfig for {dir_name}"}

    df = ncu_loader.load_concat(ncu_paths, dedupe_keys=("ID", "Kernel Name"))
    if len(df) == 0:
        return [], {"error": "empty ncu data"}
    df = _classify(df)

    flash_count = int((df["kernel_family"] == "flash_attn").sum())
    source = f"{dir_name}_prefill"

    if flash_count == 0:
        gemm_rows, g_stats = annotate_gemm_noflash_fallback(df, source, gpu, cfg)
        path_used = "noflash_fallback"
    elif cfg.is_moe:
        gemm_rows, g_stats = annotate_gemm_moe_model(df, source, gpu, cfg)
        path_used = "moe_layer_walker"
    else:
        gemm_rows, g_stats = annotate_gemm_dense_model(df, source, gpu, cfg)
        path_used = "dense_layer_walker"

    splitk_rows = annotate_splitk(df, source, gpu, cfg, gemm_rows)
    flash_rows  = annotate_flash(df, source, gpu, cfg)
    elem_rows   = annotate_elementwise(df, source, gpu, cfg)

    for r in gemm_rows:
        r.pop("_idx", None)

    rows = gemm_rows + splitk_rows + flash_rows + elem_rows
    per_fam = dict(pd.Series([r["kernel_family"] for r in rows]).value_counts()) if rows else {}
    stats = {
        "path": path_used, "flash_count": flash_count,
        "total_rows": int(len(df)), "labeled": len(rows),
        "per_family": per_fam,
        "gemm": g_stats,
    }
    return rows, stats


def run(ncu_root: Path, out_csv: Path, report_path: Path, gpus: list[str]) -> None:
    ncu_root = Path(ncu_root)
    out_csv = Path(out_csv)
    report_path = Path(report_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    report_sections: list[str] = []

    for gpu in gpus:
        gpu_root = ncu_root / gpu / "ncu"
        if not gpu_root.is_dir():
            report_sections.append(f"## {gpu}\n- no data at `{gpu_root}`, skipped\n")
            continue
        for model_dir in sorted(gpu_root.iterdir()):
            if not model_dir.is_dir():
                continue
            csvs = sorted(model_dir.glob("prefill_seq128_bs1*.csv"))
            if not csvs:
                continue
            held_out = model_specs.is_held_out(model_dir.name, gpu)
            rows, stats = label_one_csv(csvs, gpu, model_dir.name, held_out)
            all_rows.extend(rows)
            report_sections.append(
                f"### {gpu} / {model_dir.name}  (held_out={held_out})\n"
                f"- path: {stats.get('path', 'error')}\n"
                f"- flash_count: {stats.get('flash_count', '-')}\n"
                f"- ncu rows: {stats.get('total_rows', '-')}\n"
                f"- labeled: {stats.get('labeled', 0)} {stats.get('per_family', {})}\n"
                f"- gemm stats: {stats.get('gemm', {})}\n"
                + (f"- ERROR: {stats['error']}\n" if 'error' in stats else "")
            )

    df = pd.DataFrame(all_rows)
    for c in OUT_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[OUT_COLS] if len(df) else pd.DataFrame(columns=OUT_COLS)
    df.to_csv(out_csv, index=False)

    by_gpu = df.groupby("gpu").size().to_dict() if len(df) else {}
    by_family = df.groupby("kernel_family").size().to_dict() if len(df) else {}
    by_gpu_family = (df.groupby(["gpu", "kernel_family"]).size().unstack(fill_value=0)
                     if len(df) else pd.DataFrame())

    header = [
        "# Phase A1 Shape Labeling Report",
        "",
        f"- Output: `{out_csv}`",
        f"- Total labeled rows: {len(df)}",
        f"- GPUs: {', '.join(gpus)}",
        f"- Training rows (held_out=False): {int((~df['held_out'].astype(bool)).sum()) if len(df) else 0}",
        f"- Held-out rows (held_out=True): {int(df['held_out'].astype(bool).sum()) if len(df) else 0}",
        "",
        "## Counts by GPU",
        str(by_gpu),
        "",
        "## Counts by kernel_family",
        str(by_family),
        "",
        "## Counts by GPU × family",
        by_gpu_family.to_string() if not by_gpu_family.empty else "(empty)",
        "",
        "## Per-CSV details",
        *report_sections,
    ]
    report_path.write_text("\n".join(header))

    print(f"Wrote {out_csv} ({len(df)} rows)")
    print(f"Wrote {report_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase A1 shape labeler")
    ap.add_argument("--ncu-root", default="/root/llm/profiling-data",
                    help="Root directory containing {gpu}/ncu/{model_dir}/*.csv")
    ap.add_argument("--out", default=None,
                    help="Output CSV path (default: <package>/data/kernels_labeled.csv)")
    ap.add_argument("--report", default=None,
                    help="Markdown report path (default: <package>/reports/labeling_report.md)")
    ap.add_argument("--gpus", nargs="+", default=["A100", "RTX3090", "RTX2080Ti"])
    args = ap.parse_args()

    pkg_dir = Path(__file__).resolve().parent
    out_csv = Path(args.out) if args.out else pkg_dir / "data" / "kernels_labeled.csv"
    report  = Path(args.report) if args.report else pkg_dir / "reports" / "labeling_report.md"

    run(Path(args.ncu_root), out_csv, report, args.gpus)


if __name__ == "__main__":
    main()
