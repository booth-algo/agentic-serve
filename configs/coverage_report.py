#!/usr/bin/env python3
"""Per-config data coverage report — what each deployment HAS vs INHERITS vs is MISSING.

Reads the ``data`` manifests in ``configs/deployments/*.json`` (via :mod:`configs.loader`) and prints a
``config × data-input → status`` matrix, a per-config summary, and a drift check (every ``measured`` /
``derived`` artifact with a ``path`` must exist on disk). This is the canonical "which file provides what
data, and what's missing per config" view — replacing the ad-hoc ``profiling/docs/*.yaml``.

    python3 -m configs.coverage_report
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from configs.loader import REPO_ROOT, all_deployments

BENCH_BASE = Path("/mnt/100g/agent-bench/results/synthetic_distributional")
_ABBR = {"measured": "measured", "derived": "derived", "inherited": "inherit",
         "placeholder": "placehold", "missing": "MISSING"}


def _resolve(path: str) -> Path | None:
    """An existing path for a manifest entry — repo-relative, else the central bench store."""
    for cand in (REPO_ROOT / path, BENCH_BASE / path):
        if cand.exists():
            return cand
    return None


def main() -> None:
    deps = all_deployments()
    inputs: list[str] = []
    for d in deps:
        for k in d.data:
            if k not in inputs:
                inputs.append(k)
    w = max((len(i) for i in inputs), default=10) + 2

    print("\nDATA COVERAGE (configs/deployments)")
    print("input".ljust(w) + "".join(d.gpu_key.ljust(12) for d in deps))
    for inp in inputs:
        row = inp.ljust(w)
        for d in deps:
            st = (d.data.get(inp) or {}).get("status", "-")
            row += _ABBR.get(st, st).ljust(12)
        print(row)

    print("\nper-config status counts:")
    for d in deps:
        counts = Counter(e.get("status", "?") for e in d.data.values())
        miss = [k for k, e in d.data.items() if e.get("status") == "missing"]
        line = "  " + d.gpu_key.ljust(8) + ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
        if miss:
            line += f"   MISSING: {miss}"
        print(line)

    print("\ndrift check (measured/derived artifacts must exist on disk):")
    bad = 0
    for d in deps:
        for k, e in d.data.items():
            if e.get("status") in ("measured", "derived") and e.get("path"):
                if _resolve(e["path"]) is None:
                    print(f"  !! {d.gpu_key} {k}: missing file {e['path']}")
                    bad += 1
    print("  all present ✓" if bad == 0 else f"  {bad} missing file(s) — see above")


if __name__ == "__main__":
    main()
