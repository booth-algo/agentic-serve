#!/usr/bin/env python3
"""
Saturation Analysis — detect if benchmark throughput has plateaued.

For each model/engine/profile combination, checks whether output_token_throughput
is still increasing at the highest concurrency tested. Recommends whether to
extend the concurrency sweep.

Usage:
    python scripts/saturation_analysis.py [--results-dir results/] [--threshold 0.05]

Output:
    Table showing each series, its throughput curve, and saturation status:
    - SATURATED: throughput growth < threshold between last two points
    - GROWING: still increasing, recommend higher concurrency
    - DECLINING: throughput dropped (overloaded), already past saturation
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def load_results(results_dir: str) -> list[dict]:
    """Load all valid result JSON files."""
    results = []
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if not f.endswith('.json') or '_per_turn' in f:
                continue
            path = os.path.join(root, f)
            try:
                with open(path) as fh:
                    data = json.load(fh)
                if not data.get('summary') or not data.get('config'):
                    continue
                s = data['summary']
                if s.get('successful_requests', 0) == 0:
                    continue
                results.append(data)
            except (json.JSONDecodeError, KeyError):
                continue
    return results


def extract_series_key(data: dict) -> str:
    """Build a series key from the result."""
    c = data['config']
    model = c.get('model', '').split('/')[-1]
    backend = c.get('backend', '?')
    profile = c.get('profile', '?')
    # Infer TP from parent directory
    return f"{model} / {backend} / {profile}"


def analyze_saturation(results: list[dict], threshold: float) -> list[dict]:
    """Analyze each series for throughput saturation."""
    # Group by series
    series_map = defaultdict(list)
    for r in results:
        key = extract_series_key(r)
        conc = r['config'].get('concurrency', r['summary'].get('concurrency', 0))
        tput = r['summary'].get('output_token_throughput', 0)
        series_map[key].append((conc, tput))

    analysis = []
    for key, points in sorted(series_map.items()):
        points.sort(key=lambda x: x[0])

        if len(points) < 2:
            analysis.append({
                'series': key,
                'points': points,
                'status': 'INSUFFICIENT',
                'recommendation': 'Need at least 2 concurrency levels',
                'max_conc': points[-1][0] if points else 0,
                'max_tput': points[-1][1] if points else 0,
                'peak_conc': points[-1][0] if points else 0,
                'peak_tput': points[-1][1] if points else 0,
            })
            continue

        # Find peak throughput
        peak_idx = max(range(len(points)), key=lambda i: points[i][1])
        peak_conc, peak_tput = points[peak_idx]

        # Check last two points
        last_conc, last_tput = points[-1]
        prev_conc, prev_tput = points[-2]

        if prev_tput > 0:
            growth_rate = (last_tput - prev_tput) / prev_tput
        else:
            growth_rate = 0

        # Determine status
        if peak_idx < len(points) - 1:
            # Peak was before the last point — throughput is declining
            decline = (peak_tput - last_tput) / peak_tput if peak_tput > 0 else 0
            status = 'DECLINING'
            recommendation = f'Past saturation at conc={peak_conc} ({peak_tput:.0f} tok/s). Last conc={last_conc} dropped {decline:.1%}.'
        elif abs(growth_rate) < threshold:
            status = 'SATURATED'
            recommendation = f'Plateaued at conc={last_conc} ({last_tput:.0f} tok/s). Growth {growth_rate:+.1%} < {threshold:.0%} threshold.'
        else:
            status = 'GROWING'
            next_conc = int(last_conc * 1.5)
            recommendation = f'Still growing {growth_rate:+.1%} at conc={last_conc}. Try conc={next_conc}+.'

        # Build concurrency curve summary
        curve = ' → '.join(f'{c}:{t:.0f}' for c, t in points)

        analysis.append({
            'series': key,
            'points': points,
            'curve': curve,
            'status': status,
            'recommendation': recommendation,
            'growth_rate': growth_rate,
            'max_conc': last_conc,
            'max_tput': last_tput,
            'peak_conc': peak_conc,
            'peak_tput': peak_tput,
        })

    return analysis


def print_report(analysis: list[dict]):
    """Print saturation analysis report."""
    # Count by status
    counts = defaultdict(int)
    for a in analysis:
        counts[a['status']] += 1

    print(f"\n{'='*80}")
    print(f" Saturation Analysis Report")
    print(f"{'='*80}")
    print(f" Total series: {len(analysis)}")
    print(f" GROWING: {counts['GROWING']}  |  SATURATED: {counts['SATURATED']}  |  DECLINING: {counts['DECLINING']}  |  INSUFFICIENT: {counts['INSUFFICIENT']}")
    print(f"{'='*80}\n")

    # Group by status for cleaner output
    for status in ['GROWING', 'SATURATED', 'DECLINING', 'INSUFFICIENT']:
        items = [a for a in analysis if a['status'] == status]
        if not items:
            continue

        emoji = {'GROWING': '📈', 'SATURATED': '📊', 'DECLINING': '📉', 'INSUFFICIENT': '❓'}[status]
        print(f"\n{emoji} {status} ({len(items)} series)")
        print(f"{'─'*78}")

        for a in items:
            print(f"  {a['series']}")
            if 'curve' in a:
                print(f"    Curve (conc:tok/s): {a['curve']}")
            print(f"    → {a['recommendation']}")
            print()

    # Summary recommendations
    growing = [a for a in analysis if a['status'] == 'GROWING']
    if growing:
        max_concs = set(a['max_conc'] for a in growing)
        print(f"\n{'='*80}")
        print(f" RECOMMENDATION: {len(growing)} series still growing.")
        print(f" Current max concurrency tested: {sorted(max_concs)}")
        suggested = max(max_concs) * 2
        print(f" Suggest extending to: {suggested}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark throughput saturation")
    parser.add_argument('--results-dir', default='results/', help='Results directory')
    parser.add_argument('--threshold', type=float, default=0.05, help='Growth rate threshold for saturation (default: 5%%)')
    parser.add_argument('--json', action='store_true', help='Output as JSON instead of text')
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No valid results found in {args.results_dir}")
        sys.exit(1)

    print(f"Loaded {len(results)} benchmark results")

    analysis = analyze_saturation(results, args.threshold)

    if args.json:
        # JSON output (without points array for readability)
        out = [{k: v for k, v in a.items() if k != 'points'} for a in analysis]
        print(json.dumps(out, indent=2))
    else:
        print_report(analysis)


if __name__ == '__main__':
    main()
