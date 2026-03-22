#!/usr/bin/env python
"""
Master runner for the full V2 experiment suite.

Runs all 4 panels sequentially with shared output directory,
then generates plots and report.

Usage:
    # Full run (20 seeds, 4 workers, no MIP — recommended)
    python scripts/run_all_v2.py --seeds 20 --workers 4 --no-mip

    # Quick test (5 seeds, 1 worker)
    python scripts/run_all_v2.py --seeds 5 --workers 1 --no-mip

    # Dry run (shows experiment counts and time estimates)
    python scripts/run_all_v2.py --dry-run --seeds 20 --workers 4

    # Resume interrupted run
    python scripts/run_all_v2.py --seeds 20 --workers 4 --no-mip --resume \
        --output-dir artifacts/experiments/v2_benchmark_YYYYMMDD_HHMM
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(ROOT, "venv", "bin", "python")
if not os.path.exists(PYTHON):
    PYTHON = sys.executable


def run_cmd(cmd: list, description: str) -> bool:
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed with exit code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run full V2 experiment suite")
    parser.add_argument("--seeds",      type=int,  default=20)
    parser.add_argument("--workers",    type=int,  default=4)
    parser.add_argument("--output-dir", type=str,  default=None)
    parser.add_argument("--no-mip",     action="store_true")
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--skip",       type=str,  nargs="*", default=[],
                        help="Panels to skip (e.g., --skip panel2 panel3)")
    args = parser.parse_args()

    if args.output_dir:
        out_dir = args.output_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        out_dir = f"artifacts/experiments/v2_benchmark_{ts}"

    base_args = [
        "--seeds", str(args.seeds),
        "--workers", str(args.workers),
        "--output-dir", out_dir,
    ]
    if args.no_mip:
        base_args.append("--no-mip")
    if args.resume:
        base_args.append("--resume")
    if args.dry_run:
        base_args.append("--dry-run")

    panels = [
        ("panel1", "scripts/run_panel1_v2.py", "Panel 1 V2 — Core Benchmark"),
        ("panel2", "scripts/run_panel2_v2.py", "Panel 2 V2 — Mechanism Analysis"),
        ("panel3", "scripts/run_panel3_v2.py", "Panel 3 V2 — Robustness"),
        ("panel4", "scripts/run_panel4_v2.py", "Panel 4 V2 — Budget Frontier"),
    ]

    print(f"SupplySim V2 Experiment Suite")
    print(f"Output: {out_dir}")
    print(f"Seeds: {args.seeds} | Workers: {args.workers} | MIP: {'yes' if not args.no_mip else 'no'}")
    print(f"Skip: {args.skip if args.skip else 'none'}")

    for panel_name, script, description in panels:
        if panel_name in args.skip:
            print(f"\n[SKIP] {description}")
            continue
        ok = run_cmd([PYTHON, script] + base_args, description)
        if not ok and not args.dry_run:
            print(f"\n[WARN] {panel_name} failed. Continuing with remaining panels...")

    if not args.dry_run:
        # Analysis and plotting
        run_cmd(
            [PYTHON, "scripts/analyze_and_plot_v2.py", "--output-dir", out_dir],
            "Analysis and Plots"
        )

    print(f"\n{'='*70}")
    print(f"  COMPLETE")
    print(f"  Output directory: {out_dir}")
    print(f"  Report: {out_dir}/report_v2.md")
    print(f"  Plots:  {out_dir}/plots/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
