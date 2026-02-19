#!/usr/bin/env python3
"""
gpu-nvme-direct: Publication-Quality Benchmark Plots

Generates plots from benchmark results CSV:
  1. Throughput (MB/s) vs block size (grouped bar chart)
  2. Latency CDF per method
  3. CPU utilization comparison (bar chart)
  4. IOPS vs queue depth (line plot)

Usage:
    python3 plot_results.py --input aggregate_results.csv --output-dir plots/

SPDX-License-Identifier: BSD-2-Clause
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for server/CI
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("ERROR: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Note: seaborn not found, using matplotlib defaults")


# ------- Style Configuration -------

# Method display names and colors
METHOD_CONFIG = {
    "gpu_direct":   {"label": "GPU-Direct NVMe",  "color": "#e63946", "hatch": ""},
    "cufile":       {"label": "cuFile (GDS)",      "color": "#457b9d", "hatch": "//"},
    "cpu_pinned":   {"label": "CPU Pinned+Async",  "color": "#2a9d8f", "hatch": ".."},
    "cpu_memcpy":   {"label": "CPU malloc+Memcpy", "color": "#e9c46a", "hatch": "xx"},
    "cufile_posix_fallback": {"label": "cuFile (POSIX fallback)",
                              "color": "#a8dadc", "hatch": "\\\\"},
}

# Canonical method ordering
METHOD_ORDER = ["gpu_direct", "cufile", "cpu_pinned", "cpu_memcpy",
                "cufile_posix_fallback"]

BLOCK_SIZE_ORDER = [512, 4096, 16384, 65536, 262144, 1048576, 4194304]


def setup_style():
    """Configure matplotlib style for publication-quality figures."""
    if HAS_SEABORN:
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)
    else:
        plt.style.use("seaborn-v0_8-whitegrid")

    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.constrained_layout.use": True,
    })


# ------- Data Loading -------

def load_aggregate_csv(path):
    """Load the aggregate results CSV into a structured dict."""
    data = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {
                "method": row["method"],
                "block_size": int(row["block_size"]),
                "queue_depth": int(row["queue_depth"]),
                "pattern": row["pattern"],
                "num_runs": int(row.get("num_runs", 0)),
                "throughput_mbs_mean": float(row.get("throughput_mbs_mean", 0)),
                "throughput_mbs_ci_lo": float(row.get("throughput_mbs_ci_lo", 0)),
                "throughput_mbs_ci_hi": float(row.get("throughput_mbs_ci_hi", 0)),
                "iops_mean": float(row.get("iops_mean", 0)),
                "iops_ci_lo": float(row.get("iops_ci_lo", 0)),
                "iops_ci_hi": float(row.get("iops_ci_hi", 0)),
                "latency_mean_us": float(row.get("latency_mean_us", 0)),
                "latency_median_us": float(row.get("latency_median_us", 0)),
                "latency_p99_us": float(row.get("latency_p99_us", 0)),
                "latency_p999_us": float(row.get("latency_p999_us", 0)),
                "cpu_util_pct_mean": float(row.get("cpu_util_pct_mean", 0)),
            }
            data.append(entry)
    return data


def load_raw_csv(path):
    """Load raw per-run CSV for CDF plots."""
    data = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {
                "method": row["method"],
                "block_size": int(row["block_size"]),
                "queue_depth": int(row["queue_depth"]),
                "pattern": row["pattern"],
                "mean_us": float(row.get("mean_us", 0)),
                "median_us": float(row.get("median_us", 0)),
                "p99_us": float(row.get("p99_us", 0)),
                "p999_us": float(row.get("p999_us", 0)),
                "min_us": float(row.get("min_us", 0)),
                "max_us": float(row.get("max_us", 0)),
                "throughput_mbs": float(row.get("throughput_mbs", 0)),
                "iops": float(row.get("iops", 0)),
                "cpu_util_pct": float(row.get("cpu_util_pct", 0)),
            }
            data.append(entry)
    return data


def format_block_size(bs):
    """Format block size in bytes to human-readable string."""
    if bs >= 1024 * 1024:
        return f"{bs // (1024 * 1024)}M"
    elif bs >= 1024:
        return f"{bs // 1024}K"
    else:
        return str(bs)


def get_methods_in_data(data):
    """Get ordered list of methods present in the data."""
    present = set(row["method"] for row in data)
    return [m for m in METHOD_ORDER if m in present]


# ------- Plot 1: Throughput vs Block Size (Grouped Bar) -------

def plot_throughput_vs_blocksize(data, output_dir, pattern="seq", queue_depth=1):
    """Grouped bar chart: throughput (MB/s) vs block size, one bar per method."""
    # Filter data
    filtered = [r for r in data
                if r["pattern"] == pattern and r["queue_depth"] == queue_depth]
    if not filtered:
        print(f"  Skipping throughput plot (no data for pattern={pattern} qd={queue_depth})")
        return

    methods = get_methods_in_data(filtered)
    block_sizes = sorted(set(r["block_size"] for r in filtered))

    # Build lookup
    lookup = {}
    for r in filtered:
        lookup[(r["method"], r["block_size"])] = r

    fig, ax = plt.subplots(figsize=(12, 6))

    n_methods = len(methods)
    bar_width = 0.8 / n_methods
    x_positions = list(range(len(block_sizes)))

    for i, method in enumerate(methods):
        cfg = METHOD_CONFIG.get(method, {"label": method, "color": "gray", "hatch": ""})
        throughputs = []
        errors_lo = []
        errors_hi = []
        positions = []

        for j, bs in enumerate(block_sizes):
            key = (method, bs)
            if key in lookup:
                r = lookup[key]
                tp = r["throughput_mbs_mean"]
                throughputs.append(tp)
                errors_lo.append(tp - r["throughput_mbs_ci_lo"])
                errors_hi.append(r["throughput_mbs_ci_hi"] - tp)
                positions.append(j + i * bar_width - (n_methods - 1) * bar_width / 2)

        ax.bar(positions, throughputs,
               width=bar_width * 0.9,
               yerr=[errors_lo, errors_hi],
               capsize=3,
               label=cfg["label"],
               color=cfg["color"],
               hatch=cfg["hatch"],
               edgecolor="black",
               linewidth=0.5,
               error_kw={"linewidth": 1})

    ax.set_xlabel("Block Size")
    ax.set_ylabel("Throughput (MB/s)")
    ax.set_title(f"Read Throughput vs Block Size ({pattern}, QD={queue_depth})")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([format_block_size(bs) for bs in block_sizes])
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Add grid on y-axis only
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", alpha=0)

    for fmt in ["pdf", "png"]:
        path = output_dir / f"throughput_vs_blocksize_{pattern}_qd{queue_depth}.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=300 if fmt == "png" else None)
        print(f"  Saved: {path}")

    plt.close(fig)


# ------- Plot 2: Latency CDF -------

def plot_latency_cdf(data, output_dir, block_size=4096, pattern="seq", queue_depth=1):
    """CDF of latency percentiles for each method."""
    # Filter raw data
    filtered = [r for r in data
                if r["pattern"] == pattern and r["queue_depth"] == queue_depth
                and r["block_size"] == block_size]
    if not filtered:
        print(f"  Skipping latency CDF plot (no data for bs={block_size} "
              f"pattern={pattern} qd={queue_depth})")
        return

    methods = get_methods_in_data(filtered)

    fig, ax = plt.subplots(figsize=(10, 6))

    for method in methods:
        cfg = METHOD_CONFIG.get(method, {"label": method, "color": "gray"})
        method_data = [r for r in filtered if r["method"] == method]

        if not method_data:
            continue

        # Construct approximate CDF from percentile data
        # We have min, median (p50), mean, p99, p999, max
        # Use the average across runs
        min_vals = [r["min_us"] for r in method_data]
        median_vals = [r["median_us"] for r in method_data]
        mean_vals = [r["mean_us"] for r in method_data]
        p99_vals = [r["p99_us"] for r in method_data]
        p999_vals = [r["p999_us"] for r in method_data]
        max_vals = [r["max_us"] for r in method_data]

        avg = lambda vals: sum(vals) / len(vals) if vals else 0

        # Build CDF points
        cdf_x = [
            avg(min_vals),
            avg(median_vals),
            avg(mean_vals),
            avg(p99_vals),
            avg(p999_vals),
            avg(max_vals),
        ]
        cdf_y = [0.0, 0.50, 0.50, 0.99, 0.999, 1.0]

        # Sort by x for proper CDF shape
        points = sorted(zip(cdf_x, cdf_y))
        cdf_x = [p[0] for p in points]
        cdf_y = [p[1] for p in points]

        ax.plot(cdf_x, cdf_y,
                marker="o", markersize=5,
                label=cfg["label"],
                color=cfg["color"],
                linewidth=2)

    ax.set_xlabel("Latency (us)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(f"Latency CDF ({format_block_size(block_size)}, {pattern}, QD={queue_depth})")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # Add horizontal reference lines
    for p, label in [(0.5, "p50"), (0.99, "p99"), (0.999, "p99.9")]:
        ax.axhline(y=p, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.text(ax.get_xlim()[0], p + 0.01, label, fontsize=8, color="gray")

    for fmt in ["pdf", "png"]:
        path = output_dir / f"latency_cdf_{format_block_size(block_size)}_{pattern}_qd{queue_depth}.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=300 if fmt == "png" else None)
        print(f"  Saved: {path}")

    plt.close(fig)


# ------- Plot 3: CPU Utilization Comparison -------

def plot_cpu_utilization(data, output_dir, pattern="seq", queue_depth=1):
    """Bar chart comparing CPU utilization across methods and block sizes."""
    filtered = [r for r in data
                if r["pattern"] == pattern and r["queue_depth"] == queue_depth]
    if not filtered:
        print(f"  Skipping CPU utilization plot (no data)")
        return

    methods = get_methods_in_data(filtered)
    block_sizes = sorted(set(r["block_size"] for r in filtered))

    lookup = {}
    for r in filtered:
        lookup[(r["method"], r["block_size"])] = r

    fig, ax = plt.subplots(figsize=(12, 6))

    n_methods = len(methods)
    bar_width = 0.8 / n_methods
    x_positions = list(range(len(block_sizes)))

    for i, method in enumerate(methods):
        cfg = METHOD_CONFIG.get(method, {"label": method, "color": "gray", "hatch": ""})
        cpu_utils = []
        positions = []

        for j, bs in enumerate(block_sizes):
            key = (method, bs)
            if key in lookup:
                r = lookup[key]
                cpu_utils.append(r.get("cpu_util_pct_mean", 0))
                positions.append(j + i * bar_width - (n_methods - 1) * bar_width / 2)

        ax.bar(positions, cpu_utils,
               width=bar_width * 0.9,
               label=cfg["label"],
               color=cfg["color"],
               hatch=cfg["hatch"],
               edgecolor="black",
               linewidth=0.5)

    ax.set_xlabel("Block Size")
    ax.set_ylabel("CPU Utilization (%)")
    ax.set_title(f"CPU Utilization Comparison ({pattern}, QD={queue_depth})")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([format_block_size(bs) for bs in block_sizes])
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(bottom=0, top=max(100, ax.get_ylim()[1] * 1.1))
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", alpha=0)

    # Reference line at 100%
    ax.axhline(y=100, color="red", linestyle="--", alpha=0.4, linewidth=1)

    for fmt in ["pdf", "png"]:
        path = output_dir / f"cpu_utilization_{pattern}_qd{queue_depth}.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=300 if fmt == "png" else None)
        print(f"  Saved: {path}")

    plt.close(fig)


# ------- Plot 4: IOPS vs Queue Depth -------

def plot_iops_vs_queue_depth(data, output_dir, block_size=4096, pattern="rand"):
    """Line plot: IOPS vs queue depth for each method."""
    filtered = [r for r in data
                if r["pattern"] == pattern and r["block_size"] == block_size]
    if not filtered:
        print(f"  Skipping IOPS plot (no data for bs={block_size} pattern={pattern})")
        return

    methods = get_methods_in_data(filtered)

    fig, ax = plt.subplots(figsize=(10, 6))

    for method in methods:
        cfg = METHOD_CONFIG.get(method, {"label": method, "color": "gray"})
        method_data = sorted(
            [r for r in filtered if r["method"] == method],
            key=lambda r: r["queue_depth"]
        )

        if not method_data:
            continue

        qds = [r["queue_depth"] for r in method_data]
        iops_vals = [r["iops_mean"] for r in method_data]
        iops_lo = [r["iops_mean"] - r["iops_ci_lo"] for r in method_data]
        iops_hi = [r["iops_ci_hi"] - r["iops_mean"] for r in method_data]

        ax.errorbar(qds, iops_vals,
                    yerr=[iops_lo, iops_hi],
                    marker="o", markersize=6,
                    capsize=4,
                    label=cfg["label"],
                    color=cfg["color"],
                    linewidth=2,
                    markeredgecolor="black",
                    markeredgewidth=0.5)

    ax.set_xlabel("Queue Depth")
    ax.set_ylabel("IOPS")
    ax.set_title(f"IOPS vs Queue Depth ({format_block_size(block_size)}, {pattern})")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(bottom=0)

    # Mark queue depths on x-axis
    all_qds = sorted(set(r["queue_depth"] for r in filtered))
    ax.set_xticks(all_qds)

    for fmt in ["pdf", "png"]:
        path = output_dir / f"iops_vs_qd_{format_block_size(block_size)}_{pattern}.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=300 if fmt == "png" else None)
        print(f"  Saved: {path}")

    plt.close(fig)


# ------- Main -------

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality benchmark plots"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to aggregate_results.csv (from bench_sweep.py)"
    )
    parser.add_argument(
        "--raw-input", type=str, default=None,
        help="Path to combined_raw.csv for detailed CDF plots"
    )
    parser.add_argument(
        "--output-dir", type=str, default="plots",
        help="Output directory for plot files (default: plots)"
    )
    parser.add_argument(
        "--formats", nargs="+", default=["pdf", "png"],
        help="Output formats (default: pdf png)"
    )
    parser.add_argument(
        "--throughput-pattern", type=str, default="seq",
        help="Pattern for throughput plot (default: seq)"
    )
    parser.add_argument(
        "--throughput-qd", type=int, default=1,
        help="Queue depth for throughput plot (default: 1)"
    )
    parser.add_argument(
        "--cdf-blocksize", type=str, default="4K",
        help="Block size for CDF plot (default: 4K)"
    )
    parser.add_argument(
        "--iops-blocksize", type=str, default="4K",
        help="Block size for IOPS plot (default: 4K)"
    )
    parser.add_argument(
        "--iops-pattern", type=str, default="rand",
        help="Pattern for IOPS plot (default: rand)"
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"ERROR: input file not found: {args.input}")
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_style()

    # Load data
    print(f"Loading aggregate data from: {args.input}")
    agg_data = load_aggregate_csv(args.input)
    print(f"  Loaded {len(agg_data)} aggregate rows")

    raw_data = None
    if args.raw_input and os.path.exists(args.raw_input):
        print(f"Loading raw data from: {args.raw_input}")
        raw_data = load_raw_csv(args.raw_input)
        print(f"  Loaded {len(raw_data)} raw rows")

    # Parse block size arguments
    def parse_bs(s):
        s = s.strip().upper()
        if s.endswith("M"):
            return int(float(s[:-1]) * 1024 * 1024)
        elif s.endswith("K"):
            return int(float(s[:-1]) * 1024)
        return int(s)

    cdf_bs = parse_bs(args.cdf_blocksize)
    iops_bs = parse_bs(args.iops_blocksize)

    # Generate plots
    print(f"\nGenerating plots in: {output_dir}")

    print("\n1. Throughput vs Block Size")
    plot_throughput_vs_blocksize(agg_data, output_dir,
                                pattern=args.throughput_pattern,
                                queue_depth=args.throughput_qd)

    # Also generate for random pattern
    if args.throughput_pattern != "rand":
        plot_throughput_vs_blocksize(agg_data, output_dir,
                                    pattern="rand",
                                    queue_depth=args.throughput_qd)

    print("\n2. Latency CDF")
    cdf_source = raw_data if raw_data else agg_data
    plot_latency_cdf(cdf_source, output_dir,
                     block_size=cdf_bs,
                     pattern="seq", queue_depth=1)
    plot_latency_cdf(cdf_source, output_dir,
                     block_size=cdf_bs,
                     pattern="rand", queue_depth=1)

    print("\n3. CPU Utilization")
    plot_cpu_utilization(agg_data, output_dir,
                        pattern="seq", queue_depth=1)

    print("\n4. IOPS vs Queue Depth")
    plot_iops_vs_queue_depth(agg_data, output_dir,
                            block_size=iops_bs,
                            pattern=args.iops_pattern)

    # Also generate for sequential
    if args.iops_pattern != "seq":
        plot_iops_vs_queue_depth(agg_data, output_dir,
                                block_size=iops_bs,
                                pattern="seq")

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
