#!/usr/bin/env python3
"""
gpu-nvme-direct: Benchmark Parameter Sweep Orchestrator

Runs all 4 benchmark methods across a parameter sweep of block sizes,
queue depths, and access patterns. Collects CSV results, computes
aggregate statistics with confidence intervals, and detects outlier
runs (e.g., from thermal throttling).

Usage:
    python3 bench_sweep.py [options]

SPDX-License-Identifier: BSD-2-Clause
"""

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


# ------- Configuration -------

BLOCK_SIZES = ["512", "4K", "16K", "64K", "256K", "1M", "4M"]

QUEUE_DEPTHS = [1, 4, 16, 64]

PATTERNS = ["seq", "rand"]

RUNS_PER_CONFIG = 5

BENCHMARKS = {
    "gpu_direct": "bench_gpu_direct",
    "cufile": "bench_cufile",
    "cpu_memcpy": "bench_cpu_memcpy",
    "cpu_pinned": "bench_cpu_pinned",
}

# Outlier detection: if a run's throughput is below this fraction of the
# median, flag it as an outlier (likely thermal throttling or interference).
OUTLIER_THRESHOLD = 0.7


# ------- Helpers -------

def find_bench_dir():
    """Find the benchmark binary directory."""
    candidates = [
        Path(__file__).parent,
        Path(__file__).parent / ".." / "build" / "bench",
        Path(__file__).parent / "build" / "bench",
        Path("build") / "bench",
        Path("."),
    ]
    for d in candidates:
        d = d.resolve()
        # Check if at least one benchmark binary exists
        for name in BENCHMARKS.values():
            if (d / name).exists():
                return d
    return Path(__file__).parent.resolve()


def parse_block_size_bytes(s):
    """Convert block size string to integer bytes."""
    s = s.strip().upper()
    if s.endswith("M"):
        return int(float(s[:-1]) * 1024 * 1024)
    elif s.endswith("K"):
        return int(float(s[:-1]) * 1024)
    elif s.endswith("G"):
        return int(float(s[:-1]) * 1024 * 1024 * 1024)
    else:
        return int(s)


def format_block_size(bs_bytes):
    """Format byte count as human-readable block size."""
    if bs_bytes >= 1024 * 1024 and bs_bytes % (1024 * 1024) == 0:
        return f"{bs_bytes // (1024 * 1024)}M"
    elif bs_bytes >= 1024 and bs_bytes % 1024 == 0:
        return f"{bs_bytes // 1024}K"
    else:
        return str(bs_bytes)


def mean(values):
    """Arithmetic mean."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def stddev(values):
    """Sample standard deviation."""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def median(values):
    """Median value."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2
    else:
        return s[n // 2]


def confidence_interval_95(values):
    """95% confidence interval using t-distribution approximation."""
    n = len(values)
    if n < 2:
        return (mean(values), mean(values), mean(values))

    m = mean(values)
    s = stddev(values)
    # t-value for 95% CI with n-1 degrees of freedom (approximate)
    t_values = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        15: 2.131, 20: 2.086, 30: 2.042, 60: 2.000, 120: 1.980,
    }
    df = n - 1
    t = t_values.get(df, 1.96)  # Fallback to z-value
    margin = t * s / math.sqrt(n)
    return (m - margin, m, m + margin)


def detect_outliers(throughputs):
    """Detect outlier runs based on throughput deviation from median."""
    if len(throughputs) < 3:
        return []

    med = median(throughputs)
    if med <= 0:
        return []

    outliers = []
    for i, tp in enumerate(throughputs):
        if tp < med * OUTLIER_THRESHOLD:
            outliers.append(i)
    return outliers


# ------- Run a Single Benchmark -------

def run_benchmark(bench_bin, block_size, queue_depth, pattern, num_ops,
                  output_csv, device, extra_args=None):
    """Run a single benchmark invocation and return True on success."""
    cmd = [
        str(bench_bin),
        "--block-size", str(block_size),
        "--queue-depth", str(queue_depth),
        "--num-ops", str(num_ops),
        "--pattern", pattern,
        "--output", str(output_csv),
        "--device", device,
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"  CMD: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per run
        )
        if result.returncode != 0:
            print(f"  WARNING: benchmark returned {result.returncode}")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[:5]:
                    print(f"    STDERR: {line}")
            return False

        # Print first few lines of stdout for progress
        if result.stdout:
            for line in result.stdout.strip().split("\n")[:3]:
                print(f"    {line}")

        return True

    except subprocess.TimeoutExpired:
        print("  ERROR: benchmark timed out (300s)")
        return False
    except FileNotFoundError:
        print(f"  ERROR: benchmark binary not found: {bench_bin}")
        return False


# ------- Aggregate Results -------

def aggregate_results(per_run_csvs, output_csv):
    """Read per-run CSVs and produce aggregate statistics."""
    # Collect all rows keyed by (method, block_size, queue_depth, pattern)
    data = defaultdict(lambda: defaultdict(list))

    for csv_path in per_run_csvs:
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (
                    row["method"],
                    int(row["block_size"]),
                    int(row["queue_depth"]),
                    row["pattern"],
                )
                for field in ["mean_us", "median_us", "p99_us", "p999_us",
                              "throughput_mbs", "iops", "cpu_util_pct",
                              "min_us", "max_us"]:
                    if field in row and row[field]:
                        data[key][field].append(float(row[field]))

    # Write aggregate CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "block_size", "queue_depth", "pattern",
            "num_runs",
            "throughput_mbs_mean", "throughput_mbs_ci_lo", "throughput_mbs_ci_hi",
            "iops_mean", "iops_ci_lo", "iops_ci_hi",
            "latency_mean_us", "latency_median_us",
            "latency_p99_us", "latency_p999_us",
            "cpu_util_pct_mean",
            "outlier_runs",
        ])

        for key in sorted(data.keys()):
            method, block_size, queue_depth, pattern = key
            d = data[key]

            throughputs = d.get("throughput_mbs", [])
            iops_vals = d.get("iops", [])
            latency_means = d.get("mean_us", [])
            latency_medians = d.get("median_us", [])
            latency_p99s = d.get("p99_us", [])
            latency_p999s = d.get("p999_us", [])
            cpu_utils = d.get("cpu_util_pct", [])

            # Detect outlier runs
            outliers = detect_outliers(throughputs)
            outlier_str = ",".join(str(o) for o in outliers) if outliers else ""
            if outliers:
                print(f"  OUTLIER detected: {method} bs={block_size} "
                      f"qd={queue_depth} {pattern}: "
                      f"runs {outliers} (probable thermal throttling)")

            tp_ci = confidence_interval_95(throughputs)
            iops_ci = confidence_interval_95(iops_vals)

            writer.writerow([
                method, block_size, queue_depth, pattern,
                len(throughputs),
                f"{tp_ci[1]:.2f}", f"{tp_ci[0]:.2f}", f"{tp_ci[2]:.2f}",
                f"{iops_ci[1]:.1f}", f"{iops_ci[0]:.1f}", f"{iops_ci[2]:.1f}",
                f"{mean(latency_means):.3f}",
                f"{mean(latency_medians):.3f}",
                f"{mean(latency_p99s):.3f}",
                f"{mean(latency_p999s):.3f}",
                f"{mean(cpu_utils):.2f}",
                outlier_str,
            ])

    print(f"\nAggregate results written to: {output_csv}")


# ------- Main -------

def main():
    parser = argparse.ArgumentParser(
        description="gpu-nvme-direct benchmark parameter sweep"
    )
    parser.add_argument(
        "--bench-dir", type=str, default=None,
        help="Directory containing benchmark binaries"
    )
    parser.add_argument(
        "--output-dir", type=str, default="bench_results",
        help="Directory for output CSV files (default: bench_results)"
    )
    parser.add_argument(
        "--device", type=str, default="/dev/nvme0n1",
        help="NVMe device path (default: /dev/nvme0n1)"
    )
    parser.add_argument(
        "--num-ops", type=int, default=1000,
        help="Number of I/O operations per run (default: 1000)"
    )
    parser.add_argument(
        "--runs", type=int, default=RUNS_PER_CONFIG,
        help=f"Number of runs per configuration (default: {RUNS_PER_CONFIG})"
    )
    parser.add_argument(
        "--block-sizes", nargs="+", default=BLOCK_SIZES,
        help="Block sizes to test (default: 512 4K 16K 64K 256K 1M 4M)"
    )
    parser.add_argument(
        "--queue-depths", nargs="+", type=int, default=QUEUE_DEPTHS,
        help="Queue depths to test (default: 1 4 16 64)"
    )
    parser.add_argument(
        "--patterns", nargs="+", default=PATTERNS,
        help="Access patterns to test (default: seq rand)"
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=None,
        help="Subset of benchmarks to run (default: all). "
             "Options: gpu_direct, cufile, cpu_memcpy, cpu_pinned"
    )
    parser.add_argument(
        "--cooldown", type=int, default=2,
        help="Seconds to wait between runs (default: 2)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing"
    )

    args = parser.parse_args()

    # Determine benchmark directory
    if args.bench_dir:
        bench_dir = Path(args.bench_dir).resolve()
    else:
        bench_dir = find_bench_dir()

    print(f"Benchmark directory: {bench_dir}")

    # Determine which benchmarks to run
    bench_subset = args.benchmarks or list(BENCHMARKS.keys())
    active_benchmarks = {}
    for name in bench_subset:
        if name not in BENCHMARKS:
            print(f"ERROR: unknown benchmark '{name}'. "
                  f"Options: {', '.join(BENCHMARKS.keys())}")
            sys.exit(1)
        bin_path = bench_dir / BENCHMARKS[name]
        if not bin_path.exists() and not args.dry_run:
            print(f"WARNING: {bin_path} not found, skipping '{name}'")
            continue
        active_benchmarks[name] = bin_path

    if not active_benchmarks and not args.dry_run:
        print("ERROR: no benchmark binaries found. Build first with CMake.")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count total configurations
    total_configs = (
        len(active_benchmarks) *
        len(args.block_sizes) *
        len(args.queue_depths) *
        len(args.patterns) *
        args.runs
    )
    print(f"\nSweep configuration:")
    print(f"  Benchmarks:   {', '.join(active_benchmarks.keys())}")
    print(f"  Block sizes:  {', '.join(args.block_sizes)}")
    print(f"  Queue depths: {', '.join(str(q) for q in args.queue_depths)}")
    print(f"  Patterns:     {', '.join(args.patterns)}")
    print(f"  Runs/config:  {args.runs}")
    print(f"  Total runs:   {total_configs}")
    print(f"  Output dir:   {output_dir}")
    print()

    # ------- Execute Sweep -------

    per_run_csvs = []
    completed = 0
    failed = 0
    start_time = time.time()

    for bench_name, bench_bin in sorted(active_benchmarks.items()):
        for bs in args.block_sizes:
            for qd in args.queue_depths:
                for pattern in args.patterns:
                    for run_idx in range(args.runs):
                        completed += 1
                        progress = f"[{completed}/{total_configs}]"

                        run_csv = output_dir / (
                            f"{bench_name}_bs{bs}_qd{qd}_{pattern}"
                            f"_run{run_idx}.csv"
                        )
                        per_run_csvs.append(str(run_csv))

                        print(f"\n{progress} {bench_name} "
                              f"bs={bs} qd={qd} {pattern} "
                              f"run={run_idx + 1}/{args.runs}")

                        if args.dry_run:
                            print(f"  DRY RUN: would execute {bench_bin}")
                            continue

                        # Remove stale output
                        if run_csv.exists():
                            run_csv.unlink()

                        success = run_benchmark(
                            bench_bin, bs, qd, pattern,
                            args.num_ops, str(run_csv),
                            args.device,
                        )

                        if not success:
                            failed += 1

                        # Cooldown between runs to avoid thermal effects
                        if args.cooldown > 0 and completed < total_configs:
                            time.sleep(args.cooldown)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Sweep complete: {completed - failed}/{completed} successful "
          f"({elapsed:.0f}s elapsed)")

    if args.dry_run:
        print("(dry run - no results generated)")
        return

    # ------- Aggregate -------

    print(f"\nAggregating results from {len(per_run_csvs)} runs...")
    aggregate_csv = output_dir / "aggregate_results.csv"
    aggregate_results(per_run_csvs, str(aggregate_csv))

    # Also create a combined raw CSV with all individual runs
    combined_csv = output_dir / "combined_raw.csv"
    header_written = False

    with open(combined_csv, "w", newline="") as out_f:
        for csv_path in per_run_csvs:
            if not os.path.exists(csv_path):
                continue
            with open(csv_path, "r") as in_f:
                reader = csv.reader(in_f)
                header = next(reader, None)
                if header and not header_written:
                    # Add run_file column
                    writer = csv.writer(out_f)
                    writer.writerow(header + ["source_file"])
                    header_written = True
                writer = csv.writer(out_f)
                for row in reader:
                    writer.writerow(row + [os.path.basename(csv_path)])

    print(f"Combined raw results: {combined_csv}")
    print(f"Aggregate results:    {aggregate_csv}")

    if failed > 0:
        print(f"\nWARNING: {failed} runs failed. Check output above for details.")


if __name__ == "__main__":
    main()
