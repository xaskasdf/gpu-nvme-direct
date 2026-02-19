# Benchmark Methodology

## Methods Compared

| # | Method | Data Path | CPU in I/O path? |
|---|--------|-----------|-------------------|
| 1 | gpu-direct | GPU → NVMe BAR MMIO → DMA to buffer | No |
| 2 | cufile | cuFileRead() (compat mode or GDS) | Yes |
| 3 | cpu-memcpy | pread() → host buf → cudaMemcpy | Yes |
| 4 | cpu-pinned | pread() → pinned buf → cudaMemcpyAsync | Yes |

## Parameters

- **Block sizes**: 512B, 4KB, 16KB, 64KB, 256KB, 1MB, 4MB
- **Queue depths**: 1, 4, 16, 64
- **Access patterns**: Sequential, Random
- **Runs per configuration**: 5 (median reported)

## Metrics

- **Latency**: min, mean, median, p99, p99.9, max (microseconds)
- **Throughput**: MB/s (sustained) and IOPS
- **CPU utilization**: /proc/stat delta during benchmark window
- **GPU SM occupancy**: CUPTI counters (when available)

## Procedure

1. Warmup: 10 operations (discarded)
2. Measure: N operations (default 1000)
3. Per-operation latency recorded
4. 5 independent runs, detect outliers (>2σ from mean)
5. Statistics computed on pooled data

## Hardware Setup

- Dedicated test NVMe (not boot drive)
- GPU and NVMe on same PCIe root complex preferred
- Thermal steady state reached before measurement

## Output

- Raw CSV: per-operation latency, throughput, CPU util
- Aggregated CSV: per-configuration summary statistics
- Plots: throughput bars, latency CDF, CPU util, IOPS vs QD
