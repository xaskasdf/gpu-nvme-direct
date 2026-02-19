/*
 * gpu-nvme-direct: Shared Benchmark Infrastructure
 *
 * Common timing, statistics, and output helpers used by all benchmarks.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#ifndef BENCH_COMMON_H
#define BENCH_COMMON_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <vector>
#include <string>
#include <numeric>

#include <cuda_runtime.h>

/* ------- CUDA Error Checking ------- */

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err_), (int)err_);  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* ------- Constants ------- */

static constexpr int WARMUP_ITERATIONS = 10;

/* ------- GPU Event Timer ------- */

struct GpuTimer {
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    cudaStream_t stream;

    GpuTimer(cudaStream_t s = nullptr) : stream(s) {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }

    ~GpuTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    /* Record start on the associated stream */
    void start() {
        CUDA_CHECK(cudaEventRecord(start_event, stream));
    }

    /* Record stop on the associated stream */
    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_event, stream));
    }

    /* Wait for stop event and return elapsed milliseconds */
    float elapsed_ms() {
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
        return ms;
    }

    /* Elapsed in seconds */
    double elapsed_sec() {
        return static_cast<double>(elapsed_ms()) / 1000.0;
    }
};

/* ------- Wall Clock Timer (clock_gettime) ------- */

struct WallTimer {
    struct timespec ts_start;
    struct timespec ts_stop;

    void start() {
        clock_gettime(CLOCK_MONOTONIC, &ts_start);
    }

    void stop() {
        clock_gettime(CLOCK_MONOTONIC, &ts_stop);
    }

    /* Elapsed time in seconds (double precision) */
    double elapsed_sec() const {
        double s = static_cast<double>(ts_stop.tv_sec - ts_start.tv_sec);
        double ns = static_cast<double>(ts_stop.tv_nsec - ts_start.tv_nsec);
        return s + ns * 1e-9;
    }

    /* Elapsed time in milliseconds */
    double elapsed_ms() const {
        return elapsed_sec() * 1000.0;
    }

    /* Elapsed time in microseconds */
    double elapsed_us() const {
        return elapsed_sec() * 1e6;
    }
};

/* ------- CPU Utilization Sampling via /proc/stat ------- */

struct CpuSample {
    uint64_t user;
    uint64_t nice;
    uint64_t system;
    uint64_t idle;
    uint64_t iowait;
    uint64_t irq;
    uint64_t softirq;
    uint64_t steal;
    bool valid;

    CpuSample() : user(0), nice(0), system(0), idle(0),
                   iowait(0), irq(0), softirq(0), steal(0), valid(false) {}

    uint64_t total_active() const {
        return user + nice + system + irq + softirq + steal;
    }

    uint64_t total() const {
        return total_active() + idle + iowait;
    }
};

static inline CpuSample read_cpu_sample() {
    CpuSample s;
    FILE *fp = fopen("/proc/stat", "r");
    if (!fp) {
        /* /proc/stat not available (e.g., non-Linux) */
        return s;
    }

    char line[256];
    if (fgets(line, sizeof(line), fp)) {
        /* Line format: cpu  user nice system idle iowait irq softirq steal ... */
        int n = sscanf(line, "cpu %lu %lu %lu %lu %lu %lu %lu %lu",
                       &s.user, &s.nice, &s.system, &s.idle,
                       &s.iowait, &s.irq, &s.softirq, &s.steal);
        if (n >= 4) {
            s.valid = true;
        }
    }
    fclose(fp);
    return s;
}

/* Compute CPU utilization percentage between two samples */
static inline double cpu_utilization_pct(const CpuSample &before,
                                          const CpuSample &after) {
    if (!before.valid || !after.valid) return -1.0;

    uint64_t d_total = after.total() - before.total();
    uint64_t d_active = after.total_active() - before.total_active();

    if (d_total == 0) return 0.0;
    return 100.0 * static_cast<double>(d_active) / static_cast<double>(d_total);
}

/* ------- Sort Helper ------- */

static inline void sort_latencies(std::vector<double> &v) {
    std::sort(v.begin(), v.end());
}

/* ------- Percentile Calculation ------- */

/* Returns the value at the given percentile (0.0 - 100.0).
 * Vector must be pre-sorted in ascending order. */
static inline double percentile(const std::vector<double> &sorted_v, double p) {
    if (sorted_v.empty()) return 0.0;
    if (sorted_v.size() == 1) return sorted_v[0];

    double rank = (p / 100.0) * static_cast<double>(sorted_v.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(rank));
    size_t hi = static_cast<size_t>(std::ceil(rank));

    if (lo == hi || hi >= sorted_v.size()) {
        return sorted_v[lo];
    }

    double frac = rank - static_cast<double>(lo);
    return sorted_v[lo] + frac * (sorted_v[hi] - sorted_v[lo]);
}

/* ------- Statistics Struct ------- */

struct BenchStats {
    double min_us;     /* Minimum latency (microseconds) */
    double max_us;     /* Maximum latency (microseconds) */
    double mean_us;    /* Mean latency (microseconds) */
    double median_us;  /* Median latency (microseconds) */
    double p99_us;     /* 99th percentile latency (microseconds) */
    double p999_us;    /* 99.9th percentile latency (microseconds) */
    double stddev_us;  /* Standard deviation (microseconds) */

    double throughput_mbs;   /* Throughput in MB/s */
    double iops;             /* I/O operations per second */
    double cpu_util_pct;     /* CPU utilization percentage */
    double total_sec;        /* Total elapsed time (seconds) */

    uint64_t total_bytes;    /* Total bytes transferred */
    uint32_t num_ops;        /* Number of I/O operations */
};

/*
 * Compute statistics from a vector of per-operation latencies (in microseconds).
 * total_bytes: total data transferred.
 * elapsed_sec: total wall-clock time for all operations.
 * cpu_before/cpu_after: CPU samples bracketing the benchmark run.
 */
static inline BenchStats compute_stats(std::vector<double> &latencies_us,
                                        uint64_t total_bytes,
                                        double elapsed_sec,
                                        const CpuSample &cpu_before,
                                        const CpuSample &cpu_after) {
    BenchStats st = {};
    if (latencies_us.empty()) return st;

    sort_latencies(latencies_us);

    st.num_ops = static_cast<uint32_t>(latencies_us.size());
    st.total_bytes = total_bytes;
    st.total_sec = elapsed_sec;

    st.min_us = latencies_us.front();
    st.max_us = latencies_us.back();

    double sum = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
    st.mean_us = sum / static_cast<double>(st.num_ops);

    st.median_us = percentile(latencies_us, 50.0);
    st.p99_us = percentile(latencies_us, 99.0);
    st.p999_us = percentile(latencies_us, 99.9);

    /* Standard deviation */
    double sq_sum = 0.0;
    for (double v : latencies_us) {
        double d = v - st.mean_us;
        sq_sum += d * d;
    }
    st.stddev_us = std::sqrt(sq_sum / static_cast<double>(st.num_ops));

    /* Throughput and IOPS */
    if (elapsed_sec > 0.0) {
        st.throughput_mbs = (static_cast<double>(total_bytes) / (1024.0 * 1024.0))
                            / elapsed_sec;
        st.iops = static_cast<double>(st.num_ops) / elapsed_sec;
    }

    /* CPU utilization */
    st.cpu_util_pct = cpu_utilization_pct(cpu_before, cpu_after);

    return st;
}

/* ------- CSV Output ------- */

/* Write CSV header line */
static inline void write_csv_header(FILE *fp) {
    fprintf(fp,
        "method,block_size,queue_depth,pattern,num_ops,"
        "min_us,max_us,mean_us,median_us,p99_us,p999_us,stddev_us,"
        "throughput_mbs,iops,cpu_util_pct,total_sec\n");
}

/* Write one CSV row */
static inline void write_csv_row(FILE *fp,
                                   const char *method,
                                   uint64_t block_size,
                                   uint32_t queue_depth,
                                   const char *pattern,
                                   const BenchStats &stats) {
    fprintf(fp,
        "%s,%lu,%u,%s,%u,"
        "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
        "%.2f,%.1f,%.2f,%.6f\n",
        method,
        (unsigned long)block_size,
        queue_depth,
        pattern,
        stats.num_ops,
        stats.min_us, stats.max_us, stats.mean_us,
        stats.median_us, stats.p99_us, stats.p999_us, stats.stddev_us,
        stats.throughput_mbs, stats.iops, stats.cpu_util_pct, stats.total_sec);
}

/* ------- Human-Readable Output ------- */

static inline void print_stats(const char *method,
                                uint64_t block_size,
                                uint32_t queue_depth,
                                const char *pattern,
                                const BenchStats &stats) {
    printf("=== %s | bs=%lu qd=%u pattern=%s ops=%u ===\n",
           method, (unsigned long)block_size, queue_depth, pattern, stats.num_ops);
    printf("  Latency (us): min=%.1f  mean=%.1f  median=%.1f  "
           "p99=%.1f  p99.9=%.1f  max=%.1f  stddev=%.1f\n",
           stats.min_us, stats.mean_us, stats.median_us,
           stats.p99_us, stats.p999_us, stats.max_us, stats.stddev_us);
    printf("  Throughput: %.2f MB/s   IOPS: %.0f\n",
           stats.throughput_mbs, stats.iops);
    printf("  CPU util: %.1f%%   Total time: %.3f s\n",
           stats.cpu_util_pct, stats.total_sec);
    printf("\n");
}

/* ------- Throughput Helpers ------- */

static inline double bytes_to_mbs(uint64_t bytes, double elapsed_sec) {
    if (elapsed_sec <= 0.0) return 0.0;
    return (static_cast<double>(bytes) / (1024.0 * 1024.0)) / elapsed_sec;
}

static inline double ops_to_iops(uint32_t ops, double elapsed_sec) {
    if (elapsed_sec <= 0.0) return 0.0;
    return static_cast<double>(ops) / elapsed_sec;
}

/* ------- Block Size Parser ------- */

/* Parse block size string with optional suffix: 512, 4K, 16K, 1M, etc. */
static inline uint64_t parse_block_size(const char *str) {
    char *end = nullptr;
    double val = strtod(str, &end);

    if (end == str) {
        fprintf(stderr, "Error: invalid block size '%s'\n", str);
        exit(EXIT_FAILURE);
    }

    if (end && (*end == 'K' || *end == 'k')) {
        val *= 1024.0;
    } else if (end && (*end == 'M' || *end == 'm')) {
        val *= 1024.0 * 1024.0;
    } else if (end && (*end == 'G' || *end == 'g')) {
        val *= 1024.0 * 1024.0 * 1024.0;
    }

    return static_cast<uint64_t>(val);
}

/* Format block size for display: 512, 4K, 64K, 1M, etc. */
static inline std::string format_block_size(uint64_t bs) {
    char buf[32];
    if (bs >= 1024ULL * 1024ULL && (bs % (1024ULL * 1024ULL)) == 0) {
        snprintf(buf, sizeof(buf), "%luM", (unsigned long)(bs / (1024ULL * 1024ULL)));
    } else if (bs >= 1024ULL && (bs % 1024ULL) == 0) {
        snprintf(buf, sizeof(buf), "%luK", (unsigned long)(bs / 1024ULL));
    } else {
        snprintf(buf, sizeof(buf), "%lu", (unsigned long)bs);
    }
    return std::string(buf);
}

/* ------- Random LBA Generation ------- */

static inline uint64_t random_lba(uint64_t max_lba) {
    /* Combine two rand() calls for > 32-bit range */
    uint64_t r = ((uint64_t)rand() << 32) | (uint64_t)rand();
    return r % max_lba;
}

/* ------- Common CLI Argument Parsing ------- */

struct BenchConfig {
    uint64_t block_size;      /* I/O block size in bytes */
    uint32_t queue_depth;     /* Queue depth (concurrent ops) */
    uint32_t num_ops;         /* Number of I/O operations */
    std::string pattern;      /* "seq" or "rand" */
    std::string output;       /* Output CSV file path (empty = stdout) */
    std::string device;       /* NVMe device path (e.g., /dev/nvme0n1) */
    uint64_t start_lba;       /* Starting LBA for sequential I/O */
    uint64_t max_lba;         /* Maximum LBA for random I/O */

    BenchConfig()
        : block_size(4096)
        , queue_depth(1)
        , num_ops(1000)
        , pattern("seq")
        , output("")
        , device("/dev/nvme0n1")
        , start_lba(0)
        , max_lba(0)
    {}
};

static inline void print_usage(const char *prog, const char *method) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "  Benchmark: %s\n"
        "\n"
        "Options:\n"
        "  --block-size <size>   I/O block size (default: 4K)\n"
        "                        Accepts suffixes: K, M, G\n"
        "  --queue-depth <n>     Queue depth (default: 1)\n"
        "  --num-ops <n>         Number of I/O operations (default: 1000)\n"
        "  --pattern <type>      Access pattern: seq or rand (default: seq)\n"
        "  --output <file>       Output CSV file (default: stdout)\n"
        "  --device <path>       NVMe device path (default: /dev/nvme0n1)\n"
        "  --start-lba <n>       Starting LBA for sequential (default: 0)\n"
        "  --max-lba <n>         Max LBA for random (default: auto-detect)\n"
        "  --help                Show this help message\n",
        prog, method);
}

static inline BenchConfig parse_args(int argc, char **argv, const char *method) {
    BenchConfig cfg;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) {
            cfg.block_size = parse_block_size(argv[++i]);
        } else if (strcmp(argv[i], "--queue-depth") == 0 && i + 1 < argc) {
            cfg.queue_depth = static_cast<uint32_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "--num-ops") == 0 && i + 1 < argc) {
            cfg.num_ops = static_cast<uint32_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "--pattern") == 0 && i + 1 < argc) {
            cfg.pattern = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            cfg.output = argv[++i];
        } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            cfg.device = argv[++i];
        } else if (strcmp(argv[i], "--start-lba") == 0 && i + 1 < argc) {
            cfg.start_lba = strtoull(argv[++i], nullptr, 0);
        } else if (strcmp(argv[i], "--max-lba") == 0 && i + 1 < argc) {
            cfg.max_lba = strtoull(argv[++i], nullptr, 0);
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0], method);
            exit(EXIT_SUCCESS);
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0], method);
            exit(EXIT_FAILURE);
        }
    }

    /* Validate */
    if (cfg.block_size == 0 || (cfg.block_size & (cfg.block_size - 1)) != 0) {
        /* Allow non-power-of-2 but warn */
        if (cfg.block_size == 0) {
            fprintf(stderr, "Error: block size must be > 0\n");
            exit(EXIT_FAILURE);
        }
    }
    if (cfg.queue_depth == 0) {
        fprintf(stderr, "Error: queue depth must be > 0\n");
        exit(EXIT_FAILURE);
    }
    if (cfg.num_ops == 0) {
        fprintf(stderr, "Error: num-ops must be > 0\n");
        exit(EXIT_FAILURE);
    }
    if (cfg.pattern != "seq" && cfg.pattern != "rand") {
        fprintf(stderr, "Error: pattern must be 'seq' or 'rand'\n");
        exit(EXIT_FAILURE);
    }

    return cfg;
}

/* ------- File-descriptor helper: get device size in blocks ------- */

#include <sys/ioctl.h>
#include <linux/fs.h>
#include <fcntl.h>
#include <unistd.h>

static inline uint64_t get_device_size_blocks(int fd, uint64_t block_size) {
    uint64_t dev_bytes = 0;

    if (ioctl(fd, BLKGETSIZE64, &dev_bytes) < 0) {
        /* Try lseek fallback */
        off_t end = lseek(fd, 0, SEEK_END);
        if (end < 0) {
            fprintf(stderr, "Warning: cannot determine device size\n");
            return 0;
        }
        dev_bytes = static_cast<uint64_t>(end);
        lseek(fd, 0, SEEK_SET);
    }

    return dev_bytes / block_size;
}

/* Open a device for direct I/O (O_DIRECT), return fd or -1 */
static inline int open_device_direct(const char *path) {
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        /* Fallback without O_DIRECT */
        fprintf(stderr, "Warning: O_DIRECT not supported for %s, "
                "falling back to buffered I/O\n", path);
        fd = open(path, O_RDONLY);
    }
    return fd;
}

#endif /* BENCH_COMMON_H */
