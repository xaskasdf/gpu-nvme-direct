/*
 * gpu-nvme-direct: CPU Memcpy Benchmark
 *
 * Traditional I/O path: pread() from NVMe into a malloc'd host buffer,
 * then cudaMemcpy(H2D) to copy data to GPU memory.
 *
 * This is the baseline "naive" approach that most applications use.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <vector>

#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "bench_common.h"

/* ------- Main ------- */

int main(int argc, char **argv) {
    const char *METHOD = "cpu_memcpy";
    BenchConfig cfg = parse_args(argc, argv, METHOD);

    printf("[%s] block_size=%s queue_depth=%u num_ops=%u pattern=%s device=%s\n",
           METHOD, format_block_size(cfg.block_size).c_str(),
           cfg.queue_depth, cfg.num_ops, cfg.pattern.c_str(),
           cfg.device.c_str());

    /* Initialize CUDA */
    int device_id = 0;
    CUDA_CHECK(cudaSetDevice(device_id));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    printf("[%s] GPU: %s\n", METHOD, prop.name);

    /* Open the NVMe device with O_DIRECT for bypassing page cache */
    int fd = open_device_direct(cfg.device.c_str());
    if (fd < 0) {
        fprintf(stderr, "Error: cannot open %s: %s\n",
                cfg.device.c_str(), strerror(errno));
        return EXIT_FAILURE;
    }

    /* Get device size */
    uint64_t max_lba_blocks = get_device_size_blocks(fd, cfg.block_size);
    if (cfg.max_lba == 0) {
        cfg.max_lba = max_lba_blocks;
    }
    if (cfg.max_lba == 0) {
        fprintf(stderr, "Error: cannot determine device size. "
                "Use --max-lba to specify.\n");
        close(fd);
        return EXIT_FAILURE;
    }

    printf("[%s] Device size: %lu blocks of %s\n",
           METHOD, (unsigned long)cfg.max_lba,
           format_block_size(cfg.block_size).c_str());

    /* Allocate aligned host buffer for pread (O_DIRECT requires alignment) */
    void *h_buf = nullptr;
    if (posix_memalign(&h_buf, 4096, cfg.block_size) != 0) {
        fprintf(stderr, "Error: posix_memalign failed\n");
        close(fd);
        return EXIT_FAILURE;
    }
    memset(h_buf, 0, cfg.block_size);

    /* Allocate GPU buffer */
    void *d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, cfg.block_size));

    /* ------- Generate Byte Offsets ------- */

    uint64_t blocks_per_op = cfg.block_size / 512;
    if (blocks_per_op == 0) blocks_per_op = 1;

    uint64_t lba_range = cfg.max_lba - blocks_per_op;
    if (lba_range == 0) lba_range = 1;

    std::vector<uint64_t> offsets(cfg.num_ops);
    srand(42);

    if (cfg.pattern == "seq") {
        for (uint32_t i = 0; i < cfg.num_ops; i++) {
            uint64_t lba = (cfg.start_lba + (uint64_t)i * blocks_per_op) %
                           cfg.max_lba;
            offsets[i] = lba * 512;  /* Convert LBA to byte offset */
        }
    } else {
        for (uint32_t i = 0; i < cfg.num_ops; i++) {
            /* Align to block_size for O_DIRECT compatibility */
            uint64_t lba = random_lba(lba_range);
            offsets[i] = lba * 512;
        }
    }

    /* ------- Warmup ------- */

    printf("[%s] Running %d warmup iterations...\n", METHOD, WARMUP_ITERATIONS);

    for (int w = 0; w < WARMUP_ITERATIONS; w++) {
        uint64_t off = offsets[w % cfg.num_ops];

        ssize_t ret = pread(fd, h_buf, cfg.block_size, static_cast<off_t>(off));
        if (ret < 0) {
            fprintf(stderr, "pread warmup failed: %s\n", strerror(errno));
        }

        CUDA_CHECK(cudaMemcpy(d_buf, h_buf, cfg.block_size,
                               cudaMemcpyHostToDevice));
    }

    /* ------- Benchmark ------- */

    printf("[%s] Running %u operations (pread + cudaMemcpy H2D)...\n",
           METHOD, cfg.num_ops);

    std::vector<double> latencies_us;
    latencies_us.reserve(cfg.num_ops);

    /* Also track sub-timings for pread and cudaMemcpy separately */
    std::vector<double> pread_us;
    std::vector<double> memcpy_us;
    pread_us.reserve(cfg.num_ops);
    memcpy_us.reserve(cfg.num_ops);

    CpuSample cpu_before = read_cpu_sample();
    WallTimer wall;
    wall.start();

    for (uint32_t i = 0; i < cfg.num_ops; i++) {
        WallTimer total_timer, pread_timer, memcpy_timer;

        total_timer.start();

        /* Phase 1: pread() from NVMe to host buffer */
        pread_timer.start();
        ssize_t ret = pread(fd, h_buf, cfg.block_size,
                            static_cast<off_t>(offsets[i]));
        pread_timer.stop();

        if (ret < 0) {
            fprintf(stderr, "pread failed at op %u (offset=%lu): %s\n",
                    i, (unsigned long)offsets[i], strerror(errno));
            continue;
        }
        if (static_cast<uint64_t>(ret) != cfg.block_size) {
            fprintf(stderr, "pread short read at op %u: %zd / %lu\n",
                    i, ret, (unsigned long)cfg.block_size);
        }

        /* Phase 2: cudaMemcpy H2D to GPU buffer */
        memcpy_timer.start();
        CUDA_CHECK(cudaMemcpy(d_buf, h_buf, cfg.block_size,
                               cudaMemcpyHostToDevice));
        memcpy_timer.stop();

        total_timer.stop();

        latencies_us.push_back(total_timer.elapsed_us());
        pread_us.push_back(pread_timer.elapsed_us());
        memcpy_us.push_back(memcpy_timer.elapsed_us());
    }

    wall.stop();
    CpuSample cpu_after = read_cpu_sample();

    double elapsed_sec = wall.elapsed_sec();
    uint64_t total_bytes = static_cast<uint64_t>(latencies_us.size()) *
                           cfg.block_size;

    printf("[%s] Completed: %zu / %u ops (wall: %.3f s)\n",
           METHOD, latencies_us.size(), cfg.num_ops, elapsed_sec);

    /* ------- Compute and Output Stats ------- */

    BenchStats stats = compute_stats(latencies_us, total_bytes, elapsed_sec,
                                      cpu_before, cpu_after);

    print_stats(METHOD, cfg.block_size, cfg.queue_depth,
                cfg.pattern.c_str(), stats);

    /* Print breakdown of pread vs cudaMemcpy */
    if (!pread_us.empty()) {
        sort_latencies(pread_us);
        sort_latencies(memcpy_us);

        double pread_mean = std::accumulate(pread_us.begin(), pread_us.end(), 0.0)
                            / static_cast<double>(pread_us.size());
        double memcpy_mean = std::accumulate(memcpy_us.begin(), memcpy_us.end(), 0.0)
                             / static_cast<double>(memcpy_us.size());

        printf("  Breakdown (mean us): pread=%.1f  cudaMemcpy=%.1f  "
               "(pread=%.0f%% of total)\n",
               pread_mean, memcpy_mean,
               100.0 * pread_mean / (pread_mean + memcpy_mean));
    }

    /* Write CSV output */
    if (!cfg.output.empty()) {
        bool write_header = false;
        FILE *fp = fopen(cfg.output.c_str(), "r");
        if (!fp) {
            write_header = true;
        } else {
            fclose(fp);
        }

        fp = fopen(cfg.output.c_str(), "a");
        if (fp) {
            if (write_header) {
                write_csv_header(fp);
            }
            write_csv_row(fp, METHOD, cfg.block_size, cfg.queue_depth,
                          cfg.pattern.c_str(), stats);
            fclose(fp);
            printf("[%s] Results appended to %s\n", METHOD, cfg.output.c_str());
        } else {
            fprintf(stderr, "Warning: could not open %s for writing\n",
                    cfg.output.c_str());
        }
    }

    /* ------- Cleanup ------- */

    CUDA_CHECK(cudaFree(d_buf));
    free(h_buf);
    close(fd);

    return EXIT_SUCCESS;
}
