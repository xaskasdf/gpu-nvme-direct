/*
 * gpu-nvme-direct: CPU Pinned Memory Benchmark
 *
 * Optimized traditional I/O path: pread() from NVMe into a
 * cudaMallocHost pinned host buffer, then cudaMemcpyAsync(H2D)
 * to GPU memory with stream synchronization.
 *
 * Pinned memory enables DMA-based transfers that bypass the CPU,
 * resulting in higher bandwidth than pageable cudaMemcpy.
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
    const char *METHOD = "cpu_pinned";
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

    /* Create CUDA stream for async operations */
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    /* Open the NVMe device with O_DIRECT */
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

    /*
     * Allocate pinned host buffer with cudaMallocHost.
     * This memory is page-locked, which:
     *   1. Enables DMA for cudaMemcpyAsync (no staging copy)
     *   2. Must be page-aligned (cudaMallocHost guarantees this)
     *   3. Can be used with O_DIRECT pread() since it is page-aligned
     */
    void *h_pinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_pinned, cfg.block_size));
    memset(h_pinned, 0, cfg.block_size);

    /* Allocate GPU buffer */
    void *d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, cfg.block_size));

    /*
     * For higher queue depths, allocate multiple pinned buffers to enable
     * overlapping pread and cudaMemcpyAsync (double/multi-buffering).
     */
    uint32_t num_buffers = cfg.queue_depth;
    if (num_buffers > 64) num_buffers = 64;  /* Cap buffer count */

    std::vector<void *> h_buffers(num_buffers);
    std::vector<void *> d_buffers(num_buffers);
    std::vector<cudaStream_t> streams(num_buffers);

    for (uint32_t b = 0; b < num_buffers; b++) {
        CUDA_CHECK(cudaMallocHost(&h_buffers[b], cfg.block_size));
        CUDA_CHECK(cudaMalloc(&d_buffers[b], cfg.block_size));
        CUDA_CHECK(cudaStreamCreate(&streams[b]));
    }

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
            offsets[i] = lba * 512;
        }
    } else {
        for (uint32_t i = 0; i < cfg.num_ops; i++) {
            uint64_t lba = random_lba(lba_range);
            offsets[i] = lba * 512;
        }
    }

    /* ------- Warmup ------- */

    printf("[%s] Running %d warmup iterations...\n", METHOD, WARMUP_ITERATIONS);

    for (int w = 0; w < WARMUP_ITERATIONS; w++) {
        uint32_t b = w % num_buffers;
        uint64_t off = offsets[w % cfg.num_ops];

        ssize_t ret = pread(fd, h_buffers[b], cfg.block_size,
                            static_cast<off_t>(off));
        if (ret < 0) {
            fprintf(stderr, "pread warmup failed: %s\n", strerror(errno));
        }

        CUDA_CHECK(cudaMemcpyAsync(d_buffers[b], h_buffers[b], cfg.block_size,
                                    cudaMemcpyHostToDevice, streams[b]));
        CUDA_CHECK(cudaStreamSynchronize(streams[b]));
    }

    /* ------- Benchmark ------- */

    printf("[%s] Running %u operations (pread + cudaMemcpyAsync H2D)...\n",
           METHOD, cfg.num_ops);

    std::vector<double> latencies_us;
    latencies_us.reserve(cfg.num_ops);

    std::vector<double> pread_us;
    std::vector<double> memcpy_us;
    pread_us.reserve(cfg.num_ops);
    memcpy_us.reserve(cfg.num_ops);

    CpuSample cpu_before = read_cpu_sample();
    WallTimer wall;
    wall.start();

    if (num_buffers == 1) {
        /* Single-buffer path: serial pread + async memcpy + sync */
        for (uint32_t i = 0; i < cfg.num_ops; i++) {
            WallTimer total_timer, pread_timer, memcpy_timer;

            total_timer.start();

            /* Phase 1: pread() into pinned buffer */
            pread_timer.start();
            ssize_t ret = pread(fd, h_pinned, cfg.block_size,
                                static_cast<off_t>(offsets[i]));
            pread_timer.stop();

            if (ret < 0) {
                fprintf(stderr, "pread failed at op %u: %s\n",
                        i, strerror(errno));
                continue;
            }
            if (static_cast<uint64_t>(ret) != cfg.block_size) {
                fprintf(stderr, "pread short read at op %u: %zd / %lu\n",
                        i, ret, (unsigned long)cfg.block_size);
            }

            /* Phase 2: async cudaMemcpy H2D using pinned buffer */
            memcpy_timer.start();
            CUDA_CHECK(cudaMemcpyAsync(d_buf, h_pinned, cfg.block_size,
                                        cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            memcpy_timer.stop();

            total_timer.stop();

            latencies_us.push_back(total_timer.elapsed_us());
            pread_us.push_back(pread_timer.elapsed_us());
            memcpy_us.push_back(memcpy_timer.elapsed_us());
        }
    } else {
        /*
         * Multi-buffer path: overlap pread with previous cudaMemcpyAsync.
         *
         * Pipeline: while buffer[b] is being DMA'd to GPU via stream[b],
         * we issue pread() into buffer[b+1] on the CPU side.
         */
        for (uint32_t i = 0; i < cfg.num_ops; i++) {
            uint32_t b = i % num_buffers;
            WallTimer total_timer, pread_timer, memcpy_timer;

            /* Wait for the stream to be available (from a previous batch) */
            CUDA_CHECK(cudaStreamSynchronize(streams[b]));

            total_timer.start();

            /* Phase 1: pread() into pinned buffer */
            pread_timer.start();
            ssize_t ret = pread(fd, h_buffers[b], cfg.block_size,
                                static_cast<off_t>(offsets[i]));
            pread_timer.stop();

            if (ret < 0) {
                fprintf(stderr, "pread failed at op %u: %s\n",
                        i, strerror(errno));
                continue;
            }

            /* Phase 2: async cudaMemcpy H2D */
            memcpy_timer.start();
            CUDA_CHECK(cudaMemcpyAsync(d_buffers[b], h_buffers[b],
                                        cfg.block_size,
                                        cudaMemcpyHostToDevice, streams[b]));
            CUDA_CHECK(cudaStreamSynchronize(streams[b]));
            memcpy_timer.stop();

            total_timer.stop();

            latencies_us.push_back(total_timer.elapsed_us());
            pread_us.push_back(pread_timer.elapsed_us());
            memcpy_us.push_back(memcpy_timer.elapsed_us());
        }

        /* Drain all streams */
        for (uint32_t b = 0; b < num_buffers; b++) {
            CUDA_CHECK(cudaStreamSynchronize(streams[b]));
        }
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

    /* Print breakdown */
    if (!pread_us.empty()) {
        sort_latencies(pread_us);
        sort_latencies(memcpy_us);

        double pread_mean = std::accumulate(pread_us.begin(), pread_us.end(), 0.0)
                            / static_cast<double>(pread_us.size());
        double memcpy_mean = std::accumulate(memcpy_us.begin(), memcpy_us.end(), 0.0)
                             / static_cast<double>(memcpy_us.size());

        printf("  Breakdown (mean us): pread=%.1f  cudaMemcpyAsync=%.1f  "
               "(pread=%.0f%% of total)\n",
               pread_mean, memcpy_mean,
               100.0 * pread_mean / (pread_mean + memcpy_mean));
        printf("  Note: cudaMemcpyAsync with pinned memory uses DMA engine "
               "(no CPU involvement)\n");
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

    for (uint32_t b = 0; b < num_buffers; b++) {
        CUDA_CHECK(cudaStreamDestroy(streams[b]));
        CUDA_CHECK(cudaFree(d_buffers[b]));
        CUDA_CHECK(cudaFreeHost(h_buffers[b]));
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFreeHost(h_pinned));
    close(fd);

    return EXIT_SUCCESS;
}
