/*
 * gpu-nvme-direct: cuFile (GPUDirect Storage) Benchmark
 *
 * Baseline benchmark using NVIDIA's cuFile API (GDS) to read data
 * directly from NVMe into GPU memory, bypassing the CPU page cache.
 *
 * Falls back to POSIX pread() + cudaMemcpy if cuFile is not available.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cerrno>

#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "bench_common.h"

/* ------- cuFile API (conditional) ------- */

#ifdef HAVE_CUFILE
#include <cufile.h>
static bool g_cufile_available = true;
#else
/* Stub types when cuFile is not available */
static bool g_cufile_available = false;
#endif

/* ------- POSIX Fallback ------- */

/*
 * Fallback: pread() to an aligned host buffer, then cudaMemcpy H2D.
 * This is essentially what bench_cpu_memcpy does, but we include it
 * here so bench_cufile always produces output.
 */
static double posix_fallback_read(int fd, void *d_buf, void *h_buf,
                                   uint64_t offset, uint64_t size) {
    WallTimer t;
    t.start();

    ssize_t ret = pread(fd, h_buf, size, static_cast<off_t>(offset));
    if (ret < 0) {
        fprintf(stderr, "pread failed: %s\n", strerror(errno));
        return -1.0;
    }
    if (static_cast<uint64_t>(ret) != size) {
        fprintf(stderr, "pread short read: %zd / %lu\n", ret,
                (unsigned long)size);
    }

    CUDA_CHECK(cudaMemcpy(d_buf, h_buf, size, cudaMemcpyHostToDevice));

    t.stop();
    return t.elapsed_us();
}

/* ------- Main ------- */

int main(int argc, char **argv) {
    const char *METHOD = "cufile";
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

    /* Open the NVMe device */
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

    /* Allocate GPU buffer */
    void *d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, cfg.block_size));

    /* Allocate aligned host buffer for POSIX fallback */
    void *h_buf = nullptr;
    if (posix_memalign(&h_buf, 4096, cfg.block_size) != 0) {
        fprintf(stderr, "Error: posix_memalign failed\n");
        close(fd);
        return EXIT_FAILURE;
    }

    /* ------- cuFile Initialization ------- */

#ifdef HAVE_CUFILE
    CUfileDescr_t cf_descr = {};
    CUfileHandle_t cf_handle = nullptr;
    CUfileError_t cf_status;

    cf_status = cuFileDriverOpen();
    if (cf_status.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "Warning: cuFileDriverOpen failed (err=%d), "
                "falling back to POSIX\n", cf_status.err);
        g_cufile_available = false;
    }

    if (g_cufile_available) {
        cf_descr.handle.fd = fd;
        cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

        cf_status = cuFileHandleRegister(&cf_handle, &cf_descr);
        if (cf_status.err != CU_FILE_SUCCESS) {
            fprintf(stderr, "Warning: cuFileHandleRegister failed (err=%d), "
                    "falling back to POSIX\n", cf_status.err);
            g_cufile_available = false;
        }
    }

    if (g_cufile_available) {
        /* Register GPU buffer with cuFile */
        cf_status = cuFileBufRegister(d_buf, cfg.block_size, 0);
        if (cf_status.err != CU_FILE_SUCCESS) {
            fprintf(stderr, "Warning: cuFileBufRegister failed (err=%d), "
                    "falling back to POSIX\n", cf_status.err);
            cuFileHandleDeregister(cf_handle);
            g_cufile_available = false;
        }
    }

    if (g_cufile_available) {
        printf("[%s] cuFile initialized successfully (GPUDirect Storage)\n",
               METHOD);
    }
#endif

    if (!g_cufile_available) {
        printf("[%s] WARNING: cuFile not available, using POSIX fallback. "
               "Results are NOT representative of GPUDirect Storage.\n",
               METHOD);
    }

    /* ------- Generate LBA offsets ------- */

    uint64_t blocks_per_op = cfg.block_size / 512;
    if (blocks_per_op == 0) blocks_per_op = 1;

    uint64_t lba_range = cfg.max_lba - blocks_per_op;
    if (lba_range == 0) lba_range = 1;

    std::vector<uint64_t> offsets(cfg.num_ops);
    srand(42);

    if (cfg.pattern == "seq") {
        for (uint32_t i = 0; i < cfg.num_ops; i++) {
            offsets[i] = ((cfg.start_lba + (uint64_t)i * blocks_per_op) %
                         cfg.max_lba) * 512;
        }
    } else {
        for (uint32_t i = 0; i < cfg.num_ops; i++) {
            offsets[i] = random_lba(lba_range) * 512;
        }
    }

    /* ------- Warmup ------- */

    printf("[%s] Running %d warmup iterations...\n", METHOD, WARMUP_ITERATIONS);

    for (int w = 0; w < WARMUP_ITERATIONS; w++) {
#ifdef HAVE_CUFILE
        if (g_cufile_available) {
            ssize_t ret = cuFileRead(cf_handle, d_buf, cfg.block_size,
                                      static_cast<off_t>(offsets[w % cfg.num_ops]),
                                      0);
            if (ret < 0) {
                fprintf(stderr, "cuFileRead warmup failed: %zd\n", ret);
            }
        } else
#endif
        {
            posix_fallback_read(fd, d_buf, h_buf, offsets[w % cfg.num_ops],
                                cfg.block_size);
        }
    }

    /* ------- Benchmark ------- */

    printf("[%s] Running %u operations...\n", METHOD, cfg.num_ops);

    std::vector<double> latencies_us;
    latencies_us.reserve(cfg.num_ops);

    CpuSample cpu_before = read_cpu_sample();
    WallTimer wall;
    wall.start();

    for (uint32_t i = 0; i < cfg.num_ops; i++) {
        WallTimer op_timer;
        op_timer.start();

#ifdef HAVE_CUFILE
        if (g_cufile_available) {
            ssize_t ret = cuFileRead(cf_handle, d_buf, cfg.block_size,
                                      static_cast<off_t>(offsets[i]), 0);
            if (ret < 0) {
                fprintf(stderr, "cuFileRead failed at op %u: %zd\n", i, ret);
                continue;
            }
        } else
#endif
        {
            double us = posix_fallback_read(fd, d_buf, h_buf, offsets[i],
                                             cfg.block_size);
            if (us < 0) continue;
            op_timer.stop();
            latencies_us.push_back(op_timer.elapsed_us());
            continue;
        }

        op_timer.stop();
        latencies_us.push_back(op_timer.elapsed_us());
    }

    wall.stop();
    CpuSample cpu_after = read_cpu_sample();

    double elapsed_sec = wall.elapsed_sec();
    uint64_t total_bytes = static_cast<uint64_t>(latencies_us.size()) * cfg.block_size;

    printf("[%s] Completed: %zu / %u ops (wall: %.3f s)\n",
           METHOD, latencies_us.size(), cfg.num_ops, elapsed_sec);

    /* ------- Compute and Output Stats ------- */

    BenchStats stats = compute_stats(latencies_us, total_bytes, elapsed_sec,
                                      cpu_before, cpu_after);

    print_stats(METHOD, cfg.block_size, cfg.queue_depth,
                cfg.pattern.c_str(), stats);

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
            write_csv_row(fp,
                          g_cufile_available ? "cufile" : "cufile_posix_fallback",
                          cfg.block_size, cfg.queue_depth,
                          cfg.pattern.c_str(), stats);
            fclose(fp);
            printf("[%s] Results appended to %s\n", METHOD, cfg.output.c_str());
        } else {
            fprintf(stderr, "Warning: could not open %s for writing\n",
                    cfg.output.c_str());
        }
    }

    /* ------- Cleanup ------- */

#ifdef HAVE_CUFILE
    if (g_cufile_available) {
        cuFileBufDeregister(d_buf);
        cuFileHandleDeregister(cf_handle);
        cuFileDriverClose();
    }
#endif

    CUDA_CHECK(cudaFree(d_buf));
    free(h_buf);
    close(fd);

    return EXIT_SUCCESS;
}
