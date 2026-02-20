/*
 * gpu-nvme-direct: GPU-Direct NVMe I/O Benchmark
 *
 * Benchmarks GPU-initiated NVMe I/O where a single GPU thread submits
 * READ commands with pipelining, rings doorbells via MMIO, and polls
 * completions -- with zero CPU involvement in the I/O path.
 *
 * Uses the proven sq_submit_read() + cq_poll_completion() path from
 * test_large_read.cu.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include <gpunvme/error.h>
#include <gpunvme/controller.h>
#include <gpunvme/queue.h>
#include <gpunvme/nvme_regs.h>
#include <gpunvme/dma.h>

#include "bench_common.h"

#if GPUNVME_USE_SIM
#include "sim/nvme_sim.h"
#endif

#include "device/queue_state.cuh"
#include "device/mmio_ops.cuh"
#include "device/sq_submit.cuh"
#include "device/cq_poll.cuh"

#if !GPUNVME_USE_SIM
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

/* ------- Kernel Parameters ------- */

struct bench_params {
    uint64_t *lba_array;        /* Per-op starting LBA (num_ops entries) */
    uint64_t *prp1_array;       /* Per-buffer-slot PRP1 (pipeline_depth entries) */
    uint64_t *prp2_array;       /* Per-buffer-slot PRP2 (pipeline_depth entries) */
    uint32_t nlb_0based;        /* NVMe blocks per op minus 1 */
    uint32_t num_ops;           /* Total operations to run */
    uint32_t pipeline_depth;    /* Max in-flight commands */
};

struct bench_result {
    uint32_t completed;
    uint32_t error_code;        /* 0=ok, 2=timeout, 3=nvme_error */
    uint16_t cqe_status;
    uint64_t total_cycles;
};

/* ------- GPU Benchmark Kernel ------- */

/*
 * Single-thread pipelined benchmark kernel.
 *
 * Submits up to pipeline_depth commands ahead, polls completions in
 * order. Records per-completion poll latency in GPU clock cycles.
 * Buffer slots are reused cyclically: op N uses slot (N % pipeline_depth).
 */
__global__
void bench_read_pipelined(gpu_nvme_queue *q,
                          bench_params *params,
                          bench_result *result,
                          uint64_t *latency_cycles) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    result->completed = 0;
    result->error_code = 0;
    result->cqe_status = 0;

    uint32_t num_ops = params->num_ops;
    uint32_t pipe = params->pipeline_depth;
    uint32_t nlb = params->nlb_0based;

    uint32_t submitted = 0;
    uint32_t completed = 0;

    uint64_t t_start = clock64();

    while (completed < num_ops) {
        /* Fill pipeline */
        while (submitted < num_ops && (submitted - completed) < pipe) {
            uint32_t idx = submitted;
            uint32_t buf_slot = idx % pipe;

            sq_submit_read(q,
                params->lba_array[idx],
                nlb,
                params->prp1_array[buf_slot],
                params->prp2_array[buf_slot]);
            submitted++;
        }

        /* Poll for next completion */
        uint64_t poll_start = clock64();
        cq_poll_result cqr = cq_poll_completion(q, 3400000000ULL);
        uint64_t poll_end = clock64();

        if (cqr.timed_out) {
            result->completed = completed;
            result->error_code = 2;
            result->total_cycles = clock64() - t_start;
            return;
        }
        if (!cqr.success) {
            result->completed = completed;
            result->error_code = 3;
            result->cqe_status = cqr.status;
            result->total_cycles = clock64() - t_start;
            return;
        }

        latency_cycles[completed] = poll_end - poll_start;
        completed++;
    }

    result->completed = completed;
    result->total_cycles = clock64() - t_start;
}

/* ------- Helper: resolve physical address via pagemap ------- */

#if !GPUNVME_USE_SIM
static uint64_t virt_to_phys_pagemap(int pm_fd, void *vaddr) {
    long page_size = sysconf(_SC_PAGESIZE);
    uint64_t va = (uint64_t)(uintptr_t)vaddr;
    uint64_t page_idx = va / page_size;
    uint64_t entry;
    if (pread(pm_fd, &entry, 8, page_idx * 8) != 8) return 0;
    if (!(entry & (1ULL << 63))) return 0;  /* page not present */
    uint64_t pfn = entry & ((1ULL << 55) - 1);
    return pfn * page_size + (va % page_size);
}
#endif

/* ------- Main ------- */

int main(int argc, char **argv) {
    const char *METHOD = "gpu_direct";
    BenchConfig cfg = parse_args(argc, argv, METHOD);

    printf("[%s] block_size=%s queue_depth=%u num_ops=%u pattern=%s\n",
           METHOD, format_block_size(cfg.block_size).c_str(),
           cfg.queue_depth, cfg.num_ops, cfg.pattern.c_str());

    /* Query GPU clock rate */
    int device_id = 0;
    CUDA_CHECK(cudaSetDevice(device_id));
    int clock_rate_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, device_id));
    double gpu_clock_khz = static_cast<double>(clock_rate_khz);
    printf("[%s] GPU clock: %.0f MHz\n", METHOD, gpu_clock_khz / 1000.0);

    /* Shared variables across both paths */
    uint32_t pipeline_depth = cfg.queue_depth;
    gpu_nvme_queue *h_queue = nullptr;
    uint32_t nvme_block_size = 512;
    uint64_t max_lba = 0;
    uint32_t max_transfer_bytes = 0;
    int ret = EXIT_SUCCESS;

    /* Per-slot PRP arrays */
    uint64_t *prp1_arr = nullptr;
    uint64_t *prp2_arr = nullptr;

    /* HW-only resources */
    void *data_buf = nullptr;
    size_t data_buf_size = 0;
    void *prp_pool = nullptr;
    size_t prp_pool_bytes = 0;

#if GPUNVME_USE_SIM
    /* ===== Simulator path ===== */
    printf("[%s] Using NVMe simulator\n", METHOD);

    nvme_sim_config_t sim_cfg = {};
    sim_cfg.num_blocks = 1024 * 1024;  /* 512 MB virtual device */
    sim_cfg.block_size = 512;
    sim_cfg.sq_size = 256;
    sim_cfg.cq_size = 256;
    sim_cfg.latency_us = 10;

    nvme_sim_t *sim = nvme_sim_create(&sim_cfg);
    if (!sim) {
        fprintf(stderr, "Failed to create NVMe simulator\n");
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaMallocHost(&h_queue, sizeof(gpu_nvme_queue)));
    memset(h_queue, 0, sizeof(gpu_nvme_queue));

    h_queue->sq = nvme_sim_get_sq(sim);
    h_queue->cq = nvme_sim_get_cq(sim);
    h_queue->doorbell_sq = nvme_sim_get_sq_doorbell(sim);
    h_queue->doorbell_cq = nvme_sim_get_cq_doorbell(sim);
    h_queue->data_buf = nvme_sim_get_data_buf(sim);
    h_queue->data_buf_phys = nvme_sim_get_data_buf_phys(sim);
    h_queue->sq_size = nvme_sim_get_sq_size(sim);
    h_queue->cq_size = nvme_sim_get_cq_size(sim);
    h_queue->sq_tail = 0;
    h_queue->cq_head = 0;
    h_queue->qid = 1;
    h_queue->cq_phase = 1;
    h_queue->cid_counter = 0;
    h_queue->nsid = 1;
    h_queue->block_size = sim_cfg.block_size;
    h_queue->pcie_flush_addr = nullptr;
    h_queue->poll_timeout_cycles = 170000000ULL;

    nvme_block_size = sim_cfg.block_size;
    max_lba = sim_cfg.num_blocks;
    max_transfer_bytes = 1024 * 1024;

    /* Sim PRP: direct virtual addresses, single page only */
    if (cfg.block_size > 4096) {
        printf("[%s] WARNING: sim mode limited to 4K block size, clamping\n", METHOD);
        cfg.block_size = 4096;
    }

    uint64_t data_base_phys = nvme_sim_get_data_buf_phys(sim);
    CUDA_CHECK(cudaMallocHost(&prp1_arr, pipeline_depth * sizeof(uint64_t)));
    CUDA_CHECK(cudaMallocHost(&prp2_arr, pipeline_depth * sizeof(uint64_t)));
    for (uint32_t i = 0; i < pipeline_depth; i++) {
        prp1_arr[i] = data_base_phys + (uint64_t)i * cfg.block_size;
        prp2_arr[i] = 0;
    }

#else
    /* ===== Real hardware path ===== */
    printf("[%s] Using real NVMe hardware: %s\n", METHOD, cfg.device.c_str());

    /* Map BAR0 */
    char bar_path[256];
    snprintf(bar_path, sizeof(bar_path),
             "/sys/bus/pci/devices/%s/resource0", cfg.device.c_str());

    int bar_fd = open(bar_path, O_RDWR | O_SYNC);
    if (bar_fd < 0) {
        perror("open BAR0");
        fprintf(stderr, "Hint: run with sudo, ensure VFIO setup for %s\n",
                cfg.device.c_str());
        return EXIT_FAILURE;
    }

    off_t bar_size = lseek(bar_fd, 0, SEEK_END);
    volatile void *bar0 = mmap(NULL, bar_size, PROT_READ | PROT_WRITE,
                                MAP_SHARED, bar_fd, 0);
    if (bar0 == MAP_FAILED) {
        perror("mmap BAR0");
        close(bar_fd);
        return EXIT_FAILURE;
    }

    cudaError_t cerr = cudaHostRegister(
        (void *)bar0, bar_size,
        cudaHostRegisterIoMemory | cudaHostRegisterMapped);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "cudaHostRegisterIoMemory failed: %s\n",
                cudaGetErrorString(cerr));
        munmap((void *)bar0, bar_size);
        close(bar_fd);
        return EXIT_FAILURE;
    }

    void *gpu_bar0;
    cudaHostGetDevicePointer(&gpu_bar0, (void *)bar0, 0);

    /* Init controller */
    gpunvme_ctrl_t ctrl;
    gpunvme_err_t err = gpunvme_ctrl_init(&ctrl, bar0, bar_size);
    if (err != GPUNVME_OK) {
        fprintf(stderr, "Controller init failed: %s\n", gpunvme_err_str(err));
        cudaHostUnregister((void *)bar0);
        munmap((void *)bar0, bar_size);
        close(bar_fd);
        return EXIT_FAILURE;
    }
    ctrl.bar0_gpu = gpu_bar0;

    nvme_block_size = ctrl.block_size;
    max_lba = ctrl.ns_size_blocks;
    max_transfer_bytes = ctrl.max_transfer_bytes;

    printf("[%s] NVMe: %s, block_size=%u, MDTS=%uK, capacity=%u blocks\n",
           METHOD, ctrl.model, nvme_block_size,
           max_transfer_bytes / 1024, (uint32_t)max_lba);

    /* Clamp block_size to MDTS */
    if (cfg.block_size > max_transfer_bytes) {
        printf("[%s] WARNING: block_size %s exceeds MDTS %uK, clamping\n",
               METHOD, format_block_size(cfg.block_size).c_str(),
               max_transfer_bytes / 1024);
        cfg.block_size = max_transfer_bytes;
    }

    /* Create I/O queue with headroom beyond pipeline depth */
    uint16_t queue_size = (uint16_t)(pipeline_depth + 4);
    if (queue_size < 16) queue_size = 16;

    gpunvme_io_queue_t ioq;
    err = gpunvme_create_io_queue(&ctrl, 1, queue_size, 4096, GPUNVME_TIER1, &ioq);
    if (err != GPUNVME_OK) {
        fprintf(stderr, "I/O queue creation failed: %s\n", gpunvme_err_str(err));
        gpunvme_ctrl_shutdown(&ctrl);
        cudaHostUnregister((void *)bar0);
        munmap((void *)bar0, bar_size);
        close(bar_fd);
        return EXIT_FAILURE;
    }
    h_queue = ioq.gpu_queue;

    /* Allocate data buffer: pipeline_depth slots * block_size */
    data_buf_size = (size_t)pipeline_depth * cfg.block_size;
    if (posix_memalign(&data_buf, 4096, data_buf_size) != 0) {
        fprintf(stderr, "Data buffer allocation failed (%zu bytes)\n", data_buf_size);
        gpunvme_delete_io_queue(&ctrl, &ioq);
        gpunvme_ctrl_shutdown(&ctrl);
        cudaHostUnregister((void *)bar0);
        munmap((void *)bar0, bar_size);
        close(bar_fd);
        return EXIT_FAILURE;
    }
    mlock(data_buf, data_buf_size);
    cudaHostRegister(data_buf, data_buf_size, cudaHostRegisterDefault);
    memset(data_buf, 0xDE, data_buf_size);

    /* Build PRP lists for each buffer slot */
    CUDA_CHECK(cudaMallocHost(&prp1_arr, pipeline_depth * sizeof(uint64_t)));
    CUDA_CHECK(cudaMallocHost(&prp2_arr, pipeline_depth * sizeof(uint64_t)));
    memset(prp1_arr, 0, pipeline_depth * sizeof(uint64_t));
    memset(prp2_arr, 0, pipeline_depth * sizeof(uint64_t));

    {
        uint32_t pages_per_op = (cfg.block_size + ctrl.page_size - 1) / ctrl.page_size;

        /* Allocate PRP list pool for multi-page transfers */
        if (pages_per_op > 2) {
            prp_pool_bytes = (size_t)pipeline_depth * 4096;
            if (posix_memalign(&prp_pool, 4096, prp_pool_bytes) != 0) {
                fprintf(stderr, "PRP pool allocation failed\n");
                goto cleanup_hw;
            }
            mlock(prp_pool, prp_pool_bytes);
            cudaHostRegister(prp_pool, prp_pool_bytes, cudaHostRegisterDefault);
            memset(prp_pool, 0, prp_pool_bytes);
        }

        int pm_fd = open("/proc/self/pagemap", O_RDONLY);
        if (pm_fd < 0) {
            perror("pagemap");
            fprintf(stderr, "Hint: run with sudo for /proc/self/pagemap access\n");
            goto cleanup_hw;
        }

        for (uint32_t slot = 0; slot < pipeline_depth; slot++) {
            uint8_t *chunk = (uint8_t *)data_buf + (size_t)slot * cfg.block_size;

            for (uint32_t p = 0; p < pages_per_op; p++) {
                uint64_t phys = virt_to_phys_pagemap(pm_fd,
                    chunk + (size_t)p * ctrl.page_size);
                if (phys == 0) {
                    fprintf(stderr, "Failed to resolve phys addr for slot %u page %u\n",
                            slot, p);
                    close(pm_fd);
                    goto cleanup_hw;
                }

                if (p == 0) {
                    prp1_arr[slot] = phys;
                } else if (pages_per_op == 2) {
                    /* 2-page transfer: PRP2 = second page phys directly */
                    prp2_arr[slot] = phys;
                } else {
                    /* Multi-page: write into PRP list */
                    uint64_t *list = (uint64_t *)((uint8_t *)prp_pool
                                     + (size_t)slot * 4096);
                    list[p - 1] = phys;
                }
            }

            if (pages_per_op <= 1) {
                prp2_arr[slot] = 0;
            } else if (pages_per_op > 2) {
                /* PRP2 = physical address of this slot's PRP list page */
                uint64_t *list = (uint64_t *)((uint8_t *)prp_pool
                                 + (size_t)slot * 4096);
                prp2_arr[slot] = virt_to_phys_pagemap(pm_fd, list);
                if (prp2_arr[slot] == 0) {
                    fprintf(stderr, "Failed to resolve PRP list phys for slot %u\n", slot);
                    close(pm_fd);
                    goto cleanup_hw;
                }
            }
        }
        close(pm_fd);

        printf("[%s] PRP lists ready: %u buffer slots, %u pages/op\n",
               METHOD, pipeline_depth, pages_per_op);
    }

#endif /* GPUNVME_USE_SIM */

    /* ------- Generate LBA Array ------- */
    {
        uint32_t blocks_per_op = static_cast<uint32_t>(cfg.block_size / nvme_block_size);
        if (blocks_per_op == 0) blocks_per_op = 1;

        if (cfg.max_lba == 0) cfg.max_lba = max_lba;

        uint64_t lba_range = cfg.max_lba;
        if (lba_range > blocks_per_op) lba_range -= blocks_per_op;
        if (lba_range == 0) lba_range = 1;

        uint64_t *h_lba_array;
        CUDA_CHECK(cudaMallocHost(&h_lba_array, cfg.num_ops * sizeof(uint64_t)));

        srand(42);
        if (cfg.pattern == "seq") {
            for (uint32_t i = 0; i < cfg.num_ops; i++) {
                h_lba_array[i] = (cfg.start_lba + (uint64_t)i * blocks_per_op)
                                 % cfg.max_lba;
            }
        } else {
            for (uint32_t i = 0; i < cfg.num_ops; i++) {
                h_lba_array[i] = random_lba(lba_range);
            }
        }

        /* Allocate kernel params and result */
        bench_params *h_params;
        bench_result *h_result;
        CUDA_CHECK(cudaMallocHost(&h_params, sizeof(bench_params)));
        CUDA_CHECK(cudaMallocHost(&h_result, sizeof(bench_result)));

        h_params->lba_array = h_lba_array;
        h_params->prp1_array = prp1_arr;
        h_params->prp2_array = prp2_arr;
        h_params->nlb_0based = blocks_per_op - 1;
        h_params->num_ops = cfg.num_ops;
        h_params->pipeline_depth = pipeline_depth;

        /* Per-completion latency storage */
        uint64_t *h_latency_cycles;
        CUDA_CHECK(cudaMallocHost(&h_latency_cycles, cfg.num_ops * sizeof(uint64_t)));
        memset(h_latency_cycles, 0, cfg.num_ops * sizeof(uint64_t));

        /* ------- Warmup ------- */

        printf("[%s] Running %d warmup ops...\n", METHOD, WARMUP_ITERATIONS);

        bench_params *h_warmup;
        CUDA_CHECK(cudaMallocHost(&h_warmup, sizeof(bench_params)));
        *h_warmup = *h_params;
        h_warmup->num_ops = 1;
        h_warmup->pipeline_depth = 1;

        bool warmup_ok = true;
        for (int w = 0; w < WARMUP_ITERATIONS; w++) {
            memset(h_result, 0, sizeof(bench_result));
            bench_read_pipelined<<<1, 1>>>(h_queue, h_warmup, h_result, h_latency_cycles);
            CUDA_CHECK(cudaDeviceSynchronize());

            if (h_result->error_code != 0) {
                fprintf(stderr, "[%s] Warmup op %d failed: error=%u, status=0x%04x\n",
                        METHOD, w, h_result->error_code, h_result->cqe_status);
                ret = EXIT_FAILURE;
                warmup_ok = false;
                break;
            }
        }

        if (warmup_ok) {
            printf("[%s] Warmup complete (queue at sq_tail=%u, cq_head=%u)\n",
                   METHOD, h_queue->sq_tail, h_queue->cq_head);

            /* ------- Benchmark ------- */

            printf("[%s] Running %u operations (pipeline_depth=%u)...\n",
                   METHOD, cfg.num_ops, pipeline_depth);

            /* Do NOT reset queue state — the NVMe controller's internal pointers
             * are at the position left by warmup. Resetting our head/tail/phase
             * without resetting the controller causes completions to go to the
             * wrong CQ slot (timeout). Let the queue roll naturally. */

            memset(h_result, 0, sizeof(bench_result));
            memset(h_latency_cycles, 0, cfg.num_ops * sizeof(uint64_t));
            h_params->num_ops = cfg.num_ops;
            h_params->pipeline_depth = pipeline_depth;

            CpuSample cpu_before = read_cpu_sample();
            WallTimer wall;
            GpuTimer gpu_timer;

            wall.start();
            gpu_timer.start();

            bench_read_pipelined<<<1, 1>>>(h_queue, h_params, h_result, h_latency_cycles);
            CUDA_CHECK(cudaDeviceSynchronize());

            gpu_timer.stop();
            wall.stop();
            CpuSample cpu_after = read_cpu_sample();

            double wall_elapsed_sec = wall.elapsed_sec();

            printf("[%s] Completed: %u / %u ops (wall: %.3f s)\n",
                   METHOD, h_result->completed, cfg.num_ops, wall_elapsed_sec);

            if (h_result->error_code != 0) {
                fprintf(stderr, "[%s] ERROR: code=%u, cqe_status=0x%04x at op %u\n",
                        METHOD, h_result->error_code, h_result->cqe_status,
                        h_result->completed);
                ret = EXIT_FAILURE;
            }

            /* Convert per-completion latencies from GPU cycles to microseconds */
            std::vector<double> latencies_us;
            latencies_us.reserve(h_result->completed);
            for (uint32_t i = 0; i < h_result->completed; i++) {
                double us = static_cast<double>(h_latency_cycles[i])
                            / gpu_clock_khz * 1000.0;
                latencies_us.push_back(us);
            }

            /* ------- Compute and Output Stats ------- */

            uint64_t total_bytes = static_cast<uint64_t>(h_result->completed) * cfg.block_size;
            BenchStats stats = compute_stats(latencies_us, total_bytes, wall_elapsed_sec,
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
                    if (write_header) write_csv_header(fp);
                    write_csv_row(fp, METHOD, cfg.block_size, cfg.queue_depth,
                                  cfg.pattern.c_str(), stats);
                    fclose(fp);
                    printf("[%s] Results appended to %s\n", METHOD, cfg.output.c_str());
                } else {
                    fprintf(stderr, "Warning: could not open %s for writing\n",
                            cfg.output.c_str());
                }
            }
        }

        CUDA_CHECK(cudaFreeHost(h_warmup));
        CUDA_CHECK(cudaFreeHost(h_lba_array));
        CUDA_CHECK(cudaFreeHost(h_params));
        CUDA_CHECK(cudaFreeHost(h_result));
        CUDA_CHECK(cudaFreeHost(h_latency_cycles));
    }

    /* ------- Resource Cleanup ------- */

    CUDA_CHECK(cudaFreeHost(prp1_arr));
    CUDA_CHECK(cudaFreeHost(prp2_arr));

#if GPUNVME_USE_SIM
    CUDA_CHECK(cudaFreeHost(h_queue));
    nvme_sim_destroy(sim);
#else
    goto cleanup_hw_done;

cleanup_hw:
    ret = EXIT_FAILURE;

cleanup_hw_done:
    if (prp_pool) {
        cudaHostUnregister(prp_pool);
        munlock(prp_pool, prp_pool_bytes);
        free(prp_pool);
    }
    if (data_buf) {
        cudaHostUnregister(data_buf);
        munlock(data_buf, data_buf_size);
        free(data_buf);
    }
    gpunvme_delete_io_queue(&ctrl, &ioq);
    gpunvme_ctrl_shutdown(&ctrl);
    cudaHostUnregister((void *)bar0);
    munmap((void *)bar0, bar_size);
    close(bar_fd);
#endif

    return ret;
}
