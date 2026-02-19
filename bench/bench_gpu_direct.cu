/*
 * gpu-nvme-direct: GPU-Direct NVMe I/O Benchmark
 *
 * Benchmarks GPU-initiated NVMe I/O where the GPU kernel directly
 * submits READ commands to the NVMe controller, rings doorbells,
 * and polls completions -- with zero CPU involvement in the I/O path.
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
#include <gpunvme/block_io.h>
#include <gpunvme/nvme_regs.h>
#include <gpunvme/dma.h>

#include "bench_common.h"

/* For simulator mode */
#if GPUNVME_USE_SIM
#include "sim/nvme_sim.h"
#endif

#include "device/queue_state.cuh"

/* ------- Benchmark Kernel ------- */

/*
 * Per-operation latency measurement kernel.
 *
 * Each thread handles one I/O operation. For queue_depth > 1,
 * multiple threads submit concurrently. A GPU clock-based timer
 * measures per-operation latency.
 */
__global__
void bench_read_kernel(gpu_nvme_queue *q,
                       uint64_t *lba_array,
                       uint32_t blocks_per_op,
                       uint64_t data_buf_phys,
                       uint64_t data_buf_stride,
                       float *latencies_ms,
                       uint32_t num_ops,
                       uint32_t *completed_ops) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_ops) return;

    uint64_t slba = lba_array[tid];
    uint64_t my_phys = data_buf_phys + tid * data_buf_stride;

    /* Clock-based per-operation timing */
    clock_t t0 = clock();

    /* Build and submit READ command.
     * We use a simplified inline path for benchmarking. */
    volatile nvme_sq_entry_t *sqe = &q->sq[q->sq_tail % q->sq_size];

    /* Zero the entry */
    volatile uint32_t *sqe32 = (volatile uint32_t *)sqe;
    for (int w = 0; w < 16; w++) sqe32[w] = 0;

    /* Build READ command inline */
    uint16_t cid = (uint16_t)(atomicAdd((unsigned int *)&q->cid_counter, 1) & 0xFFFF);

    sqe32[0] = 0x02 | (((uint32_t)cid) << 16);  /* OPC=READ, CID */
    sqe32[1] = q->nsid;                           /* NSID */
    /* PRP1 (64-bit) at dwords 6-7 */
    sqe32[6] = (uint32_t)(my_phys & 0xFFFFFFFF);
    sqe32[7] = (uint32_t)(my_phys >> 32);
    /* CDW10-11: starting LBA */
    sqe32[10] = (uint32_t)(slba & 0xFFFFFFFF);
    sqe32[11] = (uint32_t)(slba >> 32);
    /* CDW12: NLB (0-based) */
    sqe32[12] = blocks_per_op - 1;

    /* Advance SQ tail and ring doorbell */
    __threadfence_system();
    uint16_t new_tail = (uint16_t)atomicAdd((unsigned int *)&q->sq_tail, 1);
    new_tail = (new_tail + 1) % q->sq_size;
    __threadfence_system();
    *(q->doorbell_sq) = new_tail;
    __threadfence_system();

    /* Poll CQ for our CID */
    uint64_t timeout_cycles = q->poll_timeout_cycles;
    if (timeout_cycles == 0) timeout_cycles = 170000000ULL;

    clock_t poll_start = clock();
    bool found = false;

    while (!found) {
        clock_t now = clock();
        if ((uint64_t)(now - poll_start) > timeout_cycles) {
            /* Timeout */
            break;
        }

        volatile nvme_cq_entry_t *cqe = &q->cq[q->cq_head % q->cq_size];
        uint16_t sp = cqe->status_phase;
        uint8_t phase = sp & 1;

        if (phase == q->cq_phase) {
            if (cqe->cid == cid) {
                found = true;

                /* Advance CQ head */
                uint16_t new_head = (q->cq_head + 1) % q->cq_size;
                q->cq_head = new_head;
                if (new_head == 0) q->cq_phase ^= 1;

                __threadfence_system();
                *(q->doorbell_cq) = new_head;
                __threadfence_system();
            }
        }
    }

    clock_t t1 = clock();

    /* Convert GPU clock cycles to milliseconds.
     * Note: clock() returns SM clock cycles. We store raw delta;
     * the host converts to real time using the GPU clock rate. */
    latencies_ms[tid] = (float)(t1 - t0);

    if (found) {
        atomicAdd(completed_ops, 1);
    }
}

/* ------- Main ------- */

int main(int argc, char **argv) {
    const char *METHOD = "gpu_direct";
    BenchConfig cfg = parse_args(argc, argv, METHOD);

    printf("[%s] block_size=%s queue_depth=%u num_ops=%u pattern=%s\n",
           METHOD, format_block_size(cfg.block_size).c_str(),
           cfg.queue_depth, cfg.num_ops, cfg.pattern.c_str());

    /* Query GPU clock rate for converting clock() ticks to time */
    int device_id = 0;
    CUDA_CHECK(cudaSetDevice(device_id));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    int clock_rate_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, device_id));
    double gpu_clock_khz = static_cast<double>(clock_rate_khz);  /* kHz */
    printf("[%s] GPU: %s, clock: %.0f MHz\n", METHOD, prop.name,
           gpu_clock_khz / 1000.0);

    /* ------- Controller and Queue Setup ------- */

#if GPUNVME_USE_SIM
    printf("[%s] Using NVMe simulator\n", METHOD);

    uint32_t blocks_per_op = static_cast<uint32_t>(cfg.block_size / 512);
    if (blocks_per_op == 0) blocks_per_op = 1;

    /* Create simulator with enough capacity */
    nvme_sim_config_t sim_cfg = {};
    sim_cfg.num_blocks = 1024 * 1024;  /* 512 MB virtual device */
    sim_cfg.block_size = 512;
    sim_cfg.sq_size = 256;
    sim_cfg.cq_size = 256;
    sim_cfg.latency_us = 10;  /* 10 us simulated latency */

    nvme_sim_t *sim = nvme_sim_create(&sim_cfg);
    if (!sim) {
        fprintf(stderr, "Failed to create NVMe simulator\n");
        return EXIT_FAILURE;
    }

    /* Set up GPU queue state */
    gpu_nvme_queue *h_queue;
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
    h_queue->poll_timeout_cycles = 170000000ULL;

    uint64_t data_buf_phys = nvme_sim_get_data_buf_phys(sim);
    uint64_t max_lba = sim_cfg.num_blocks;

#else
    /* Real hardware path */
    printf("[%s] Using real NVMe hardware: %s\n", METHOD, cfg.device.c_str());

    /* Map BAR0 -- requires gpunvme_host library and root privileges */
    /* This is a placeholder for the real hardware init path.
     * In production, this would:
     *   1. Open VFIO device
     *   2. Map BAR0 for CPU
     *   3. Map BAR0 for GPU (via CUDA external memory or P2P)
     *   4. Init controller
     *   5. Create I/O queue
     */
    fprintf(stderr, "Error: real hardware mode requires bare-metal setup.\n"
                    "Build with -DGPUNVME_USE_SIM=ON for simulator mode.\n");
    return EXIT_FAILURE;
#endif

    /* ------- Generate LBA Array ------- */

    uint64_t data_stride = cfg.block_size;

    if (cfg.max_lba == 0) {
        cfg.max_lba = max_lba;
    }

    /* Ensure we don't exceed device capacity */
    uint64_t lba_range = cfg.max_lba - blocks_per_op;
    if (lba_range == 0) lba_range = 1;

    uint64_t *h_lba_array;
    CUDA_CHECK(cudaMallocHost(&h_lba_array, cfg.num_ops * sizeof(uint64_t)));

    srand(42);  /* Deterministic seed */

    if (cfg.pattern == "seq") {
        for (uint32_t i = 0; i < cfg.num_ops; i++) {
            h_lba_array[i] = (cfg.start_lba + (uint64_t)i * blocks_per_op) % cfg.max_lba;
        }
    } else {
        for (uint32_t i = 0; i < cfg.num_ops; i++) {
            h_lba_array[i] = random_lba(lba_range);
        }
    }

    /* Copy LBA array to device */
    uint64_t *d_lba_array;
    CUDA_CHECK(cudaMalloc(&d_lba_array, cfg.num_ops * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_lba_array, h_lba_array,
                           cfg.num_ops * sizeof(uint64_t),
                           cudaMemcpyHostToDevice));

    /* Allocate latency array and completion counter */
    float *d_latencies;
    CUDA_CHECK(cudaMalloc(&d_latencies, cfg.num_ops * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_latencies, 0, cfg.num_ops * sizeof(float)));

    uint32_t *d_completed;
    CUDA_CHECK(cudaMalloc(&d_completed, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_completed, 0, sizeof(uint32_t)));

    /* ------- Warmup ------- */

    printf("[%s] Running %d warmup iterations...\n", METHOD, WARMUP_ITERATIONS);

    for (int w = 0; w < WARMUP_ITERATIONS; w++) {
        CUDA_CHECK(cudaMemset(d_completed, 0, sizeof(uint32_t)));

        /* Launch with 1 thread for warmup (single op) */
        bench_read_kernel<<<1, 1>>>(h_queue,
                                     d_lba_array,
                                     blocks_per_op,
                                     data_buf_phys,
                                     data_stride,
                                     d_latencies,
                                     1,
                                     d_completed);
        CUDA_CHECK(cudaDeviceSynchronize());

        /* Reset queue state for next warmup */
        h_queue->sq_tail = 0;
        h_queue->cq_head = 0;
        h_queue->cq_phase = 1;
        h_queue->cid_counter = 0;
    }

    /* ------- Benchmark ------- */

    printf("[%s] Running %u operations...\n", METHOD, cfg.num_ops);

    /* Reset state */
    h_queue->sq_tail = 0;
    h_queue->cq_head = 0;
    h_queue->cq_phase = 1;
    h_queue->cid_counter = 0;
    CUDA_CHECK(cudaMemset(d_completed, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_latencies, 0, cfg.num_ops * sizeof(float)));

    /* Determine launch geometry.
     * For qd=1: launch ops sequentially (1 thread at a time).
     * For qd>1: launch batches of qd threads concurrently. */
    CpuSample cpu_before = read_cpu_sample();
    WallTimer wall;
    GpuTimer gpu_timer;

    wall.start();
    gpu_timer.start();

    if (cfg.queue_depth == 1) {
        /* Serial: one op at a time for accurate per-op latency */
        for (uint32_t op = 0; op < cfg.num_ops; op++) {
            bench_read_kernel<<<1, 1>>>(h_queue,
                                         d_lba_array + op,
                                         blocks_per_op,
                                         data_buf_phys,
                                         data_stride,
                                         d_latencies + op,
                                         1,
                                         d_completed);
            CUDA_CHECK(cudaDeviceSynchronize());

            /* Reset queue head/tail for next operation in serial mode */
            h_queue->sq_tail = 0;
            h_queue->cq_head = 0;
            h_queue->cq_phase = 1;
        }
    } else {
        /* Batched: launch queue_depth threads per batch */
        uint32_t ops_done = 0;
        while (ops_done < cfg.num_ops) {
            uint32_t batch = cfg.queue_depth;
            if (ops_done + batch > cfg.num_ops) {
                batch = cfg.num_ops - ops_done;
            }

            /* Reset queue for each batch */
            h_queue->sq_tail = 0;
            h_queue->cq_head = 0;
            h_queue->cq_phase = 1;

            bench_read_kernel<<<1, batch>>>(h_queue,
                                              d_lba_array + ops_done,
                                              blocks_per_op,
                                              data_buf_phys,
                                              data_stride,
                                              d_latencies + ops_done,
                                              batch,
                                              d_completed);
            CUDA_CHECK(cudaDeviceSynchronize());

            ops_done += batch;
        }
    }

    gpu_timer.stop();
    wall.stop();
    CpuSample cpu_after = read_cpu_sample();

    double gpu_elapsed_sec = gpu_timer.elapsed_sec();
    double wall_elapsed_sec = wall.elapsed_sec();

    /* Read back completions */
    uint32_t h_completed = 0;
    CUDA_CHECK(cudaMemcpy(&h_completed, d_completed, sizeof(uint32_t),
                           cudaMemcpyDeviceToHost));

    printf("[%s] Completed: %u / %u ops (GPU: %.3f s, wall: %.3f s)\n",
           METHOD, h_completed, cfg.num_ops, gpu_elapsed_sec, wall_elapsed_sec);

    /* Read back per-op latencies and convert from clock ticks to microseconds */
    std::vector<float> h_latencies_raw(cfg.num_ops);
    CUDA_CHECK(cudaMemcpy(h_latencies_raw.data(), d_latencies,
                           cfg.num_ops * sizeof(float),
                           cudaMemcpyDeviceToHost));

    std::vector<double> latencies_us;
    latencies_us.reserve(h_completed);

    for (uint32_t i = 0; i < cfg.num_ops; i++) {
        if (h_latencies_raw[i] > 0.0f) {
            /* Convert GPU clock ticks to microseconds:
             * ticks / (clock_rate_kHz * 1000) * 1e6 = ticks / clock_rate_kHz * 1000 */
            double us = static_cast<double>(h_latencies_raw[i]) / gpu_clock_khz * 1000.0;
            latencies_us.push_back(us);
        }
    }

    /* ------- Compute and Output Stats ------- */

    uint64_t total_bytes = static_cast<uint64_t>(h_completed) * cfg.block_size;
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

    CUDA_CHECK(cudaFree(d_lba_array));
    CUDA_CHECK(cudaFree(d_latencies));
    CUDA_CHECK(cudaFree(d_completed));
    CUDA_CHECK(cudaFreeHost(h_lba_array));
    CUDA_CHECK(cudaFreeHost(h_queue));

#if GPUNVME_USE_SIM
    nvme_sim_destroy(sim);
#endif

    return EXIT_SUCCESS;
}
