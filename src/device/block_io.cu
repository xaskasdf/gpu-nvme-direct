/*
 * gpu-nvme-direct: GPU-Side Block I/O Kernels
 *
 * High-level CUDA kernels for reading/writing NVMe blocks.
 * These kernels operate autonomously: build commands, ring doorbells,
 * poll completions — no CPU involvement in the I/O path.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <cstdio>
#include <cstdint>
#include <gpunvme/nvme_regs.h>
#include "queue_state.cuh"
#include "sq_submit.cuh"
#include "cq_poll.cuh"

/* Default poll timeout: ~100ms at 1.7 GHz GPU clock */
#define DEFAULT_TIMEOUT_CYCLES (170000000ULL)

/*
 * Result structure written by GPU kernel, read by host.
 * Placed in host pinned memory for visibility.
 */
struct gpunvme_io_result {
    uint32_t status;       /* 0 = success, nonzero = NVMe status or error */
    uint32_t blocks_done;  /* Number of blocks successfully read/written */
    uint32_t error_code;   /* Internal error code if status != 0 */
    uint32_t cqe_status;   /* Raw CQE status field on error */
};

/*
 * GPU kernel: read a contiguous range of blocks.
 *
 * Launched with 1 thread (single-threaded I/O for simplicity in Phase 0).
 * Future phases will use multiple threads for queue-depth parallelism.
 *
 * q:          GPU-visible queue state (in pinned or device memory)
 * slba:       Starting LBA
 * num_blocks: Number of blocks to read (1-based)
 * data_phys:  Physical/bus address of destination buffer
 * result:     Output result struct (in pinned memory)
 */
__global__
void gpunvme_read_blocks_kernel(gpu_nvme_queue *q,
                                 uint64_t slba,
                                 uint32_t num_blocks,
                                 uint64_t data_phys,
                                 gpunvme_io_result *result) {
    /* Only thread 0 operates */
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    result->status = 0;
    result->blocks_done = 0;
    result->error_code = 0;
    result->cqe_status = 0;

    uint64_t timeout = q->poll_timeout_cycles;
    if (timeout == 0) timeout = DEFAULT_TIMEOUT_CYCLES;

    /*
     * For Tier 1: data_phys is a host pinned DMA address.
     * We issue reads in chunks that fit in a single PRP1 (one page = 4096 bytes).
     * For simplicity, we read one block at a time if block_size <= 4096.
     * Multi-page reads with PRP lists are a Phase 3 optimization.
     */
    uint32_t block_size = q->block_size;
    if (block_size == 0) block_size = 512;  /* Safe default */

    uint32_t blocks_per_page = 4096 / block_size;
    if (blocks_per_page == 0) blocks_per_page = 1;

    uint32_t done = 0;
    while (done < num_blocks) {
        /* How many blocks in this batch (limited to one page for single PRP1) */
        uint32_t batch = num_blocks - done;
        if (batch > blocks_per_page) batch = blocks_per_page;

        uint64_t cur_lba = slba + done;
        uint64_t cur_phys = data_phys + (uint64_t)done * block_size;

        /* Submit READ command */
        uint16_t cid = sq_submit_read(q, cur_lba, (uint16_t)(batch - 1),
                                       cur_phys, 0 /* no PRP2 for single page */);

        /* Poll for completion */
        cq_poll_result cqr = cq_poll_for_cid(q, cid, timeout);

        if (cqr.timed_out) {
            result->status = 1;
            result->error_code = 2;  /* timeout */
            result->blocks_done = done;
            return;
        }

        if (!cqr.success) {
            result->status = 1;
            result->error_code = 3;  /* NVMe error */
            result->cqe_status = cqr.status;
            result->blocks_done = done;
            return;
        }

        done += batch;
    }

    result->blocks_done = done;
}

/*
 * GPU kernel: write a contiguous range of blocks.
 * Same structure as read, but uses WRITE opcode.
 */
__global__
void gpunvme_write_blocks_kernel(gpu_nvme_queue *q,
                                  uint64_t slba,
                                  uint32_t num_blocks,
                                  uint64_t data_phys,
                                  gpunvme_io_result *result) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    result->status = 0;
    result->blocks_done = 0;
    result->error_code = 0;
    result->cqe_status = 0;

    uint64_t timeout = q->poll_timeout_cycles;
    if (timeout == 0) timeout = DEFAULT_TIMEOUT_CYCLES;

    uint32_t block_size = q->block_size;
    if (block_size == 0) block_size = 512;

    uint32_t blocks_per_page = 4096 / block_size;
    if (blocks_per_page == 0) blocks_per_page = 1;

    uint32_t done = 0;
    while (done < num_blocks) {
        uint32_t batch = num_blocks - done;
        if (batch > blocks_per_page) batch = blocks_per_page;

        uint64_t cur_lba = slba + done;
        uint64_t cur_phys = data_phys + (uint64_t)done * block_size;

        uint16_t cid = sq_submit_write(q, cur_lba, (uint16_t)(batch - 1),
                                        cur_phys, 0);

        cq_poll_result cqr = cq_poll_for_cid(q, cid, timeout);

        if (cqr.timed_out) {
            result->status = 1;
            result->error_code = 2;
            result->blocks_done = done;
            return;
        }

        if (!cqr.success) {
            result->status = 1;
            result->error_code = 3;
            result->cqe_status = cqr.status;
            result->blocks_done = done;
            return;
        }

        done += batch;
    }

    result->blocks_done = done;
}
