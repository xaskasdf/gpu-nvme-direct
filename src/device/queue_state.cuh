/*
 * gpu-nvme-direct: GPU-Side Queue State
 *
 * Passed to GPU kernels so they can independently manage NVMe I/O queues.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#ifndef GPUNVME_QUEUE_STATE_CUH
#define GPUNVME_QUEUE_STATE_CUH

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#include <stdbool.h>
#endif
#include <gpunvme/nvme_regs.h>

/* GPU-visible NVMe queue pair state.
 * This struct lives in device-accessible memory (pinned host or GPU VRAM).
 * The GPU kernel uses it to submit commands and poll completions. */
typedef struct gpu_nvme_queue {
    /* Queue memory pointers (mapped into GPU address space) */
    volatile nvme_sq_entry_t *sq;        /* Submission queue entries */
    volatile nvme_cq_entry_t *cq;        /* Completion queue entries */

    /* Doorbell register pointers (BAR0 MMIO, mapped into GPU address space) */
    volatile uint32_t *doorbell_sq;      /* SQ tail doorbell */
    volatile uint32_t *doorbell_cq;      /* CQ head doorbell */

    /* Data buffer (where NVMe DMA writes read data / reads write data) */
    volatile void *data_buf;             /* Data buffer base */
    uint64_t data_buf_phys;              /* Physical/bus address of data buffer */

    /* Queue dimensions */
    uint16_t sq_size;                    /* Number of SQ entries (actual count, not 0-based) */
    uint16_t cq_size;                    /* Number of CQ entries */

    /* Queue state (mutable by GPU kernel) */
    uint16_t sq_tail;                    /* Next SQ entry to write */
    uint16_t cq_head;                    /* Next CQ entry to read */
    uint16_t qid;                        /* Queue pair identifier */
    uint8_t  cq_phase;                   /* Expected CQ phase bit (starts at 1) */
    uint16_t cid_counter;                /* Rolling command ID */

    /* Namespace info (needed for building commands) */
    uint32_t nsid;                       /* Namespace ID (usually 1) */
    uint32_t block_size;                 /* Bytes per LBA (typically 512 or 4096) */

    /* Timeout for polling (in GPU clock cycles, 0 = no timeout) */
    uint64_t poll_timeout_cycles;
} gpu_nvme_queue;

/* Device-side helper functions (only available in CUDA compilation) */
#ifdef __CUDACC__

/* Advance SQ tail with wrap-around */
__device__ __forceinline__
uint16_t gpu_nvme_advance_sq_tail(gpu_nvme_queue *q) {
    uint16_t cur = q->sq_tail;
    q->sq_tail = (cur + 1) % q->sq_size;
    return cur;
}

/* Advance CQ head with wrap-around and phase bit flip */
__device__ __forceinline__
void gpu_nvme_advance_cq_head(gpu_nvme_queue *q) {
    q->cq_head++;
    if (q->cq_head >= q->cq_size) {
        q->cq_head = 0;
        q->cq_phase ^= 1;  /* Flip expected phase on wrap */
    }
}

/* Generate next command ID (wraps at 65535) */
__device__ __forceinline__
uint16_t gpu_nvme_next_cid(gpu_nvme_queue *q) {
    return q->cid_counter++;
}

/* Check if SQ has room for one more entry */
__device__ __forceinline__
bool gpu_nvme_sq_has_room(const gpu_nvme_queue *q, uint16_t sq_head_from_cq) {
    uint16_t next = (q->sq_tail + 1) % q->sq_size;
    return next != sq_head_from_cq;
}

#endif /* __CUDACC__ */

#endif /* GPUNVME_QUEUE_STATE_CUH */
