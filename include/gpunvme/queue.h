/*
 * gpu-nvme-direct: I/O Queue Management API
 *
 * Creates and manages NVMe I/O queue pairs for GPU-driven I/O.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#ifndef GPUNVME_QUEUE_H
#define GPUNVME_QUEUE_H

#include <stdint.h>
#include "error.h"
#include "controller.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declare GPU queue state (defined in device/queue_state.cuh) */
struct gpu_nvme_queue;

/* Queue allocation tier */
typedef enum {
    GPUNVME_TIER1 = 1,  /* Queues + data in host pinned memory */
    GPUNVME_TIER2 = 2,  /* Queues in host, data in GPU VRAM */
    GPUNVME_TIER3 = 3,  /* Queues + data in GPU VRAM */
} gpunvme_tier_t;

/* I/O queue pair context (host-side tracking) */
typedef struct {
    uint16_t qid;                  /* Queue ID (1-based for I/O queues) */
    uint16_t sq_size;              /* SQ entry count */
    uint16_t cq_size;              /* CQ entry count */
    gpunvme_tier_t tier;

    /* Queue memory */
    volatile nvme_sq_entry_t *sq;
    volatile nvme_cq_entry_t *cq;
    uint64_t sq_phys;              /* Physical/DMA address */
    uint64_t cq_phys;

    /* Data buffer */
    void *data_buf;
    uint64_t data_buf_phys;
    size_t data_buf_size;

    /* Doorbell offsets (relative to BAR0) */
    uint32_t sq_doorbell_off;
    uint32_t cq_doorbell_off;

    /* GPU queue state struct (in pinned memory, passed to kernels) */
    struct gpu_nvme_queue *gpu_queue;
} gpunvme_io_queue_t;

/*
 * Create an I/O queue pair.
 *
 * Issues admin commands to create I/O CQ and I/O SQ.
 * Allocates queue memory based on tier:
 *   TIER1: cudaMallocHost for SQ, CQ, and data buffer
 *   TIER2: cudaMallocHost for SQ/CQ, cudaMalloc for data (needs P2P)
 *   TIER3: cudaMalloc for everything (needs P2P)
 *
 * ctrl:        Initialized controller
 * qid:         Queue ID (1+)
 * depth:       Queue depth (entries)
 * data_buf_sz: Size of data buffer to allocate (bytes)
 * tier:        Allocation tier
 * out:         Output queue context
 */
gpunvme_err_t gpunvme_create_io_queue(gpunvme_ctrl_t *ctrl,
                                       uint16_t qid,
                                       uint16_t depth,
                                       size_t data_buf_sz,
                                       gpunvme_tier_t tier,
                                       gpunvme_io_queue_t *out);

/*
 * Delete an I/O queue pair.
 * Issues admin Delete I/O SQ and Delete I/O CQ commands.
 */
gpunvme_err_t gpunvme_delete_io_queue(gpunvme_ctrl_t *ctrl,
                                       gpunvme_io_queue_t *q);

/*
 * Get the GPU-visible queue state struct.
 * Pass this pointer to CUDA kernels.
 */
struct gpu_nvme_queue *gpunvme_get_gpu_queue(gpunvme_io_queue_t *q);

#ifdef __cplusplus
}
#endif

#endif /* GPUNVME_QUEUE_H */
