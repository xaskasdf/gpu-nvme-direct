/*
 * gpu-nvme-direct: I/O Queue Creation and Management
 *
 * Creates NVMe I/O queue pairs and sets up the GPU-visible queue state
 * structure that CUDA kernels use for autonomous I/O.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <gpunvme/queue.h>
#include <gpunvme/controller.h>
#include <gpunvme/nvme_regs.h>
#include <gpunvme/nvme_cmds.h>
#include <gpunvme/dma.h>
#include <gpunvme/error.h>

#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

/* Forward declare admin helpers (from admin.c) */
extern gpunvme_err_t gpunvme_admin_create_io_cq(gpunvme_ctrl_t *ctrl,
    uint16_t qid, uint16_t size, uint64_t cq_phys);
extern gpunvme_err_t gpunvme_admin_create_io_sq(gpunvme_ctrl_t *ctrl,
    uint16_t qid, uint16_t size, uint64_t sq_phys, uint16_t cqid);
extern gpunvme_err_t gpunvme_admin_delete_io_sq(gpunvme_ctrl_t *ctrl, uint16_t qid);
extern gpunvme_err_t gpunvme_admin_delete_io_cq(gpunvme_ctrl_t *ctrl, uint16_t qid);

/* Include gpu_nvme_queue definition */
#include "device/queue_state.cuh"

gpunvme_err_t gpunvme_create_io_queue(gpunvme_ctrl_t *ctrl,
                                       uint16_t qid,
                                       uint16_t depth,
                                       size_t data_buf_sz,
                                       gpunvme_tier_t tier,
                                       gpunvme_io_queue_t *out) {
    if (!ctrl || !out || qid == 0 || depth == 0)
        return GPUNVME_ERR_INVALID_PARAM;

    if (depth > ctrl->max_queue_entries)
        depth = ctrl->max_queue_entries;

    memset(out, 0, sizeof(*out));
    out->qid = qid;
    out->sq_size = depth;
    out->cq_size = depth;
    out->tier = tier;

    gpunvme_err_t err;

    /* Allocate SQ/CQ — NVMe requires page-aligned base addresses */
    size_t sq_bytes = depth * sizeof(nvme_sq_entry_t);
    size_t cq_bytes = depth * sizeof(nvme_cq_entry_t);
    if (sq_bytes < 4096) sq_bytes = 4096;
    if (cq_bytes < 4096) cq_bytes = 4096;

    if (tier == GPUNVME_TIER1 || tier == GPUNVME_TIER2) {
        /* Queues in host pinned memory */
        if (cudaMallocHost((void **)&out->sq, sq_bytes) != cudaSuccess)
            return GPUNVME_ERR_NOMEM;
        if (cudaMallocHost((void **)&out->cq, cq_bytes) != cudaSuccess) {
            cudaFreeHost((void *)out->sq);
            return GPUNVME_ERR_NOMEM;
        }
    } else {
        /* TIER3: queues in GPU VRAM (requires P2P DMA mapping) */
        if (cudaMalloc((void **)&out->sq, sq_bytes) != cudaSuccess)
            return GPUNVME_ERR_NOMEM;
        if (cudaMalloc((void **)&out->cq, cq_bytes) != cudaSuccess) {
            cudaFree((void *)out->sq);
            return GPUNVME_ERR_NOMEM;
        }
    }

    memset((void *)out->sq, 0, sq_bytes);
    memset((void *)out->cq, 0, cq_bytes);

    /* Resolve physical addresses */
    if (tier == GPUNVME_TIER1 || tier == GPUNVME_TIER2) {
        gpunvme_virt_to_phys((void *)out->sq, &out->sq_phys);
        gpunvme_virt_to_phys((void *)out->cq, &out->cq_phys);
    }
    /* TIER3 requires kernel module for GPU DMA addresses */

    /* Allocate data buffer */
    if (data_buf_sz > 0) {
        if (tier == GPUNVME_TIER1) {
            if (cudaMallocHost(&out->data_buf, data_buf_sz) != cudaSuccess)
                goto fail;
            gpunvme_virt_to_phys(out->data_buf, &out->data_buf_phys);
        } else {
            if (cudaMalloc(&out->data_buf, data_buf_sz) != cudaSuccess)
                goto fail;
            /* GPU DMA addr via kernel module */
        }
        out->data_buf_size = data_buf_sz;
    }

    /* Create CQ via admin command (must be created before SQ) */
    err = gpunvme_admin_create_io_cq(ctrl, qid, depth, out->cq_phys);
    if (err != GPUNVME_OK) goto fail;

    /* Create SQ, linked to our CQ */
    err = gpunvme_admin_create_io_sq(ctrl, qid, depth, out->sq_phys, qid);
    if (err != GPUNVME_OK) {
        gpunvme_admin_delete_io_cq(ctrl, qid);
        goto fail;
    }

    /* Calculate doorbell offsets */
    out->sq_doorbell_off = nvme_sq_doorbell_offset(qid, ctrl->dstrd);
    out->cq_doorbell_off = nvme_cq_doorbell_offset(qid, ctrl->dstrd);

    /* Allocate and initialize GPU queue state struct */
    if (cudaMallocHost((void **)&out->gpu_queue, sizeof(gpu_nvme_queue)) != cudaSuccess)
        goto fail;

    gpu_nvme_queue *gq = out->gpu_queue;
    memset(gq, 0, sizeof(*gq));
    gq->sq = out->sq;
    gq->cq = out->cq;
    gq->sq_size = out->sq_size;
    gq->cq_size = out->cq_size;
    gq->sq_tail = 0;
    gq->cq_head = 0;
    gq->qid = qid;
    gq->cq_phase = 1;
    gq->cid_counter = 0;
    gq->nsid = 1;
    gq->block_size = ctrl->block_size;
    gq->data_buf = (volatile void *)out->data_buf;
    gq->data_buf_phys = out->data_buf_phys;
    gq->poll_timeout_cycles = 0;  /* Use default in kernel */

    /* Set doorbell pointers (BAR0 + offset, GPU-accessible if mapped) */
    if (ctrl->bar0_gpu) {
        gq->doorbell_sq = (volatile uint32_t *)
            ((uint8_t *)ctrl->bar0_gpu + out->sq_doorbell_off);
        gq->doorbell_cq = (volatile uint32_t *)
            ((uint8_t *)ctrl->bar0_gpu + out->cq_doorbell_off);
    }

    fprintf(stderr, "io_queue: Created I/O queue pair %u (depth=%u, tier=%d)\n",
            qid, depth, tier);

    return GPUNVME_OK;

fail:
    /* Cleanup on failure */
    if (out->data_buf) {
        if (tier == GPUNVME_TIER1) cudaFreeHost(out->data_buf);
        else cudaFree(out->data_buf);
    }
    if (tier == GPUNVME_TIER1 || tier == GPUNVME_TIER2) {
        if (out->sq) cudaFreeHost((void *)out->sq);
        if (out->cq) cudaFreeHost((void *)out->cq);
    } else {
        if (out->sq) cudaFree((void *)out->sq);
        if (out->cq) cudaFree((void *)out->cq);
    }
    memset(out, 0, sizeof(*out));
    return GPUNVME_ERR_NOMEM;
}

gpunvme_err_t gpunvme_delete_io_queue(gpunvme_ctrl_t *ctrl,
                                       gpunvme_io_queue_t *q) {
    if (!ctrl || !q) return GPUNVME_ERR_INVALID_PARAM;

    /* Delete SQ first, then CQ (NVMe spec order) */
    gpunvme_admin_delete_io_sq(ctrl, q->qid);
    gpunvme_admin_delete_io_cq(ctrl, q->qid);

    /* Free memory */
    if (q->gpu_queue) cudaFreeHost(q->gpu_queue);

    if (q->tier == GPUNVME_TIER1 || q->tier == GPUNVME_TIER2) {
        if (q->sq) cudaFreeHost((void *)q->sq);
        if (q->cq) cudaFreeHost((void *)q->cq);
    } else {
        if (q->sq) cudaFree((void *)q->sq);
        if (q->cq) cudaFree((void *)q->cq);
    }

    if (q->data_buf) {
        if (q->tier == GPUNVME_TIER1) cudaFreeHost(q->data_buf);
        else cudaFree(q->data_buf);
    }

    memset(q, 0, sizeof(*q));
    return GPUNVME_OK;
}

struct gpu_nvme_queue *gpunvme_get_gpu_queue(gpunvme_io_queue_t *q) {
    return q ? q->gpu_queue : NULL;
}
