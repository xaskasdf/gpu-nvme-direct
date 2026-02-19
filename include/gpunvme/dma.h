/*
 * gpu-nvme-direct: DMA Buffer Management
 *
 * Allocates DMA-accessible buffers for NVMe data transfers.
 * Supports host pinned memory (Tier 1) and GPU VRAM (Tier 2+).
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#ifndef GPUNVME_DMA_H
#define GPUNVME_DMA_H

#include <stdint.h>
#include <stddef.h>
#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* DMA buffer type */
typedef enum {
    GPUNVME_DMA_HOST = 0,  /* Host pinned memory (cudaMallocHost) */
    GPUNVME_DMA_GPU  = 1,  /* GPU VRAM (cudaMalloc, needs P2P DMA mapping) */
} gpunvme_dma_type_t;

/* DMA buffer descriptor */
typedef struct {
    void *vaddr;              /* Virtual address (host or device) */
    uint64_t phys_addr;       /* Physical/bus/DMA address for NVMe PRP */
    size_t size;              /* Buffer size in bytes */
    gpunvme_dma_type_t type;
} gpunvme_dma_buf_t;

/*
 * Allocate a DMA buffer in host pinned memory.
 * The physical address is obtained via /proc/self/pagemap.
 */
gpunvme_err_t gpunvme_dma_alloc_host(size_t size, gpunvme_dma_buf_t *buf);

/*
 * Allocate a DMA buffer in GPU VRAM.
 * Requires nvidia_p2p_get_pages via kernel module for DMA address.
 * Returns GPUNVME_ERR_P2P if P2P is not available.
 */
gpunvme_err_t gpunvme_dma_alloc_gpu(size_t size, gpunvme_dma_buf_t *buf);

/*
 * Free a DMA buffer.
 */
void gpunvme_dma_free(gpunvme_dma_buf_t *buf);

/*
 * Get physical address of a host virtual address via /proc/self/pagemap.
 * Requires root or appropriate capabilities.
 */
gpunvme_err_t gpunvme_virt_to_phys(void *vaddr, uint64_t *phys);

#ifdef __cplusplus
}
#endif

#endif /* GPUNVME_DMA_H */
