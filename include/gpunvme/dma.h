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

/*
 * PRP list for multi-page NVMe transfers.
 *
 * For transfers > 2 pages (8KB with 4KB pages), NVMe requires PRP2 to point
 * to a PRP list: a page-aligned array of uint64_t physical addresses, one
 * per data page (starting from the second page; PRP1 covers the first).
 *
 * Usage:
 *   gpunvme_prp_list_t prp;
 *   gpunvme_prp_list_alloc(&prp, max_pages);
 *   gpunvme_prp_list_build(&prp, data_vaddr, transfer_bytes, page_size);
 *   // prp.prp1 = first page phys addr
 *   // prp.prp2 = PRP list phys addr (or second page phys for 2-page xfer)
 *   // Submit NVMe command with prp1, prp2
 *   gpunvme_prp_list_free(&prp);
 */
typedef struct {
    uint64_t *list;           /* Virtual address of PRP list entries */
    uint64_t  list_phys;      /* Physical address of PRP list (for PRP2) */
    uint64_t  prp1;           /* Physical address of first data page */
    uint64_t  prp2;           /* PRP2 value (second page phys or list phys) */
    uint32_t  n_entries;      /* Number of entries in list */
    uint32_t  max_entries;    /* Allocated capacity */
} gpunvme_prp_list_t;

/*
 * Allocate a PRP list buffer. max_pages = max data pages the list can describe.
 * The list itself is allocated in page-aligned host pinned memory.
 */
gpunvme_err_t gpunvme_prp_list_alloc(gpunvme_prp_list_t *prp, uint32_t max_pages);

/*
 * Build a PRP list for a transfer.
 * Resolves physical addresses of each page in the data buffer.
 * Sets prp1 and prp2 ready for use in an NVMe command.
 *
 * data_vaddr:      Virtual address of data buffer (must be page-aligned)
 * transfer_bytes:  Total transfer size in bytes
 * page_size:       NVMe page size (usually 4096)
 */
gpunvme_err_t gpunvme_prp_list_build(gpunvme_prp_list_t *prp,
                                      void *data_vaddr,
                                      size_t transfer_bytes,
                                      uint32_t page_size);

/*
 * Free a PRP list buffer.
 */
void gpunvme_prp_list_free(gpunvme_prp_list_t *prp);

#ifdef __cplusplus
}
#endif

#endif /* GPUNVME_DMA_H */
