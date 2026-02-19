/*
 * gpu-nvme-direct: DMA Buffer Allocation
 *
 * Allocates DMA-accessible buffers and resolves their physical addresses
 * for use in NVMe PRP entries.
 *
 * Physical address resolution:
 *   - Host memory: /proc/self/pagemap (requires root)
 *   - GPU memory: nvidia_p2p_get_pages via kernel module (Tier 2+)
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <gpunvme/dma.h>
#include <gpunvme/error.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/mman.h>
#include <cuda_runtime.h>

/*
 * Translate virtual address to physical address using /proc/self/pagemap.
 *
 * pagemap format (per entry, 8 bytes):
 *   Bit 63: page present
 *   Bit 62: swapped
 *   Bits 54:0: page frame number (PFN) if present
 */
gpunvme_err_t gpunvme_virt_to_phys(void *vaddr, uint64_t *phys) {
    if (!vaddr || !phys) return GPUNVME_ERR_INVALID_PARAM;

    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) {
        perror("dma: open /proc/self/pagemap");
        return GPUNVME_ERR_IO;
    }

    uint64_t vaddr_int = (uint64_t)(uintptr_t)vaddr;
    long page_size = sysconf(_SC_PAGESIZE);
    uint64_t page_index = vaddr_int / page_size;
    uint64_t page_offset = vaddr_int % page_size;

    uint64_t entry;
    if (pread(fd, &entry, sizeof(entry), page_index * sizeof(entry)) != sizeof(entry)) {
        close(fd);
        return GPUNVME_ERR_IO;
    }
    close(fd);

    /* Check if page is present */
    if (!(entry & (1ULL << 63))) {
        fprintf(stderr, "dma: page not present for vaddr %p\n", vaddr);
        return GPUNVME_ERR_DMA;
    }

    /* Extract PFN (bits 54:0) */
    uint64_t pfn = entry & ((1ULL << 55) - 1);
    *phys = pfn * page_size + page_offset;

    return GPUNVME_OK;
}

gpunvme_err_t gpunvme_dma_alloc_host(size_t size, gpunvme_dma_buf_t *buf) {
    if (!buf || size == 0) return GPUNVME_ERR_INVALID_PARAM;

    memset(buf, 0, sizeof(*buf));

    /* Allocate host pinned memory (page-aligned, DMA-friendly) */
    cudaError_t err = cudaMallocHost(&buf->vaddr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "dma: cudaMallocHost(%zu) failed: %s\n",
                size, cudaGetErrorString(err));
        return GPUNVME_ERR_NOMEM;
    }

    memset(buf->vaddr, 0, size);
    buf->size = size;
    buf->type = GPUNVME_DMA_HOST;

    /* Resolve physical address */
    gpunvme_err_t ret = gpunvme_virt_to_phys(buf->vaddr, &buf->phys_addr);
    if (ret != GPUNVME_OK) {
        fprintf(stderr, "dma: WARNING - could not resolve phys addr (need root?)\n");
        /* Don't fail — the buffer is still usable for simulation */
        buf->phys_addr = (uint64_t)(uintptr_t)buf->vaddr;
    }

    return GPUNVME_OK;
}

gpunvme_err_t gpunvme_dma_alloc_gpu(size_t size, gpunvme_dma_buf_t *buf) {
    if (!buf || size == 0) return GPUNVME_ERR_INVALID_PARAM;

    memset(buf, 0, sizeof(*buf));

    cudaError_t err = cudaMalloc(&buf->vaddr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "dma: cudaMalloc(%zu) failed: %s\n",
                size, cudaGetErrorString(err));
        return GPUNVME_ERR_NOMEM;
    }

    cudaMemset(buf->vaddr, 0, size);
    buf->size = size;
    buf->type = GPUNVME_DMA_GPU;

    /* GPU physical/DMA address requires nvidia_p2p_get_pages via kernel module.
     * This is not available in userspace directly.
     * For now, return an error for the physical address — the kernel module
     * (kmod/gpunvme_dma.c) handles this via ioctl. */
    buf->phys_addr = 0;
    fprintf(stderr, "dma: GPU buffer allocated at device ptr %p\n", buf->vaddr);
    fprintf(stderr, "dma: Physical address requires kernel module (Tier 2+)\n");

    return GPUNVME_OK;
}

void gpunvme_dma_free(gpunvme_dma_buf_t *buf) {
    if (!buf || !buf->vaddr) return;

    if (buf->type == GPUNVME_DMA_HOST) {
        cudaFreeHost(buf->vaddr);
    } else if (buf->type == GPUNVME_DMA_GPU) {
        cudaFree(buf->vaddr);
    }

    memset(buf, 0, sizeof(*buf));
}

/* ---- PRP List ---- */

gpunvme_err_t gpunvme_prp_list_alloc(gpunvme_prp_list_t *prp, uint32_t max_pages) {
    if (!prp || max_pages == 0) return GPUNVME_ERR_INVALID_PARAM;

    memset(prp, 0, sizeof(*prp));

    /* PRP list needs (max_pages - 1) entries: PRP1 covers the first page.
     * Round up to whole page for alignment. */
    uint32_t n_entries = max_pages - 1;
    if (n_entries == 0) n_entries = 1;
    size_t list_bytes = n_entries * sizeof(uint64_t);
    if (list_bytes < 4096) list_bytes = 4096;  /* page-align */

    /* Use posix_memalign to guarantee page alignment for NVMe PRP list.
     * cudaMallocHost's suballocator may return sub-page offsets after
     * many allocations, violating NVMe PRP list alignment requirements. */
    void *list_ptr = NULL;
    if (posix_memalign(&list_ptr, 4096, list_bytes) != 0)
        return GPUNVME_ERR_NOMEM;
    mlock(list_ptr, list_bytes);
    if (cudaHostRegister(list_ptr, list_bytes, cudaHostRegisterDefault) != cudaSuccess) {
        munlock(list_ptr, list_bytes);
        free(list_ptr);
        return GPUNVME_ERR_NOMEM;
    }
    prp->list = (uint64_t *)list_ptr;

    memset(prp->list, 0, list_bytes);
    prp->max_entries = n_entries;
    prp->list_bytes = list_bytes;  /* Save for cleanup */

    /* Resolve physical address of the PRP list itself */
    gpunvme_err_t err = gpunvme_virt_to_phys(prp->list, &prp->list_phys);
    if (err != GPUNVME_OK) {
        cudaHostUnregister(prp->list);
        munlock(prp->list, list_bytes);
        free(prp->list);
        memset(prp, 0, sizeof(*prp));
        return err;
    }

    /* Verify page alignment */
    if (prp->list_phys & 0xFFF) {
        fprintf(stderr, "dma: WARNING: PRP list phys 0x%lx not page-aligned!\n",
                (unsigned long)prp->list_phys);
    }

    return GPUNVME_OK;
}

gpunvme_err_t gpunvme_prp_list_build(gpunvme_prp_list_t *prp,
                                      void *data_vaddr,
                                      size_t transfer_bytes,
                                      uint32_t page_size) {
    if (!prp || !prp->list || !data_vaddr || transfer_bytes == 0)
        return GPUNVME_ERR_INVALID_PARAM;

    uint32_t n_pages = (transfer_bytes + page_size - 1) / page_size;
    if (n_pages > prp->max_entries + 1)
        return GPUNVME_ERR_INVALID_PARAM;

    /* Open pagemap once for efficiency */
    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) return GPUNVME_ERR_IO;

    long sys_page_size = sysconf(_SC_PAGESIZE);
    uint8_t *base = (uint8_t *)data_vaddr;

    for (uint32_t i = 0; i < n_pages; i++) {
        uint64_t vaddr_int = (uint64_t)(uintptr_t)(base + (uint64_t)i * page_size);
        uint64_t page_index = vaddr_int / sys_page_size;

        uint64_t entry;
        if (pread(fd, &entry, sizeof(entry), page_index * sizeof(entry)) != sizeof(entry)) {
            close(fd);
            return GPUNVME_ERR_IO;
        }

        if (!(entry & (1ULL << 63))) {
            fprintf(stderr, "dma: page %u not present (vaddr=%p)\n", i, (void *)vaddr_int);
            close(fd);
            return GPUNVME_ERR_DMA;
        }

        uint64_t pfn = entry & ((1ULL << 55) - 1);
        uint64_t phys = pfn * sys_page_size + (vaddr_int % sys_page_size);

        if (i == 0) {
            prp->prp1 = phys;
        } else {
            prp->list[i - 1] = phys;
        }
    }

    close(fd);

    prp->n_entries = (n_pages > 1) ? n_pages - 1 : 0;

    /* Set PRP2 */
    if (n_pages <= 1) {
        prp->prp2 = 0;
    } else if (n_pages == 2) {
        prp->prp2 = prp->list[0];  /* Direct physical address of second page */
    } else {
        prp->prp2 = prp->list_phys;  /* Physical address of PRP list */
    }

    return GPUNVME_OK;
}

void gpunvme_prp_list_free(gpunvme_prp_list_t *prp) {
    if (!prp) return;
    if (prp->list) {
        cudaHostUnregister(prp->list);
        munlock(prp->list, prp->list_bytes);
        free(prp->list);
    }
    memset(prp, 0, sizeof(*prp));
}
