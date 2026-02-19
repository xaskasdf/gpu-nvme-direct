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
