/*
 * gpu-nvme-direct: Layer Loader API
 *
 * High-level API for GPU-initiated NVMe reads of large contiguous regions
 * (e.g., model layers). Encapsulates BAR0 mapping, controller init, PRP
 * building, and pipelined GPU kernel launch into a 3-call interface:
 *   init → load_layer (repeated) → destroy
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#ifndef GPUNVME_LAYER_LOADER_H
#define GPUNVME_LAYER_LOADER_H

#include <stdint.h>
#include <stddef.h>
#include "error.h"
#include "controller.h"
#include "queue.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gpunvme_layer_loader {
    gpunvme_ctrl_t ctrl;
    gpunvme_io_queue_t ioq;
    int bar0_fd;
    volatile void *bar0;
    void *bar0_gpu;
    size_t bar0_size;

    /* PRP pool: single page-aligned alloc, one 4KB page per command */
    void *prp_pool;
    size_t prp_pool_bytes;
    uint64_t *prp1_array;     /* pinned, max_commands entries */
    uint64_t *prp2_array;     /* pinned, max_commands entries */

    uint32_t max_commands;     /* ceil(max_layer_bytes / MDTS) */
    uint32_t blocks_per_cmd;   /* MDTS / block_size */
    uint32_t pipeline_depth;

    void *kernel_params;       /* layer_read_params* (pinned) */
    void *kernel_result;       /* layer_read_result* (pinned) */

    int pagemap_fd;            /* cached /proc/self/pagemap fd */
    int gpu_clock_khz;
} gpunvme_layer_loader_t;

/*
 * Initialize the layer loader.
 *
 * Opens BAR0, registers with CUDA, initializes the NVMe controller,
 * creates an I/O queue, and pre-allocates all PRP/kernel buffers.
 *
 * pci_bdf:         PCI BDF string (e.g., "0000:0b:00.0")
 * max_layer_bytes: Maximum layer size to support (determines PRP pool size)
 * pipeline_depth:  Number of NVMe commands in flight (recommended: 32)
 */
gpunvme_err_t gpunvme_layer_loader_init(gpunvme_layer_loader_t *loader,
                                         const char *pci_bdf,
                                         size_t max_layer_bytes,
                                         uint32_t pipeline_depth);

/*
 * Load a contiguous region from NVMe into host pinned memory.
 *
 * Rebuilds PRP entries for dest_pinned, launches the GPU kernel,
 * and waits for completion. Queue state rolls naturally between calls.
 *
 * start_lba:   First LBA to read
 * size_bytes:  Number of bytes to read (will be rounded up to block size)
 * dest_pinned: Destination buffer (must be page-aligned, cudaMallocHost or
 *              posix_memalign + mlock + cudaHostRegister)
 */
gpunvme_err_t gpunvme_load_layer(gpunvme_layer_loader_t *loader,
                                  uint64_t start_lba,
                                  size_t size_bytes,
                                  void *dest_pinned);

/* Query helpers */
uint32_t gpunvme_layer_loader_block_size(const gpunvme_layer_loader_t *loader);
uint32_t gpunvme_layer_loader_max_transfer(const gpunvme_layer_loader_t *loader);
uint64_t gpunvme_layer_loader_ns_blocks(const gpunvme_layer_loader_t *loader);

/*
 * Destroy the layer loader.
 * Frees all resources, deletes the I/O queue, shuts down the controller.
 */
void gpunvme_layer_loader_destroy(gpunvme_layer_loader_t *loader);

#ifdef __cplusplus
}
#endif

#endif /* GPUNVME_LAYER_LOADER_H */
