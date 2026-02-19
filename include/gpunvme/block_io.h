/*
 * gpu-nvme-direct: High-Level Block I/O API
 *
 * Convenience API for GPU-initiated block reads/writes.
 * Wraps the setup of queues and kernel launches.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#ifndef GPUNVME_BLOCK_IO_H
#define GPUNVME_BLOCK_IO_H

#include <stdint.h>
#include "error.h"
#include "controller.h"
#include "queue.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Read blocks from NVMe using GPU-initiated I/O.
 *
 * Launches a CUDA kernel that:
 *   1. Builds NVMe READ commands
 *   2. Writes them to the SQ
 *   3. Rings doorbells
 *   4. Polls CQ for completion
 *
 * ctrl:      Initialized controller
 * q:         I/O queue pair
 * slba:      Starting LBA
 * num_blocks: Number of blocks to read
 * stream:    CUDA stream (0 for default)
 *
 * On completion, data is in q->data_buf (host pinned for Tier 1).
 *
 * Returns GPUNVME_OK on success.
 */
gpunvme_err_t gpunvme_read_blocks(gpunvme_ctrl_t *ctrl,
                                   gpunvme_io_queue_t *q,
                                   uint64_t slba,
                                   uint32_t num_blocks,
                                   void *cuda_stream);

/*
 * Write blocks to NVMe using GPU-initiated I/O.
 * Data must be in q->data_buf before calling.
 */
gpunvme_err_t gpunvme_write_blocks(gpunvme_ctrl_t *ctrl,
                                    gpunvme_io_queue_t *q,
                                    uint64_t slba,
                                    uint32_t num_blocks,
                                    void *cuda_stream);

/*
 * Synchronous read: launches kernel, waits for completion, checks result.
 * Simpler API for testing.
 */
gpunvme_err_t gpunvme_read_blocks_sync(gpunvme_ctrl_t *ctrl,
                                        gpunvme_io_queue_t *q,
                                        uint64_t slba,
                                        uint32_t num_blocks);

#ifdef __cplusplus
}
#endif

#endif /* GPUNVME_BLOCK_IO_H */
