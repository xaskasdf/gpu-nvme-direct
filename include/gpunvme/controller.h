/*
 * gpu-nvme-direct: Controller Management API
 *
 * Host-side API for initializing, configuring, and shutting down
 * the NVMe controller.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#ifndef GPUNVME_CONTROLLER_H
#define GPUNVME_CONTROLLER_H

#include <stdint.h>
#include <stddef.h>
#include "error.h"
#include "nvme_regs.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declaration of BAR map (defined in bar_map.c) */
typedef struct gpunvme_bar_map gpunvme_bar_map_t;

/* NVMe controller context */
typedef struct {
    volatile void *bar0;          /* CPU-accessible BAR0 base */
    void *bar0_gpu;               /* GPU-accessible BAR0 base (or NULL) */
    size_t bar0_size;

    /* Controller capabilities (read once at init) */
    nvme_cap_t cap;
    uint8_t dstrd;                /* Doorbell stride */
    uint32_t page_size;           /* Memory page size (bytes) */
    uint32_t timeout_ms;          /* Controller timeout (ms) */
    uint16_t max_queue_entries;   /* Max entries per queue */

    /* Admin queue */
    volatile nvme_sq_entry_t *admin_sq;
    volatile nvme_cq_entry_t *admin_cq;
    uint64_t admin_sq_phys;       /* Physical/DMA address */
    uint64_t admin_cq_phys;
    uint16_t admin_sq_size;
    uint16_t admin_cq_size;
    uint16_t admin_sq_tail;
    uint16_t admin_cq_head;
    uint8_t  admin_cq_phase;
    uint16_t admin_cid;

    /* Identify data */
    uint32_t ns_size_blocks;      /* Namespace size in blocks */
    uint32_t block_size;          /* LBA data size (bytes) */
    uint8_t  mdts;                /* Max Data Transfer Size (log2 pages, 0=no limit) */
    uint32_t max_transfer_bytes;  /* Computed: min(2^mdts * page_size, 1MB fallback) */
    char model[41];               /* Controller model string */
    char serial[21];              /* Controller serial number */
    char firmware[9];             /* Firmware revision */
} gpunvme_ctrl_t;

/*
 * Initialize the NVMe controller.
 *
 * Performs the full init sequence:
 *   1. Read CAP register
 *   2. Disable controller (CC.EN=0)
 *   3. Wait for CSTS.RDY=0
 *   4. Allocate admin SQ/CQ in host pinned memory
 *   5. Configure AQA, ASQ, ACQ registers
 *   6. Configure CC (page size, queue entry sizes)
 *   7. Enable controller (CC.EN=1)
 *   8. Wait for CSTS.RDY=1
 *   9. Issue Identify Controller + Namespace commands
 *
 * bar0: CPU-accessible BAR0 mapping
 * bar0_size: Size of BAR0 mapping in bytes
 */
gpunvme_err_t gpunvme_ctrl_init(gpunvme_ctrl_t *ctrl,
                                 volatile void *bar0,
                                 size_t bar0_size);

/*
 * Shutdown the controller cleanly.
 *
 *   1. Delete I/O queues (if any)
 *   2. Set CC.SHN = Normal Shutdown
 *   3. Wait for CSTS.SHST = Shutdown Complete
 *   4. Disable (CC.EN=0)
 *   5. Free admin queues
 */
gpunvme_err_t gpunvme_ctrl_shutdown(gpunvme_ctrl_t *ctrl);

/*
 * Submit an admin command and wait for completion.
 * Returns the CQE status (0 = success).
 */
gpunvme_err_t gpunvme_admin_submit(gpunvme_ctrl_t *ctrl,
                                    nvme_sq_entry_t *cmd,
                                    uint32_t *cdw0_out);

/*
 * Check if controller has fatal error (CSTS.CFS=1).
 */
int gpunvme_ctrl_is_fatal(gpunvme_ctrl_t *ctrl);

#ifdef __cplusplus
}
#endif

#endif /* GPUNVME_CONTROLLER_H */
