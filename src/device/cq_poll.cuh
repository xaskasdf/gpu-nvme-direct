/*
 * gpu-nvme-direct: GPU-Side Completion Queue Polling
 *
 * After submitting a command, the GPU polls the CQ for a matching
 * completion entry. Detection uses the NVMe phase bit protocol.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#ifndef GPUNVME_CQ_POLL_CUH
#define GPUNVME_CQ_POLL_CUH

#include <cstdint>
#include <gpunvme/nvme_regs.h>
#include "queue_state.cuh"
#include "mmio_ops.cuh"
#include "doorbell.cuh"

/* Status return from CQ poll */
struct cq_poll_result {
    uint32_t cdw0;          /* Command-specific result */
    uint16_t sqhd;          /* Updated SQ head pointer */
    uint16_t status;        /* Status field (without phase bit) */
    uint16_t cid;           /* Command ID of completed command */
    bool     success;       /* true if status == 0 (no error) */
    bool     timed_out;     /* true if poll timed out */
};

/*
 * Poll CQ for the next completion.
 *
 * Uses the phase bit protocol:
 *   - Controller writes CQE with phase bit matching current round
 *   - Phase flips each time CQ wraps around
 *   - New completion detected when (cqe.status_phase & 1) == expected_phase
 *
 * After consuming a CQE, we advance cq_head and write the CQ head doorbell
 * to tell the controller we've freed that slot.
 *
 * timeout_cycles: GPU clock cycles to wait (0 = spin forever, dangerous on real hw)
 */
__device__
cq_poll_result cq_poll_completion(gpu_nvme_queue *q, uint64_t timeout_cycles) {
    cq_poll_result result;
    result.cdw0 = 0;
    result.sqhd = 0;
    result.status = 0;
    result.cid = 0;
    result.success = false;
    result.timed_out = false;

    uint64_t start = clock64();

    while (true) {
        /* Read the status_phase word of the current CQ entry.
         * On real hardware this goes through PCIe MMIO.
         * On simulator this reads from host pinned memory. */
        volatile nvme_cq_entry_t *cqe = &q->cq[q->cq_head];

        /* CQE DW3 (offset 12) contains cid[15:0] | status_phase[31:16].
         * Read as a single aligned 32-bit word to avoid misaligned access
         * on the uint16_t fields. */
        volatile uint32_t *cqe_base = (volatile uint32_t *)cqe;
        uint32_t dw3 = mmio_read32(&cqe_base[3]);
        uint16_t sp = (uint16_t)(dw3 >> 16);

        /* Check phase bit */
        uint8_t phase = sp & 1;
        if (phase == q->cq_phase) {
            /* New completion! Read the full CQE via aligned 32-bit reads. */
            result.cdw0   = mmio_read32(&cqe_base[0]);
            uint32_t dw2  = mmio_read32(&cqe_base[2]);
            result.sqhd   = (uint16_t)(dw2 & 0xFFFF);
            result.cid    = (uint16_t)(dw3 & 0xFFFF);
            result.status = (sp >> 1) & 0x7FFF;
            result.success = (result.status == 0);
            result.timed_out = false;

            /* Advance CQ head (handles wrap + phase flip) */
            gpu_nvme_advance_cq_head(q);

            /* Write CQ head doorbell to free the slot */
            doorbell_write_cq_head(q->doorbell_cq, q->cq_head);

            return result;
        }

        /* Check timeout */
        if (timeout_cycles > 0) {
            uint64_t elapsed = clock64() - start;
            if (elapsed > timeout_cycles) {
                result.timed_out = true;
                return result;
            }
        }
    }
}

/*
 * Poll CQ for a specific command ID.
 * Consumes and discards completions for other CIDs (they should not appear
 * if only one command is in flight, but this handles queue reuse).
 */
__device__
cq_poll_result cq_poll_for_cid(gpu_nvme_queue *q, uint16_t target_cid,
                                uint64_t timeout_cycles) {
    uint64_t start = clock64();

    while (true) {
        uint64_t remaining = 0;
        if (timeout_cycles > 0) {
            uint64_t elapsed = clock64() - start;
            if (elapsed > timeout_cycles) {
                cq_poll_result r;
                r.timed_out = true;
                r.success = false;
                return r;
            }
            remaining = timeout_cycles - elapsed;
        }

        cq_poll_result r = cq_poll_completion(q, remaining);
        if (r.timed_out) return r;
        if (r.cid == target_cid) return r;
        /* Got completion for different CID — continue polling */
    }
}

#endif /* GPUNVME_CQ_POLL_CUH */
