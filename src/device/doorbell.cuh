/*
 * gpu-nvme-direct: GPU-Side Doorbell Write
 *
 * Writing the SQ tail doorbell tells the NVMe controller there are new
 * commands to process. Writing the CQ head doorbell tells the controller
 * we've consumed completions.
 *
 * Memory ordering is critical:
 *   1. SQ entry writes must be visible before doorbell write
 *   2. Doorbell write must reach PCIe (not stuck in GPU write buffer)
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#ifndef GPUNVME_DOORBELL_CUH
#define GPUNVME_DOORBELL_CUH

#include <cstdint>
#include "mmio_ops.cuh"

/*
 * Ring the SQ tail doorbell.
 * Call __threadfence_system() BEFORE this to ensure SQ entry writes
 * are globally visible. The MMIO write itself is ordered via PTX.
 */
__device__ __forceinline__
void doorbell_write_sq_tail(volatile uint32_t *doorbell, uint16_t tail) {
    mmio_write32(doorbell, (uint32_t)tail);
}

/*
 * Ring the CQ head doorbell.
 * This tells the controller we've consumed entries up to (but not including) head.
 */
__device__ __forceinline__
void doorbell_write_cq_head(volatile uint32_t *doorbell, uint16_t head) {
    mmio_write32(doorbell, (uint32_t)head);
}

#endif /* GPUNVME_DOORBELL_CUH */
