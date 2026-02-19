/*
 * gpu-nvme-direct: Host-Side MMIO Helpers
 *
 * CPU-side volatile MMIO read/write with compiler barriers.
 * GPU-side MMIO uses PTX inline asm (see src/device/mmio_ops.cuh).
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#ifndef GPUNVME_MMIO_H
#define GPUNVME_MMIO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Compiler barrier to prevent reordering around MMIO */
#define mmio_barrier() __asm__ __volatile__("" ::: "memory")

/* On x86, reads/writes to WC/UC memory are naturally ordered.
 * We still use volatile + compiler barrier for correctness. */

static inline uint32_t host_mmio_read32(volatile void *addr) {
    uint32_t val;
    mmio_barrier();
    val = *(volatile uint32_t *)addr;
    mmio_barrier();
    return val;
}

static inline void host_mmio_write32(volatile void *addr, uint32_t val) {
    mmio_barrier();
    *(volatile uint32_t *)addr = val;
    mmio_barrier();
}

static inline uint64_t host_mmio_read64(volatile void *addr) {
    uint64_t val;
    mmio_barrier();
    val = *(volatile uint64_t *)addr;
    mmio_barrier();
    return val;
}

static inline void host_mmio_write64(volatile void *addr, uint64_t val) {
    /* NVMe spec: 64-bit registers may not support atomic 64-bit writes.
     * Write as two 32-bit writes, low dword first (NVMe spec 3.1.1.1). */
    volatile uint32_t *addr32 = (volatile uint32_t *)addr;
    mmio_barrier();
    addr32[0] = (uint32_t)(val & 0xFFFFFFFF);
    mmio_barrier();
    addr32[1] = (uint32_t)(val >> 32);
    mmio_barrier();
}

#ifdef __cplusplus
}
#endif

#endif /* GPUNVME_MMIO_H */
