/*
 * gpu-nvme-direct: GPU-Side PTX MMIO Operations
 *
 * Standard CUDA volatile is NOT sufficient for PCIe BAR MMIO.
 * We need PTX ld.mmio.sys / st.mmio.sys instructions to ensure
 * reads/writes bypass the GPU cache hierarchy and reach the PCIe bus.
 *
 * On the simulator path, we fall back to volatile reads/writes since
 * we're accessing host pinned memory, not actual MMIO space.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#ifndef GPUNVME_MMIO_OPS_CUH
#define GPUNVME_MMIO_OPS_CUH

#include <cstdint>

#ifdef GPUNVME_USE_SIM

/* Simulator: no real MMIO, use volatile accesses to host pinned memory */

__device__ __forceinline__
void mmio_write32(volatile uint32_t *addr, uint32_t val) {
    __threadfence_system();
    *addr = val;
    __threadfence_system();
}

__device__ __forceinline__
uint32_t mmio_read32(volatile uint32_t *addr) {
    __threadfence_system();
    uint32_t val = *addr;
    __threadfence_system();
    return val;
}

__device__ __forceinline__
void mmio_write64(volatile uint64_t *addr, uint64_t val) {
    __threadfence_system();
    *addr = val;
    __threadfence_system();
}

__device__ __forceinline__
uint64_t mmio_read64(volatile uint64_t *addr) {
    __threadfence_system();
    uint64_t val = *addr;
    __threadfence_system();
    return val;
}

#else

/* Real hardware: PTX MMIO instructions for PCIe BAR access */

__device__ __forceinline__
void mmio_write32(volatile uint32_t *addr, uint32_t val) {
    asm volatile(
        "st.relaxed.mmio.sys.u32 [%0], %1;"
        :: "l"(addr), "r"(val) : "memory"
    );
}

__device__ __forceinline__
uint32_t mmio_read32(volatile uint32_t *addr) {
    uint32_t val;
    asm volatile(
        "ld.relaxed.mmio.sys.u32 %0, [%1];"
        : "=r"(val) : "l"(addr) : "memory"
    );
    return val;
}

__device__ __forceinline__
void mmio_write64(volatile uint64_t *addr, uint64_t val) {
    /* NVMe spec: 64-bit registers may need to be written as two 32-bit writes.
     * However, on x86 PCIe with natural alignment, 64-bit write should work.
     * We provide both options. */
    asm volatile(
        "st.relaxed.mmio.sys.u64 [%0], %1;"
        :: "l"(addr), "l"(val) : "memory"
    );
}

__device__ __forceinline__
uint64_t mmio_read64(volatile uint64_t *addr) {
    uint64_t val;
    asm volatile(
        "ld.relaxed.mmio.sys.u64 %0, [%1];"
        : "=l"(val) : "l"(addr) : "memory"
    );
    return val;
}

#endif /* GPUNVME_USE_SIM */

#endif /* GPUNVME_MMIO_OPS_CUH */
