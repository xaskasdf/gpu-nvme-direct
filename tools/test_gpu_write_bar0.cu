/*
 * gpu-nvme-direct: Test GPU Write to NVMe BAR0
 *
 * PCIe Memory Writes are POSTED (no completion needed), so they may work
 * even when reads fail (reads need a completion TLP).
 *
 * Test strategy:
 *   1. CPU reads CC register (current value)
 *   2. GPU writes CC.EN=0 (disable controller) via MMIO write
 *   3. CPU reads CSTS to see if RDY went to 0 (proves write arrived)
 *   4. GPU writes CC.EN=1 (re-enable) to restore state
 *
 * Also tests doorbell write (offset 0x1000) which is the only MMIO write
 * needed for Tier 1 operation.
 *
 * Usage: sudo ./test_gpu_write_bar0 <PCI_BDF>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <cuda_runtime.h>

/* Inline PTX MMIO write */
__device__ __forceinline__
void mmio_write32_dev(volatile uint32_t *addr, uint32_t val) {
    asm volatile(
        "st.relaxed.mmio.sys.u32 [%0], %1;"
        :: "l"(addr), "r"(val) : "memory"
    );
}

__device__ __forceinline__
uint32_t mmio_read32_dev(volatile uint32_t *addr) {
    uint32_t val;
    asm volatile(
        "ld.relaxed.mmio.sys.u32 %0, [%1];"
        : "=r"(val) : "l"(addr) : "memory"
    );
    return val;
}

/* GPU kernel: write a value to a BAR0 register */
__global__
void gpu_write_reg(volatile uint32_t *reg, uint32_t value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __threadfence_system();
        mmio_write32_dev(reg, value);
        __threadfence_system();
    }
}

/* GPU kernel: write doorbell (the only write needed for Tier 1) */
__global__
void gpu_write_doorbell(volatile uint32_t *doorbell, uint32_t tail_value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __threadfence_system();
        mmio_write32_dev(doorbell, tail_value);
        __threadfence_system();
    }
}

/* CPU-side MMIO read */
static inline uint32_t cpu_read32(volatile void *base, uint32_t offset) {
    volatile uint32_t *reg = (volatile uint32_t *)((volatile uint8_t *)base + offset);
    uint32_t val;
    asm volatile("" ::: "memory");
    val = *reg;
    asm volatile("" ::: "memory");
    return val;
}

/* CPU-side MMIO write */
static inline void cpu_write32(volatile void *base, uint32_t offset, uint32_t val) {
    volatile uint32_t *reg = (volatile uint32_t *)((volatile uint8_t *)base + offset);
    asm volatile("" ::: "memory");
    *reg = val;
    asm volatile("" ::: "memory");
}

#define NVME_REG_CAP  0x00
#define NVME_REG_VS   0x08
#define NVME_REG_CC   0x14
#define NVME_REG_CSTS 0x1C
#define NVME_REG_NSSR 0x20
#define NVME_DB_BASE  0x1000

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <PCI_BDF>\n", argv[0]);
        return 1;
    }

    const char *bdf = argv[1];
    char path[256];

    printf("=== GPU Write to NVMe BAR0 Test ===\n\n");

    /* CUDA init */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("NVMe BDF: %s\n\n", bdf);

    /* Map BAR0 */
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource0", bdf);
    int fd = open(path, O_RDWR | O_SYNC);
    if (fd < 0) { perror("open resource0"); return 1; }

    off_t bar_size = lseek(fd, 0, SEEK_END);
    volatile void *bar0 = mmap(NULL, bar_size, PROT_READ | PROT_WRITE,
                                MAP_SHARED, fd, 0);
    if (bar0 == MAP_FAILED) { perror("mmap BAR0"); close(fd); return 1; }

    printf("BAR0 mapped at %p, size %ld\n", bar0, (long)bar_size);

    /* CPU reads before test */
    uint32_t vs  = cpu_read32(bar0, NVME_REG_VS);
    uint32_t cc  = cpu_read32(bar0, NVME_REG_CC);
    uint32_t csts = cpu_read32(bar0, NVME_REG_CSTS);
    printf("CPU: VS=0x%08x CC=0x%08x CSTS=0x%08x\n", vs, cc, csts);

    if (vs == 0xffffffff) {
        printf("ERROR: NVMe not responding (BAR0 reads 0xffffffff)\n");
        munmap((void*)bar0, bar_size);
        close(fd);
        return 1;
    }

    printf("  CC.EN=%d  CSTS.RDY=%d  CSTS.SHST=%d\n",
           cc & 1, csts & 1, (csts >> 2) & 3);

    /* Register BAR0 with CUDA */
    printf("\n--- Registering BAR0 with cudaHostRegisterIoMemory ---\n");
    cudaError_t err = cudaHostRegister(
        (void *)bar0, bar_size,
        cudaHostRegisterIoMemory | cudaHostRegisterMapped
    );
    if (err != cudaSuccess) {
        printf("FAILED: %s\n", cudaGetErrorString(err));
        munmap((void*)bar0, bar_size);
        close(fd);
        return 1;
    }
    printf("OK\n");

    void *gpu_bar0 = NULL;
    err = cudaHostGetDevicePointer(&gpu_bar0, (void *)bar0, 0);
    if (err != cudaSuccess) {
        printf("cudaHostGetDevicePointer FAILED: %s\n", cudaGetErrorString(err));
        cudaHostUnregister((void *)bar0);
        munmap((void*)bar0, bar_size);
        close(fd);
        return 1;
    }
    printf("GPU device pointer: %p\n", gpu_bar0);

    /* ========== TEST 1: GPU Write to CC register ========== */
    printf("\n--- TEST 1: GPU Write CC.EN=0 (disable controller) ---\n");

    /* Target: CC register */
    volatile uint32_t *gpu_cc = (volatile uint32_t *)
        ((uint8_t *)gpu_bar0 + NVME_REG_CC);

    /* Write CC with EN=0 (clear bit 0), keep other bits */
    uint32_t cc_disabled = cc & ~(uint32_t)1;  /* clear EN bit */
    printf("Writing CC=0x%08x (EN=0) from GPU...\n", cc_disabled);

    gpu_write_reg<<<1, 1>>>(gpu_cc, cc_disabled);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("GPU kernel FAILED: %s\n", cudaGetErrorString(err));
    } else {
        printf("GPU kernel completed OK\n");

        /* Wait for controller to process */
        usleep(100000); /* 100ms */

        /* CPU reads CC and CSTS to verify */
        uint32_t cc_after  = cpu_read32(bar0, NVME_REG_CC);
        uint32_t csts_after = cpu_read32(bar0, NVME_REG_CSTS);
        printf("CPU after write: CC=0x%08x CSTS=0x%08x\n", cc_after, csts_after);
        printf("  CC.EN=%d  CSTS.RDY=%d\n", cc_after & 1, csts_after & 1);

        if ((cc_after & 1) == 0) {
            printf("*** WRITE SUCCESS: CC.EN changed to 0! GPU writes reach NVMe! ***\n");
        } else if (cc_after == 0xffffffff) {
            printf("NVMe went offline (reads 0xffffffff)\n");
        } else {
            printf("CC unchanged - write may not have arrived\n");
        }
    }

    /* ========== TEST 2: GPU Write to doorbell ========== */
    printf("\n--- TEST 2: GPU Write to Admin SQ Tail Doorbell (0x1000) ---\n");

    volatile uint32_t *gpu_db = (volatile uint32_t *)
        ((uint8_t *)gpu_bar0 + NVME_DB_BASE);

    printf("Writing doorbell value 0 from GPU...\n");
    gpu_write_doorbell<<<1, 1>>>(gpu_db, 0);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("GPU kernel FAILED: %s\n", cudaGetErrorString(err));
    } else {
        printf("GPU kernel completed OK\n");
        usleep(50000); /* 50ms */

        /* Verify NVMe still alive */
        uint32_t vs_after = cpu_read32(bar0, NVME_REG_VS);
        if (vs_after == vs) {
            printf("NVMe still responding (VS=0x%08x) - doorbell write didn't crash it\n", vs_after);
            printf("*** DOORBELL WRITE TEST PASSED ***\n");
        } else if (vs_after == 0xffffffff) {
            printf("NVMe went offline after doorbell write\n");
        } else {
            printf("VS changed unexpectedly: 0x%08x\n", vs_after);
        }
    }

    /* ========== Restore: re-enable controller ========== */
    printf("\n--- Restoring CC.EN=1 from CPU ---\n");
    uint32_t cc_check = cpu_read32(bar0, NVME_REG_CC);
    if (cc_check != 0xffffffff && (cc_check & 1) == 0) {
        cpu_write32(bar0, NVME_REG_CC, cc);  /* restore original CC */
        usleep(500000); /* 500ms for controller to re-enable */
        uint32_t csts_restore = cpu_read32(bar0, NVME_REG_CSTS);
        printf("After restore: CC=0x%08x CSTS=0x%08x\n",
               cpu_read32(bar0, NVME_REG_CC), csts_restore);
    } else {
        printf("Skipping restore (NVMe offline or already enabled)\n");
    }

    /* Cleanup */
    cudaHostUnregister((void *)bar0);
    munmap((void *)bar0, bar_size);
    close(fd);

    printf("\n=== Done ===\n");
    return 0;
}
