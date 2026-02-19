/*
 * gpu-nvme-direct: GPU BAR0 Read Test
 *
 * THE Phase 1 milestone test: GPU reads an NVMe register via MMIO
 * and the result matches the CPU-read value.
 *
 * This proves that a CUDA kernel on the GPU can talk directly to the
 * NVMe controller through PCIe BAR0 registers.
 *
 * Usage: sudo ./test_bar_read <PCI_BDF>
 *   e.g.: sudo ./test_bar_read 0000:03:00.0
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <cuda_runtime.h>

#include <gpunvme/nvme_regs.h>
#include <gpunvme/mmio.h>
#include "device/mmio_ops.cuh"

/* GPU kernel: read multiple NVMe registers */
__global__
void gpu_read_registers(volatile void *bar0_gpu, uint32_t *results) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    volatile uint8_t *base = (volatile uint8_t *)bar0_gpu;

    /* Read VS (Version) at offset 0x08 */
    results[0] = mmio_read32((volatile uint32_t *)(base + NVME_REG_VS));

    /* Read CC (Controller Configuration) at offset 0x14 */
    results[1] = mmio_read32((volatile uint32_t *)(base + NVME_REG_CC));

    /* Read CSTS (Controller Status) at offset 0x1C */
    results[2] = mmio_read32((volatile uint32_t *)(base + NVME_REG_CSTS));

    /* Read CAP low 32 bits at offset 0x00 */
    results[3] = mmio_read32((volatile uint32_t *)(base + NVME_REG_CAP));

    /* Read CAP high 32 bits at offset 0x04 */
    results[4] = mmio_read32((volatile uint32_t *)(base + NVME_REG_CAP + 4));
}

#define NUM_REGS 5

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <PCI_BDF>\n", argv[0]);
        return 1;
    }

    const char *bdf = argv[1];
    int passed = 0;
    int total = NUM_REGS;

    printf("=== GPU BAR0 Register Read Test ===\n\n");

    /* Map BAR0 */
    char path[256];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource0", bdf);

    int fd = open(path, O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    off_t bar_size = lseek(fd, 0, SEEK_END);
    volatile void *bar0 = mmap(NULL, bar_size, PROT_READ | PROT_WRITE,
                                MAP_SHARED, fd, 0);
    if (bar0 == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    /* CPU reads */
    uint32_t cpu_vals[NUM_REGS];
    cpu_vals[0] = host_mmio_read32(nvme_reg_ptr(bar0, NVME_REG_VS));
    cpu_vals[1] = host_mmio_read32(nvme_reg_ptr(bar0, NVME_REG_CC));
    cpu_vals[2] = host_mmio_read32(nvme_reg_ptr(bar0, NVME_REG_CSTS));
    cpu_vals[3] = host_mmio_read32(nvme_reg_ptr(bar0, NVME_REG_CAP));
    cpu_vals[4] = host_mmio_read32(nvme_reg_ptr(bar0, NVME_REG_CAP + 4));

    const char *reg_names[NUM_REGS] = {
        "VS   (0x08)", "CC   (0x14)", "CSTS (0x1C)",
        "CAP_L(0x00)", "CAP_H(0x04)"
    };

    /* Register BAR0 for GPU access */
    cudaError_t err = cudaHostRegister(
        (void *)bar0, bar_size,
        cudaHostRegisterIoMemory | cudaHostRegisterMapped
    );
    if (err != cudaSuccess) {
        printf("cudaHostRegisterIoMemory FAILED: %s\n", cudaGetErrorString(err));
        printf("Cannot perform GPU BAR read test.\n");
        munmap((void *)bar0, bar_size);
        close(fd);
        return 1;
    }

    void *gpu_bar0;
    cudaHostGetDevicePointer(&gpu_bar0, (void *)bar0, 0);

    /* Allocate GPU results */
    uint32_t *d_results;
    cudaMallocHost(&d_results, sizeof(uint32_t) * NUM_REGS);
    memset(d_results, 0, sizeof(uint32_t) * NUM_REGS);

    /* Launch GPU kernel */
    gpu_read_registers<<<1, 1>>>((volatile void *)gpu_bar0, d_results);
    err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        printf("GPU kernel FAILED: %s\n", cudaGetErrorString(err));
        cudaFreeHost(d_results);
        cudaHostUnregister((void *)bar0);
        munmap((void *)bar0, bar_size);
        close(fd);
        return 1;
    }

    /* Compare */
    printf("%-14s  %-12s  %-12s  %s\n", "Register", "CPU", "GPU", "Match");
    printf("%-14s  %-12s  %-12s  %s\n", "--------", "---", "---", "-----");

    for (int i = 0; i < NUM_REGS; i++) {
        int match = (cpu_vals[i] == d_results[i]);
        if (match) passed++;

        printf("%-14s  0x%08x    0x%08x    %s\n",
               reg_names[i], cpu_vals[i], d_results[i],
               match ? "OK" : "MISMATCH");
    }

    printf("\n=== Results: %d/%d registers match ===\n", passed, total);

    if (passed == total) {
        printf("\n*** MILESTONE: GPU successfully reads NVMe registers via MMIO! ***\n");
    }

    /* Cleanup */
    cudaFreeHost(d_results);
    cudaHostUnregister((void *)bar0);
    munmap((void *)bar0, bar_size);
    close(fd);

    return (passed == total) ? 0 : 1;
}
