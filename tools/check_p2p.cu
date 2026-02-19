/*
 * gpu-nvme-direct: Check GPU-to-NVMe P2P Capability
 *
 * Probes whether the GPU can access NVMe BAR0 via:
 *   1. cudaHostRegisterIoMemory (maps MMIO BAR into GPU address space)
 *   2. Simple MMIO read from GPU kernel (read NVMe Version register)
 *
 * Usage: sudo ./check_p2p <PCI_BDF>
 *   e.g.: sudo ./check_p2p 0000:03:00.0
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

/* GPU kernel: read NVMe Version register via MMIO */
__global__
void gpu_read_version(volatile uint32_t *vs_reg, uint32_t *result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = mmio_read32(vs_reg);
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <PCI_BDF>\n", argv[0]);
        return 1;
    }

    const char *bdf = argv[1];
    char path[256];
    int ret = 1;

    printf("=== GPU-NVMe P2P Capability Check ===\n\n");

    /* Check CUDA device */
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found.\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("NVMe BDF: %s\n\n", bdf);

    /* Map BAR0 */
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource0", bdf);
    int fd = open(path, O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open resource0");
        return 1;
    }

    off_t bar_size = lseek(fd, 0, SEEK_END);
    volatile void *bar0 = mmap(NULL, bar_size, PROT_READ | PROT_WRITE,
                                MAP_SHARED, fd, 0);
    if (bar0 == MAP_FAILED) {
        perror("mmap BAR0");
        close(fd);
        return 1;
    }

    printf("BAR0 mapped at CPU addr %p, size %ld bytes\n", bar0, (long)bar_size);

    /* CPU-side read of Version register */
    nvme_vs_t cpu_vs;
    cpu_vs.raw = host_mmio_read32(nvme_reg_ptr(bar0, NVME_REG_VS));
    printf("CPU read NVMe Version: %u.%u.%u (raw: 0x%08x)\n",
           cpu_vs.bits.mjr, cpu_vs.bits.mnr, cpu_vs.bits.ter, cpu_vs.raw);

    /* Attempt cudaHostRegister with IoMemory flag */
    printf("\n--- Testing cudaHostRegisterIoMemory ---\n");
    cudaError_t err = cudaHostRegister(
        (void *)bar0, bar_size,
        cudaHostRegisterIoMemory | cudaHostRegisterMapped
    );

    if (err != cudaSuccess) {
        printf("RESULT: cudaHostRegisterIoMemory FAILED: %s\n", cudaGetErrorString(err));
        printf("\nThis is expected on GeForce consumer GPUs.\n");
        printf("Possible workarounds:\n");
        printf("  1. Use NVIDIA open-source kernel modules with P2P patch\n");
        printf("  2. Use a custom kernel module to bypass cudaHostRegister\n");
        printf("  3. Use a Tesla/A-series GPU\n");
        goto cleanup;
    }

    printf("RESULT: cudaHostRegisterIoMemory SUCCEEDED!\n");

    /* Get GPU device pointer */
    {
        void *gpu_bar0 = NULL;
        err = cudaHostGetDevicePointer(&gpu_bar0, (void *)bar0, 0);
        if (err != cudaSuccess) {
            printf("cudaHostGetDevicePointer failed: %s\n", cudaGetErrorString(err));
            cudaHostUnregister((void *)bar0);
            goto cleanup;
        }

        printf("GPU device pointer: %p\n", gpu_bar0);

        /* Allocate result buffer */
        uint32_t *d_result;
        cudaMallocHost(&d_result, sizeof(uint32_t));
        *d_result = 0;

        /* Compute VS register GPU address */
        volatile uint32_t *gpu_vs = (volatile uint32_t *)
            ((uint8_t *)gpu_bar0 + NVME_REG_VS);

        /* Launch GPU kernel to read Version register */
        printf("\n--- Testing GPU MMIO Read ---\n");
        gpu_read_version<<<1, 1>>>(gpu_vs, d_result);
        err = cudaDeviceSynchronize();

        if (err != cudaSuccess) {
            printf("RESULT: GPU MMIO read FAILED: %s\n", cudaGetErrorString(err));
            printf("The GPU kernel crashed while reading BAR0.\n");
        } else {
            nvme_vs_t gpu_vs_val;
            gpu_vs_val.raw = *d_result;
            printf("GPU read NVMe Version: %u.%u.%u (raw: 0x%08x)\n",
                   gpu_vs_val.bits.mjr, gpu_vs_val.bits.mnr,
                   gpu_vs_val.bits.ter, gpu_vs_val.raw);

            if (gpu_vs_val.raw == cpu_vs.raw) {
                printf("\nRESULT: *** SUCCESS *** GPU and CPU read matching Version registers!\n");
                printf("GPU can directly access NVMe BAR0 registers via MMIO.\n");
                ret = 0;
            } else {
                printf("\nRESULT: MISMATCH - CPU=0x%08x, GPU=0x%08x\n",
                       cpu_vs.raw, gpu_vs_val.raw);
                printf("GPU read did not return the expected value.\n");
            }
        }

        cudaFreeHost(d_result);
        cudaHostUnregister((void *)bar0);
    }

cleanup:
    munmap((void *)bar0, bar_size);
    close(fd);

    printf("\n=== Done ===\n");
    return ret;
}
