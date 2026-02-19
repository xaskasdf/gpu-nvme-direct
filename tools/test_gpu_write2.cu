/*
 * gpu-nvme-direct: Test GPU access to NVMe BAR0 via multiple strategies
 *
 * Tries different cudaHostRegister flags to map BAR0 into GPU address space.
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

/* GPU kernel: read a 32-bit value */
__global__
void gpu_read_reg(volatile uint32_t *reg, uint32_t *result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __threadfence_system();
        *result = *reg;
        __threadfence_system();
    }
}

/* GPU kernel: write a 32-bit value */
__global__
void gpu_write_reg(volatile uint32_t *reg, uint32_t value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __threadfence_system();
        *reg = value;
        __threadfence_system();
    }
}

struct strategy {
    const char *name;
    unsigned int flags;
};

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <PCI_BDF>\n", argv[0]);
        return 1;
    }

    const char *bdf = argv[1];
    char path[256];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource0", bdf);

    /* Initialize CUDA */
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);

    /* Map BAR0 */
    int fd = open(path, O_RDWR | O_SYNC);
    if (fd < 0) { perror("open"); return 1; }
    off_t bar_size = lseek(fd, 0, SEEK_END);
    volatile void *bar0 = mmap(NULL, bar_size, PROT_READ | PROT_WRITE,
                                MAP_SHARED, fd, 0);
    if (bar0 == MAP_FAILED) { perror("mmap"); close(fd); return 1; }

    /* CPU-side sanity check */
    nvme_vs_t cpu_vs;
    cpu_vs.raw = host_mmio_read32(nvme_reg_ptr(bar0, NVME_REG_VS));
    printf("CPU reads VS = 0x%08x (NVMe %u.%u)\n\n", cpu_vs.raw,
           cpu_vs.bits.mjr, cpu_vs.bits.mnr);

    /* Try different registration strategies */
    strategy strategies[] = {
        {"IoMemory|Mapped",  cudaHostRegisterIoMemory | cudaHostRegisterMapped},
        {"Mapped only",      cudaHostRegisterMapped},
        {"Default (0)",      cudaHostRegisterDefault},
    };

    for (auto &s : strategies) {
        printf("--- Strategy: %s (flags=0x%x) ---\n", s.name, s.flags);

        cudaError_t err = cudaHostRegister((void *)bar0, bar_size, s.flags);
        if (err != cudaSuccess) {
            printf("  cudaHostRegister: FAILED (%s)\n\n", cudaGetErrorString(err));
            continue;
        }
        printf("  cudaHostRegister: OK\n");

        void *gpu_ptr = NULL;
        if (s.flags & cudaHostRegisterMapped) {
            err = cudaHostGetDevicePointer(&gpu_ptr, (void *)bar0, 0);
            if (err != cudaSuccess) {
                printf("  cudaHostGetDevicePointer: FAILED (%s)\n\n", cudaGetErrorString(err));
                cudaHostUnregister((void *)bar0);
                continue;
            }
        } else {
            gpu_ptr = (void *)bar0;
        }
        printf("  GPU ptr: %p\n", gpu_ptr);

        /* GPU read test */
        uint32_t *d_result;
        cudaMallocHost(&d_result, sizeof(uint32_t));
        *d_result = 0xDEADBEEF;

        volatile uint32_t *gpu_vs = (volatile uint32_t *)
            ((uint8_t *)gpu_ptr + NVME_REG_VS);

        gpu_read_reg<<<1, 1>>>(gpu_vs, d_result);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  GPU read: KERNEL CRASH (%s)\n", cudaGetErrorString(err));
            /* Reset device after crash */
            cudaDeviceReset();
            cudaSetDevice(0);
        } else {
            printf("  GPU read VS = 0x%08x %s\n", *d_result,
                   (*d_result == cpu_vs.raw) ? "*** MATCH ***" :
                   (*d_result == 0xFFFFFFFF) ? "(all Fs)" : "(MISMATCH)");
        }

        /* GPU write test: write CC=0 (controller reset) */
        nvme_cc_t cc_before;
        cc_before.raw = host_mmio_read32(nvme_reg_ptr(bar0, NVME_REG_CC));

        volatile uint32_t *gpu_cc = (volatile uint32_t *)
            ((uint8_t *)gpu_ptr + NVME_REG_CC);

        gpu_write_reg<<<1, 1>>>(gpu_cc, 0x00000000);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  GPU write: KERNEL CRASH (%s)\n", cudaGetErrorString(err));
            cudaDeviceReset();
            cudaSetDevice(0);
        } else {
            usleep(100000);
            nvme_cc_t cc_after;
            cc_after.raw = host_mmio_read32(nvme_reg_ptr(bar0, NVME_REG_CC));
            printf("  GPU write CC=0: before=0x%08x after=0x%08x %s\n",
                   cc_before.raw, cc_after.raw,
                   (cc_after.raw == 0) ? "*** WRITE WORKS ***" :
                   (cc_after.raw == cc_before.raw) ? "(no change)" : "(changed!)");
        }

        cudaFreeHost(d_result);
        cudaHostUnregister((void *)bar0);
        printf("\n");
    }

    munmap((void *)bar0, bar_size);
    close(fd);
    return 0;
}
