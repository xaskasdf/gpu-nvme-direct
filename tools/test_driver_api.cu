/*
 * gpu-nvme-direct: Test BAR0 access via CUDA Driver API
 *
 * Bypasses cudaHostRegister (runtime API) and uses cuMemHostRegister
 * (driver API) directly. Also tests cuMemCreate/cuMemMap as alternative.
 */

#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <gpunvme/nvme_regs.h>
#include <gpunvme/mmio.h>

#define CU_CHECK(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char *str = NULL; \
        cuGetErrorString(err, &str); \
        printf("  Driver API error: %s (%d)\n", str ? str : "?", err); \
        goto next; \
    } \
} while(0)

/* GPU kernel: read a register */
__global__
void gpu_read32(volatile uint32_t *addr, uint32_t *out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __threadfence_system();
        *out = *addr;
        __threadfence_system();
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <PCI_BDF>\n", argv[0]);
        return 1;
    }

    /* Init CUDA Driver API */
    cuInit(0);

    CUdevice dev;
    cuDeviceGet(&dev, 0);
    char name[256];
    cuDeviceGetName(name, sizeof(name), dev);
    printf("GPU: %s\n", name);

    CUcontext ctx;
    CUctxCreateParams ctxParams = {};
    cuCtxCreate(&ctx, &ctxParams, 0, dev);

    /* Map BAR0 */
    char path[256];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource0", argv[1]);
    int fd = open(path, O_RDWR | O_SYNC);
    if (fd < 0) { perror("open"); return 1; }
    off_t bar_size = lseek(fd, 0, SEEK_END);
    volatile void *bar0 = mmap(NULL, bar_size, PROT_READ | PROT_WRITE,
                                MAP_SHARED, fd, 0);
    if (bar0 == MAP_FAILED) { perror("mmap"); close(fd); return 1; }

    nvme_vs_t vs;
    vs.raw = host_mmio_read32(nvme_reg_ptr(bar0, NVME_REG_VS));
    printf("CPU read VS = 0x%08x (NVMe %u.%u)\n\n", vs.raw, vs.bits.mjr, vs.bits.mnr);

    /* Strategy 1: cuMemHostRegister with IOMEMORY */
    printf("=== Strategy 1: cuMemHostRegister + CU_MEMHOSTREGISTER_IOMEMORY ===\n");
    {
        CUresult err = cuMemHostRegister(
            (void *)bar0, bar_size,
            CU_MEMHOSTREGISTER_IOMEMORY | CU_MEMHOSTREGISTER_DEVICEMAP
        );
        if (err != CUDA_SUCCESS) {
            const char *str = NULL;
            cuGetErrorString(err, &str);
            printf("  FAILED: %s (%d)\n", str ? str : "?", err);
        } else {
            printf("  OK!\n");
            CUdeviceptr dptr;
            cuMemHostGetDevicePointer(&dptr, (void *)bar0, 0);
            printf("  GPU ptr: 0x%llx\n", (unsigned long long)dptr);

            /* Try GPU read */
            uint32_t *h_result;
            cudaMallocHost(&h_result, sizeof(uint32_t));
            *h_result = 0xDEAD;

            volatile uint32_t *gpu_vs = (volatile uint32_t *)(dptr + NVME_REG_VS);
            gpu_read32<<<1, 1>>>(gpu_vs, h_result);
            cudaError_t cerr = cudaDeviceSynchronize();
            if (cerr != cudaSuccess) {
                printf("  GPU kernel: %s\n", cudaGetErrorString(cerr));
            } else {
                printf("  GPU read VS = 0x%08x %s\n", *h_result,
                       (*h_result == vs.raw) ? "*** MATCH ***" :
                       (*h_result == 0xFFFFFFFF) ? "(all Fs - P2P read blocked)" : "(mismatch)");
            }
            cudaFreeHost(h_result);
            cuMemHostUnregister((void *)bar0);
        }
    }

next:
    /* Strategy 2: cuMemHostRegister WITHOUT IOMEMORY flag */
    printf("\n=== Strategy 2: cuMemHostRegister + DEVICEMAP only ===\n");
    {
        CUresult err = cuMemHostRegister(
            (void *)bar0, bar_size,
            CU_MEMHOSTREGISTER_DEVICEMAP
        );
        if (err != CUDA_SUCCESS) {
            const char *str = NULL;
            cuGetErrorString(err, &str);
            printf("  FAILED: %s (%d)\n", str ? str : "?", err);
        } else {
            printf("  OK!\n");
            CUdeviceptr dptr;
            cuMemHostGetDevicePointer(&dptr, (void *)bar0, 0);
            printf("  GPU ptr: 0x%llx\n", (unsigned long long)dptr);

            uint32_t *h_result;
            cudaMallocHost(&h_result, sizeof(uint32_t));
            *h_result = 0xDEAD;

            volatile uint32_t *gpu_vs = (volatile uint32_t *)(dptr + NVME_REG_VS);
            gpu_read32<<<1, 1>>>(gpu_vs, h_result);
            cudaError_t cerr = cudaDeviceSynchronize();
            if (cerr != cudaSuccess) {
                printf("  GPU kernel: %s\n", cudaGetErrorString(cerr));
            } else {
                printf("  GPU read VS = 0x%08x %s\n", *h_result,
                       (*h_result == vs.raw) ? "*** MATCH ***" :
                       (*h_result == 0xFFFFFFFF) ? "(all Fs)" : "(mismatch)");
            }
            cudaFreeHost(h_result);
            cuMemHostUnregister((void *)bar0);
        }
    }

    /* Strategy 3: cuMemHostRegister with just PORTABLE */
    printf("\n=== Strategy 3: cuMemHostRegister + PORTABLE ===\n");
    {
        CUresult err = cuMemHostRegister(
            (void *)bar0, bar_size,
            CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP
        );
        if (err != CUDA_SUCCESS) {
            const char *str = NULL;
            cuGetErrorString(err, &str);
            printf("  FAILED: %s (%d)\n", str ? str : "?", err);
        } else {
            printf("  OK!\n");
            cuMemHostUnregister((void *)bar0);
        }
    }

    munmap((void *)bar0, bar_size);
    close(fd);
    cuCtxDestroy(ctx);
    return 0;
}
