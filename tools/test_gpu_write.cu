/*
 * gpu-nvme-direct: Test if GPU writes reach NVMe BAR0
 *
 * The controller is in shutdown state (SHST=2). We need to reset it
 * by clearing CC.EN anyway. Let's test if the GPU can do this write.
 *
 * 1. CPU reads CC (should have EN=1)
 * 2. GPU writes CC=0 (clears EN)
 * 3. CPU reads CC to verify EN went to 0
 *
 * Usage: sudo ./test_gpu_write <PCI_BDF>
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

/* GPU kernel: write a 32-bit value to a BAR0 register */
__global__
void gpu_write_reg(volatile uint32_t *reg, uint32_t value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __threadfence_system();
        *reg = value;
        __threadfence_system();
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <PCI_BDF>\n", argv[0]);
        return 1;
    }

    const char *bdf = argv[1];
    char path[256];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource0", bdf);

    int fd = open(path, O_RDWR | O_SYNC);
    if (fd < 0) { perror("open"); return 1; }

    off_t bar_size = lseek(fd, 0, SEEK_END);
    volatile void *bar0 = mmap(NULL, bar_size, PROT_READ | PROT_WRITE,
                                MAP_SHARED, fd, 0);
    if (bar0 == MAP_FAILED) { perror("mmap"); close(fd); return 1; }

    printf("=== GPU Write to NVMe BAR0 Test ===\n\n");

    /* Step 1: CPU reads CC */
    nvme_cc_t cc;
    cc.raw = host_mmio_read32(nvme_reg_ptr(bar0, NVME_REG_CC));
    printf("Before: CC = 0x%08x (EN=%u, SHN=%u)\n", cc.raw, cc.bits.en, cc.bits.shn);

    nvme_csts_t csts;
    csts.raw = host_mmio_read32(nvme_reg_ptr(bar0, NVME_REG_CSTS));
    printf("Before: CSTS = 0x%08x (RDY=%u, SHST=%u)\n", csts.raw, csts.bits.rdy, csts.bits.shst);

    /* Register BAR0 with CUDA */
    cudaError_t err = cudaHostRegister(
        (void *)bar0, bar_size,
        cudaHostRegisterIoMemory | cudaHostRegisterMapped
    );
    if (err != cudaSuccess) {
        printf("cudaHostRegisterIoMemory failed: %s\n", cudaGetErrorString(err));
        munmap((void *)bar0, bar_size);
        close(fd);
        return 1;
    }

    void *gpu_bar0 = NULL;
    cudaHostGetDevicePointer(&gpu_bar0, (void *)bar0, 0);

    /* Step 2: GPU writes CC = 0 (clear EN, SHN, everything) */
    volatile uint32_t *gpu_cc = (volatile uint32_t *)
        ((uint8_t *)gpu_bar0 + NVME_REG_CC);

    printf("\nGPU writing CC = 0x00000000 (EN=0)...\n");
    gpu_write_reg<<<1, 1>>>(gpu_cc, 0x00000000);
    err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        printf("GPU kernel failed: %s\n", cudaGetErrorString(err));
    } else {
        printf("GPU kernel completed OK.\n");
    }

    /* Step 3: CPU reads CC again */
    usleep(100000); /* 100ms wait */
    cc.raw = host_mmio_read32(nvme_reg_ptr(bar0, NVME_REG_CC));
    printf("\nAfter:  CC = 0x%08x (EN=%u, SHN=%u)\n", cc.raw, cc.bits.en, cc.bits.shn);

    csts.raw = host_mmio_read32(nvme_reg_ptr(bar0, NVME_REG_CSTS));
    printf("After:  CSTS = 0x%08x (RDY=%u, SHST=%u)\n", csts.raw, csts.bits.rdy, csts.bits.shst);

    if (cc.bits.en == 0) {
        printf("\n*** SUCCESS: GPU write reached NVMe! CC.EN is now 0. ***\n");
        printf("GPU-to-NVMe posted PCIe writes WORK on this platform.\n");
    } else {
        printf("\n*** FAIL: CC.EN is still 1. GPU write did NOT reach NVMe. ***\n");
    }

    cudaHostUnregister((void *)bar0);
    munmap((void *)bar0, bar_size);
    close(fd);
    return 0;
}
