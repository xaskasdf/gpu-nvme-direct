/*
 * gpu-nvme-direct: Test BAR0 mapping via /dev/mem
 *
 * Maps the NVMe BAR0 physical address directly through /dev/mem,
 * then tries cudaHostRegister on the resulting mapping.
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

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <BAR0_PHYS_ADDR_HEX>\n", argv[0]);
        fprintf(stderr, "  e.g.: %s 0xfc700000\n", argv[0]);
        return 1;
    }

    uint64_t bar_phys = strtoull(argv[1], NULL, 16);
    size_t bar_size = 16384;

    printf("=== /dev/mem BAR0 Test ===\n");
    printf("BAR0 phys: 0x%lx, size: %zu\n\n", bar_phys, bar_size);

    cudaSetDevice(0);

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open /dev/mem");
        printf("Try with: CONFIG_STRICT_DEVMEM=n or iomem=relaxed kernel param\n");
        return 1;
    }

    volatile void *bar0 = mmap(NULL, bar_size, PROT_READ | PROT_WRITE,
                                MAP_SHARED, fd, bar_phys);
    if (bar0 == MAP_FAILED) {
        perror("mmap /dev/mem");
        printf("Kernel may block MMIO access via /dev/mem (CONFIG_STRICT_DEVMEM)\n");
        close(fd);
        return 1;
    }

    printf("/dev/mem mmap at %p\n", bar0);

    /* CPU read test */
    nvme_vs_t vs;
    vs.raw = host_mmio_read32(nvme_reg_ptr(bar0, NVME_REG_VS));
    printf("CPU read VS = 0x%08x (NVMe %u.%u)\n\n", vs.raw, vs.bits.mjr, vs.bits.mnr);

    /* Try cudaHostRegister strategies */
    unsigned int flags_list[] = {
        cudaHostRegisterIoMemory | cudaHostRegisterMapped,
        cudaHostRegisterMapped,
        cudaHostRegisterDefault,
    };
    const char *names[] = {"IoMemory|Mapped", "Mapped", "Default"};

    for (int i = 0; i < 3; i++) {
        printf("Strategy %s (0x%x): ", names[i], flags_list[i]);
        cudaError_t err = cudaHostRegister((void *)bar0, bar_size, flags_list[i]);
        if (err == cudaSuccess) {
            printf("OK!\n");
            cudaHostUnregister((void *)bar0);
        } else {
            printf("FAIL (%s)\n", cudaGetErrorString(err));
        }
    }

    munmap((void *)bar0, bar_size);
    close(fd);
    return 0;
}
