/*
 * gpu-nvme-direct: BAR0 Memory Mapping
 *
 * Maps the NVMe controller's BAR0 registers into both CPU and GPU
 * address spaces. Two mapping strategies:
 *
 *   1. VFIO: mmap /sys/bus/pci/devices/<bdf>/resource0
 *   2. Custom kmod: mmap /dev/gpunvme0
 *
 * GPU mapping uses cudaHostRegister with cudaHostRegisterIoMemory flag,
 * which tells CUDA the memory is MMIO (not regular DRAM).
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <gpunvme/error.h>
#include <gpunvme/nvme_regs.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <cuda_runtime.h>

/* BAR0 mapping context */
typedef struct {
    volatile void *cpu_addr;   /* CPU virtual address of BAR0 */
    void *gpu_addr;            /* GPU-accessible address of BAR0 */
    size_t size;               /* Mapping size */
    int fd;                    /* File descriptor (resource0 or /dev/gpunvme0) */
    char bdf[16];              /* PCI BDF string */
} gpunvme_bar_map_t;

/*
 * Map NVMe BAR0 into CPU address space via sysfs resource0.
 * Requires the device to be bound to vfio-pci or no driver.
 */
gpunvme_err_t gpunvme_mmap_bar0(const char *bdf, gpunvme_bar_map_t *map) {
    if (!bdf || !map) return GPUNVME_ERR_INVALID_PARAM;

    memset(map, 0, sizeof(*map));
    strncpy(map->bdf, bdf, sizeof(map->bdf) - 1);

    /* Open resource0 */
    char path[256];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource0", bdf);

    map->fd = open(path, O_RDWR | O_SYNC);
    if (map->fd < 0) {
        fprintf(stderr, "bar_map: cannot open %s (need root or vfio-pci)\n", path);
        return GPUNVME_ERR_BAR_MAP;
    }

    /* Get file size = BAR0 size */
    off_t end = lseek(map->fd, 0, SEEK_END);
    if (end <= 0) {
        /* Fallback: read from resource file */
        close(map->fd);
        return GPUNVME_ERR_BAR_MAP;
    }
    map->size = (size_t)end;

    /* mmap BAR0 — use MAP_SHARED for real MMIO writes to reach the device */
    map->cpu_addr = mmap(NULL, map->size, PROT_READ | PROT_WRITE,
                         MAP_SHARED, map->fd, 0);
    if (map->cpu_addr == MAP_FAILED) {
        fprintf(stderr, "bar_map: mmap failed for %s (size=%zu)\n", path, map->size);
        close(map->fd);
        return GPUNVME_ERR_BAR_MAP;
    }

    fprintf(stderr, "bar_map: mapped BAR0 at %p, size=%zu bytes\n",
            map->cpu_addr, map->size);
    return GPUNVME_OK;
}

/*
 * Register the BAR0 mapping with CUDA so GPU kernels can access it.
 * Uses cudaHostRegisterIoMemory flag.
 *
 * This is the critical operation that may fail on GeForce GPUs.
 */
gpunvme_err_t gpunvme_map_bar0_to_gpu(gpunvme_bar_map_t *map) {
    if (!map || !map->cpu_addr) return GPUNVME_ERR_INVALID_PARAM;

    cudaError_t err = cudaHostRegister(
        (void *)map->cpu_addr,
        map->size,
        cudaHostRegisterIoMemory | cudaHostRegisterMapped
    );

    if (err != cudaSuccess) {
        fprintf(stderr, "bar_map: cudaHostRegister(IoMemory) failed: %s\n",
                cudaGetErrorString(err));
        fprintf(stderr, "bar_map: This may be a GeForce P2P restriction.\n");
        fprintf(stderr, "bar_map: Try NVIDIA open-source kernel modules.\n");
        return GPUNVME_ERR_P2P;
    }

    /* Get device pointer */
    err = cudaHostGetDevicePointer(&map->gpu_addr, (void *)map->cpu_addr, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "bar_map: cudaHostGetDevicePointer failed: %s\n",
                cudaGetErrorString(err));
        cudaHostUnregister((void *)map->cpu_addr);
        return GPUNVME_ERR_CUDA;
    }

    fprintf(stderr, "bar_map: GPU can access BAR0 at device ptr %p\n", map->gpu_addr);
    return GPUNVME_OK;
}

/*
 * Get a GPU-accessible pointer to a specific BAR0 register offset.
 */
volatile void *gpunvme_bar0_gpu_ptr(gpunvme_bar_map_t *map, uint32_t offset) {
    if (!map || !map->gpu_addr) return NULL;
    if (offset >= map->size) return NULL;
    return (volatile void *)((uint8_t *)map->gpu_addr + offset);
}

/*
 * Get a CPU-accessible pointer to a specific BAR0 register offset.
 */
volatile void *gpunvme_bar0_cpu_ptr(gpunvme_bar_map_t *map, uint32_t offset) {
    if (!map || !map->cpu_addr) return NULL;
    if (offset >= map->size) return NULL;
    return (volatile void *)((volatile uint8_t *)map->cpu_addr + offset);
}

/*
 * Unmap BAR0 and clean up.
 */
void gpunvme_unmap_bar0(gpunvme_bar_map_t *map) {
    if (!map) return;

    if (map->gpu_addr) {
        cudaHostUnregister((void *)map->cpu_addr);
        map->gpu_addr = NULL;
    }

    if (map->cpu_addr && map->cpu_addr != MAP_FAILED) {
        munmap((void *)map->cpu_addr, map->size);
        map->cpu_addr = NULL;
    }

    if (map->fd >= 0) {
        close(map->fd);
        map->fd = -1;
    }
}
