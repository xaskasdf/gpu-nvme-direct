/*
 * test_p2p_map.cu — Test GPU P2P memory mapping via gpunvme kernel module
 *
 * Tests whether nvidia_p2p_get_pages() works on GeForce RTX 3090
 * by allocating GPU memory, then asking the kernel module to pin it
 * and return bus addresses.
 *
 * Usage: sudo ./test_p2p_map
 *   Requires: gpunvme.ko loaded and bound to an NVMe device
 *             (/dev/gpunvme0 must exist)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <cuda_runtime.h>

// Include the ioctl definitions
#include "../kmod/gpunvme_ioctl.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

int main() {
    printf("=== GPU P2P Memory Mapping Test ===\n\n");

    // Step 1: Initialize CUDA
    printf("[1] Initializing CUDA...\n");
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("    GPU: %s\n", prop.name);

    // Step 2: Allocate GPU memory (must be 64KB aligned, which cudaMalloc guarantees)
    size_t alloc_size = 64 * 1024;  // 64 KB = 1 GPU page
    void* d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, alloc_size));
    printf("[2] cudaMalloc: %zu bytes at GPU vaddr %p\n", alloc_size, d_buf);

    // Step 3: Open the gpunvme device
    printf("[3] Opening /dev/gpunvme0...\n");
    int fd = open("/dev/gpunvme0", O_RDWR);
    if (fd < 0) {
        perror("    open /dev/gpunvme0 failed");
        printf("    Make sure gpunvme.ko is loaded and bound to an NVMe device.\n");
        printf("    Try: sudo insmod kmod/gpunvme.ko target_bdf=0000:01:00.0\n");
        cudaFree(d_buf);
        return 1;
    }
    printf("    OK (fd=%d)\n", fd);

    // Step 4: Get BAR info (sanity check)
    struct gpunvme_bar_info bar_info;
    if (ioctl(fd, GPUNVME_IOCTL_GET_BAR_INFO, &bar_info) == 0) {
        printf("[4] BAR0: phys=0x%llx size=0x%llx (%llu KB)\n",
               (unsigned long long)bar_info.phys_addr,
               (unsigned long long)bar_info.size,
               (unsigned long long)bar_info.size / 1024);
    } else {
        perror("    GET_BAR_INFO failed");
    }

    // Step 5: Map GPU memory — THE CRITICAL TEST
    printf("[5] Mapping GPU memory via nvidia_p2p_get_pages...\n");
    printf("    gpu_vaddr = 0x%llx\n", (unsigned long long)(uintptr_t)d_buf);
    printf("    size      = 0x%llx (%zu bytes)\n",
           (unsigned long long)alloc_size, alloc_size);

    struct gpunvme_gpu_mem_map_req map_req;
    memset(&map_req, 0, sizeof(map_req));
    map_req.gpu_vaddr = (uint64_t)(uintptr_t)d_buf;
    map_req.size = alloc_size;

    int ret = ioctl(fd, GPUNVME_IOCTL_MAP_GPU_MEM, &map_req);
    if (ret == 0) {
        printf("\n    *** SUCCESS ***\n");
        printf("    bus_addr = 0x%llx\n", (unsigned long long)map_req.bus_addr);
        printf("    handle   = %llu\n", (unsigned long long)map_req.handle);
        printf("\n    nvidia_p2p_get_pages WORKS on GeForce RTX 3090!\n");
        printf("    NVMe can DMA to GPU VRAM at bus address 0x%llx\n",
               (unsigned long long)map_req.bus_addr);

        // Step 6: Try a larger mapping (1 MB)
        size_t large_size = 1024 * 1024;  // 1 MB
        void* d_large = nullptr;
        CUDA_CHECK(cudaMalloc(&d_large, large_size));

        struct gpunvme_gpu_mem_map_req large_req;
        memset(&large_req, 0, sizeof(large_req));
        large_req.gpu_vaddr = (uint64_t)(uintptr_t)d_large;
        large_req.size = large_size;

        printf("\n[6] Mapping 1 MB GPU buffer...\n");
        if (ioctl(fd, GPUNVME_IOCTL_MAP_GPU_MEM, &large_req) == 0) {
            printf("    bus_addr = 0x%llx (1 MB)\n",
                   (unsigned long long)large_req.bus_addr);

            // Unmap large
            struct gpunvme_gpu_mem_unmap_req unmap_large;
            unmap_large.handle = large_req.handle;
            ioctl(fd, GPUNVME_IOCTL_UNMAP_GPU_MEM, &unmap_large);
        } else {
            perror("    1 MB mapping failed");
        }
        cudaFree(d_large);

        // Unmap the first one
        struct gpunvme_gpu_mem_unmap_req unmap_req;
        unmap_req.handle = map_req.handle;
        if (ioctl(fd, GPUNVME_IOCTL_UNMAP_GPU_MEM, &unmap_req) == 0) {
            printf("\n[7] Unmapped successfully\n");
        } else {
            perror("    Unmap failed");
        }

    } else {
        perror("    MAP_GPU_MEM failed");
        printf("\n    nvidia_p2p_get_pages does NOT work on this GPU.\n");
        printf("    Check dmesg for details (likely -ENOTSUP for GeForce).\n");
        printf("    This is expected — GeForce GPUs typically block P2P.\n");
    }

    close(fd);
    cudaFree(d_buf);

    printf("\n=== Test complete ===\n");
    return (ret == 0) ? 0 : 1;
}
