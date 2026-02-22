// Quick test: does nvidia_p2p_get_pages work on our patched GeForce RTX 3090?
// This test allocates GPU memory, then asks our kernel module to pin it via P2P API.
// If the patch works, we get physical addresses for DMA.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <string.h>
#include <errno.h>

// From gpunvme ioctl interface
#define GPUNVME_IOC_MAGIC 'G'
#define GPUNVME_IOC_GPU_MEM_MAP    _IOWR(GPUNVME_IOC_MAGIC, 10, struct gpunvme_gpu_mem_map_req)
#define GPUNVME_IOC_GPU_MEM_UNMAP  _IOW(GPUNVME_IOC_MAGIC, 11, struct gpunvme_gpu_mem_unmap_req)

struct gpunvme_gpu_mem_map_req {
    uint64_t gpu_vaddr;  // in: CUDA device pointer
    uint64_t size;       // in: size in bytes
    uint64_t bus_addr;   // out: PCIe bus address for DMA
    uint32_t handle;     // out: mapping handle
    uint32_t pad;
};

struct gpunvme_gpu_mem_unmap_req {
    uint32_t handle;
    uint32_t pad;
};

int main(int argc, char **argv) {
    printf("=== nvidia_p2p_get_pages test (P2P patch) ===\n\n");

    // Check CUDA
    int count = 0;
    cudaGetDeviceCount(&count);
    if (count == 0) {
        printf("FAIL: No CUDA devices\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);

    // Allocate 64KB GPU memory (minimum P2P granularity)
    void *d_buf = NULL;
    size_t alloc_size = 64 * 1024; // 64KB = NVRM_P2P_PAGESIZE_BIG_64K
    cudaError_t err = cudaMalloc(&d_buf, alloc_size);
    if (err != cudaSuccess) {
        printf("FAIL: cudaMalloc: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("cudaMalloc: %p (%zu KB)\n", d_buf, alloc_size / 1024);

    // Fill with pattern
    cudaMemset(d_buf, 0xAA, alloc_size);
    cudaDeviceSynchronize();

    // Try to open gpunvme device
    const char *dev_path = "/dev/gpunvme0";
    int fd = open(dev_path, O_RDWR);
    if (fd < 0) {
        printf("\nINFO: %s not available (gpunvme.ko not loaded)\n", dev_path);
        printf("Testing nvidia_p2p_get_pages requires gpunvme.ko loaded.\n");
        printf("But the BAR1 static mapping test doesn't need it.\n\n");

        // Just report the GPU VA for manual BAR1 test
        printf("GPU device pointer: 0x%llx\n", (unsigned long long)d_buf);
        printf("BAR1 physical base: 0x7000000000 (from lspci)\n");
        printf("Expected BAR1 addr: 0x%llx (if static 1:1 mapping)\n",
               0x7000000000ULL + (unsigned long long)d_buf);

        cudaFree(d_buf);
        return 0;
    }

    // Call nvidia_p2p_get_pages via gpunvme ioctl
    struct gpunvme_gpu_mem_map_req req;
    memset(&req, 0, sizeof(req));
    req.gpu_vaddr = (uint64_t)d_buf;
    req.size = alloc_size;

    printf("\nCalling nvidia_p2p_get_pages via gpunvme ioctl...\n");
    int ret = ioctl(fd, GPUNVME_IOC_GPU_MEM_MAP, &req);
    if (ret == 0) {
        printf("SUCCESS! nvidia_p2p_get_pages works on GeForce RTX 3090!\n");
        printf("  GPU VA:   0x%llx\n", (unsigned long long)req.gpu_vaddr);
        printf("  Bus addr: 0x%llx\n", (unsigned long long)req.bus_addr);
        printf("  Handle:   %u\n", req.handle);

        // Unmap
        struct gpunvme_gpu_mem_unmap_req unreq;
        unreq.handle = req.handle;
        ioctl(fd, GPUNVME_IOC_GPU_MEM_UNMAP, &unreq);
    } else {
        printf("FAILED: ioctl returned %d (errno=%d: %s)\n", ret, errno, strerror(errno));
        printf("Check dmesg for nvidia_p2p_get_pages error details.\n");
    }

    close(fd);
    cudaFree(d_buf);
    return (ret == 0) ? 0 : 1;
}
