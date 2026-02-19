/*
 * gpu-nvme-direct: Single Block GPU Read Test (Real Hardware)
 *
 * THE CORE Phase 3 milestone test:
 *   1. CPU initializes NVMe controller
 *   2. CPU creates I/O queue pair
 *   3. GPU kernel reads LBA 0 autonomously
 *   4. Result compared against dd-read baseline
 *
 * This proves the GPU can independently drive NVMe I/O on real hardware.
 *
 * Usage: sudo ./test_single_block <PCI_BDF> [baseline_file]
 *   baseline: captured with `dd if=/dev/nvmeXn1 bs=512 count=1 of=baseline.bin`
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
#include <gpunvme/controller.h>
#include <gpunvme/queue.h>
#include <gpunvme/error.h>
#include "device/queue_state.cuh"
#include "device/mmio_ops.cuh"
#include "device/sq_submit.cuh"
#include "device/cq_poll.cuh"

struct gpunvme_io_result {
    uint32_t status;
    uint32_t blocks_done;
    uint32_t error_code;
    uint32_t cqe_status;
};

__global__
void gpu_read_lba0(gpu_nvme_queue *q, uint64_t data_phys,
                    gpunvme_io_result *result) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    result->status = 0;
    result->blocks_done = 0;
    result->error_code = 0;
    result->cqe_status = 0;

    uint16_t cid = sq_submit_read(q, 0, 0, data_phys, 0);

    cq_poll_result cqr = cq_poll_for_cid(q, cid, 1700000000ULL);  /* ~1s timeout */

    if (cqr.timed_out) {
        result->status = 1;
        result->error_code = 2;
        return;
    }
    if (!cqr.success) {
        result->status = 1;
        result->error_code = 3;
        result->cqe_status = cqr.status;
        return;
    }

    result->blocks_done = 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <PCI_BDF> [baseline_file]\n", argv[0]);
        return 1;
    }

    const char *bdf = argv[1];
    const char *baseline_path = argc > 2 ? argv[2] : NULL;

    printf("=== GPU Single Block Read Test ===\n\n");

    /* Map BAR0 */
    char path[256];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource0", bdf);

    int fd = open(path, O_RDWR | O_SYNC);
    if (fd < 0) { perror("open"); return 1; }

    off_t bar_size = lseek(fd, 0, SEEK_END);
    volatile void *bar0 = mmap(NULL, bar_size, PROT_READ | PROT_WRITE,
                                MAP_SHARED, fd, 0);
    if (bar0 == MAP_FAILED) { perror("mmap"); close(fd); return 1; }

    /* Register BAR0 for GPU access */
    cudaError_t cerr = cudaHostRegister(
        (void *)bar0, bar_size,
        cudaHostRegisterIoMemory | cudaHostRegisterMapped
    );
    if (cerr != cudaSuccess) {
        printf("cudaHostRegisterIoMemory failed: %s\n", cudaGetErrorString(cerr));
        munmap((void *)bar0, bar_size);
        close(fd);
        return 1;
    }

    void *gpu_bar0;
    cudaHostGetDevicePointer(&gpu_bar0, (void *)bar0, 0);

    /* Init controller */
    gpunvme_ctrl_t ctrl;
    gpunvme_err_t err = gpunvme_ctrl_init(&ctrl, bar0, bar_size);
    if (err != GPUNVME_OK) {
        printf("Controller init failed: %s\n", gpunvme_err_str(err));
        goto cleanup;
    }
    ctrl.bar0_gpu = gpu_bar0;

    /* Create I/O queue */
    gpunvme_io_queue_t ioq;
    err = gpunvme_create_io_queue(&ctrl, 1, 64, ctrl.block_size * 8,
                                   GPUNVME_TIER1, &ioq);
    if (err != GPUNVME_OK) {
        printf("I/O queue creation failed: %s\n", gpunvme_err_str(err));
        gpunvme_ctrl_shutdown(&ctrl);
        goto cleanup;
    }

    /* Launch GPU kernel */
    {
        gpunvme_io_result *h_result;
        cudaMallocHost(&h_result, sizeof(gpunvme_io_result));
        memset(h_result, 0, sizeof(gpunvme_io_result));

        printf("Launching GPU kernel to read LBA 0...\n");
        gpu_read_lba0<<<1, 1>>>(ioq.gpu_queue, ioq.data_buf_phys, h_result);
        cerr = cudaDeviceSynchronize();

        if (cerr != cudaSuccess) {
            printf("GPU kernel crashed: %s\n", cudaGetErrorString(cerr));
        } else if (h_result->status != 0) {
            printf("GPU read failed: error_code=%u, cqe_status=0x%04x\n",
                   h_result->error_code, h_result->cqe_status);
        } else {
            printf("GPU read succeeded! %u blocks read.\n", h_result->blocks_done);

            /* Compare with baseline if provided */
            if (baseline_path) {
                FILE *bf = fopen(baseline_path, "rb");
                if (bf) {
                    uint8_t expected[4096];
                    size_t bsz = ctrl.block_size > 4096 ? 4096 : ctrl.block_size;
                    size_t nread = fread(expected, 1, bsz, bf);
                    fclose(bf);

                    if (nread == bsz &&
                        memcmp(ioq.data_buf, expected, bsz) == 0) {
                        printf("*** MILESTONE: Data matches baseline! ***\n");
                        printf("GPU autonomously read a block from NVMe.\n");
                    } else {
                        printf("Data mismatch with baseline.\n");
                        printf("First bytes - GPU: %02x %02x %02x %02x\n",
                               ((uint8_t *)ioq.data_buf)[0],
                               ((uint8_t *)ioq.data_buf)[1],
                               ((uint8_t *)ioq.data_buf)[2],
                               ((uint8_t *)ioq.data_buf)[3]);
                        printf("First bytes - Expected: %02x %02x %02x %02x\n",
                               expected[0], expected[1], expected[2], expected[3]);
                    }
                }
            } else {
                printf("No baseline file provided. First 16 bytes of data:\n  ");
                for (int i = 0; i < 16; i++)
                    printf("%02x ", ((uint8_t *)ioq.data_buf)[i]);
                printf("\n");
            }
        }

        cudaFreeHost(h_result);
    }

    /* Cleanup */
    gpunvme_delete_io_queue(&ctrl, &ioq);
    gpunvme_ctrl_shutdown(&ctrl);

cleanup:
    cudaHostUnregister((void *)bar0);
    munmap((void *)bar0, bar_size);
    close(fd);

    return 0;
}
