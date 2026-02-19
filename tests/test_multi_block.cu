/*
 * gpu-nvme-direct: Multi-Block GPU Read Test
 *
 * Tests reading multiple blocks (up to MDTS) with PRP lists.
 * Progressively tests: 1 block, 8 blocks (4KB), 16 blocks (8KB),
 * then up to MDTS. Verifies NVMe DMA by pre-filling buffer with 0xDE.
 *
 * Usage: sudo ./test_multi_block <PCI_BDF>
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
#include <gpunvme/dma.h>
#include <gpunvme/error.h>
#include "device/queue_state.cuh"
#include "device/mmio_ops.cuh"
#include "device/sq_submit.cuh"
#include "device/cq_poll.cuh"

struct read_result {
    uint32_t status;       /* 0 = success */
    uint32_t error_code;   /* 1=crash, 2=timeout, 3=nvme_error */
    uint16_t cqe_status;
};

__global__
void gpu_read_blocks(gpu_nvme_queue *q,
                     uint64_t slba,
                     uint16_t nlb_0based,
                     uint64_t prp1,
                     uint64_t prp2,
                     read_result *result) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    result->status = 0;
    result->error_code = 0;
    result->cqe_status = 0;

    uint16_t cid = sq_submit_read(q, slba, nlb_0based, prp1, prp2);

    /* ~2 second timeout at ~1.7 GHz GPU clock */
    cq_poll_result cqr = cq_poll_for_cid(q, cid, 3400000000ULL);

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
}

static int run_read_test(gpunvme_io_queue_t *ioq,
                         gpunvme_ctrl_t *ctrl,
                         uint32_t n_blocks,
                         void *data_buf,
                         size_t data_buf_size,
                         read_result *h_result) {
    uint32_t transfer_bytes = n_blocks * ctrl->block_size;
    uint32_t page_size = ctrl->page_size;

    if (transfer_bytes > data_buf_size) {
        printf("  SKIP: transfer %u bytes > buffer %zu bytes\n",
               transfer_bytes, data_buf_size);
        return -1;
    }

    /* Build PRP list */
    uint32_t n_pages = (transfer_bytes + page_size - 1) / page_size;
    gpunvme_prp_list_t prp;
    gpunvme_err_t err = gpunvme_prp_list_alloc(&prp, n_pages > 0 ? n_pages : 1);
    if (err != GPUNVME_OK) {
        printf("  FAIL: PRP list alloc failed\n");
        return -1;
    }

    err = gpunvme_prp_list_build(&prp, data_buf, transfer_bytes, page_size);
    if (err != GPUNVME_OK) {
        printf("  FAIL: PRP list build failed\n");
        gpunvme_prp_list_free(&prp);
        return -1;
    }

    /* Pre-fill buffer with 0xDE to verify DMA overwrites it */
    memset(data_buf, 0xDE, transfer_bytes);

    /* Clear result */
    memset(h_result, 0, sizeof(*h_result));

    /* Launch GPU kernel */
    gpu_read_blocks<<<1, 1>>>(ioq->gpu_queue,
                               0,                    /* LBA 0 */
                               n_blocks - 1,         /* 0-based count */
                               prp.prp1,
                               prp.prp2,
                               h_result);
    cudaError_t cerr = cudaDeviceSynchronize();

    gpunvme_prp_list_free(&prp);

    if (cerr != cudaSuccess) {
        printf("  FAIL: GPU kernel error: %s\n", cudaGetErrorString(cerr));
        return -1;
    }

    if (h_result->status != 0) {
        printf("  FAIL: error_code=%u, cqe_status=0x%04x\n",
               h_result->error_code, h_result->cqe_status);
        return -1;
    }

    /* Verify DMA wrote something (check that 0xDE was overwritten) */
    uint8_t *buf = (uint8_t *)data_buf;
    int all_de = 1;
    for (uint32_t i = 0; i < transfer_bytes && i < 4096; i++) {
        if (buf[i] != 0xDE) { all_de = 0; break; }
    }

    if (all_de && transfer_bytes > 0) {
        printf("  FAIL: buffer still 0xDE — NVMe DMA did not write\n");
        return -1;
    }

    /* Print first and last 16 bytes */
    printf("  OK: %u blocks (%u KB) read successfully\n",
           n_blocks, transfer_bytes / 1024);
    printf("  First 16B: ");
    for (int i = 0; i < 16 && i < (int)transfer_bytes; i++)
        printf("%02x ", buf[i]);
    printf("\n");
    if (transfer_bytes > 16) {
        printf("  Last 16B:  ");
        uint32_t off = transfer_bytes - 16;
        for (int i = 0; i < 16; i++)
            printf("%02x ", buf[off + i]);
        printf("\n");
    }

    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <PCI_BDF>\n", argv[0]);
        return 1;
    }

    const char *bdf = argv[1];
    printf("=== GPU Multi-Block Read Test ===\n\n");

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

    printf("\nMDTS=%u → max %u KB per command\n",
           ctrl.mdts, ctrl.max_transfer_bytes / 1024);
    printf("Block size=%u, Page size=%u\n\n", ctrl.block_size, ctrl.page_size);

    /* Create I/O queue with large data buffer */
    {
        gpunvme_io_queue_t ioq;
        size_t buf_size = ctrl.max_transfer_bytes;
        if (buf_size < 4096) buf_size = 4096;

        /* Allocate separate data buffer (page-aligned) */
        void *data_buf;
        if (cudaMallocHost(&data_buf, buf_size) != cudaSuccess) {
            printf("Failed to allocate data buffer\n");
            gpunvme_ctrl_shutdown(&ctrl);
            goto cleanup;
        }

        err = gpunvme_create_io_queue(&ctrl, 1, 64, 4096,
                                       GPUNVME_TIER1, &ioq);
        if (err != GPUNVME_OK) {
            printf("I/O queue creation failed: %s\n", gpunvme_err_str(err));
            cudaFreeHost(data_buf);
            gpunvme_ctrl_shutdown(&ctrl);
            goto cleanup;
        }

        read_result *h_result;
        cudaMallocHost(&h_result, sizeof(read_result));

        /* Progressive size tests */
        uint32_t block_size = ctrl.block_size;
        uint32_t page_size = ctrl.page_size;
        uint32_t blocks_per_page = page_size / block_size;
        uint32_t mdts_blocks = ctrl.max_transfer_bytes / block_size;

        struct test_case {
            const char *name;
            uint32_t n_blocks;
        };

        test_case tests[] = {
            {"1 block (512B)",                    1},
            {"1 page (4KB)",                      blocks_per_page},
            {"2 pages (8KB, PRP2=phys)",          blocks_per_page * 2},
            {"4 pages (16KB, PRP list)",           blocks_per_page * 4},
            {"8 pages (32KB)",                    blocks_per_page * 8},
            {"MDTS",                              mdts_blocks},
        };
        int n_tests = sizeof(tests) / sizeof(tests[0]);

        int pass = 0, fail = 0, skip = 0;

        for (int t = 0; t < n_tests; t++) {
            uint32_t xfer = tests[t].n_blocks * block_size;
            printf("TEST %d: %s (%u blocks, %u KB)\n",
                   t + 1, tests[t].name, tests[t].n_blocks, xfer / 1024);

            if (xfer > buf_size) {
                printf("  SKIP: exceeds buffer\n");
                skip++;
                continue;
            }

            int ret = run_read_test(&ioq, &ctrl, tests[t].n_blocks,
                                    data_buf, buf_size, h_result);
            if (ret == 0) pass++;
            else if (ret == -1) fail++;
            else skip++;
            printf("\n");
        }

        printf("=== Results: %d PASS, %d FAIL, %d SKIP ===\n", pass, fail, skip);

        cudaFreeHost(h_result);
        cudaFreeHost(data_buf);
        gpunvme_delete_io_queue(&ctrl, &ioq);
        gpunvme_ctrl_shutdown(&ctrl);
    }

cleanup:
    cudaHostUnregister((void *)bar0);
    munmap((void *)bar0, bar_size);
    close(fd);

    return 0;
}
