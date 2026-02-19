/*
 * gpu-nvme-direct: Large Sequential Read Test
 *
 * Reads multiple megabytes by issuing back-to-back NVMe commands at MDTS
 * granularity with queue depth pipelining. The GPU submits the next command
 * before polling the previous one, maximizing NVMe utilization.
 *
 * This is the building block for layer streaming in ntransformer.
 *
 * Usage: sudo ./test_large_read <PCI_BDF> [size_mb]
 *   size_mb: total read size in MB (default: 4)
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

/*
 * GPU kernel: sequential read with multi-command pipelining.
 *
 * Issues commands at MDTS granularity. Submits up to `queue_depth` commands
 * ahead, then polls completions in order. This keeps the NVMe busy while
 * the GPU processes completions.
 *
 * All PRP lists are pre-built by the CPU. The GPU just picks the right
 * prp1/prp2 for each chunk.
 */

struct large_read_params {
    uint64_t start_lba;         /* First LBA to read */
    uint32_t total_blocks;      /* Total blocks to read */
    uint32_t blocks_per_cmd;    /* Blocks per NVMe command (MDTS / block_size) */
    uint32_t n_commands;        /* Total commands to issue */
    uint32_t pipeline_depth;    /* How many commands to have in flight */

    /* Per-command PRP info (pre-built by CPU) */
    uint64_t *prp1_array;      /* prp1[i] for command i */
    uint64_t *prp2_array;      /* prp2[i] for command i */
};

struct large_read_result {
    uint32_t status;            /* 0 = success */
    uint32_t commands_completed;
    uint32_t error_code;        /* 0=ok, 2=timeout, 3=nvme_error */
    uint16_t cqe_status;        /* NVMe status of failed command */
    uint64_t gpu_cycles;        /* Total GPU clock cycles for the read */
};

__global__
void gpu_large_read(gpu_nvme_queue *q,
                    large_read_params *params,
                    large_read_result *result) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    result->status = 0;
    result->commands_completed = 0;
    result->error_code = 0;
    result->cqe_status = 0;

    uint32_t n_cmds = params->n_commands;
    uint32_t pipe_depth = params->pipeline_depth;
    uint32_t blocks_per = params->blocks_per_cmd;
    uint64_t base_lba = params->start_lba;

    uint64_t t_start = clock64();

    /* Track CIDs for in-flight commands */
    uint16_t cids[64];  /* max pipeline depth */
    uint32_t submitted = 0;
    uint32_t completed = 0;

    while (completed < n_cmds) {
        /* Submit commands up to pipeline depth */
        while (submitted < n_cmds && (submitted - completed) < pipe_depth) {
            uint32_t cmd_idx = submitted;
            uint64_t lba = base_lba + (uint64_t)cmd_idx * blocks_per;

            /* Last command may be smaller */
            uint32_t remaining = params->total_blocks - cmd_idx * blocks_per;
            uint32_t nlb = (remaining < blocks_per) ? remaining : blocks_per;

            sq_submit_read(
                q, lba, nlb - 1,
                params->prp1_array[cmd_idx],
                params->prp2_array[cmd_idx]
            );
            submitted++;
        }

        /* Poll for ANY next completion (NVMe may complete out of order) */
        cq_poll_result cqr = cq_poll_completion(q, 3400000000ULL);

        if (cqr.timed_out) {
            result->status = 1;
            result->error_code = 2;
            result->commands_completed = completed;
            result->gpu_cycles = clock64() - t_start;
            return;
        }
        if (!cqr.success) {
            result->status = 1;
            result->error_code = 3;
            result->cqe_status = cqr.status;
            result->commands_completed = completed;
            result->gpu_cycles = clock64() - t_start;
            return;
        }

        completed++;
    }

    result->commands_completed = completed;
    result->gpu_cycles = clock64() - t_start;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <PCI_BDF> [size_mb] [pipeline_depth]\n", argv[0]);
        return 1;
    }

    const char *bdf = argv[1];
    uint32_t size_mb = (argc > 2) ? atoi(argv[2]) : 4;
    uint32_t pipe_depth_arg = (argc > 3) ? atoi(argv[3]) : 32;

    printf("=== GPU Large Sequential Read Test ===\n\n");

    /* Map BAR0 */
    char path[256];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource0", bdf);

    int fd = open(path, O_RDWR | O_SYNC);
    if (fd < 0) { perror("open"); return 1; }

    off_t bar_size = lseek(fd, 0, SEEK_END);
    volatile void *bar0 = mmap(NULL, bar_size, PROT_READ | PROT_WRITE,
                                MAP_SHARED, fd, 0);
    if (bar0 == MAP_FAILED) { perror("mmap"); close(fd); return 1; }

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

    {
        uint32_t total_bytes = size_mb * 1024u * 1024u;
        uint32_t blocks_per_cmd = ctrl.max_transfer_bytes / ctrl.block_size;
        uint32_t total_blocks = total_bytes / ctrl.block_size;
        uint32_t n_commands = (total_blocks + blocks_per_cmd - 1) / blocks_per_cmd;

        /* Limit to available LBAs */
        if (total_blocks > ctrl.ns_size_blocks) {
            total_blocks = ctrl.ns_size_blocks;
            total_bytes = total_blocks * ctrl.block_size;
            n_commands = (total_blocks + blocks_per_cmd - 1) / blocks_per_cmd;
            size_mb = total_bytes / (1024 * 1024);
        }

        /* Pipeline depth: configurable via 3rd arg, limited by queue depth */
        uint32_t pipe_depth = pipe_depth_arg;
        if (pipe_depth > 60) pipe_depth = 60;  /* leave room in queue */

        printf("Reading %u MB (%u blocks) in %u commands of %u KB\n",
               size_mb, total_blocks, n_commands, ctrl.max_transfer_bytes / 1024);
        printf("Pipeline depth: %u commands in flight\n", pipe_depth);
        printf("Block size: %u, MDTS: %u KB\n\n", ctrl.block_size,
               ctrl.max_transfer_bytes / 1024);

        /* Allocate data buffer */
        void *data_buf;
        if (cudaMallocHost(&data_buf, total_bytes) != cudaSuccess) {
            printf("Failed to allocate %u MB data buffer\n", size_mb);
            gpunvme_ctrl_shutdown(&ctrl);
            goto cleanup;
        }
        memset(data_buf, 0xDE, total_bytes);

        /* Pre-build PRP lists for all commands.
         * Allocate ONE contiguous page-aligned buffer for all PRP lists
         * to avoid excessive cudaHostRegister calls (which cause NVMe
         * timeouts with many commands). Each PRP list needs one 4KB page. */
        uint64_t *prp1_arr, *prp2_arr;
        cudaMallocHost(&prp1_arr, n_commands * sizeof(uint64_t));
        cudaMallocHost(&prp2_arr, n_commands * sizeof(uint64_t));

        uint32_t pages_per_cmd = ctrl.max_transfer_bytes / ctrl.page_size;
        size_t prp_list_page = 4096;  /* one page per PRP list */
        size_t prp_pool_bytes = n_commands * prp_list_page;
        void *prp_pool = NULL;
        int pm_fd = -1;
        long sys_page_size = sysconf(_SC_PAGESIZE);

        if (posix_memalign(&prp_pool, 4096, prp_pool_bytes) != 0) {
            printf("PRP pool alloc failed\n");
            goto cleanup_bufs;
        }
        mlock(prp_pool, prp_pool_bytes);
        cudaHostRegister(prp_pool, prp_pool_bytes, cudaHostRegisterDefault);
        memset(prp_pool, 0, prp_pool_bytes);

        /* Open pagemap once for all PRP list + data page lookups */
        pm_fd = open("/proc/self/pagemap", O_RDONLY);
        if (pm_fd < 0) { perror("pagemap"); goto cleanup_bufs; }

        printf("Building PRP lists for %u commands...\n", n_commands);
        for (uint32_t i = 0; i < n_commands; i++) {
            uint32_t remaining_blocks = total_blocks - i * blocks_per_cmd;
            uint32_t cmd_blocks = (remaining_blocks < blocks_per_cmd)
                                  ? remaining_blocks : blocks_per_cmd;
            uint32_t cmd_bytes = cmd_blocks * ctrl.block_size;
            uint32_t cmd_pages = (cmd_bytes + ctrl.page_size - 1) / ctrl.page_size;

            /* This command's PRP list lives at offset i*4096 in the pool */
            uint64_t *list_virt = (uint64_t *)((uint8_t *)prp_pool + i * prp_list_page);

            /* Resolve PRP list physical address */
            uint64_t list_vaddr = (uint64_t)(uintptr_t)list_virt;
            uint64_t list_page_idx = list_vaddr / sys_page_size;
            uint64_t pm_entry;
            pread(pm_fd, &pm_entry, 8, list_page_idx * 8);
            uint64_t list_phys = (pm_entry & ((1ULL << 55) - 1)) * sys_page_size;

            /* Build PRP entries: resolve phys addr for each data page */
            uint8_t *chunk = (uint8_t *)data_buf + (uint64_t)i * ctrl.max_transfer_bytes;
            uint64_t prp1 = 0, prp2 = 0;

            for (uint32_t p = 0; p < cmd_pages; p++) {
                uint64_t va = (uint64_t)(uintptr_t)(chunk + (uint64_t)p * ctrl.page_size);
                uint64_t pidx = va / sys_page_size;
                uint64_t entry;
                pread(pm_fd, &entry, 8, pidx * 8);
                uint64_t phys = (entry & ((1ULL << 55) - 1)) * sys_page_size + (va % sys_page_size);

                if (p == 0) {
                    prp1 = phys;
                } else {
                    list_virt[p - 1] = phys;
                }
            }

            if (cmd_pages <= 1) {
                prp2 = 0;
            } else if (cmd_pages == 2) {
                prp2 = list_virt[0];  /* Direct second page phys */
            } else {
                prp2 = list_phys;  /* PRP list phys addr */
            }

            prp1_arr[i] = prp1;
            prp2_arr[i] = prp2;
        }
        close(pm_fd);
        printf("PRP lists ready.\n\n");

        /* Create I/O queue */
        gpunvme_io_queue_t ioq;
        err = gpunvme_create_io_queue(&ctrl, 1, 64, 4096, GPUNVME_TIER1, &ioq);
        if (err != GPUNVME_OK) {
            printf("I/O queue creation failed: %s\n", gpunvme_err_str(err));
            goto cleanup_bufs;
        }

        /* Set up kernel params */
        large_read_params *h_params;
        large_read_result *h_result;
        cudaMallocHost(&h_params, sizeof(large_read_params));
        cudaMallocHost(&h_result, sizeof(large_read_result));

        h_params->start_lba = 0;
        h_params->total_blocks = total_blocks;
        h_params->blocks_per_cmd = blocks_per_cmd;
        h_params->n_commands = n_commands;
        h_params->pipeline_depth = pipe_depth;
        h_params->prp1_array = prp1_arr;
        h_params->prp2_array = prp2_arr;
        memset(h_result, 0, sizeof(large_read_result));

        /* Get GPU clock rate for time calculation */
        int gpu_clock_khz;
        cudaDeviceGetAttribute(&gpu_clock_khz, cudaDevAttrClockRate, 0);

        printf("Launching GPU kernel...\n");
        gpu_large_read<<<1, 1>>>(ioq.gpu_queue, h_params, h_result);
        cerr = cudaDeviceSynchronize();

        if (cerr != cudaSuccess) {
            printf("GPU kernel error: %s\n", cudaGetErrorString(cerr));
        } else if (h_result->status != 0) {
            printf("Read failed at command %u: error=%u, cqe_status=0x%04x\n",
                   h_result->commands_completed, h_result->error_code,
                   h_result->cqe_status);
            /* Check NVMe controller status */
            uint32_t csts = *(volatile uint32_t *)((uint8_t *)bar0 + NVME_REG_CSTS);
            printf("CSTS=0x%08x (RDY=%u, CFS=%u, SHST=%u)\n",
                   csts, csts & 1, (csts >> 1) & 1, (csts >> 2) & 3);
            /* Print PRP info for failing command */
            uint32_t fc = h_result->commands_completed;
            if (fc < n_commands) {
                printf("Failing cmd %u: prp1=0x%lx, prp2=0x%lx\n",
                       fc, prp1_arr[fc], prp2_arr[fc]);
                if (fc > 0)
                    printf("Last ok cmd %u: prp1=0x%lx, prp2=0x%lx\n",
                           fc-1, prp1_arr[fc-1], prp2_arr[fc-1]);
            }
        } else {
            double seconds = (double)h_result->gpu_cycles / ((double)gpu_clock_khz * 1000.0);
            double mb_per_sec = (double)total_bytes / (1024.0 * 1024.0) / seconds;

            printf("\n=== SUCCESS ===\n");
            printf("Read %u MB in %u commands\n", size_mb, h_result->commands_completed);
            printf("GPU cycles: %lu (%.3f ms)\n", h_result->gpu_cycles, seconds * 1000.0);
            printf("Throughput: %.1f MB/s\n", mb_per_sec);

            /* Verify data was written (check 0xDE overwritten) */
            uint8_t *buf = (uint8_t *)data_buf;
            int verified = 1;
            /* Check first, middle, last pages */
            size_t check_offsets[] = {0, total_bytes / 2, total_bytes - 512};
            for (int c = 0; c < 3; c++) {
                size_t off = check_offsets[c];
                int all_de = 1;
                for (int j = 0; j < 16; j++) {
                    if (buf[off + j] != 0xDE) { all_de = 0; break; }
                }
                if (all_de) {
                    printf("WARNING: data at offset %zu still 0xDE\n", off);
                    verified = 0;
                }
            }
            if (verified) {
                printf("Data verification: PASS (0xDE overwritten at start/mid/end)\n");
            }

            /* Print MBR signature from LBA 0 */
            printf("LBA 0 bytes 510-511: %02x %02x", buf[510], buf[511]);
            if (buf[510] == 0x55 && buf[511] == 0xAA)
                printf(" (MBR signature)\n");
            else
                printf("\n");
        }

        cudaFreeHost(h_params);
        cudaFreeHost(h_result);
        gpunvme_delete_io_queue(&ctrl, &ioq);

cleanup_bufs:
        if (prp_pool) {
            cudaHostUnregister(prp_pool);
            munlock(prp_pool, prp_pool_bytes);
            free(prp_pool);
        }
        cudaFreeHost(prp1_arr);
        cudaFreeHost(prp2_arr);
        cudaFreeHost(data_buf);
        gpunvme_ctrl_shutdown(&ctrl);
    }

cleanup:
    cudaHostUnregister((void *)bar0);
    munmap((void *)bar0, bar_size);
    close(fd);

    return 0;
}
