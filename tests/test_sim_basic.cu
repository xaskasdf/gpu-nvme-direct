/*
 * gpu-nvme-direct: Simulator-Based GPU I/O Test
 *
 * Tests the full path: GPU kernel builds NVMe READ command → writes SQ →
 * rings doorbell → simulator processes command → GPU polls CQ → data verified.
 *
 * This proves the GPU can autonomously drive NVMe I/O (through simulation).
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#include <gpunvme/nvme_regs.h>
#include <gpunvme/nvme_cmds.h>
#include <gpunvme/error.h>

/* Simulator */
#include "sim/nvme_sim.h"

/* GPU device code */
#include "device/queue_state.cuh"
#include "device/mmio_ops.cuh"
#include "device/doorbell.cuh"
#include "device/sq_submit.cuh"
#include "device/cq_poll.cuh"

/* ---- Result struct (same as in block_io.cu) ---- */
struct gpunvme_io_result {
    uint32_t status;
    uint32_t blocks_done;
    uint32_t error_code;
    uint32_t cqe_status;
};

/* ---- Test kernel: read a single block ---- */
__global__
void test_read_single_block(gpu_nvme_queue *q, uint64_t data_phys,
                             gpunvme_io_result *result) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    result->status = 0;
    result->blocks_done = 0;
    result->error_code = 0;
    result->cqe_status = 0;

    /* Submit a READ for LBA 0, 1 block */
    uint16_t cid = sq_submit_read(q, 0, 0, data_phys, 0);

    /* Poll for completion with 500M cycle timeout (~300ms at 1.7GHz) */
    cq_poll_result cqr = cq_poll_for_cid(q, cid, 500000000ULL);

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

/* ---- Test kernel: read multiple blocks ---- */
__global__
void test_read_multi_block(gpu_nvme_queue *q, uint64_t data_phys,
                            uint32_t num_blocks, uint32_t block_size,
                            gpunvme_io_result *result) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    result->status = 0;
    result->blocks_done = 0;
    result->error_code = 0;
    result->cqe_status = 0;

    /* Read blocks one at a time */
    for (uint32_t i = 0; i < num_blocks; i++) {
        uint64_t lba = i;
        uint64_t phys = data_phys + (uint64_t)i * block_size;

        uint16_t cid = sq_submit_read(q, lba, 0, phys, 0);
        cq_poll_result cqr = cq_poll_for_cid(q, cid, 500000000ULL);

        if (cqr.timed_out) {
            result->status = 1;
            result->error_code = 2;
            result->blocks_done = i;
            return;
        }

        if (!cqr.success) {
            result->status = 1;
            result->error_code = 3;
            result->cqe_status = cqr.status;
            result->blocks_done = i;
            return;
        }
    }

    result->blocks_done = num_blocks;
}

/* ---- Pattern fill callback ---- */
static void fill_pattern(uint32_t lba, void *buf, uint32_t size) {
    uint8_t *p = (uint8_t *)buf;
    for (uint32_t i = 0; i < size; i++) {
        /* Deterministic pattern: each byte = (lba * 7 + byte_offset) & 0xFF */
        p[i] = (uint8_t)((lba * 7 + i) & 0xFF);
    }
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

/* ---- Test runner ---- */
static int tests_run = 0;
static int tests_passed = 0;

static void test_single_block_read() {
    printf("  TEST: Single block read through simulator... ");
    tests_run++;

    /* Create simulator */
    nvme_sim_config_t cfg;
    cfg.num_blocks = 1024;
    cfg.block_size = 512;
    cfg.sq_size = 64;
    cfg.cq_size = 64;
    cfg.latency_us = 10;  /* 10us simulated latency */

    nvme_sim_t *sim = nvme_sim_create(&cfg);
    if (!sim) {
        printf("[FAIL] simulator creation failed\n");
        return;
    }

    /* Fill block 0 with known pattern */
    nvme_sim_fill_blocks(sim, 0, 1, fill_pattern);

    /* Set up GPU queue state in pinned memory */
    gpu_nvme_queue *h_queue;
    CUDA_CHECK(cudaMallocHost(&h_queue, sizeof(gpu_nvme_queue)));
    memset(h_queue, 0, sizeof(gpu_nvme_queue));

    h_queue->sq = nvme_sim_get_sq(sim);
    h_queue->cq = nvme_sim_get_cq(sim);
    h_queue->doorbell_sq = nvme_sim_get_sq_doorbell(sim);
    h_queue->doorbell_cq = nvme_sim_get_cq_doorbell(sim);
    h_queue->data_buf = nvme_sim_get_data_buf(sim);
    h_queue->data_buf_phys = nvme_sim_get_data_buf_phys(sim);
    h_queue->sq_size = nvme_sim_get_sq_size(sim);
    h_queue->cq_size = nvme_sim_get_cq_size(sim);
    h_queue->sq_tail = 0;
    h_queue->cq_head = 0;
    h_queue->qid = 1;
    h_queue->cq_phase = 1;  /* Must match controller initial phase */
    h_queue->cid_counter = 0;
    h_queue->nsid = 1;
    h_queue->block_size = cfg.block_size;
    h_queue->poll_timeout_cycles = 500000000ULL;

    /* Allocate result in pinned memory */
    gpunvme_io_result *h_result;
    CUDA_CHECK(cudaMallocHost(&h_result, sizeof(gpunvme_io_result)));
    memset(h_result, 0, sizeof(gpunvme_io_result));

    /* Launch GPU kernel */
    test_read_single_block<<<1, 1>>>(h_queue, h_queue->data_buf_phys, h_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Check result */
    if (h_result->status != 0) {
        printf("[FAIL] GPU kernel returned error: code=%u, cqe_status=0x%04x\n",
               h_result->error_code, h_result->cqe_status);
        goto cleanup_single;
    }

    if (h_result->blocks_done != 1) {
        printf("[FAIL] Expected 1 block done, got %u\n", h_result->blocks_done);
        goto cleanup_single;
    }

    /* Verify data: compare data_buf with direct read from simulator */
    {
        uint8_t expected[512];
        nvme_sim_direct_read(sim, 0, 1, expected);

        uint8_t *actual = (uint8_t *)h_queue->data_buf;
        if (memcmp(actual, expected, 512) != 0) {
            printf("[FAIL] Data mismatch!\n");
            printf("    First differing byte: ");
            for (int i = 0; i < 512; i++) {
                if (actual[i] != expected[i]) {
                    printf("offset %d: expected 0x%02x, got 0x%02x\n",
                           i, expected[i], actual[i]);
                    break;
                }
            }
            goto cleanup_single;
        }
    }

    printf("[PASS]\n");
    tests_passed++;

cleanup_single:
    cudaFreeHost(h_result);
    cudaFreeHost(h_queue);
    nvme_sim_destroy(sim);
}

static void test_multi_block_read() {
    printf("  TEST: Multi-block read (8 blocks, 4KB)...    ");
    tests_run++;

    nvme_sim_config_t cfg;
    cfg.num_blocks = 1024;
    cfg.block_size = 512;
    cfg.sq_size = 64;
    cfg.cq_size = 64;
    cfg.latency_us = 5;

    nvme_sim_t *sim = nvme_sim_create(&cfg);
    if (!sim) {
        printf("[FAIL] simulator creation failed\n");
        return;
    }

    uint32_t num_blocks = 8;  /* 8 * 512 = 4KB */

    /* Fill blocks 0-7 with known pattern */
    nvme_sim_fill_blocks(sim, 0, num_blocks, fill_pattern);

    gpu_nvme_queue *h_queue;
    CUDA_CHECK(cudaMallocHost(&h_queue, sizeof(gpu_nvme_queue)));
    memset(h_queue, 0, sizeof(gpu_nvme_queue));

    h_queue->sq = nvme_sim_get_sq(sim);
    h_queue->cq = nvme_sim_get_cq(sim);
    h_queue->doorbell_sq = nvme_sim_get_sq_doorbell(sim);
    h_queue->doorbell_cq = nvme_sim_get_cq_doorbell(sim);
    h_queue->data_buf = nvme_sim_get_data_buf(sim);
    h_queue->data_buf_phys = nvme_sim_get_data_buf_phys(sim);
    h_queue->sq_size = nvme_sim_get_sq_size(sim);
    h_queue->cq_size = nvme_sim_get_cq_size(sim);
    h_queue->sq_tail = 0;
    h_queue->cq_head = 0;
    h_queue->qid = 1;
    h_queue->cq_phase = 1;
    h_queue->cid_counter = 0;
    h_queue->nsid = 1;
    h_queue->block_size = cfg.block_size;
    h_queue->poll_timeout_cycles = 500000000ULL;

    gpunvme_io_result *h_result;
    CUDA_CHECK(cudaMallocHost(&h_result, sizeof(gpunvme_io_result)));
    memset(h_result, 0, sizeof(gpunvme_io_result));

    test_read_multi_block<<<1, 1>>>(h_queue, h_queue->data_buf_phys,
                                     num_blocks, cfg.block_size, h_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (h_result->status != 0) {
        printf("[FAIL] GPU kernel error: code=%u, cqe=0x%04x, done=%u\n",
               h_result->error_code, h_result->cqe_status, h_result->blocks_done);
        goto cleanup_multi;
    }

    if (h_result->blocks_done != num_blocks) {
        printf("[FAIL] Expected %u blocks done, got %u\n",
               num_blocks, h_result->blocks_done);
        goto cleanup_multi;
    }

    /* Verify all blocks */
    {
        uint32_t total_bytes = num_blocks * cfg.block_size;
        uint8_t *expected = (uint8_t *)malloc(total_bytes);
        nvme_sim_direct_read(sim, 0, num_blocks, expected);

        uint8_t *actual = (uint8_t *)h_queue->data_buf;
        if (memcmp(actual, expected, total_bytes) != 0) {
            printf("[FAIL] Data mismatch in multi-block read!\n");
            for (uint32_t i = 0; i < total_bytes; i++) {
                if (actual[i] != expected[i]) {
                    printf("    First diff at byte %u (block %u, offset %u): "
                           "expected 0x%02x, got 0x%02x\n",
                           i, i / cfg.block_size, i % cfg.block_size,
                           expected[i], actual[i]);
                    break;
                }
            }
            free(expected);
            goto cleanup_multi;
        }
        free(expected);
    }

    printf("[PASS]\n");
    tests_passed++;

cleanup_multi:
    cudaFreeHost(h_result);
    cudaFreeHost(h_queue);
    nvme_sim_destroy(sim);
}

static void test_4k_block_read() {
    printf("  TEST: 4KB block size read...                 ");
    tests_run++;

    nvme_sim_config_t cfg;
    cfg.num_blocks = 256;
    cfg.block_size = 4096;
    cfg.sq_size = 32;
    cfg.cq_size = 32;
    cfg.latency_us = 10;

    nvme_sim_t *sim = nvme_sim_create(&cfg);
    if (!sim) {
        printf("[FAIL] simulator creation failed\n");
        return;
    }

    nvme_sim_fill_blocks(sim, 0, 1, fill_pattern);

    gpu_nvme_queue *h_queue;
    CUDA_CHECK(cudaMallocHost(&h_queue, sizeof(gpu_nvme_queue)));
    memset(h_queue, 0, sizeof(gpu_nvme_queue));

    h_queue->sq = nvme_sim_get_sq(sim);
    h_queue->cq = nvme_sim_get_cq(sim);
    h_queue->doorbell_sq = nvme_sim_get_sq_doorbell(sim);
    h_queue->doorbell_cq = nvme_sim_get_cq_doorbell(sim);
    h_queue->data_buf = nvme_sim_get_data_buf(sim);
    h_queue->data_buf_phys = nvme_sim_get_data_buf_phys(sim);
    h_queue->sq_size = nvme_sim_get_sq_size(sim);
    h_queue->cq_size = nvme_sim_get_cq_size(sim);
    h_queue->sq_tail = 0;
    h_queue->cq_head = 0;
    h_queue->qid = 1;
    h_queue->cq_phase = 1;
    h_queue->cid_counter = 0;
    h_queue->nsid = 1;
    h_queue->block_size = cfg.block_size;
    h_queue->poll_timeout_cycles = 500000000ULL;

    gpunvme_io_result *h_result;
    CUDA_CHECK(cudaMallocHost(&h_result, sizeof(gpunvme_io_result)));
    memset(h_result, 0, sizeof(gpunvme_io_result));

    test_read_single_block<<<1, 1>>>(h_queue, h_queue->data_buf_phys, h_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (h_result->status != 0) {
        printf("[FAIL] error_code=%u\n", h_result->error_code);
        goto cleanup_4k;
    }

    {
        uint8_t expected[4096];
        nvme_sim_direct_read(sim, 0, 1, expected);
        uint8_t *actual = (uint8_t *)h_queue->data_buf;
        if (memcmp(actual, expected, 4096) != 0) {
            printf("[FAIL] 4KB data mismatch\n");
            goto cleanup_4k;
        }
    }

    printf("[PASS]\n");
    tests_passed++;

cleanup_4k:
    cudaFreeHost(h_result);
    cudaFreeHost(h_queue);
    nvme_sim_destroy(sim);
}

int main() {
    printf("=== GPU NVMe Simulator Tests ===\n\n");

    /* Check CUDA device */
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found. Skipping GPU tests.\n");
        return 0;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    test_single_block_read();
    test_multi_block_read();
    test_4k_block_read();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
