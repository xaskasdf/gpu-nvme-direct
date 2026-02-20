/*
 * gpu-nvme-direct: Layer Loader API Test
 *
 * Validates the layer loader API with three tests:
 *   1. Load LBA 0 into buf_a — basic read
 *   2. Load next region into buf_b — PRP rebuild with different dest
 *   3. Re-load different LBA into buf_a — queue continuity + buffer reuse
 *
 * Usage: sudo ./test_layer_loader <PCI_BDF> [size_mb]
 *   size_mb: read size per test in MB (default: 4)
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#include <gpunvme/layer_loader.h>
#include <gpunvme/error.h>

static int verify_not_pattern(const void *buf, size_t size, uint8_t pattern) {
    const uint8_t *p = (const uint8_t *)buf;
    /* Check first, middle, last 16 bytes */
    size_t offsets[] = {0, size / 2, size - 16};
    for (int i = 0; i < 3; i++) {
        int all_pattern = 1;
        for (int j = 0; j < 16; j++) {
            if (p[offsets[i] + j] != pattern) { all_pattern = 0; break; }
        }
        if (all_pattern) {
            fprintf(stderr, "  FAIL: data at offset %zu still 0x%02X\n",
                    offsets[i], pattern);
            return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <PCI_BDF> [size_mb]\n", argv[0]);
        return 1;
    }

    const char *bdf = argv[1];
    uint32_t size_mb = (argc > 2) ? atoi(argv[2]) : 4;
    size_t size_bytes = (size_t)size_mb * 1024 * 1024;

    printf("=== Layer Loader API Test ===\n");
    printf("Device: %s, size per test: %u MB\n\n", bdf, size_mb);

    /* Initialize loader */
    gpunvme_layer_loader_t loader;
    gpunvme_err_t err = gpunvme_layer_loader_init(&loader, bdf, size_bytes, 32);
    if (err != GPUNVME_OK) {
        fprintf(stderr, "Init failed: %s\n", gpunvme_err_str(err));
        return 1;
    }

    uint32_t block_size = gpunvme_layer_loader_block_size(&loader);
    uint32_t max_transfer = gpunvme_layer_loader_max_transfer(&loader);
    uint64_t ns_blocks = gpunvme_layer_loader_ns_blocks(&loader);

    printf("Block size: %u, MDTS: %u KB, namespace: %lu blocks\n\n",
           block_size, max_transfer / 1024, (unsigned long)ns_blocks);

    /* Clamp size to device capacity */
    uint32_t total_blocks = (uint32_t)(size_bytes / block_size);
    if (total_blocks > ns_blocks / 3) {
        /* Need 3x worth of blocks for 3 tests at different LBAs */
        total_blocks = (uint32_t)(ns_blocks / 3);
        size_bytes = (size_t)total_blocks * block_size;
        size_mb = (uint32_t)(size_bytes / (1024 * 1024));
        printf("Clamped to %u MB to fit 3 test regions\n\n", size_mb);
    }

    /* Compute LBA offsets for each test region */
    uint32_t blocks_per_test = total_blocks;
    uint64_t lba_0 = 0;
    uint64_t lba_1 = (uint64_t)blocks_per_test;
    uint64_t lba_2 = (uint64_t)blocks_per_test * 2;

    /* Allocate two destination buffers */
    void *buf_a, *buf_b;
    if (cudaMallocHost(&buf_a, size_bytes) != cudaSuccess ||
        cudaMallocHost(&buf_b, size_bytes) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate %zu byte buffers\n", size_bytes);
        gpunvme_layer_loader_destroy(&loader);
        return 1;
    }

    int passed = 0;
    int failed = 0;

    /* Test 1: Load LBA 0 → buf_a */
    printf("--- Test 1: Load LBA 0 → buf_a (%u MB) ---\n", size_mb);
    memset(buf_a, 0xDE, size_bytes);
    err = gpunvme_load_layer(&loader, lba_0, size_bytes, buf_a);
    if (err != GPUNVME_OK) {
        printf("  FAIL: %s\n\n", gpunvme_err_str(err));
        failed++;
    } else if (!verify_not_pattern(buf_a, size_bytes, 0xDE)) {
        printf("  FAIL: data not overwritten\n\n");
        failed++;
    } else {
        printf("  PASS\n\n");
        passed++;
    }

    /* Test 2: Load next region → buf_b (different destination) */
    printf("--- Test 2: Load LBA %lu → buf_b (%u MB) ---\n",
           (unsigned long)lba_1, size_mb);
    memset(buf_b, 0xAB, size_bytes);
    err = gpunvme_load_layer(&loader, lba_1, size_bytes, buf_b);
    if (err != GPUNVME_OK) {
        printf("  FAIL: %s\n\n", gpunvme_err_str(err));
        failed++;
    } else if (!verify_not_pattern(buf_b, size_bytes, 0xAB)) {
        printf("  FAIL: data not overwritten\n\n");
        failed++;
    } else {
        /* Verify buf_a wasn't corrupted by test 2 */
        if (!verify_not_pattern(buf_a, size_bytes, 0xDE)) {
            printf("  FAIL: buf_a corrupted during test 2\n\n");
            failed++;
        } else {
            printf("  PASS\n\n");
            passed++;
        }
    }

    /* Test 3: Re-load different LBA → buf_a (queue continuity + reuse) */
    printf("--- Test 3: Load LBA %lu → buf_a (%u MB, reuse) ---\n",
           (unsigned long)lba_2, size_mb);
    memset(buf_a, 0xCD, size_bytes);
    err = gpunvme_load_layer(&loader, lba_2, size_bytes, buf_a);
    if (err != GPUNVME_OK) {
        printf("  FAIL: %s\n\n", gpunvme_err_str(err));
        failed++;
    } else if (!verify_not_pattern(buf_a, size_bytes, 0xCD)) {
        printf("  FAIL: data not overwritten\n\n");
        failed++;
    } else {
        printf("  PASS\n\n");
        passed++;
    }

    /* Summary */
    printf("=== Results: %d/3 passed", passed);
    if (failed > 0) printf(", %d failed", failed);
    printf(" ===\n");

    cudaFreeHost(buf_a);
    cudaFreeHost(buf_b);
    gpunvme_layer_loader_destroy(&loader);

    return (failed > 0) ? 1 : 0;
}
