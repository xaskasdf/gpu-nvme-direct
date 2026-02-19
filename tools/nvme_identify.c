/*
 * gpu-nvme-direct: NVMe Identify Tool
 *
 * Standalone tool that initializes the NVMe controller and prints
 * Identify Controller and Identify Namespace data.
 *
 * Usage: sudo ./nvme_identify <PCI_BDF>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include <gpunvme/nvme_regs.h>
#include <gpunvme/nvme_cmds.h>
#include <gpunvme/mmio.h>
#include <gpunvme/error.h>
#include <gpunvme/controller.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <PCI_BDF>\n", argv[0]);
        fprintf(stderr, "  e.g.: %s 0000:03:00.0\n", argv[0]);
        return 1;
    }

    const char *bdf = argv[1];
    char path[256];

    printf("=== NVMe Identify: %s ===\n\n", bdf);

    /* Map BAR0 */
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource0", bdf);
    int fd = open(path, O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open resource0");
        return 1;
    }

    off_t bar_size = lseek(fd, 0, SEEK_END);
    volatile void *bar0 = mmap(NULL, bar_size, PROT_READ | PROT_WRITE,
                                MAP_SHARED, fd, 0);
    if (bar0 == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    /* Initialize controller */
    gpunvme_ctrl_t ctrl;
    gpunvme_err_t err = gpunvme_ctrl_init(&ctrl, bar0, bar_size);

    if (err != GPUNVME_OK) {
        fprintf(stderr, "Controller init failed: %s\n", gpunvme_err_str(err));
        munmap((void *)bar0, bar_size);
        close(fd);
        return 1;
    }

    /* Print results */
    printf("Controller Information:\n");
    printf("  Model:        %s\n", ctrl.model);
    printf("  Serial:       %s\n", ctrl.serial);
    printf("  Firmware:     %s\n", ctrl.firmware);
    printf("  Max Queues:   %u entries\n", ctrl.max_queue_entries);
    printf("  Page Size:    %u bytes\n", ctrl.page_size);
    printf("  DB Stride:    %u\n", ctrl.dstrd);
    printf("\n");
    printf("Namespace 1:\n");
    printf("  Size:         %u blocks\n", ctrl.ns_size_blocks);
    printf("  Block Size:   %u bytes\n", ctrl.block_size);
    if (ctrl.ns_size_blocks > 0 && ctrl.block_size > 0) {
        uint64_t total_mb = (uint64_t)ctrl.ns_size_blocks * ctrl.block_size / (1024 * 1024);
        printf("  Capacity:     %llu MB (%.1f GB)\n",
               (unsigned long long)total_mb, total_mb / 1024.0);
    }

    /* Shutdown */
    gpunvme_ctrl_shutdown(&ctrl);

    munmap((void *)bar0, bar_size);
    close(fd);

    printf("\nDone.\n");
    return 0;
}
