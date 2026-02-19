/*
 * gpu-nvme-direct: Dump NVMe BAR0 Registers
 *
 * Reads and prints all standard NVMe registers from BAR0.
 * CPU-side only, no CUDA needed. Useful for verification.
 *
 * Usage: sudo ./dump_bar0 <PCI_BDF>
 *   e.g.: sudo ./dump_bar0 0000:03:00.0
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdint.h>

#include <gpunvme/nvme_regs.h>
#include <gpunvme/mmio.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <PCI_BDF>\n", argv[0]);
        fprintf(stderr, "  e.g.: %s 0000:03:00.0\n", argv[0]);
        return 1;
    }

    const char *bdf = argv[1];
    char path[256];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource0", bdf);

    int fd = open(path, O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open resource0");
        fprintf(stderr, "Need root and device bound to vfio-pci or no driver.\n");
        return 1;
    }

    off_t size = lseek(fd, 0, SEEK_END);
    if (size <= 0) {
        fprintf(stderr, "Cannot determine BAR0 size\n");
        close(fd);
        return 1;
    }

    volatile void *bar0 = mmap(NULL, size, PROT_READ | PROT_WRITE,
                                MAP_SHARED, fd, 0);
    if (bar0 == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    printf("=== NVMe BAR0 Register Dump for %s ===\n", bdf);
    printf("BAR0 size: %ld bytes (0x%lx)\n\n", (long)size, (long)size);

    /* CAP — Controller Capabilities (64-bit, offset 0x00) */
    nvme_cap_t cap;
    cap.raw = host_mmio_read64(nvme_reg_ptr(bar0, NVME_REG_CAP));
    printf("CAP (0x%02X) = 0x%016llx\n", NVME_REG_CAP, (unsigned long long)cap.raw);
    printf("  MQES  = %u (max queue entries = %u)\n", cap.bits.mqes, cap.bits.mqes + 1);
    printf("  CQR   = %u\n", cap.bits.cqr);
    printf("  AMS   = %u\n", cap.bits.ams);
    printf("  TO    = %u (%u ms)\n", cap.bits.to, cap.bits.to * 500);
    printf("  DSTRD = %u (stride = %u bytes)\n", cap.bits.dstrd, 4 << cap.bits.dstrd);
    printf("  NSSRS = %u\n", cap.bits.nssrs);
    printf("  CSS   = 0x%02x\n", cap.bits.css);
    printf("  MPSMIN= %u (min page = %u bytes)\n", cap.bits.mpsmin, 1 << (12 + cap.bits.mpsmin));
    printf("  MPSMAX= %u (max page = %u bytes)\n", cap.bits.mpsmax, 1 << (12 + cap.bits.mpsmax));
    printf("  PMRS  = %u\n", cap.bits.pmrs);
    printf("  CMBS  = %u\n", cap.bits.cmbs);

    /* VS — Version (32-bit, offset 0x08) */
    nvme_vs_t vs;
    vs.raw = host_mmio_read32(nvme_reg_ptr(bar0, NVME_REG_VS));
    printf("\nVS  (0x%02X) = 0x%08x\n", NVME_REG_VS, vs.raw);
    printf("  Version: %u.%u.%u\n", vs.bits.mjr, vs.bits.mnr, vs.bits.ter);

    /* CC — Controller Configuration (32-bit, offset 0x14) */
    nvme_cc_t cc;
    cc.raw = host_mmio_read32(nvme_reg_ptr(bar0, NVME_REG_CC));
    printf("\nCC  (0x%02X) = 0x%08x\n", NVME_REG_CC, cc.raw);
    printf("  EN     = %u\n", cc.bits.en);
    printf("  CSS    = %u\n", cc.bits.css);
    printf("  MPS    = %u (page = %u bytes)\n", cc.bits.mps, 1 << (12 + cc.bits.mps));
    printf("  AMS    = %u\n", cc.bits.ams);
    printf("  SHN    = %u\n", cc.bits.shn);
    printf("  IOSQES = %u (%u bytes)\n", cc.bits.iosqes, 1 << cc.bits.iosqes);
    printf("  IOCQES = %u (%u bytes)\n", cc.bits.iocqes, 1 << cc.bits.iocqes);

    /* CSTS — Controller Status (32-bit, offset 0x1C) */
    nvme_csts_t csts;
    csts.raw = host_mmio_read32(nvme_reg_ptr(bar0, NVME_REG_CSTS));
    printf("\nCSTS(0x%02X) = 0x%08x\n", NVME_REG_CSTS, csts.raw);
    printf("  RDY   = %u\n", csts.bits.rdy);
    printf("  CFS   = %u%s\n", csts.bits.cfs, csts.bits.cfs ? " *** FATAL ***" : "");
    printf("  SHST  = %u\n", csts.bits.shst);
    printf("  NSSRO = %u\n", csts.bits.nssro);
    printf("  PP    = %u\n", csts.bits.pp);

    /* AQA — Admin Queue Attributes (32-bit, offset 0x24) */
    nvme_aqa_t aqa;
    aqa.raw = host_mmio_read32(nvme_reg_ptr(bar0, NVME_REG_AQA));
    printf("\nAQA (0x%02X) = 0x%08x\n", NVME_REG_AQA, aqa.raw);
    printf("  ASQS = %u (admin SQ size = %u)\n", aqa.bits.asqs, aqa.bits.asqs + 1);
    printf("  ACQS = %u (admin CQ size = %u)\n", aqa.bits.acqs, aqa.bits.acqs + 1);

    /* ASQ — Admin SQ Base Address (64-bit, offset 0x28) */
    uint64_t asq = host_mmio_read64(nvme_reg_ptr(bar0, NVME_REG_ASQ));
    printf("\nASQ (0x%02X) = 0x%016llx\n", NVME_REG_ASQ, (unsigned long long)asq);

    /* ACQ — Admin CQ Base Address (64-bit, offset 0x30) */
    uint64_t acq = host_mmio_read64(nvme_reg_ptr(bar0, NVME_REG_ACQ));
    printf("\nACQ (0x%02X) = 0x%016llx\n", NVME_REG_ACQ, (unsigned long long)acq);

    /* Doorbell region info */
    printf("\nDoorbell base: 0x%04X\n", NVME_REG_DOORBELL_BASE);
    printf("  Admin SQ Tail DB offset: 0x%04X\n", nvme_sq_doorbell_offset(0, cap.bits.dstrd));
    printf("  Admin CQ Head DB offset: 0x%04X\n", nvme_cq_doorbell_offset(0, cap.bits.dstrd));
    printf("  IO Q1 SQ Tail DB offset: 0x%04X\n", nvme_sq_doorbell_offset(1, cap.bits.dstrd));
    printf("  IO Q1 CQ Head DB offset: 0x%04X\n", nvme_cq_doorbell_offset(1, cap.bits.dstrd));

    munmap((void *)bar0, size);
    close(fd);
    return 0;
}
