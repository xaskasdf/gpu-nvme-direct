/*
 * gpu-nvme-direct: NVMe Register Definitions
 *
 * NVMe PCIe Transport register layout per NVMe Base Specification 2.0.
 * All structs are packed and sized with static assertions.
 *
 * BAR0 Register Map:
 *   0x00  CAP    Controller Capabilities (64-bit)
 *   0x08  VS     Version (32-bit)
 *   0x0C  INTMS  Interrupt Mask Set (32-bit)
 *   0x10  INTMC  Interrupt Mask Clear (32-bit)
 *   0x14  CC     Controller Configuration (32-bit)
 *   0x1C  CSTS   Controller Status (32-bit)
 *   0x20  NSSR   NVM Subsystem Reset (32-bit)
 *   0x24  AQA    Admin Queue Attributes (32-bit)
 *   0x28  ASQ    Admin Submission Queue Base Address (64-bit)
 *   0x30  ACQ    Admin Completion Queue Base Address (64-bit)
 *   0x1000+ Doorbell registers
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#ifndef GPUNVME_NVME_REGS_H
#define GPUNVME_NVME_REGS_H

#include <stdint.h>

/* _Static_assert is C11; C++ uses static_assert */
#ifdef __cplusplus
#define _Static_assert static_assert
extern "C" {
#endif

/* ------- BAR0 Register Offsets ------- */

#define NVME_REG_CAP    0x00  /* Controller Capabilities */
#define NVME_REG_VS     0x08  /* Version */
#define NVME_REG_INTMS  0x0C  /* Interrupt Mask Set */
#define NVME_REG_INTMC  0x10  /* Interrupt Mask Clear */
#define NVME_REG_CC     0x14  /* Controller Configuration */
#define NVME_REG_CSTS   0x1C  /* Controller Status */
#define NVME_REG_NSSR   0x20  /* NVM Subsystem Reset */
#define NVME_REG_AQA    0x24  /* Admin Queue Attributes */
#define NVME_REG_ASQ    0x28  /* Admin Submission Queue Base Address */
#define NVME_REG_ACQ    0x30  /* Admin Completion Queue Base Address */
#define NVME_REG_DOORBELL_BASE 0x1000

/* ------- Controller Capabilities (CAP) — 64-bit ------- */

typedef union {
    uint64_t raw;
    struct {
        uint64_t mqes   : 16;  /* Maximum Queue Entries Supported (0-based) */
        uint64_t cqr    : 1;   /* Contiguous Queues Required */
        uint64_t ams    : 2;   /* Arbitration Mechanism Supported */
        uint64_t rsvd0  : 5;
        uint64_t to     : 8;   /* Timeout (in 500ms units) */
        uint64_t dstrd  : 4;   /* Doorbell Stride (2^(2+DSTRD) bytes) */
        uint64_t nssrs  : 1;   /* NVM Subsystem Reset Supported */
        uint64_t css    : 8;   /* Command Sets Supported */
        uint64_t bps    : 1;   /* Boot Partition Support */
        uint64_t rsvd1  : 2;
        uint64_t mpsmin : 4;   /* Memory Page Size Minimum (2^(12+MPSMIN)) */
        uint64_t mpsmax : 4;   /* Memory Page Size Maximum (2^(12+MPSMAX)) */
        uint64_t pmrs   : 1;   /* Persistent Memory Region Supported */
        uint64_t cmbs   : 1;   /* Controller Memory Buffer Supported */
        uint64_t rsvd2  : 6;
    } bits;
} nvme_cap_t;

_Static_assert(sizeof(nvme_cap_t) == 8, "nvme_cap_t must be 8 bytes");

/* ------- Version (VS) — 32-bit ------- */

typedef union {
    uint32_t raw;
    struct {
        uint32_t ter : 8;   /* Tertiary Version */
        uint32_t mnr : 8;   /* Minor Version */
        uint32_t mjr : 16;  /* Major Version */
    } bits;
} nvme_vs_t;

_Static_assert(sizeof(nvme_vs_t) == 4, "nvme_vs_t must be 4 bytes");

/* ------- Controller Configuration (CC) — 32-bit ------- */

typedef union {
    uint32_t raw;
    struct {
        uint32_t en     : 1;   /* Enable */
        uint32_t rsvd0  : 3;
        uint32_t css    : 3;   /* I/O Command Set Selected */
        uint32_t mps    : 4;   /* Memory Page Size (2^(12+MPS)) */
        uint32_t ams    : 3;   /* Arbitration Mechanism Selected */
        uint32_t shn    : 2;   /* Shutdown Notification */
        uint32_t iosqes : 4;   /* I/O Submission Queue Entry Size (2^IOSQES) */
        uint32_t iocqes : 4;   /* I/O Completion Queue Entry Size (2^IOCQES) */
        uint32_t rsvd1  : 8;
    } bits;
} nvme_cc_t;

_Static_assert(sizeof(nvme_cc_t) == 4, "nvme_cc_t must be 4 bytes");

/* CC.SHN values */
#define NVME_CC_SHN_NONE    0
#define NVME_CC_SHN_NORMAL  1
#define NVME_CC_SHN_ABRUPT  2

/* ------- Controller Status (CSTS) — 32-bit ------- */

typedef union {
    uint32_t raw;
    struct {
        uint32_t rdy   : 1;  /* Ready */
        uint32_t cfs   : 1;  /* Controller Fatal Status */
        uint32_t shst  : 2;  /* Shutdown Status */
        uint32_t nssro : 1;  /* NVM Subsystem Reset Occurred */
        uint32_t pp    : 1;  /* Processing Paused */
        uint32_t rsvd  : 26;
    } bits;
} nvme_csts_t;

_Static_assert(sizeof(nvme_csts_t) == 4, "nvme_csts_t must be 4 bytes");

/* CSTS.SHST values */
#define NVME_CSTS_SHST_NONE      0
#define NVME_CSTS_SHST_OCCURRING 1
#define NVME_CSTS_SHST_COMPLETE  2

/* ------- Admin Queue Attributes (AQA) — 32-bit ------- */

typedef union {
    uint32_t raw;
    struct {
        uint32_t asqs : 12;  /* Admin Submission Queue Size (0-based) */
        uint32_t rsvd0 : 4;
        uint32_t acqs : 12;  /* Admin Completion Queue Size (0-based) */
        uint32_t rsvd1 : 4;
    } bits;
} nvme_aqa_t;

_Static_assert(sizeof(nvme_aqa_t) == 4, "nvme_aqa_t must be 4 bytes");

/* ------- Submission Queue Entry — 64 bytes ------- */

typedef struct {
    /* Command Dword 0 */
    uint32_t opc    : 8;   /* Opcode */
    uint32_t fuse   : 2;   /* Fused Operation */
    uint32_t rsvd0  : 4;
    uint32_t psdt   : 2;   /* PRP or SGL for Data Transfer */
    uint32_t cid    : 16;  /* Command Identifier */

    /* Command Dword 1 */
    uint32_t nsid;          /* Namespace Identifier */

    /* Command Dwords 2-3: Reserved */
    uint32_t cdw2;
    uint32_t cdw3;

    /* Metadata Pointer */
    uint64_t mptr;

    /* Data Pointer: PRP Entry 1 and PRP Entry 2 (or SGL) */
    uint64_t prp1;
    uint64_t prp2;

    /* Command-specific Dwords 10-15 */
    uint32_t cdw10;
    uint32_t cdw11;
    uint32_t cdw12;
    uint32_t cdw13;
    uint32_t cdw14;
    uint32_t cdw15;
} nvme_sq_entry_t;

_Static_assert(sizeof(nvme_sq_entry_t) == 64, "nvme_sq_entry_t must be 64 bytes");

/* ------- Completion Queue Entry — 16 bytes ------- */

typedef struct {
    uint32_t cdw0;          /* Command-specific result */
    uint32_t cdw1;          /* Reserved */
    uint16_t sqhd;          /* SQ Head Pointer */
    uint16_t sqid;          /* SQ Identifier */
    uint16_t cid;           /* Command Identifier */
    uint16_t status_phase;  /* Status Field [15:1] and Phase Tag [0] */
} nvme_cq_entry_t;

_Static_assert(sizeof(nvme_cq_entry_t) == 16, "nvme_cq_entry_t must be 16 bytes");

/* Extract status code from CQE status_phase field */
#define NVME_CQE_PHASE(sp)       ((sp) & 0x1)
#define NVME_CQE_STATUS(sp)      (((sp) >> 1) & 0x7FFF)
#define NVME_CQE_SC(sp)          (((sp) >> 1) & 0xFF)     /* Status Code */
#define NVME_CQE_SCT(sp)         (((sp) >> 9) & 0x7)      /* Status Code Type */
#define NVME_CQE_MORE(sp)        (((sp) >> 14) & 0x1)     /* More */
#define NVME_CQE_DNR(sp)         (((sp) >> 15) & 0x1)     /* Do Not Retry */

/* ------- Doorbell Offset Calculation ------- */

/*
 * SQ Y Tail Doorbell = 0x1000 + ((2*Y)     * (4 << DSTRD))
 * CQ Y Head Doorbell = 0x1000 + ((2*Y + 1) * (4 << DSTRD))
 *
 * Most controllers use DSTRD=0, giving stride=4 bytes.
 */

static inline uint32_t nvme_sq_doorbell_offset(uint16_t qid, uint8_t dstrd) {
    return NVME_REG_DOORBELL_BASE + ((2u * qid) * (4u << dstrd));
}

static inline uint32_t nvme_cq_doorbell_offset(uint16_t qid, uint8_t dstrd) {
    return NVME_REG_DOORBELL_BASE + (((2u * qid) + 1u) * (4u << dstrd));
}

/* ------- Convenience: byte offset to pointer ------- */

static inline volatile void *nvme_reg_ptr(volatile void *bar0, uint32_t offset) {
    return (volatile void *)((volatile uint8_t *)bar0 + offset);
}

#ifdef __cplusplus
}
#endif

#endif /* GPUNVME_NVME_REGS_H */
