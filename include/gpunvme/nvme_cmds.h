/*
 * gpu-nvme-direct: NVMe Command Definitions and Builder Helpers
 *
 * Opcodes and helpers for building NVMe admin and I/O commands.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#ifndef GPUNVME_NVME_CMDS_H
#define GPUNVME_NVME_CMDS_H

#include <string.h>
#include "nvme_regs.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------- Admin Command Opcodes ------- */

#define NVME_ADMIN_OPC_DELETE_IO_SQ  0x00
#define NVME_ADMIN_OPC_CREATE_IO_SQ  0x01
#define NVME_ADMIN_OPC_GET_LOG_PAGE  0x02
#define NVME_ADMIN_OPC_DELETE_IO_CQ  0x04
#define NVME_ADMIN_OPC_CREATE_IO_CQ  0x05
#define NVME_ADMIN_OPC_IDENTIFY      0x06
#define NVME_ADMIN_OPC_ABORT         0x08
#define NVME_ADMIN_OPC_SET_FEATURES  0x09
#define NVME_ADMIN_OPC_GET_FEATURES  0x0A

/* ------- I/O Command Opcodes (NVM Command Set) ------- */

#define NVME_IO_OPC_FLUSH   0x00
#define NVME_IO_OPC_WRITE   0x01
#define NVME_IO_OPC_READ    0x02

/* ------- Identify CNS Values ------- */

#define NVME_IDENTIFY_CNS_NAMESPACE  0x00
#define NVME_IDENTIFY_CNS_CONTROLLER 0x01

/* ------- Create I/O CQ Flags (CDW11) ------- */

#define NVME_CQ_PC         (1u << 0)  /* Physically Contiguous */
#define NVME_CQ_IEN        (1u << 1)  /* Interrupts Enabled */

/* ------- Create I/O SQ Flags (CDW11) ------- */

#define NVME_SQ_PC         (1u << 0)  /* Physically Contiguous */

/* ------- Command Builders ------- */

/* Zero out an SQ entry */
static inline void nvme_cmd_init(nvme_sq_entry_t *cmd) {
    memset(cmd, 0, sizeof(*cmd));
}

/*
 * Build an Identify Controller command.
 * PRP1 must point to a 4KB DMA-accessible buffer.
 */
static inline void nvme_cmd_identify_controller(nvme_sq_entry_t *cmd,
                                                 uint16_t cid,
                                                 uint64_t prp1_phys) {
    nvme_cmd_init(cmd);
    cmd->opc  = NVME_ADMIN_OPC_IDENTIFY;
    cmd->cid  = cid;
    cmd->nsid = 0;
    cmd->prp1 = prp1_phys;
    cmd->cdw10 = NVME_IDENTIFY_CNS_CONTROLLER;
}

/*
 * Build an Identify Namespace command.
 * PRP1 must point to a 4KB DMA-accessible buffer.
 */
static inline void nvme_cmd_identify_namespace(nvme_sq_entry_t *cmd,
                                                uint16_t cid,
                                                uint32_t nsid,
                                                uint64_t prp1_phys) {
    nvme_cmd_init(cmd);
    cmd->opc  = NVME_ADMIN_OPC_IDENTIFY;
    cmd->cid  = cid;
    cmd->nsid = nsid;
    cmd->prp1 = prp1_phys;
    cmd->cdw10 = NVME_IDENTIFY_CNS_NAMESPACE;
}

/*
 * Build a Create I/O Completion Queue command.
 * PRP1 = physical address of CQ memory.
 * CDW10: QSIZE (15:0 = size-1) | QID (31:16) — but spec says QID in 15:0, QSIZE in 31:16
 *   Actually NVMe spec 1.4+:
 *     CDW10[15:0]  = Queue Identifier (QID)
 *     CDW10[31:16] = Queue Size (0-based)
 *   CDW11[0] = PC (Physically Contiguous)
 *   CDW11[1] = IEN (Interrupt Enable)
 *   CDW11[31:16] = Interrupt Vector
 */
static inline void nvme_cmd_create_io_cq(nvme_sq_entry_t *cmd,
                                          uint16_t cid,
                                          uint16_t qid,
                                          uint16_t qsize_0based,
                                          uint64_t cq_phys,
                                          uint16_t iv) {
    nvme_cmd_init(cmd);
    cmd->opc  = NVME_ADMIN_OPC_CREATE_IO_CQ;
    cmd->cid  = cid;
    cmd->prp1 = cq_phys;
    cmd->cdw10 = ((uint32_t)qsize_0based << 16) | qid;
    cmd->cdw11 = NVME_CQ_PC | ((uint32_t)iv << 16);  /* PC=1, no interrupts for polling */
}

/*
 * Build a Create I/O Submission Queue command.
 * CDW10[15:0]  = QID
 * CDW10[31:16] = Queue Size (0-based)
 * CDW11[0]     = PC (Physically Contiguous)
 * CDW11[31:16] = CQID (Completion Queue Identifier)
 */
static inline void nvme_cmd_create_io_sq(nvme_sq_entry_t *cmd,
                                          uint16_t cid,
                                          uint16_t qid,
                                          uint16_t qsize_0based,
                                          uint64_t sq_phys,
                                          uint16_t cqid) {
    nvme_cmd_init(cmd);
    cmd->opc  = NVME_ADMIN_OPC_CREATE_IO_SQ;
    cmd->cid  = cid;
    cmd->prp1 = sq_phys;
    cmd->cdw10 = ((uint32_t)qsize_0based << 16) | qid;
    cmd->cdw11 = NVME_SQ_PC | ((uint32_t)cqid << 16);
}

/*
 * Build an NVMe Read command.
 *
 * prp1_phys:  Physical/DMA address of destination data buffer
 * prp2_phys:  Second PRP entry or PRP list address (for transfers > 1 page)
 * slba:       Starting Logical Block Address
 * nlb:        Number of Logical Blocks minus 1 (0-based)
 */
static inline void nvme_cmd_read(nvme_sq_entry_t *cmd,
                                  uint16_t cid,
                                  uint32_t nsid,
                                  uint64_t prp1_phys,
                                  uint64_t prp2_phys,
                                  uint64_t slba,
                                  uint16_t nlb_0based) {
    nvme_cmd_init(cmd);
    cmd->opc  = NVME_IO_OPC_READ;
    cmd->cid  = cid;
    cmd->nsid = nsid;
    cmd->prp1 = prp1_phys;
    cmd->prp2 = prp2_phys;
    cmd->cdw10 = (uint32_t)(slba & 0xFFFFFFFF);
    cmd->cdw11 = (uint32_t)(slba >> 32);
    cmd->cdw12 = nlb_0based;  /* bits 15:0 = NLB (0-based) */
}

/*
 * Build an NVMe Write command.
 */
static inline void nvme_cmd_write(nvme_sq_entry_t *cmd,
                                   uint16_t cid,
                                   uint32_t nsid,
                                   uint64_t prp1_phys,
                                   uint64_t prp2_phys,
                                   uint64_t slba,
                                   uint16_t nlb_0based) {
    nvme_cmd_init(cmd);
    cmd->opc  = NVME_IO_OPC_WRITE;
    cmd->cid  = cid;
    cmd->nsid = nsid;
    cmd->prp1 = prp1_phys;
    cmd->prp2 = prp2_phys;
    cmd->cdw10 = (uint32_t)(slba & 0xFFFFFFFF);
    cmd->cdw11 = (uint32_t)(slba >> 32);
    cmd->cdw12 = nlb_0based;
}

/*
 * Build a Flush command.
 */
static inline void nvme_cmd_flush(nvme_sq_entry_t *cmd,
                                   uint16_t cid,
                                   uint32_t nsid) {
    nvme_cmd_init(cmd);
    cmd->opc  = NVME_IO_OPC_FLUSH;
    cmd->cid  = cid;
    cmd->nsid = nsid;
}

/*
 * Build a Delete I/O Submission Queue command.
 */
static inline void nvme_cmd_delete_io_sq(nvme_sq_entry_t *cmd,
                                          uint16_t cid,
                                          uint16_t qid) {
    nvme_cmd_init(cmd);
    cmd->opc  = NVME_ADMIN_OPC_DELETE_IO_SQ;
    cmd->cid  = cid;
    cmd->cdw10 = qid;
}

/*
 * Build a Delete I/O Completion Queue command.
 */
static inline void nvme_cmd_delete_io_cq(nvme_sq_entry_t *cmd,
                                          uint16_t cid,
                                          uint16_t qid) {
    nvme_cmd_init(cmd);
    cmd->opc  = NVME_ADMIN_OPC_DELETE_IO_CQ;
    cmd->cid  = cid;
    cmd->cdw10 = qid;
}

#ifdef __cplusplus
}
#endif

#endif /* GPUNVME_NVME_CMDS_H */
