/*
 * gpu-nvme-direct: NVMe Controller Initialization
 *
 * Full controller init/shutdown sequence per NVMe specification.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <gpunvme/controller.h>
#include <gpunvme/dma.h>
#include <gpunvme/mmio.h>
#include <gpunvme/nvme_regs.h>
#include <gpunvme/nvme_cmds.h>
#include <gpunvme/error.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

/* Helper: sleep for milliseconds */
static void sleep_ms(uint32_t ms) {
    struct timespec ts = { ms / 1000, (ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}

/* Wait for CSTS.RDY to reach target value, with timeout */
static gpunvme_err_t wait_csts_rdy(volatile void *bar0, int target, uint32_t timeout_ms) {
    for (uint32_t i = 0; i < timeout_ms; i++) {
        nvme_csts_t csts;
        csts.raw = host_mmio_read32(nvme_reg_ptr(bar0, NVME_REG_CSTS));

        if (csts.bits.cfs) {
            fprintf(stderr, "ctrl: Controller Fatal Status during RDY wait\n");
            return GPUNVME_ERR_CTRL_FATAL;
        }

        if (csts.bits.rdy == (unsigned)target) return GPUNVME_OK;
        sleep_ms(1);
    }

    fprintf(stderr, "ctrl: Timeout waiting for CSTS.RDY=%d\n", target);
    return GPUNVME_ERR_TIMEOUT;
}

gpunvme_err_t gpunvme_ctrl_init(gpunvme_ctrl_t *ctrl,
                                 volatile void *bar0,
                                 size_t bar0_size) {
    if (!ctrl || !bar0) return GPUNVME_ERR_INVALID_PARAM;

    memset(ctrl, 0, sizeof(*ctrl));
    ctrl->bar0 = bar0;
    ctrl->bar0_size = bar0_size;

    /* 1. Read Controller Capabilities */
    ctrl->cap.raw = host_mmio_read64(nvme_reg_ptr(bar0, NVME_REG_CAP));
    ctrl->dstrd = ctrl->cap.bits.dstrd;
    ctrl->timeout_ms = ctrl->cap.bits.to * 500;
    ctrl->max_queue_entries = ctrl->cap.bits.mqes + 1;
    ctrl->page_size = 1u << (12 + ctrl->cap.bits.mpsmin);

    fprintf(stderr, "ctrl: CAP: MQES=%u, DSTRD=%u, TO=%ums, page=%uB\n",
            ctrl->max_queue_entries, ctrl->dstrd, ctrl->timeout_ms, ctrl->page_size);

    /* 2. Disable controller (CC.EN=0) */
    nvme_cc_t cc;
    cc.raw = host_mmio_read32(nvme_reg_ptr(bar0, NVME_REG_CC));
    if (cc.bits.en) {
        cc.bits.en = 0;
        host_mmio_write32(nvme_reg_ptr(bar0, NVME_REG_CC), cc.raw);

        /* 3. Wait CSTS.RDY=0 */
        gpunvme_err_t err = wait_csts_rdy(bar0, 0, ctrl->timeout_ms);
        if (err != GPUNVME_OK) return err;
    }

    /* 4. Allocate Admin SQ/CQ in host pinned memory.
     * NVMe spec requires ASQ and ACQ to be page-aligned (bits 11:0 = 0).
     * cudaMallocHost may pack small allocations into the same page,
     * so we allocate at least one full page (4096) for each queue. */
    ctrl->admin_sq_size = 32;
    ctrl->admin_cq_size = 32;

    size_t sq_bytes = ctrl->admin_sq_size * sizeof(nvme_sq_entry_t);
    size_t cq_bytes = ctrl->admin_cq_size * sizeof(nvme_cq_entry_t);
    if (sq_bytes < 4096) sq_bytes = 4096;
    if (cq_bytes < 4096) cq_bytes = 4096;

    if (cudaMallocHost((void **)&ctrl->admin_sq, sq_bytes) != cudaSuccess) {
        return GPUNVME_ERR_NOMEM;
    }
    memset((void *)ctrl->admin_sq, 0, sq_bytes);

    if (cudaMallocHost((void **)&ctrl->admin_cq, cq_bytes) != cudaSuccess) {
        cudaFreeHost((void *)ctrl->admin_sq);
        return GPUNVME_ERR_NOMEM;
    }
    memset((void *)ctrl->admin_cq, 0, cq_bytes);

    /* Get physical addresses for admin queues via /proc/self/pagemap */
    gpunvme_err_t phys_err;
    phys_err = gpunvme_virt_to_phys((void *)ctrl->admin_sq, &ctrl->admin_sq_phys);
    if (phys_err != GPUNVME_OK) {
        fprintf(stderr, "ctrl: Failed to resolve admin SQ physical address (need root?)\n");
        return phys_err;
    }
    phys_err = gpunvme_virt_to_phys((void *)ctrl->admin_cq, &ctrl->admin_cq_phys);
    if (phys_err != GPUNVME_OK) {
        fprintf(stderr, "ctrl: Failed to resolve admin CQ physical address (need root?)\n");
        return phys_err;
    }
    fprintf(stderr, "ctrl: Admin SQ phys=0x%lx, CQ phys=0x%lx\n",
            (unsigned long)ctrl->admin_sq_phys, (unsigned long)ctrl->admin_cq_phys);

    /* 5. Set AQA (Admin Queue Attributes) */
    nvme_aqa_t aqa;
    aqa.raw = 0;
    aqa.bits.asqs = ctrl->admin_sq_size - 1;
    aqa.bits.acqs = ctrl->admin_cq_size - 1;
    host_mmio_write32(nvme_reg_ptr(bar0, NVME_REG_AQA), aqa.raw);

    /* Set ASQ and ACQ base addresses */
    host_mmio_write64(nvme_reg_ptr(bar0, NVME_REG_ASQ), ctrl->admin_sq_phys);
    host_mmio_write64(nvme_reg_ptr(bar0, NVME_REG_ACQ), ctrl->admin_cq_phys);

    /* 6. Configure CC */
    cc.raw = 0;
    cc.bits.en = 0;  /* Don't enable yet */
    cc.bits.css = 0;  /* NVM Command Set */
    cc.bits.mps = 0;  /* 4KB pages (2^(12+0)) */
    cc.bits.ams = 0;  /* Round Robin */
    cc.bits.shn = NVME_CC_SHN_NONE;
    cc.bits.iosqes = 6;  /* 2^6 = 64 bytes */
    cc.bits.iocqes = 4;  /* 2^4 = 16 bytes */
    host_mmio_write32(nvme_reg_ptr(bar0, NVME_REG_CC), cc.raw);

    /* 7. Enable (CC.EN=1) */
    cc.bits.en = 1;
    host_mmio_write32(nvme_reg_ptr(bar0, NVME_REG_CC), cc.raw);

    /* 8. Wait CSTS.RDY=1 */
    gpunvme_err_t err = wait_csts_rdy(bar0, 1, ctrl->timeout_ms);
    if (err != GPUNVME_OK) return err;

    ctrl->admin_sq_tail = 0;
    ctrl->admin_cq_head = 0;
    ctrl->admin_cq_phase = 1;
    ctrl->admin_cid = 0;

    fprintf(stderr, "ctrl: Controller enabled and ready\n");

    /* 9. Identify Controller */
    {
        void *id_buf;
        if (cudaMallocHost(&id_buf, 4096) != cudaSuccess)
            return GPUNVME_ERR_NOMEM;

        nvme_sq_entry_t cmd;
        uint64_t id_phys = 0;
        gpunvme_virt_to_phys(id_buf, &id_phys);
        nvme_cmd_identify_controller(&cmd, ctrl->admin_cid++, id_phys);

        uint32_t cdw0;
        err = gpunvme_admin_submit(ctrl, &cmd, &cdw0);
        if (err == GPUNVME_OK) {
            /* Parse identify data */
            uint8_t *id = (uint8_t *)id_buf;
            memcpy(ctrl->serial, id + 4, 20);
            ctrl->serial[20] = '\0';
            memcpy(ctrl->model, id + 24, 40);
            ctrl->model[40] = '\0';
            memcpy(ctrl->firmware, id + 64, 8);
            ctrl->firmware[8] = '\0';

            /* Trim trailing spaces */
            for (int i = 19; i >= 0 && ctrl->serial[i] == ' '; i--)
                ctrl->serial[i] = '\0';
            for (int i = 39; i >= 0 && ctrl->model[i] == ' '; i--)
                ctrl->model[i] = '\0';

            /* MDTS at offset 77 */
            ctrl->mdts = id[77];
            if (ctrl->mdts > 0) {
                ctrl->max_transfer_bytes = ctrl->page_size * (1u << ctrl->mdts);
            } else {
                ctrl->max_transfer_bytes = 1u << 20; /* 1MB fallback if no limit */
            }

            fprintf(stderr, "ctrl: Model:    %s\n", ctrl->model);
            fprintf(stderr, "ctrl: Serial:   %s\n", ctrl->serial);
            fprintf(stderr, "ctrl: Firmware: %s\n", ctrl->firmware);
            fprintf(stderr, "ctrl: MDTS:     %u (%u KB max per command)\n",
                    ctrl->mdts, ctrl->max_transfer_bytes / 1024);
        }

        cudaFreeHost(id_buf);
    }

    /* Identify Namespace 1 */
    {
        void *ns_buf;
        if (cudaMallocHost(&ns_buf, 4096) != cudaSuccess)
            return GPUNVME_ERR_NOMEM;

        nvme_sq_entry_t cmd;
        uint64_t ns_phys = 0;
        gpunvme_virt_to_phys(ns_buf, &ns_phys);
        nvme_cmd_identify_namespace(&cmd, ctrl->admin_cid++, 1, ns_phys);

        uint32_t cdw0;
        err = gpunvme_admin_submit(ctrl, &cmd, &cdw0);
        if (err == GPUNVME_OK) {
            uint8_t *ns = (uint8_t *)ns_buf;
            /* NSZE at offset 0, 8 bytes */
            uint64_t nsze;
            memcpy(&nsze, ns, 8);
            ctrl->ns_size_blocks = (uint32_t)nsze; /* truncate for simplicity */

            /* FLBAS at offset 26 */
            uint8_t flbas = ns[26];
            uint8_t lba_idx = flbas & 0x0F;

            /* LBA Format at offset 128 + lba_idx*4 */
            uint32_t lbaf;
            memcpy(&lbaf, ns + 128 + lba_idx * 4, 4);
            uint8_t lba_ds = (lbaf >> 16) & 0xFF;
            ctrl->block_size = 1u << lba_ds;

            fprintf(stderr, "ctrl: Namespace 1: %u blocks, %u bytes/block\n",
                    ctrl->ns_size_blocks, ctrl->block_size);
        }

        cudaFreeHost(ns_buf);
    }

    return GPUNVME_OK;
}

gpunvme_err_t gpunvme_admin_submit(gpunvme_ctrl_t *ctrl,
                                    nvme_sq_entry_t *cmd,
                                    uint32_t *cdw0_out) {
    if (!ctrl || !cmd) return GPUNVME_ERR_INVALID_PARAM;

    /* Write command to admin SQ */
    memcpy((void *)&ctrl->admin_sq[ctrl->admin_sq_tail], cmd, sizeof(*cmd));

    /* Advance tail */
    ctrl->admin_sq_tail = (ctrl->admin_sq_tail + 1) % ctrl->admin_sq_size;

    /* Ring admin SQ doorbell */
    uint32_t db_off = nvme_sq_doorbell_offset(0, ctrl->dstrd);
    host_mmio_write32(nvme_reg_ptr(ctrl->bar0, db_off), ctrl->admin_sq_tail);

    /* Poll admin CQ for completion */
    for (uint32_t i = 0; i < ctrl->timeout_ms * 1000; i++) {
        volatile nvme_cq_entry_t *cqe = &ctrl->admin_cq[ctrl->admin_cq_head];
        uint16_t sp = cqe->status_phase;

        if (NVME_CQE_PHASE(sp) == ctrl->admin_cq_phase) {
            /* Got completion */
            uint16_t status = NVME_CQE_SC(sp);
            if (cdw0_out) *cdw0_out = cqe->cdw0;

            /* Advance CQ head */
            ctrl->admin_cq_head++;
            if (ctrl->admin_cq_head >= ctrl->admin_cq_size) {
                ctrl->admin_cq_head = 0;
                ctrl->admin_cq_phase ^= 1;
            }

            /* Ring admin CQ doorbell */
            uint32_t cq_db_off = nvme_cq_doorbell_offset(0, ctrl->dstrd);
            host_mmio_write32(nvme_reg_ptr(ctrl->bar0, cq_db_off), ctrl->admin_cq_head);

            if (status != 0) {
                fprintf(stderr, "ctrl: Admin command failed, status=0x%02x\n", status);
                return GPUNVME_ERR_NVME_STATUS;
            }

            return GPUNVME_OK;
        }

        /* Brief spin (1us resolution) */
        struct timespec ts = { 0, 1000 };
        nanosleep(&ts, NULL);
    }

    return GPUNVME_ERR_TIMEOUT;
}

gpunvme_err_t gpunvme_ctrl_shutdown(gpunvme_ctrl_t *ctrl) {
    if (!ctrl || !ctrl->bar0) return GPUNVME_ERR_INVALID_PARAM;

    /* Set CC.SHN = Normal Shutdown */
    nvme_cc_t cc;
    cc.raw = host_mmio_read32(nvme_reg_ptr(ctrl->bar0, NVME_REG_CC));
    cc.bits.shn = NVME_CC_SHN_NORMAL;
    host_mmio_write32(nvme_reg_ptr(ctrl->bar0, NVME_REG_CC), cc.raw);

    /* Wait for CSTS.SHST = Complete */
    for (uint32_t i = 0; i < ctrl->timeout_ms; i++) {
        nvme_csts_t csts;
        csts.raw = host_mmio_read32(nvme_reg_ptr(ctrl->bar0, NVME_REG_CSTS));
        if (csts.bits.shst == NVME_CSTS_SHST_COMPLETE) {
            fprintf(stderr, "ctrl: Shutdown complete\n");
            break;
        }
        sleep_ms(1);
    }

    /* Disable controller */
    cc.raw = host_mmio_read32(nvme_reg_ptr(ctrl->bar0, NVME_REG_CC));
    cc.bits.en = 0;
    host_mmio_write32(nvme_reg_ptr(ctrl->bar0, NVME_REG_CC), cc.raw);

    /* Free admin queues */
    if (ctrl->admin_sq) { cudaFreeHost((void *)ctrl->admin_sq); ctrl->admin_sq = NULL; }
    if (ctrl->admin_cq) { cudaFreeHost((void *)ctrl->admin_cq); ctrl->admin_cq = NULL; }

    return GPUNVME_OK;
}

int gpunvme_ctrl_is_fatal(gpunvme_ctrl_t *ctrl) {
    if (!ctrl || !ctrl->bar0) return 1;
    nvme_csts_t csts;
    csts.raw = host_mmio_read32(nvme_reg_ptr(ctrl->bar0, NVME_REG_CSTS));
    return csts.bits.cfs;
}
