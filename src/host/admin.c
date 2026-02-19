/*
 * gpu-nvme-direct: Admin Command Helpers
 *
 * Higher-level wrappers around admin commands: Identify, Create/Delete queues.
 * Uses gpunvme_admin_submit() from controller.c for the actual submission.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <gpunvme/controller.h>
#include <gpunvme/nvme_cmds.h>
#include <gpunvme/error.h>
#include <gpunvme/dma.h>

#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

/*
 * Identify Controller: returns 4KB identify data structure.
 * Caller provides a 4KB buffer (must be DMA-accessible).
 */
gpunvme_err_t gpunvme_admin_identify_ctrl(gpunvme_ctrl_t *ctrl,
                                           void *id_buf,
                                           uint64_t id_buf_phys) {
    nvme_sq_entry_t cmd;
    nvme_cmd_identify_controller(&cmd, ctrl->admin_cid++, id_buf_phys);

    uint32_t cdw0;
    return gpunvme_admin_submit(ctrl, &cmd, &cdw0);
}

/*
 * Identify Namespace: returns 4KB namespace data structure.
 */
gpunvme_err_t gpunvme_admin_identify_ns(gpunvme_ctrl_t *ctrl,
                                         uint32_t nsid,
                                         void *ns_buf,
                                         uint64_t ns_buf_phys) {
    nvme_sq_entry_t cmd;
    nvme_cmd_identify_namespace(&cmd, ctrl->admin_cid++, nsid, ns_buf_phys);

    uint32_t cdw0;
    return gpunvme_admin_submit(ctrl, &cmd, &cdw0);
}

/*
 * Create I/O Completion Queue.
 */
gpunvme_err_t gpunvme_admin_create_io_cq(gpunvme_ctrl_t *ctrl,
                                          uint16_t qid,
                                          uint16_t size,
                                          uint64_t cq_phys) {
    nvme_sq_entry_t cmd;
    nvme_cmd_create_io_cq(&cmd, ctrl->admin_cid++, qid, size - 1, cq_phys, 0);

    uint32_t cdw0;
    gpunvme_err_t err = gpunvme_admin_submit(ctrl, &cmd, &cdw0);
    if (err == GPUNVME_OK) {
        fprintf(stderr, "admin: Created I/O CQ %u (size=%u)\n", qid, size);
    }
    return err;
}

/*
 * Create I/O Submission Queue.
 */
gpunvme_err_t gpunvme_admin_create_io_sq(gpunvme_ctrl_t *ctrl,
                                          uint16_t qid,
                                          uint16_t size,
                                          uint64_t sq_phys,
                                          uint16_t cqid) {
    nvme_sq_entry_t cmd;
    nvme_cmd_create_io_sq(&cmd, ctrl->admin_cid++, qid, size - 1, sq_phys, cqid);

    uint32_t cdw0;
    gpunvme_err_t err = gpunvme_admin_submit(ctrl, &cmd, &cdw0);
    if (err == GPUNVME_OK) {
        fprintf(stderr, "admin: Created I/O SQ %u (size=%u, cqid=%u)\n", qid, size, cqid);
    }
    return err;
}

/*
 * Delete I/O Submission Queue.
 */
gpunvme_err_t gpunvme_admin_delete_io_sq(gpunvme_ctrl_t *ctrl, uint16_t qid) {
    nvme_sq_entry_t cmd;
    nvme_cmd_delete_io_sq(&cmd, ctrl->admin_cid++, qid);

    uint32_t cdw0;
    return gpunvme_admin_submit(ctrl, &cmd, &cdw0);
}

/*
 * Delete I/O Completion Queue.
 */
gpunvme_err_t gpunvme_admin_delete_io_cq(gpunvme_ctrl_t *ctrl, uint16_t qid) {
    nvme_sq_entry_t cmd;
    nvme_cmd_delete_io_cq(&cmd, ctrl->admin_cid++, qid);

    uint32_t cdw0;
    return gpunvme_admin_submit(ctrl, &cmd, &cdw0);
}
