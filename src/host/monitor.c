/*
 * gpu-nvme-direct: Error Monitoring Thread
 *
 * Continuously polls CSTS.CFS (Controller Fatal Status) to detect
 * hardware errors. Provides a callback mechanism for error handling.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <gpunvme/controller.h>
#include <gpunvme/mmio.h>
#include <gpunvme/nvme_regs.h>
#include <gpunvme/error.h>

#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <stdatomic.h>

typedef struct {
    gpunvme_ctrl_t *ctrl;
    pthread_t thread;
    atomic_int running;
    atomic_int fatal_detected;
    uint32_t poll_interval_ms;
    void (*error_callback)(gpunvme_ctrl_t *ctrl, void *userdata);
    void *callback_userdata;
} gpunvme_monitor_t;

static void *monitor_thread_func(void *arg) {
    gpunvme_monitor_t *mon = (gpunvme_monitor_t *)arg;

    struct timespec ts;
    ts.tv_sec = mon->poll_interval_ms / 1000;
    ts.tv_nsec = (mon->poll_interval_ms % 1000) * 1000000L;

    while (atomic_load(&mon->running)) {
        nvme_csts_t csts;
        csts.raw = host_mmio_read32(nvme_reg_ptr(mon->ctrl->bar0, NVME_REG_CSTS));

        if (csts.bits.cfs) {
            if (!atomic_exchange(&mon->fatal_detected, 1)) {
                fprintf(stderr, "monitor: *** CONTROLLER FATAL STATUS DETECTED ***\n");
                fprintf(stderr, "monitor: CSTS=0x%08x\n", csts.raw);

                if (mon->error_callback) {
                    mon->error_callback(mon->ctrl, mon->callback_userdata);
                }
            }
        }

        nanosleep(&ts, NULL);
    }

    return NULL;
}

/*
 * Start error monitoring thread.
 */
gpunvme_err_t gpunvme_monitor_start(gpunvme_monitor_t *mon,
                                     gpunvme_ctrl_t *ctrl,
                                     uint32_t poll_interval_ms,
                                     void (*error_cb)(gpunvme_ctrl_t *, void *),
                                     void *userdata) {
    if (!mon || !ctrl) return GPUNVME_ERR_INVALID_PARAM;

    memset(mon, 0, sizeof(*mon));
    mon->ctrl = ctrl;
    mon->poll_interval_ms = poll_interval_ms > 0 ? poll_interval_ms : 1;
    mon->error_callback = error_cb;
    mon->callback_userdata = userdata;
    atomic_store(&mon->running, 1);
    atomic_store(&mon->fatal_detected, 0);

    if (pthread_create(&mon->thread, NULL, monitor_thread_func, mon) != 0) {
        return GPUNVME_ERR_IO;
    }

    fprintf(stderr, "monitor: Started (poll every %u ms)\n", mon->poll_interval_ms);
    return GPUNVME_OK;
}

/*
 * Stop error monitoring thread.
 */
void gpunvme_monitor_stop(gpunvme_monitor_t *mon) {
    if (!mon) return;
    atomic_store(&mon->running, 0);
    pthread_join(mon->thread, NULL);
    fprintf(stderr, "monitor: Stopped\n");
}

/*
 * Check if fatal error was detected.
 */
int gpunvme_monitor_is_fatal(gpunvme_monitor_t *mon) {
    return mon ? atomic_load(&mon->fatal_detected) : 0;
}
