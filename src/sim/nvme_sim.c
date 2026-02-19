/*
 * gpu-nvme-direct: Software NVMe Simulator Implementation
 *
 * A CPU-side thread polls the simulated SQ doorbell, processes commands,
 * and writes CQ entries with proper phase bit tracking.
 *
 * PRP handling in simulator mode:
 *   PRP addresses are treated as host virtual addresses pointing into
 *   the data buffer. The GPU kernel writes the pinned buffer address
 *   it got from nvme_sim_get_data_buf_phys() + offset.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "nvme_sim.h"
#include <gpunvme/nvme_regs.h>
#include <gpunvme/nvme_cmds.h>
#include <gpunvme/error.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <cuda_runtime.h>

struct nvme_sim {
    /* Configuration */
    nvme_sim_config_t cfg;

    /* Virtual block device storage (regular host memory) */
    uint8_t *vdev;
    uint64_t vdev_size;

    /* SQ and CQ in CUDA pinned memory (GPU-accessible) */
    nvme_sq_entry_t *sq;
    nvme_cq_entry_t *cq;

    /* Simulated doorbell registers in CUDA pinned memory */
    uint32_t *sq_doorbell;  /* Written by GPU: new SQ tail */
    uint32_t *cq_doorbell;  /* Written by GPU: new CQ head */

    /* Data buffer in CUDA pinned memory (GPU-accessible, DMA target) */
    uint8_t *data_buf;
    uint64_t data_buf_size;

    /* Simulated BAR0 register space in CUDA pinned memory */
    uint8_t *bar0;
    size_t   bar0_size;

    /* Simulator state */
    uint16_t sq_head;       /* Next SQ entry to process (simulator tracks this) */
    uint16_t cq_tail;       /* Next CQ entry to write */
    uint8_t  cq_phase;      /* Current CQ phase bit */

    /* Poller thread */
    pthread_t poller_thread;
    volatile int running;
};

/* Simulated BAR0 layout:
 * We allocate a 8KB region. Registers at standard offsets.
 * Doorbells start at 0x1000 (standard NVMe).
 */
#define SIM_BAR0_SIZE 8192

static void sim_init_bar0(nvme_sim_t *sim) {
    memset(sim->bar0, 0, SIM_BAR0_SIZE);

    /* CAP register at 0x00 */
    nvme_cap_t cap;
    cap.raw = 0;
    cap.bits.mqes = sim->cfg.sq_size - 1;  /* 0-based */
    cap.bits.cqr = 1;
    cap.bits.to = 10;  /* 5 second timeout (10 * 500ms) */
    cap.bits.dstrd = 0;
    cap.bits.mpsmin = 0;  /* 4KB pages */
    cap.bits.mpsmax = 0;
    cap.bits.css = 1;  /* NVM command set */
    memcpy(sim->bar0 + NVME_REG_CAP, &cap.raw, 8);

    /* VS register at 0x08: NVMe 1.4 */
    nvme_vs_t vs;
    vs.raw = 0;
    vs.bits.mjr = 1;
    vs.bits.mnr = 4;
    vs.bits.ter = 0;
    memcpy(sim->bar0 + NVME_REG_VS, &vs.raw, 4);

    /* CSTS at 0x1C: ready */
    nvme_csts_t csts;
    csts.raw = 0;
    csts.bits.rdy = 1;
    memcpy(sim->bar0 + NVME_REG_CSTS, &csts.raw, 4);

    /* CC at 0x14: enabled, IOSQES=6 (64B), IOCQES=4 (16B) */
    nvme_cc_t cc;
    cc.raw = 0;
    cc.bits.en = 1;
    cc.bits.iosqes = 6;
    cc.bits.iocqes = 4;
    memcpy(sim->bar0 + NVME_REG_CC, &cc.raw, 4);

    /* Point doorbell pointers into BAR0 at standard offsets.
     * For I/O queue 1 with DSTRD=0:
     *   SQ1 tail doorbell = 0x1000 + (2*1 * 4) = 0x1008
     *   CQ1 head doorbell = 0x1000 + (3   * 4) = 0x100C
     * But we use dedicated allocations for the doorbells the GPU actually
     * writes to. The BAR0 doorbells are for tools that read BAR0. */
}

static void sim_sleep_us(uint32_t us) {
    struct timespec ts;
    ts.tv_sec = us / 1000000;
    ts.tv_nsec = (us % 1000000) * 1000;
    nanosleep(&ts, NULL);
}

static void sim_process_read(nvme_sim_t *sim, const nvme_sq_entry_t *cmd) {
    uint64_t slba = ((uint64_t)cmd->cdw11 << 32) | cmd->cdw10;
    uint32_t nlb = (cmd->cdw12 & 0xFFFF) + 1;  /* 0-based in command */
    uint64_t prp1 = cmd->prp1;

    /* Simulate latency */
    if (sim->cfg.latency_us > 0) {
        sim_sleep_us(sim->cfg.latency_us);
    }

    /* Validate LBA range */
    if (slba + nlb > sim->cfg.num_blocks) {
        /* Write error CQE */
        volatile nvme_cq_entry_t *cqe = &sim->cq[sim->cq_tail];
        cqe->cdw0 = 0;
        cqe->cdw1 = 0;
        cqe->sqhd = sim->sq_head;
        cqe->sqid = 1;
        cqe->cid = cmd->cid;
        /* Status: Generic Command Status, Invalid Field (0x02), phase bit */
        cqe->status_phase = (0x02 << 1) | sim->cq_phase;
        return;
    }

    /* Copy data from virtual device to data buffer at PRP1 address.
     * In sim mode, PRP1 is a host virtual address in the pinned data buffer. */
    uint64_t byte_offset = slba * sim->cfg.block_size;
    uint64_t byte_count = (uint64_t)nlb * sim->cfg.block_size;

    void *dst = (void *)prp1;
    memcpy(dst, sim->vdev + byte_offset, byte_count);

    /* Write success CQE */
    volatile nvme_cq_entry_t *cqe = &sim->cq[sim->cq_tail];
    cqe->cdw0 = 0;
    cqe->cdw1 = 0;
    cqe->sqhd = sim->sq_head;
    cqe->sqid = 1;
    cqe->cid = cmd->cid;
    cqe->status_phase = sim->cq_phase;  /* Status = 0 (success) | phase */
}

static void sim_process_write(nvme_sim_t *sim, const nvme_sq_entry_t *cmd) {
    uint64_t slba = ((uint64_t)cmd->cdw11 << 32) | cmd->cdw10;
    uint32_t nlb = (cmd->cdw12 & 0xFFFF) + 1;
    uint64_t prp1 = cmd->prp1;

    if (sim->cfg.latency_us > 0) {
        sim_sleep_us(sim->cfg.latency_us);
    }

    if (slba + nlb > sim->cfg.num_blocks) {
        volatile nvme_cq_entry_t *cqe = &sim->cq[sim->cq_tail];
        cqe->cdw0 = 0;
        cqe->cdw1 = 0;
        cqe->sqhd = sim->sq_head;
        cqe->sqid = 1;
        cqe->cid = cmd->cid;
        cqe->status_phase = (0x02 << 1) | sim->cq_phase;
        return;
    }

    uint64_t byte_offset = slba * sim->cfg.block_size;
    uint64_t byte_count = (uint64_t)nlb * sim->cfg.block_size;

    void *src = (void *)prp1;
    memcpy(sim->vdev + byte_offset, src, byte_count);

    volatile nvme_cq_entry_t *cqe = &sim->cq[sim->cq_tail];
    cqe->cdw0 = 0;
    cqe->cdw1 = 0;
    cqe->sqhd = sim->sq_head;
    cqe->sqid = 1;
    cqe->cid = cmd->cid;
    cqe->status_phase = sim->cq_phase;
}

static void sim_advance_cq_tail(nvme_sim_t *sim) {
    sim->cq_tail++;
    if (sim->cq_tail >= sim->cfg.cq_size) {
        sim->cq_tail = 0;
        sim->cq_phase ^= 1;
    }
}

static void *sim_poller_thread(void *arg) {
    nvme_sim_t *sim = (nvme_sim_t *)arg;

    while (sim->running) {
        /* Read current SQ tail doorbell (written by GPU) */
        uint32_t sq_tail = __atomic_load_n(sim->sq_doorbell, __ATOMIC_ACQUIRE);

        if (sim->sq_head != sq_tail) {
            /* Process the command at sq_head */
            nvme_sq_entry_t cmd;
            memcpy(&cmd, (void *)&sim->sq[sim->sq_head], sizeof(cmd));

            uint8_t opcode = cmd.opc;

            switch (opcode) {
            case NVME_IO_OPC_READ:
                sim_process_read(sim, &cmd);
                sim_advance_cq_tail(sim);
                break;
            case NVME_IO_OPC_WRITE:
                sim_process_write(sim, &cmd);
                sim_advance_cq_tail(sim);
                break;
            case NVME_IO_OPC_FLUSH:
                /* Flush is a no-op in simulator */
                {
                    volatile nvme_cq_entry_t *cqe = &sim->cq[sim->cq_tail];
                    cqe->cdw0 = 0;
                    cqe->cdw1 = 0;
                    cqe->sqhd = sim->sq_head;
                    cqe->sqid = 1;
                    cqe->cid = cmd.cid;
                    cqe->status_phase = sim->cq_phase;
                    sim_advance_cq_tail(sim);
                }
                break;
            default:
                /* Unknown opcode — return error */
                {
                    volatile nvme_cq_entry_t *cqe = &sim->cq[sim->cq_tail];
                    cqe->cdw0 = 0;
                    cqe->cdw1 = 0;
                    cqe->sqhd = sim->sq_head;
                    cqe->sqid = 1;
                    cqe->cid = cmd.cid;
                    cqe->status_phase = (NVME_SC_INVALID_OPCODE << 1) | sim->cq_phase;
                    sim_advance_cq_tail(sim);
                }
                break;
            }

            /* Advance SQ head */
            sim->sq_head = (sim->sq_head + 1) % sim->cfg.sq_size;

            /* Memory fence to ensure CQE is visible before GPU polls */
            __atomic_thread_fence(__ATOMIC_RELEASE);
        } else {
            /* No work — brief sleep to avoid spinning at 100% CPU */
            usleep(1);
        }
    }

    return NULL;
}

nvme_sim_t *nvme_sim_create(const nvme_sim_config_t *cfg) {
    nvme_sim_t *sim = (nvme_sim_t *)calloc(1, sizeof(nvme_sim_t));
    if (!sim) return NULL;

    sim->cfg = *cfg;

    /* Allocate virtual block device in regular host memory */
    sim->vdev_size = (uint64_t)cfg->num_blocks * cfg->block_size;
    sim->vdev = (uint8_t *)calloc(1, sim->vdev_size);
    if (!sim->vdev) goto fail;

    /* Allocate SQ in CUDA pinned memory */
    if (cudaMallocHost((void **)&sim->sq,
                       cfg->sq_size * sizeof(nvme_sq_entry_t)) != cudaSuccess) {
        fprintf(stderr, "nvme_sim: cudaMallocHost SQ failed\n");
        goto fail;
    }
    memset((void *)sim->sq, 0, cfg->sq_size * sizeof(nvme_sq_entry_t));

    /* Allocate CQ in CUDA pinned memory */
    if (cudaMallocHost((void **)&sim->cq,
                       cfg->cq_size * sizeof(nvme_cq_entry_t)) != cudaSuccess) {
        fprintf(stderr, "nvme_sim: cudaMallocHost CQ failed\n");
        goto fail;
    }
    memset((void *)sim->cq, 0, cfg->cq_size * sizeof(nvme_cq_entry_t));

    /* Allocate doorbell registers in CUDA pinned memory */
    if (cudaMallocHost((void **)&sim->sq_doorbell, sizeof(uint32_t) * 2) != cudaSuccess) {
        fprintf(stderr, "nvme_sim: cudaMallocHost doorbells failed\n");
        goto fail;
    }
    sim->cq_doorbell = sim->sq_doorbell + 1;
    *sim->sq_doorbell = 0;
    *sim->cq_doorbell = 0;

    /* Allocate data buffer in CUDA pinned memory */
    sim->data_buf_size = sim->vdev_size;
    if (sim->data_buf_size > 64 * 1024 * 1024) {
        /* Cap at 64MB for the data buffer — virtual device can be larger */
        sim->data_buf_size = 64 * 1024 * 1024;
    }
    if (sim->data_buf_size < (uint64_t)cfg->block_size * 256) {
        sim->data_buf_size = (uint64_t)cfg->block_size * 256;
    }
    if (cudaMallocHost((void **)&sim->data_buf, sim->data_buf_size) != cudaSuccess) {
        fprintf(stderr, "nvme_sim: cudaMallocHost data_buf failed\n");
        goto fail;
    }
    memset(sim->data_buf, 0, sim->data_buf_size);

    /* Allocate simulated BAR0 in CUDA pinned memory */
    sim->bar0_size = SIM_BAR0_SIZE;
    if (cudaMallocHost((void **)&sim->bar0, sim->bar0_size) != cudaSuccess) {
        fprintf(stderr, "nvme_sim: cudaMallocHost bar0 failed\n");
        goto fail;
    }
    sim_init_bar0(sim);

    /* Initialize queue state */
    sim->sq_head = 0;
    sim->cq_tail = 0;
    sim->cq_phase = 1;  /* NVMe spec: controller starts with phase = 1 */

    /* Start poller thread */
    sim->running = 1;
    if (pthread_create(&sim->poller_thread, NULL, sim_poller_thread, sim) != 0) {
        fprintf(stderr, "nvme_sim: pthread_create failed\n");
        goto fail;
    }

    return sim;

fail:
    nvme_sim_destroy(sim);
    return NULL;
}

void nvme_sim_destroy(nvme_sim_t *sim) {
    if (!sim) return;

    if (sim->running) {
        sim->running = 0;
        pthread_join(sim->poller_thread, NULL);
    }

    if (sim->bar0)        cudaFreeHost(sim->bar0);
    if (sim->data_buf)    cudaFreeHost(sim->data_buf);
    if (sim->sq_doorbell) cudaFreeHost(sim->sq_doorbell);
    if (sim->cq)          cudaFreeHost((void *)sim->cq);
    if (sim->sq)          cudaFreeHost((void *)sim->sq);
    free(sim->vdev);
    free(sim);
}

volatile nvme_sq_entry_t *nvme_sim_get_sq(nvme_sim_t *sim) {
    return (volatile nvme_sq_entry_t *)sim->sq;
}

volatile nvme_cq_entry_t *nvme_sim_get_cq(nvme_sim_t *sim) {
    return (volatile nvme_cq_entry_t *)sim->cq;
}

volatile uint32_t *nvme_sim_get_sq_doorbell(nvme_sim_t *sim) {
    return (volatile uint32_t *)sim->sq_doorbell;
}

volatile uint32_t *nvme_sim_get_cq_doorbell(nvme_sim_t *sim) {
    return (volatile uint32_t *)sim->cq_doorbell;
}

volatile void *nvme_sim_get_data_buf(nvme_sim_t *sim) {
    return (volatile void *)sim->data_buf;
}

uint64_t nvme_sim_get_data_buf_phys(nvme_sim_t *sim) {
    /* In sim mode, "physical" address = host virtual address */
    return (uint64_t)(uintptr_t)sim->data_buf;
}

volatile void *nvme_sim_get_bar0(nvme_sim_t *sim) {
    return (volatile void *)sim->bar0;
}

uint16_t nvme_sim_get_sq_size(nvme_sim_t *sim) {
    return sim->cfg.sq_size;
}

uint16_t nvme_sim_get_cq_size(nvme_sim_t *sim) {
    return sim->cfg.cq_size;
}

void nvme_sim_fill_blocks(nvme_sim_t *sim,
                          uint32_t start_lba,
                          uint32_t count,
                          void (*pattern_fn)(uint32_t lba, void *buf, uint32_t size)) {
    for (uint32_t i = 0; i < count; i++) {
        uint32_t lba = start_lba + i;
        if (lba >= sim->cfg.num_blocks) break;
        uint8_t *block = sim->vdev + (uint64_t)lba * sim->cfg.block_size;
        pattern_fn(lba, block, sim->cfg.block_size);
    }
}

int nvme_sim_direct_read(nvme_sim_t *sim,
                         uint32_t lba,
                         uint32_t count,
                         void *buf) {
    if (lba + count > sim->cfg.num_blocks) return -1;
    uint64_t offset = (uint64_t)lba * sim->cfg.block_size;
    uint64_t size = (uint64_t)count * sim->cfg.block_size;
    memcpy(buf, sim->vdev + offset, size);
    return 0;
}
