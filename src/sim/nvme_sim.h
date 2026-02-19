/*
 * gpu-nvme-direct: Software NVMe Simulator
 *
 * Simulates an NVMe controller in software so GPU CUDA kernels can be
 * developed and tested without real NVMe hardware (e.g., in WSL).
 *
 * The simulator:
 *   - Allocates SQ/CQ in host pinned memory (cudaMallocHost)
 *   - Provides a simulated BAR0 register space in pinned memory
 *   - Runs a CPU thread that polls the SQ for new entries
 *   - Simulates configurable latency
 *   - Writes CQ entries with correct phase bit protocol
 *   - Manages a virtual block device backed by host memory
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#ifndef GPUNVME_NVME_SIM_H
#define GPUNVME_NVME_SIM_H

#include <stdint.h>
#include <stdbool.h>
#include <gpunvme/nvme_regs.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Simulator configuration */
typedef struct {
    uint32_t num_blocks;      /* Total blocks in virtual device */
    uint32_t block_size;      /* Bytes per block (512 or 4096) */
    uint16_t sq_size;         /* SQ entries (actual count, power of 2 preferred) */
    uint16_t cq_size;         /* CQ entries */
    uint32_t latency_us;      /* Simulated per-command latency in microseconds */
} nvme_sim_config_t;

/* Opaque simulator handle */
typedef struct nvme_sim nvme_sim_t;

/*
 * Create and start the NVMe simulator.
 *
 * Allocates:
 *   - Virtual block device (cfg->num_blocks * cfg->block_size bytes) in host mem
 *   - SQ, CQ, and BAR0 registers in CUDA pinned memory (GPU-accessible)
 *   - Starts a CPU poller thread
 *
 * Returns NULL on failure.
 */
nvme_sim_t *nvme_sim_create(const nvme_sim_config_t *cfg);

/*
 * Stop and destroy the simulator. Frees all resources.
 */
void nvme_sim_destroy(nvme_sim_t *sim);

/*
 * Get the GPU-accessible SQ base pointer.
 * The GPU kernel writes SQ entries here.
 */
volatile nvme_sq_entry_t *nvme_sim_get_sq(nvme_sim_t *sim);

/*
 * Get the GPU-accessible CQ base pointer.
 * The GPU kernel reads CQ entries here.
 */
volatile nvme_cq_entry_t *nvme_sim_get_cq(nvme_sim_t *sim);

/*
 * Get the GPU-accessible SQ tail doorbell pointer.
 * The GPU kernel writes the SQ tail index here after submitting commands.
 */
volatile uint32_t *nvme_sim_get_sq_doorbell(nvme_sim_t *sim);

/*
 * Get the GPU-accessible CQ head doorbell pointer.
 * The GPU kernel writes the CQ head index here after consuming completions.
 */
volatile uint32_t *nvme_sim_get_cq_doorbell(nvme_sim_t *sim);

/*
 * Get the GPU-accessible data buffer pointer.
 * NVMe read commands place data here; write commands read from here.
 * In the simulator, PRP addresses are offsets into this buffer.
 */
volatile void *nvme_sim_get_data_buf(nvme_sim_t *sim);

/*
 * Get the "physical" address of the data buffer.
 * In the simulator, this is just the host virtual address cast to uint64_t,
 * since PRP entries are interpreted as offsets from data_buf base or
 * absolute host virtual addresses depending on sim mode.
 */
uint64_t nvme_sim_get_data_buf_phys(nvme_sim_t *sim);

/*
 * Get the simulated BAR0 register space (GPU-accessible pinned memory).
 * Contains CAP, VS, CC, CSTS, doorbells, etc.
 */
volatile void *nvme_sim_get_bar0(nvme_sim_t *sim);

/*
 * Get simulator queue sizes.
 */
uint16_t nvme_sim_get_sq_size(nvme_sim_t *sim);
uint16_t nvme_sim_get_cq_size(nvme_sim_t *sim);

/*
 * Pre-fill virtual device blocks with known data pattern.
 * Useful for read verification in tests.
 * pattern_fn is called with (block_index, block_buffer, block_size).
 */
void nvme_sim_fill_blocks(nvme_sim_t *sim,
                          uint32_t start_lba,
                          uint32_t count,
                          void (*pattern_fn)(uint32_t lba, void *buf, uint32_t size));

/*
 * Read blocks directly from virtual device (for test verification).
 * Bypasses the NVMe queue path.
 */
int nvme_sim_direct_read(nvme_sim_t *sim,
                         uint32_t lba,
                         uint32_t count,
                         void *buf);

#ifdef __cplusplus
}
#endif

#endif /* GPUNVME_NVME_SIM_H */
