/* SPDX-License-Identifier: GPL-2.0-only WITH Linux-syscall-note */
/*
 * gpu-nvme-direct: ioctl ABI definitions
 *
 * Shared header between kernel module and userspace.
 * Defines the ioctl commands for /dev/gpunvme<N>.
 *
 * Copyright (c) 2024-2026 gpu-nvme-direct contributors
 */

#ifndef GPUNVME_IOCTL_H
#define GPUNVME_IOCTL_H

#ifdef __KERNEL__
#include <linux/types.h>
#include <linux/ioctl.h>
#else
#include <stdint.h>
#include <sys/ioctl.h>
#endif

#define GPUNVME_IOCTL_MAGIC	'G'

/*
 * struct gpunvme_bar_info - BAR0 information returned to userspace
 * @phys_addr: physical address of BAR0
 * @size:      BAR0 region size in bytes
 */
struct gpunvme_bar_info {
	uint64_t phys_addr;
	uint64_t size;
};

/*
 * struct gpunvme_gpu_mem_map_req - GPU memory mapping request
 * @gpu_vaddr:   CUDA device pointer (virtual address on GPU)
 * @size:        size of the GPU memory region in bytes (must be page-aligned)
 * @bus_addr:    [out] DMA/bus address usable by the NVMe controller
 * @handle:      [out] opaque handle for unmapping
 *
 * The bus_addr is the address of the first page. For multi-page mappings,
 * individual page bus addresses can be retrieved via additional ioctls.
 */
struct gpunvme_gpu_mem_map_req {
	uint64_t gpu_vaddr;
	uint64_t size;
	uint64_t bus_addr;
	uint64_t handle;
};

/*
 * struct gpunvme_gpu_mem_unmap_req - GPU memory unmap request
 * @handle: handle returned by MAP_GPU_MEM
 */
struct gpunvme_gpu_mem_unmap_req {
	uint64_t handle;
};

/*
 * struct gpunvme_phys_addr_req - Virtual to physical address translation
 * @vaddr:     userspace virtual address
 * @phys_addr: [out] physical address
 */
struct gpunvme_phys_addr_req {
	uint64_t vaddr;
	uint64_t phys_addr;
};

/* ioctl commands */

/* Get BAR0 physical address and size */
#define GPUNVME_IOCTL_GET_BAR_INFO \
	_IOR(GPUNVME_IOCTL_MAGIC, 0x01, struct gpunvme_bar_info)

/* Map GPU VRAM via nvidia_p2p, returns bus address */
#define GPUNVME_IOCTL_MAP_GPU_MEM \
	_IOWR(GPUNVME_IOCTL_MAGIC, 0x02, struct gpunvme_gpu_mem_map_req)

/* Unmap previously mapped GPU memory */
#define GPUNVME_IOCTL_UNMAP_GPU_MEM \
	_IOW(GPUNVME_IOCTL_MAGIC, 0x03, struct gpunvme_gpu_mem_unmap_req)

/* Translate a virtual address to physical */
#define GPUNVME_IOCTL_GET_PHYS_ADDR \
	_IOWR(GPUNVME_IOCTL_MAGIC, 0x04, struct gpunvme_phys_addr_req)

/* Maximum number of simultaneous GPU memory mappings per device */
#define GPUNVME_MAX_GPU_MAPPINGS	16

#endif /* GPUNVME_IOCTL_H */
