/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * gpu-nvme-direct: Internal kernel module header
 *
 * Shared structures and function declarations between the kmod source
 * files.  Not exported to userspace.
 *
 * Copyright (c) 2024-2026 gpu-nvme-direct contributors
 */

#ifndef GPUNVME_INTERNAL_H
#define GPUNVME_INTERNAL_H

#include <linux/pci.h>
#include <linux/cdev.h>
#include <linux/mutex.h>
#include <linux/types.h>

#include "gpunvme_ioctl.h"

#define GPUNVME_DRV_NAME	"gpunvme"
#define GPUNVME_MAX_DEVICES	4

/**
 * struct gpunvme_gpu_mapping - Tracks a single GPU P2P memory mapping
 * @gpu_vaddr:     CUDA virtual address of the GPU buffer
 * @size:          Size in bytes
 * @page_table:    nvidia_p2p_page_table_t from nvidia_p2p_get_pages()
 * @dma_mapping:   nvidia_p2p_dma_mapping_t from nvidia_p2p_dma_map_pages()
 * @bus_addr:      First bus/DMA address for the mapping
 * @in_use:        Whether this slot is active
 */
struct gpunvme_gpu_mapping {
	uint64_t		gpu_vaddr;
	uint64_t		size;
	void			*page_table;   /* nvidia_p2p_page_table_t * */
	void			*dma_mapping;  /* nvidia_p2p_dma_mapping_t * */
	uint64_t		bus_addr;
	bool			in_use;
};

/**
 * struct gpunvme_dev - Per-device context
 * @pdev:        PCI device
 * @bar0:        ioremap'd BAR0 pointer
 * @bar0_phys:   Physical address of BAR0
 * @bar0_size:   Size of BAR0 region in bytes
 * @cdev:        Character device
 * @dev:         Device node in sysfs
 * @dev_num:     Device number (major:minor)
 * @index:       Device index (0, 1, ...)
 * @gpu_maps:    Array of GPU P2P memory mappings
 * @gpu_map_lock: Protects @gpu_maps
 */
struct gpunvme_dev {
	struct pci_dev		*pdev;
	void __iomem		*bar0;
	resource_size_t		bar0_phys;
	resource_size_t		bar0_size;

	struct cdev		cdev;
	struct device		*dev;
	dev_t			dev_num;
	int			index;

	struct gpunvme_gpu_mapping gpu_maps[GPUNVME_MAX_GPU_MAPPINGS];
	struct mutex		gpu_map_lock;
};

/* ---- gpunvme_main.c ---- */
extern struct class *gpunvme_class;
extern int gpunvme_major;

/* ---- gpunvme_pci.c ---- */
int gpunvme_pci_register(void);
void gpunvme_pci_unregister(void);

/* ---- gpunvme_chrdev.c ---- */
int gpunvme_chrdev_create(struct gpunvme_dev *gdev);
void gpunvme_chrdev_destroy(struct gpunvme_dev *gdev);
extern const struct file_operations gpunvme_fops;

/* ---- gpunvme_dma.c ---- */
int gpunvme_gpu_mem_map(struct gpunvme_dev *gdev,
			struct gpunvme_gpu_mem_map_req *req);
int gpunvme_gpu_mem_unmap(struct gpunvme_dev *gdev,
			  struct gpunvme_gpu_mem_unmap_req *req);
void gpunvme_gpu_mem_unmap_all(struct gpunvme_dev *gdev);

#endif /* GPUNVME_INTERNAL_H */
