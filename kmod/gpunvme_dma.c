// SPDX-License-Identifier: GPL-2.0-only
/*
 * gpu-nvme-direct: GPU P2P DMA memory mapping
 *
 * Uses the NVIDIA GPU Direct RDMA API (nvidia_p2p_get_pages /
 * nvidia_p2p_dma_map_pages) to pin GPU VRAM and obtain bus addresses
 * that the NVMe controller can DMA to/from directly.
 *
 * IMPORTANT: This functionality requires:
 *   - NVIDIA driver with P2P support (nv-p2p.h)
 *   - Tesla / data-center class GPU (Quadro, A100, H100, etc.)
 *   - GeForce / consumer GPUs do NOT support P2P and will return errors
 *
 * The module compiles and loads without NVIDIA headers, but the GPU
 * mapping ioctls will return -ENOTSUP at runtime.
 *
 * Copyright (c) 2024-2026 gpu-nvme-direct contributors
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/mutex.h>

#include "gpunvme_internal.h"

/*
 * Conditionally include the NVIDIA P2P header.
 *
 * When building without the NVIDIA driver source, define stubs that
 * return -ENOTSUP so the module can still load for BAR0 mmap use.
 */
#if defined(CONFIG_NVIDIA_P2P) || defined(HAVE_NV_P2P_H)
#include <nv-p2p.h>
#define GPUNVME_HAVE_GPU_P2P	1
#else
#define GPUNVME_HAVE_GPU_P2P	0
#endif

/* GPU page size used by nvidia_p2p (64 KiB) */
#define GPUNVME_GPU_PAGE_SHIFT	16
#define GPUNVME_GPU_PAGE_SIZE	(1ULL << GPUNVME_GPU_PAGE_SHIFT)

#if GPUNVME_HAVE_GPU_P2P

/**
 * gpunvme_p2p_free_callback() - Called by NVIDIA driver when GPU memory is freed
 * @data: Pointer to the gpunvme_gpu_mapping that holds this mapping
 *
 * This callback is invoked if the CUDA application frees the GPU memory
 * while it is still pinned.  We must release our reference to the
 * page table without calling nvidia_p2p_put_pages (the driver handles
 * that internally when invoking this callback).
 */
static void gpunvme_p2p_free_callback(void *data)
{
	struct gpunvme_gpu_mapping *mapping = data;

	pr_warn(GPUNVME_DRV_NAME
		": GPU memory freed while still mapped (vaddr=0x%llx)\n",
		mapping->gpu_vaddr);

	/*
	 * The NVIDIA driver has already freed the pages.  We just mark
	 * the mapping as invalid so we do not double-free.
	 */
	mapping->page_table = NULL;
	mapping->dma_mapping = NULL;
	mapping->in_use = false;
}

/**
 * gpunvme_gpu_mem_map() - Pin GPU VRAM and obtain DMA bus addresses
 * @gdev: gpu-nvme device context
 * @req:  mapping request (gpu_vaddr, size in; bus_addr, handle out)
 *
 * Calls nvidia_p2p_get_pages() to pin the GPU virtual address range,
 * then nvidia_p2p_dma_map_pages() to get PCIe bus addresses that the
 * NVMe controller can target for DMA.
 *
 * Return: 0 on success, negative errno on failure
 */
int gpunvme_gpu_mem_map(struct gpunvme_dev *gdev,
			struct gpunvme_gpu_mem_map_req *req)
{
	nvidia_p2p_page_table_t *page_table = NULL;
	nvidia_p2p_dma_mapping_t *dma_mapping = NULL;
	struct gpunvme_gpu_mapping *mapping = NULL;
	int slot = -1;
	int ret;
	int i;

	if (!req->gpu_vaddr || !req->size) {
		dev_err(&gdev->pdev->dev,
			"invalid GPU mapping request: vaddr=0x%llx size=0x%llx\n",
			req->gpu_vaddr, req->size);
		return -EINVAL;
	}

	/* Ensure size is aligned to GPU page size */
	if (req->size & (GPUNVME_GPU_PAGE_SIZE - 1)) {
		dev_err(&gdev->pdev->dev,
			"GPU mapping size 0x%llx not aligned to 0x%llx\n",
			req->size, GPUNVME_GPU_PAGE_SIZE);
		return -EINVAL;
	}

	/* Find a free mapping slot */
	mutex_lock(&gdev->gpu_map_lock);
	for (i = 0; i < GPUNVME_MAX_GPU_MAPPINGS; i++) {
		if (!gdev->gpu_maps[i].in_use) {
			slot = i;
			gdev->gpu_maps[i].in_use = true;
			break;
		}
	}
	mutex_unlock(&gdev->gpu_map_lock);

	if (slot < 0) {
		dev_err(&gdev->pdev->dev,
			"no free GPU mapping slots (max=%d)\n",
			GPUNVME_MAX_GPU_MAPPINGS);
		return -ENOSPC;
	}

	mapping = &gdev->gpu_maps[slot];
	mapping->gpu_vaddr = req->gpu_vaddr;
	mapping->size = req->size;

	/* Pin GPU pages */
	ret = nvidia_p2p_get_pages(0, 0, req->gpu_vaddr, req->size,
				   &page_table,
				   gpunvme_p2p_free_callback, mapping);
	if (ret) {
		dev_err(&gdev->pdev->dev,
			"nvidia_p2p_get_pages failed: %d "
			"(GeForce/consumer GPUs do not support P2P)\n", ret);
		goto err_slot;
	}

	mapping->page_table = page_table;

	/* Map pages for DMA by the NVMe controller's PCI device */
	ret = nvidia_p2p_dma_map_pages(gdev->pdev, page_table, &dma_mapping);
	if (ret) {
		dev_err(&gdev->pdev->dev,
			"nvidia_p2p_dma_map_pages failed: %d\n", ret);
		goto err_put_pages;
	}

	mapping->dma_mapping = dma_mapping;
	mapping->bus_addr = dma_mapping->dma_addresses[0];

	/* Return results to userspace */
	req->bus_addr = mapping->bus_addr;
	req->handle = (uint64_t)slot;

	dev_info(&gdev->pdev->dev,
		 "GPU P2P mapped: vaddr=0x%llx size=0x%llx bus_addr=0x%llx slot=%d pages=%u\n",
		 req->gpu_vaddr, req->size, req->bus_addr, slot,
		 page_table->entries);

	return 0;

err_put_pages:
	nvidia_p2p_put_pages(0, 0, mapping->gpu_vaddr, mapping->page_table);
	mapping->page_table = NULL;
err_slot:
	mutex_lock(&gdev->gpu_map_lock);
	mapping->in_use = false;
	mutex_unlock(&gdev->gpu_map_lock);
	return ret;
}

/**
 * gpunvme_gpu_mem_unmap() - Release a GPU P2P DMA mapping
 * @gdev: gpu-nvme device context
 * @req:  unmap request (handle in)
 *
 * Return: 0 on success, negative errno on failure
 */
int gpunvme_gpu_mem_unmap(struct gpunvme_dev *gdev,
			  struct gpunvme_gpu_mem_unmap_req *req)
{
	struct gpunvme_gpu_mapping *mapping;
	int slot = (int)req->handle;

	if (slot < 0 || slot >= GPUNVME_MAX_GPU_MAPPINGS) {
		dev_err(&gdev->pdev->dev,
			"invalid GPU mapping handle: %d\n", slot);
		return -EINVAL;
	}

	mutex_lock(&gdev->gpu_map_lock);
	mapping = &gdev->gpu_maps[slot];

	if (!mapping->in_use) {
		mutex_unlock(&gdev->gpu_map_lock);
		dev_err(&gdev->pdev->dev,
			"GPU mapping slot %d not in use\n", slot);
		return -ENOENT;
	}

	/* DMA unmap first, then release pages */
	if (mapping->dma_mapping) {
		nvidia_p2p_dma_unmap_pages(gdev->pdev,
					   mapping->page_table,
					   mapping->dma_mapping);
		mapping->dma_mapping = NULL;
	}

	if (mapping->page_table) {
		nvidia_p2p_put_pages(0, 0, mapping->gpu_vaddr,
				     mapping->page_table);
		mapping->page_table = NULL;
	}

	dev_info(&gdev->pdev->dev,
		 "GPU P2P unmapped: vaddr=0x%llx slot=%d\n",
		 mapping->gpu_vaddr, slot);

	mapping->in_use = false;
	mutex_unlock(&gdev->gpu_map_lock);

	return 0;
}

/**
 * gpunvme_gpu_mem_unmap_all() - Release all GPU mappings for a device
 * @gdev: gpu-nvme device context
 *
 * Called during device removal to ensure all GPU memory is released.
 */
void gpunvme_gpu_mem_unmap_all(struct gpunvme_dev *gdev)
{
	struct gpunvme_gpu_mem_unmap_req req;
	int i;

	mutex_lock(&gdev->gpu_map_lock);
	for (i = 0; i < GPUNVME_MAX_GPU_MAPPINGS; i++) {
		if (gdev->gpu_maps[i].in_use) {
			mutex_unlock(&gdev->gpu_map_lock);
			req.handle = (uint64_t)i;
			gpunvme_gpu_mem_unmap(gdev, &req);
			mutex_lock(&gdev->gpu_map_lock);
		}
	}
	mutex_unlock(&gdev->gpu_map_lock);
}

#else /* !GPUNVME_HAVE_GPU_P2P */

/*
 * Stub implementations when NVIDIA P2P headers are not available.
 * The module still loads and provides BAR0 mmap functionality.
 */

int gpunvme_gpu_mem_map(struct gpunvme_dev *gdev,
			struct gpunvme_gpu_mem_map_req *req)
{
	dev_err(&gdev->pdev->dev,
		"GPU P2P DMA not available: module built without NVIDIA P2P support\n");
	dev_err(&gdev->pdev->dev,
		"rebuild with NVIDIA_SRC pointing to the NVIDIA driver source\n");
	return -EOPNOTSUPP;
}

int gpunvme_gpu_mem_unmap(struct gpunvme_dev *gdev,
			  struct gpunvme_gpu_mem_unmap_req *req)
{
	return -EOPNOTSUPP;
}

void gpunvme_gpu_mem_unmap_all(struct gpunvme_dev *gdev)
{
	/* Nothing to do — no mappings possible without P2P support */
}

#endif /* GPUNVME_HAVE_GPU_P2P */
