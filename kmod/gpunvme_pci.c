// SPDX-License-Identifier: GPL-2.0-only
/*
 * gpu-nvme-direct: PCI device probe and removal
 *
 * Binds to NVMe-class PCI devices, maps BAR0, and creates the
 * corresponding character device.  Respects the optional target_bdf
 * module parameter to restrict binding to a specific controller.
 *
 * Copyright (c) 2024-2026 gpu-nvme-direct contributors
 */

#include <linux/module.h>
#include <linux/pci.h>
#include <linux/kernel.h>
#include <linux/slab.h>

#include "gpunvme_internal.h"

/* Atomic counter for device indices */
static atomic_t gpunvme_dev_count = ATOMIC_INIT(0);

/* Forward declaration for BDF filter */
extern const char *gpunvme_get_target_bdf(void);

/**
 * gpunvme_pci_probe() - Called when a matching PCI device is found
 * @pdev: PCI device being probed
 * @id:   Matching PCI device ID entry
 *
 * Performs the following steps:
 *   1. Check target_bdf filter
 *   2. Enable the PCI device
 *   3. Request BAR0 region
 *   4. Map BAR0 into kernel virtual address space
 *   5. Enable bus mastering (required for DMA)
 *   6. Set 64-bit DMA mask
 *   7. Create the character device
 *
 * Return: 0 on success, negative errno on failure
 */
static int gpunvme_pci_probe(struct pci_dev *pdev,
			     const struct pci_device_id *id)
{
	struct gpunvme_dev *gdev;
	const char *filter;
	int ret;

	/* Check BDF filter if specified */
	filter = gpunvme_get_target_bdf();
	if (filter) {
		const char *devname = dev_name(&pdev->dev);

		if (strcmp(devname, filter) != 0) {
			dev_dbg(&pdev->dev,
				"skipping (BDF %s does not match filter %s)\n",
				devname, filter);
			return -ENODEV;
		}
	}

	dev_info(&pdev->dev, "probing NVMe device %04x:%04x\n",
		 pdev->vendor, pdev->device);

	/* Allocate per-device context */
	gdev = kzalloc(sizeof(*gdev), GFP_KERNEL);
	if (!gdev)
		return -ENOMEM;

	gdev->pdev = pdev;
	gdev->index = atomic_fetch_add(1, &gpunvme_dev_count);
	mutex_init(&gdev->gpu_map_lock);

	if (gdev->index >= GPUNVME_MAX_DEVICES) {
		dev_err(&pdev->dev,
			"maximum device count (%d) reached\n",
			GPUNVME_MAX_DEVICES);
		ret = -ENOSPC;
		goto err_free;
	}

	/* Enable the PCI device */
	ret = pci_enable_device(pdev);
	if (ret) {
		dev_err(&pdev->dev, "pci_enable_device failed: %d\n", ret);
		goto err_free;
	}

	/* Request BAR0 region */
	ret = pci_request_region(pdev, 0, GPUNVME_DRV_NAME);
	if (ret) {
		dev_err(&pdev->dev, "pci_request_region(BAR0) failed: %d\n",
			ret);
		goto err_disable;
	}

	/* Record BAR0 physical address and size */
	gdev->bar0_phys = pci_resource_start(pdev, 0);
	gdev->bar0_size = pci_resource_len(pdev, 0);

	if (gdev->bar0_size == 0) {
		dev_err(&pdev->dev, "BAR0 has zero size\n");
		ret = -ENODEV;
		goto err_release;
	}

	/* Map BAR0 into kernel address space */
	gdev->bar0 = pci_iomap(pdev, 0, 0);
	if (!gdev->bar0) {
		dev_err(&pdev->dev, "pci_iomap(BAR0) failed\n");
		ret = -ENOMEM;
		goto err_release;
	}

	dev_info(&pdev->dev, "BAR0 mapped: phys=0x%llx size=0x%llx virt=%p\n",
		 (unsigned long long)gdev->bar0_phys,
		 (unsigned long long)gdev->bar0_size,
		 gdev->bar0);

	/* Enable bus mastering for DMA */
	pci_set_master(pdev);

	/* Set 64-bit DMA mask, fall back to 32-bit */
	ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(64));
	if (ret) {
		dev_warn(&pdev->dev,
			 "64-bit DMA mask failed, trying 32-bit\n");
		ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(32));
		if (ret) {
			dev_err(&pdev->dev, "DMA mask setup failed: %d\n",
				ret);
			goto err_unmap;
		}
	}

	/* Store context in PCI device private data */
	pci_set_drvdata(pdev, gdev);

	/* Create character device /dev/gpunvme<N> */
	ret = gpunvme_chrdev_create(gdev);
	if (ret) {
		dev_err(&pdev->dev,
			"failed to create character device: %d\n", ret);
		goto err_unmap;
	}

	dev_info(&pdev->dev,
		 "gpu-nvme-direct device ready: /dev/gpunvme%d\n",
		 gdev->index);

	return 0;

err_unmap:
	pci_clear_master(pdev);
	pci_iounmap(pdev, gdev->bar0);
err_release:
	pci_release_region(pdev, 0);
err_disable:
	pci_disable_device(pdev);
err_free:
	kfree(gdev);
	return ret;
}

/**
 * gpunvme_pci_remove() - Called when the PCI device is removed or driver unloads
 * @pdev: PCI device being removed
 *
 * Reverses the probe sequence: destroys the character device, unmaps BAR0,
 * releases regions, and disables the device.
 */
static void gpunvme_pci_remove(struct pci_dev *pdev)
{
	struct gpunvme_dev *gdev = pci_get_drvdata(pdev);

	if (!gdev)
		return;

	dev_info(&pdev->dev, "removing /dev/gpunvme%d\n", gdev->index);

	/* Tear down GPU P2P mappings */
	gpunvme_gpu_mem_unmap_all(gdev);

	/* Destroy character device */
	gpunvme_chrdev_destroy(gdev);

	/* Unmap BAR0 */
	pci_clear_master(pdev);
	if (gdev->bar0) {
		pci_iounmap(pdev, gdev->bar0);
		gdev->bar0 = NULL;
	}

	/* Release PCI resources */
	pci_release_region(pdev, 0);
	pci_disable_device(pdev);

	pci_set_drvdata(pdev, NULL);
	kfree(gdev);
}

/*
 * PCI device ID table.
 *
 * Match any device in the NVMe storage class (01:08:02).
 * The kernel's PCI subsystem uses class/class_mask to match.
 */
static const struct pci_device_id gpunvme_pci_ids[] = {
	{ PCI_DEVICE_CLASS(PCI_CLASS_STORAGE_EXPRESS << 8, 0xFFFF00) },
	{ 0 }
};
MODULE_DEVICE_TABLE(pci, gpunvme_pci_ids);

static struct pci_driver gpunvme_pci_driver = {
	.name     = GPUNVME_DRV_NAME,
	.id_table = gpunvme_pci_ids,
	.probe    = gpunvme_pci_probe,
	.remove   = gpunvme_pci_remove,
};

/**
 * gpunvme_pci_register() - Register the PCI driver
 *
 * Return: 0 on success, negative errno on failure
 */
int gpunvme_pci_register(void)
{
	return pci_register_driver(&gpunvme_pci_driver);
}

/**
 * gpunvme_pci_unregister() - Unregister the PCI driver
 */
void gpunvme_pci_unregister(void)
{
	pci_unregister_driver(&gpunvme_pci_driver);
}
