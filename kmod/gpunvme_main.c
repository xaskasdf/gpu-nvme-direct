// SPDX-License-Identifier: GPL-2.0-only
/*
 * gpu-nvme-direct: Main module entry point
 *
 * Registers the PCI driver for NVMe-class devices and creates the
 * "gpunvme" character device class.  The optional "target_bdf" module
 * parameter restricts which NVMe controller the driver will claim.
 *
 * Copyright (c) 2024-2026 gpu-nvme-direct contributors
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/device.h>
#include <linux/fs.h>

#include "gpunvme_internal.h"

/* ---- Module parameters ---- */

static char *target_bdf;
module_param(target_bdf, charp, 0444);
MODULE_PARM_DESC(target_bdf,
	"PCI BDF of the NVMe device to claim, e.g. \"0000:03:00.0\". "
	"If unset, the driver claims the first unbound NVMe device.");

/* ---- Globals ---- */

struct class *gpunvme_class;
int gpunvme_major;

/**
 * gpunvme_get_target_bdf() - Return the target BDF filter string
 *
 * Returns the module parameter value, or NULL if not set.
 */
const char *gpunvme_get_target_bdf(void)
{
	if (target_bdf && target_bdf[0] != '\0')
		return target_bdf;
	return NULL;
}
EXPORT_SYMBOL_GPL(gpunvme_get_target_bdf);

/* ---- Module init / exit ---- */

static int __init gpunvme_init(void)
{
	dev_t dev_num;
	int ret;

	pr_info(GPUNVME_DRV_NAME ": initializing gpu-nvme-direct kernel module\n");

	if (target_bdf && target_bdf[0] != '\0')
		pr_info(GPUNVME_DRV_NAME ": target BDF filter: %s\n",
			target_bdf);

	/* Allocate a range of character device numbers */
	ret = alloc_chrdev_region(&dev_num, 0, GPUNVME_MAX_DEVICES,
				  GPUNVME_DRV_NAME);
	if (ret < 0) {
		pr_err(GPUNVME_DRV_NAME ": failed to allocate chrdev region: %d\n",
		       ret);
		return ret;
	}
	gpunvme_major = MAJOR(dev_num);

	/* Create device class for /dev/gpunvme<N> nodes */
	gpunvme_class = class_create(GPUNVME_DRV_NAME);
	if (IS_ERR(gpunvme_class)) {
		ret = PTR_ERR(gpunvme_class);
		pr_err(GPUNVME_DRV_NAME ": failed to create device class: %d\n",
		       ret);
		goto err_chrdev;
	}

	/* Register PCI driver */
	ret = gpunvme_pci_register();
	if (ret < 0) {
		pr_err(GPUNVME_DRV_NAME ": failed to register PCI driver: %d\n",
		       ret);
		goto err_class;
	}

	pr_info(GPUNVME_DRV_NAME ": module loaded (major=%d)\n",
		gpunvme_major);
	return 0;

err_class:
	class_destroy(gpunvme_class);
err_chrdev:
	unregister_chrdev_region(MKDEV(gpunvme_major, 0), GPUNVME_MAX_DEVICES);
	return ret;
}

static void __exit gpunvme_exit(void)
{
	gpunvme_pci_unregister();
	class_destroy(gpunvme_class);
	unregister_chrdev_region(MKDEV(gpunvme_major, 0), GPUNVME_MAX_DEVICES);

	pr_info(GPUNVME_DRV_NAME ": module unloaded\n");
}

module_init(gpunvme_init);
module_exit(gpunvme_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("gpu-nvme-direct contributors");
MODULE_DESCRIPTION("GPU-NVMe Direct: BAR0 mmap and GPU P2P DMA for userspace NVMe drivers");
MODULE_VERSION("0.1.0");
