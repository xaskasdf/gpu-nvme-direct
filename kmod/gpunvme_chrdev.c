// SPDX-License-Identifier: GPL-2.0-only
/*
 * gpu-nvme-direct: Character device operations
 *
 * Implements /dev/gpunvme<N> with:
 *   - mmap:  Maps NVMe BAR0 into userspace (MMIO access to controller regs)
 *   - ioctl: GET_BAR_INFO, MAP_GPU_MEM, UNMAP_GPU_MEM, GET_PHYS_ADDR
 *
 * BAR0 is mapped uncacheable (UC) via io_remap_pfn_range, which is the
 * correct semantic for MMIO registers.  Userspace can then directly
 * read/write NVMe registers including doorbell registers.
 *
 * Copyright (c) 2024-2026 gpu-nvme-direct contributors
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/mm.h>
#include <linux/uaccess.h>
#include <linux/pci.h>
#include <linux/pagemap.h>

#include "gpunvme_internal.h"

/**
 * gpunvme_open() - Open the character device
 * @inode: inode for the device file
 * @filp:  file pointer
 *
 * Stores the per-device context in filp->private_data for use by
 * subsequent mmap/ioctl calls.
 *
 * Return: 0 on success
 */
static int gpunvme_open(struct inode *inode, struct file *filp)
{
	struct gpunvme_dev *gdev;

	gdev = container_of(inode->i_cdev, struct gpunvme_dev, cdev);
	filp->private_data = gdev;

	dev_dbg(&gdev->pdev->dev, "device opened (pid=%d)\n",
		current->pid);

	return 0;
}

/**
 * gpunvme_release() - Release / close the character device
 * @inode: inode for the device file
 * @filp:  file pointer
 *
 * Return: 0
 */
static int gpunvme_release(struct inode *inode, struct file *filp)
{
	struct gpunvme_dev *gdev = filp->private_data;

	dev_dbg(&gdev->pdev->dev, "device closed (pid=%d)\n",
		current->pid);

	return 0;
}

/**
 * gpunvme_mmap() - Map BAR0 into userspace
 * @filp: file pointer
 * @vma:  virtual memory area descriptor
 *
 * Maps the NVMe controller's BAR0 registers into the calling process's
 * address space.  The mapping uses uncacheable (UC) memory type, which
 * is required for correct MMIO semantics on x86.
 *
 * Userspace should mmap with MAP_SHARED and the desired offset/length.
 * Offset 0 corresponds to the start of BAR0.
 *
 * Return: 0 on success, negative errno on failure
 */
static int gpunvme_mmap(struct file *filp, struct vm_area_struct *vma)
{
	struct gpunvme_dev *gdev = filp->private_data;
	unsigned long requested_size;
	unsigned long offset;
	unsigned long pfn;

	offset = vma->vm_pgoff << PAGE_SHIFT;
	requested_size = vma->vm_end - vma->vm_start;

	/* Validate that the request fits within BAR0 */
	if (offset + requested_size > gdev->bar0_size) {
		dev_err(&gdev->pdev->dev,
			"mmap request out of range: offset=0x%lx size=0x%lx bar0_size=0x%llx\n",
			offset, requested_size,
			(unsigned long long)gdev->bar0_size);
		return -EINVAL;
	}

	/* Set uncacheable for MMIO */
	vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);

	/* Prevent the VMA from being copied on fork, expanded, etc. */
	vm_flags_set(vma, VM_IO | VM_DONTEXPAND | VM_DONTDUMP);

	/* Convert BAR0 physical address to page frame number */
	pfn = (gdev->bar0_phys + offset) >> PAGE_SHIFT;

	if (io_remap_pfn_range(vma, vma->vm_start, pfn, requested_size,
			       vma->vm_page_prot)) {
		dev_err(&gdev->pdev->dev, "io_remap_pfn_range failed\n");
		return -EAGAIN;
	}

	dev_dbg(&gdev->pdev->dev,
		"mmap: BAR0 offset=0x%lx size=0x%lx -> user=0x%lx\n",
		offset, requested_size, vma->vm_start);

	return 0;
}

/**
 * gpunvme_ioctl_get_bar_info() - Handle GPUNVME_IOCTL_GET_BAR_INFO
 * @gdev: device context
 * @arg:  userspace pointer to struct gpunvme_bar_info
 *
 * Return: 0 on success, negative errno on failure
 */
static int gpunvme_ioctl_get_bar_info(struct gpunvme_dev *gdev,
				      unsigned long arg)
{
	struct gpunvme_bar_info info;

	info.phys_addr = gdev->bar0_phys;
	info.size = gdev->bar0_size;

	if (copy_to_user((void __user *)arg, &info, sizeof(info)))
		return -EFAULT;

	return 0;
}

/**
 * gpunvme_ioctl_map_gpu_mem() - Handle GPUNVME_IOCTL_MAP_GPU_MEM
 * @gdev: device context
 * @arg:  userspace pointer to struct gpunvme_gpu_mem_map_req
 *
 * Return: 0 on success, negative errno on failure
 */
static int gpunvme_ioctl_map_gpu_mem(struct gpunvme_dev *gdev,
				     unsigned long arg)
{
	struct gpunvme_gpu_mem_map_req req;
	int ret;

	if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
		return -EFAULT;

	ret = gpunvme_gpu_mem_map(gdev, &req);
	if (ret)
		return ret;

	if (copy_to_user((void __user *)arg, &req, sizeof(req)))
		return -EFAULT;

	return 0;
}

/**
 * gpunvme_ioctl_unmap_gpu_mem() - Handle GPUNVME_IOCTL_UNMAP_GPU_MEM
 * @gdev: device context
 * @arg:  userspace pointer to struct gpunvme_gpu_mem_unmap_req
 *
 * Return: 0 on success, negative errno on failure
 */
static int gpunvme_ioctl_unmap_gpu_mem(struct gpunvme_dev *gdev,
				       unsigned long arg)
{
	struct gpunvme_gpu_mem_unmap_req req;

	if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
		return -EFAULT;

	return gpunvme_gpu_mem_unmap(gdev, &req);
}

/**
 * gpunvme_ioctl_get_phys_addr() - Handle GPUNVME_IOCTL_GET_PHYS_ADDR
 * @gdev: device context
 * @arg:  userspace pointer to struct gpunvme_phys_addr_req
 *
 * Translates a userspace virtual address to its physical address by
 * walking the process page tables.  Requires that the page is currently
 * present and pinned (e.g., via mlock or CUDA pinned allocation).
 *
 * Return: 0 on success, negative errno on failure
 */
static int gpunvme_ioctl_get_phys_addr(struct gpunvme_dev *gdev,
					unsigned long arg)
{
	struct gpunvme_phys_addr_req req;
	struct page *page = NULL;
	int ret;

	if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
		return -EFAULT;

	/* Pin the user page to ensure it is present */
	ret = pin_user_pages_fast(req.vaddr, 1, 0, &page);
	if (ret < 0) {
		dev_err(&gdev->pdev->dev,
			"pin_user_pages_fast failed for vaddr=0x%llx: %d\n",
			req.vaddr, ret);
		return ret;
	}

	if (ret != 1) {
		dev_err(&gdev->pdev->dev,
			"pin_user_pages_fast returned %d pages (expected 1)\n",
			ret);
		return -EFAULT;
	}

	/* Compute physical address: page base + offset within page */
	req.phys_addr = page_to_phys(page) +
			offset_in_page((unsigned long)req.vaddr);

	/* Unpin — the page stays in memory as long as the user holds it */
	unpin_user_page(page);

	if (copy_to_user((void __user *)arg, &req, sizeof(req)))
		return -EFAULT;

	return 0;
}

/**
 * gpunvme_ioctl() - Dispatch ioctl commands
 * @filp: file pointer
 * @cmd:  ioctl command number
 * @arg:  userspace argument pointer
 *
 * Return: 0 on success, negative errno on failure
 */
static long gpunvme_ioctl(struct file *filp, unsigned int cmd,
			   unsigned long arg)
{
	struct gpunvme_dev *gdev = filp->private_data;

	switch (cmd) {
	case GPUNVME_IOCTL_GET_BAR_INFO:
		return gpunvme_ioctl_get_bar_info(gdev, arg);

	case GPUNVME_IOCTL_MAP_GPU_MEM:
		return gpunvme_ioctl_map_gpu_mem(gdev, arg);

	case GPUNVME_IOCTL_UNMAP_GPU_MEM:
		return gpunvme_ioctl_unmap_gpu_mem(gdev, arg);

	case GPUNVME_IOCTL_GET_PHYS_ADDR:
		return gpunvme_ioctl_get_phys_addr(gdev, arg);

	default:
		dev_dbg(&gdev->pdev->dev,
			"unknown ioctl cmd=0x%x\n", cmd);
		return -ENOTTY;
	}
}

/* File operations for /dev/gpunvme<N> */
const struct file_operations gpunvme_fops = {
	.owner          = THIS_MODULE,
	.open           = gpunvme_open,
	.release        = gpunvme_release,
	.mmap           = gpunvme_mmap,
	.unlocked_ioctl = gpunvme_ioctl,
	.compat_ioctl   = compat_ptr_ioctl,
};

/**
 * gpunvme_chrdev_create() - Create the character device for a gpunvme device
 * @gdev: per-device context (must have index and pdev set)
 *
 * Registers a cdev and creates a device node at /dev/gpunvme<N>.
 *
 * Return: 0 on success, negative errno on failure
 */
int gpunvme_chrdev_create(struct gpunvme_dev *gdev)
{
	int ret;

	gdev->dev_num = MKDEV(gpunvme_major, gdev->index);

	cdev_init(&gdev->cdev, &gpunvme_fops);
	gdev->cdev.owner = THIS_MODULE;

	ret = cdev_add(&gdev->cdev, gdev->dev_num, 1);
	if (ret) {
		dev_err(&gdev->pdev->dev,
			"cdev_add failed: %d\n", ret);
		return ret;
	}

	gdev->dev = device_create(gpunvme_class, &gdev->pdev->dev,
				  gdev->dev_num, gdev,
				  GPUNVME_DRV_NAME "%d", gdev->index);
	if (IS_ERR(gdev->dev)) {
		ret = PTR_ERR(gdev->dev);
		dev_err(&gdev->pdev->dev,
			"device_create failed: %d\n", ret);
		cdev_del(&gdev->cdev);
		return ret;
	}

	return 0;
}

/**
 * gpunvme_chrdev_destroy() - Destroy the character device
 * @gdev: per-device context
 */
void gpunvme_chrdev_destroy(struct gpunvme_dev *gdev)
{
	if (gdev->dev) {
		device_destroy(gpunvme_class, gdev->dev_num);
		gdev->dev = NULL;
	}
	cdev_del(&gdev->cdev);
}
