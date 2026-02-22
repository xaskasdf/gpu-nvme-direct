// Minimal kernel module to test nvidia_p2p_get_pages on GeForce RTX 3090
// Usage: echo <gpu_vaddr> <size> > /proc/test_p2p
// Read results from dmesg

#include <linux/module.h>
#include <linux/proc_fs.h>
#include <linux/uaccess.h>
#include <linux/pci.h>

#if __has_include("nv-p2p.h")
#include "nv-p2p.h"
#define HAVE_NV_P2P_H 1
#else
#define HAVE_NV_P2P_H 0
#endif

MODULE_LICENSE("Dual MIT/GPL");
MODULE_DESCRIPTION("Test nvidia_p2p_get_pages on GeForce");

#if HAVE_NV_P2P_H

static void p2p_free_callback(void *data) {
    pr_info("test_p2p: free callback invoked\n");
}

static ssize_t test_p2p_write(struct file *file, const char __user *buf,
                              size_t count, loff_t *ppos)
{
    char kbuf[64];
    uint64_t gpu_vaddr;
    uint64_t size;
    nvidia_p2p_page_table_t *page_table = NULL;
    int ret, i;

    if (count >= sizeof(kbuf))
        return -EINVAL;

    if (copy_from_user(kbuf, buf, count))
        return -EFAULT;
    kbuf[count] = '\0';

    if (sscanf(kbuf, "%llx %llu", &gpu_vaddr, &size) != 2) {
        pr_err("test_p2p: usage: echo <gpu_vaddr_hex> <size_dec> > /proc/test_p2p\n");
        return -EINVAL;
    }

    pr_info("test_p2p: calling nvidia_p2p_get_pages(0, 0, 0x%llx, %llu)\n",
            gpu_vaddr, size);

    ret = nvidia_p2p_get_pages(0, 0, gpu_vaddr, size,
                               &page_table, p2p_free_callback, NULL);

    if (ret) {
        pr_err("test_p2p: nvidia_p2p_get_pages FAILED: %d\n", ret);
        return count; // still consume input
    }

    pr_info("test_p2p: SUCCESS! Got %d pages (page_size=%d)\n",
            page_table->entries, page_table->page_size);

    for (i = 0; i < page_table->entries && i < 4; i++) {
        pr_info("test_p2p:   page[%d] physical_addr = 0x%llx\n",
                i, page_table->pages[i]->physical_address);
    }

    // Release the pages
    nvidia_p2p_put_pages(0, 0, gpu_vaddr, page_table);
    pr_info("test_p2p: pages released\n");

    return count;
}

static const struct proc_ops test_p2p_ops = {
    .proc_write = test_p2p_write,
};

static struct proc_dir_entry *proc_entry;

static int __init test_p2p_init(void) {
    proc_entry = proc_create("test_p2p", 0222, NULL, &test_p2p_ops);
    if (!proc_entry)
        return -ENOMEM;
    pr_info("test_p2p: loaded. Write to /proc/test_p2p to test.\n");
    return 0;
}

static void __exit test_p2p_exit(void) {
    proc_remove(proc_entry);
    pr_info("test_p2p: unloaded\n");
}

#else

static int __init test_p2p_init(void) {
    pr_err("test_p2p: compiled without nv-p2p.h support\n");
    return -ENOTSUP;
}

static void __exit test_p2p_exit(void) {}

#endif

module_init(test_p2p_init);
module_exit(test_p2p_exit);
