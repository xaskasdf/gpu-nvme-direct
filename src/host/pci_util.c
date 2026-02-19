/*
 * gpu-nvme-direct: PCI Utility Functions
 *
 * Discovers NVMe devices via sysfs, reads BAR information,
 * and provides helpers for VFIO-based userspace access.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <gpunvme/error.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <stdint.h>

/* NVMe PCI class code: 0x010802 (Mass storage > NVM > NVMe) */
#define PCI_CLASS_NVME 0x010802

/* PCI BAR info */
typedef struct {
    char bdf[16];            /* e.g., "0000:03:00.0" */
    uint64_t bar0_phys;      /* Physical address of BAR0 */
    uint64_t bar0_size;      /* Size of BAR0 in bytes */
    uint16_t vendor_id;
    uint16_t device_id;
    char driver[64];         /* Current bound driver */
} gpunvme_pci_dev_t;

/*
 * Read a hex value from a sysfs file.
 */
static int read_sysfs_hex(const char *path, uint64_t *val) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;

    char buf[64];
    if (!fgets(buf, sizeof(buf), f)) {
        fclose(f);
        return -1;
    }
    fclose(f);

    *val = strtoull(buf, NULL, 16);
    return 0;
}

/*
 * Read a string from a sysfs file.
 */
static int read_sysfs_str(const char *path, char *buf, size_t bufsz) {
    FILE *f = fopen(path, "r");
    if (!f) {
        buf[0] = '\0';
        return -1;
    }

    if (!fgets(buf, bufsz, f)) {
        fclose(f);
        buf[0] = '\0';
        return -1;
    }
    fclose(f);

    /* Strip trailing newline */
    size_t len = strlen(buf);
    if (len > 0 && buf[len - 1] == '\n') buf[len - 1] = '\0';
    return 0;
}

/*
 * Parse BAR0 address and size from /sys/bus/pci/devices/<bdf>/resource.
 * The resource file has one line per BAR:
 *   start_addr end_addr flags
 * BAR0 is the first line.
 */
static int parse_bar0_resource(const char *bdf, uint64_t *phys, uint64_t *size) {
    char path[256];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource", bdf);

    FILE *f = fopen(path, "r");
    if (!f) return -1;

    char line[256];
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return -1;
    }
    fclose(f);

    uint64_t start, end, flags;
    if (sscanf(line, "%llx %llx %llx",
               (unsigned long long *)&start,
               (unsigned long long *)&end,
               (unsigned long long *)&flags) != 3) {
        return -1;
    }

    *phys = start;
    *size = (start && end) ? (end - start + 1) : 0;
    return 0;
}

/*
 * Scan /sys/bus/pci/devices for NVMe controllers.
 * Returns number of devices found (up to max_devs).
 */
int gpunvme_find_nvme_devices(gpunvme_pci_dev_t *devs, int max_devs) {
    DIR *dir = opendir("/sys/bus/pci/devices");
    if (!dir) return 0;

    int count = 0;
    struct dirent *ent;

    while ((ent = readdir(dir)) != NULL && count < max_devs) {
        if (ent->d_name[0] == '.') continue;

        char path[256];
        uint64_t class_code;

        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/class", ent->d_name);
        if (read_sysfs_hex(path, &class_code) != 0) continue;

        /* Check NVMe class code (top 24 bits) */
        if ((class_code >> 8) != PCI_CLASS_NVME) continue;

        gpunvme_pci_dev_t *dev = &devs[count];
        strncpy(dev->bdf, ent->d_name, sizeof(dev->bdf) - 1);
        dev->bdf[sizeof(dev->bdf) - 1] = '\0';

        /* Read vendor/device IDs */
        uint64_t vid = 0, did = 0;
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/vendor", ent->d_name);
        read_sysfs_hex(path, &vid);
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/device", ent->d_name);
        read_sysfs_hex(path, &did);
        dev->vendor_id = (uint16_t)vid;
        dev->device_id = (uint16_t)did;

        /* Read BAR0 */
        parse_bar0_resource(ent->d_name, &dev->bar0_phys, &dev->bar0_size);

        /* Read current driver */
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/driver", ent->d_name);
        char link[256];
        ssize_t len = readlink(path, link, sizeof(link) - 1);
        if (len > 0) {
            link[len] = '\0';
            char *base = strrchr(link, '/');
            strncpy(dev->driver, base ? base + 1 : link, sizeof(dev->driver) - 1);
        } else {
            strcpy(dev->driver, "none");
        }

        count++;
    }

    closedir(dir);
    return count;
}

/*
 * Get BAR0 physical address and size for a specific BDF.
 */
gpunvme_err_t gpunvme_get_bar_info(const char *bdf, uint64_t *phys, uint64_t *size) {
    if (!bdf || !phys || !size) return GPUNVME_ERR_INVALID_PARAM;

    if (parse_bar0_resource(bdf, phys, size) != 0) {
        return GPUNVME_ERR_PCI;
    }

    if (*size == 0) {
        return GPUNVME_ERR_PCI;
    }

    return GPUNVME_OK;
}

/*
 * Enable bus mastering for PCI device (required for DMA).
 * Writes to PCI config space command register.
 */
gpunvme_err_t gpunvme_enable_bus_master(const char *bdf) {
    char path[256];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/config", bdf);

    int fd = open(path, O_RDWR);
    if (fd < 0) return GPUNVME_ERR_PCI;

    /* PCI command register at offset 0x04, 16-bit */
    uint16_t cmd;
    if (pread(fd, &cmd, 2, 0x04) != 2) {
        close(fd);
        return GPUNVME_ERR_PCI;
    }

    /* Set bit 2 (Bus Master Enable) and bit 1 (Memory Space Enable) */
    cmd |= 0x06;

    if (pwrite(fd, &cmd, 2, 0x04) != 2) {
        close(fd);
        return GPUNVME_ERR_PCI;
    }

    close(fd);
    return GPUNVME_OK;
}
