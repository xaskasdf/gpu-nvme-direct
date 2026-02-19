/*
 * gpu-nvme-direct: Error Codes
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#ifndef GPUNVME_ERROR_H
#define GPUNVME_ERROR_H

#ifdef __cplusplus
extern "C" {
#endif

/* Library error codes (negative values; 0 = success) */
typedef enum {
    GPUNVME_OK                  =  0,
    GPUNVME_ERR_NOMEM           = -1,  /* Memory allocation failed */
    GPUNVME_ERR_TIMEOUT         = -2,  /* Operation timed out */
    GPUNVME_ERR_NVME_STATUS     = -3,  /* NVMe command returned error status */
    GPUNVME_ERR_CTRL_FATAL      = -4,  /* Controller Fatal Status (CSTS.CFS) */
    GPUNVME_ERR_BAR_MAP         = -5,  /* BAR0 mapping failed */
    GPUNVME_ERR_CUDA            = -6,  /* CUDA operation failed */
    GPUNVME_ERR_PCI             = -7,  /* PCI/VFIO operation failed */
    GPUNVME_ERR_INVALID_PARAM   = -8,  /* Invalid parameter */
    GPUNVME_ERR_NOT_READY       = -9,  /* Controller not in expected state */
    GPUNVME_ERR_QUEUE_FULL      = -10, /* Submission queue is full */
    GPUNVME_ERR_DMA             = -11, /* DMA mapping failed */
    GPUNVME_ERR_P2P             = -12, /* PCIe peer-to-peer not available */
    GPUNVME_ERR_IO              = -13, /* I/O error (file, sysfs, etc.) */
    GPUNVME_ERR_NOT_SUPPORTED   = -14, /* Feature not supported */
} gpunvme_err_t;

/* Convert error code to string */
static inline const char *gpunvme_err_str(gpunvme_err_t err) {
    switch (err) {
    case GPUNVME_OK:                return "success";
    case GPUNVME_ERR_NOMEM:         return "out of memory";
    case GPUNVME_ERR_TIMEOUT:       return "timeout";
    case GPUNVME_ERR_NVME_STATUS:   return "NVMe error status";
    case GPUNVME_ERR_CTRL_FATAL:    return "controller fatal status";
    case GPUNVME_ERR_BAR_MAP:       return "BAR mapping failed";
    case GPUNVME_ERR_CUDA:          return "CUDA error";
    case GPUNVME_ERR_PCI:           return "PCI error";
    case GPUNVME_ERR_INVALID_PARAM: return "invalid parameter";
    case GPUNVME_ERR_NOT_READY:     return "not ready";
    case GPUNVME_ERR_QUEUE_FULL:    return "queue full";
    case GPUNVME_ERR_DMA:           return "DMA error";
    case GPUNVME_ERR_P2P:           return "P2P not available";
    case GPUNVME_ERR_IO:            return "I/O error";
    case GPUNVME_ERR_NOT_SUPPORTED: return "not supported";
    default:                        return "unknown error";
    }
}

/* NVMe status code types */
#define NVME_SCT_GENERIC    0x0
#define NVME_SCT_CMDSPECIFIC 0x1
#define NVME_SCT_MEDIA      0x2
#define NVME_SCT_PATH       0x3
#define NVME_SCT_VENDOR     0x7

/* Common NVMe generic status codes */
#define NVME_SC_SUCCESS             0x00
#define NVME_SC_INVALID_OPCODE      0x01
#define NVME_SC_INVALID_FIELD       0x02
#define NVME_SC_DATA_XFER_ERROR     0x04
#define NVME_SC_INTERNAL_ERROR      0x06

#ifdef __cplusplus
}
#endif

#endif /* GPUNVME_ERROR_H */
