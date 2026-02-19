/*
 * rm_bar0_register: Register NVMe BAR0 with GPU via direct RM ioctls
 *
 * Bypasses libcuda.so's capability check by calling the NVIDIA kernel
 * module's RM (Resource Manager) ioctls directly.
 *
 * Flow:
 *   1. mmap NVMe BAR0 via sysfs resource0
 *   2. Open /dev/nvidiactl (control device)
 *   3. Allocate RM root client (NV01_ROOT_CLIENT = 0x41)
 *   4. Attach GPU to FD
 *   5. Allocate device (NV01_DEVICE_0 = 0x80) on /dev/nvidia0
 *   6. Call NV_ESC_RM_ALLOC_MEMORY with NV01_MEMORY_SYSTEM_OS_DESCRIPTOR
 *      using the mmap'd BAR0 pointer
 *   7. Kernel auto-detects IO memory via os_lookup_user_io_memory()
 *
 * Build: gcc-14 -o rm_bar0_register rm_bar0_register.c
 * Usage: sudo ./rm_bar0_register <NVMe_PCI_BDF>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <errno.h>

/* ── NVIDIA ioctl constants (from open-gpu-kernel-modules) ────────── */

#define NV_IOCTL_MAGIC      'F'
#define NV_IOCTL_BASE       200

/* nv-ioctl-numbers.h */
#define NV_ESC_REGISTER_FD          (NV_IOCTL_BASE + 1)
#define NV_ESC_IOCTL_XFER_CMD      (NV_IOCTL_BASE + 11)
#define NV_ESC_ATTACH_GPUS_TO_FD   (NV_IOCTL_BASE + 12)

/* nv_escape.h */
#define NV_ESC_RM_ALLOC_MEMORY     0x27
#define NV_ESC_RM_ALLOC            0x2B
#define NV_ESC_RM_FREE             0x29
#define NV_ESC_RM_MAP_MEMORY       0x4E

/* Class IDs */
#define NV01_ROOT                           0x00000000
#define NV01_ROOT_CLIENT                    0x00000041
#define NV01_DEVICE_0                       0x00000080
#define NV20_SUBDEVICE_0                    0x00002080
#define NV01_MEMORY_SYSTEM_OS_DESCRIPTOR    0x00000071

/* NVOS02 flags (for NV_ESC_RM_ALLOC_MEMORY) */
#define NVOS02_FLAGS_PHYSICALITY_CONTIGUOUS     (0x00000000 << 4)
#define NVOS02_FLAGS_LOCATION_PCI               (0x00000000 << 8)
#define NVOS02_FLAGS_COHERENCY_UNCACHED         (0x00000000 << 12)
#define NVOS02_FLAGS_ALLOC_NONE                 (0x00000001 << 16)
#define NVOS02_FLAGS_GPU_CACHEABLE_NO           (0x00000000 << 18)
#define NVOS02_FLAGS_PEER_MAP_OVERRIDE_REQUIRED (0x00000001 << 23)
#define NVOS02_FLAGS_MAPPING_NO_MAP             (0x00000001 << 30)

/* ── NVIDIA ioctl structures ─────────────────────────────────────── */

/* nv_ioctl_xfer_t: wrapper for large parameter structures */
typedef struct {
    uint32_t  cmd;
    uint32_t  size;
    uint64_t  ptr;    /* NvP64 — user pointer to actual params */
} __attribute__((packed)) nv_ioctl_xfer_t;

/* nv_ioctl_register_fd_t */
typedef struct {
    int ctl_fd;
} nv_ioctl_register_fd_t;

/*
 * NVOS21_PARAMETERS: for NV_ESC_RM_ALLOC
 * Matches kernel layout from nvos.h — NO packed attribute.
 * NV_ALIGN_BYTES(8) = __attribute__((aligned(8))) on 64-bit fields.
 */
typedef struct {
    uint32_t  hRoot;
    uint32_t  hObjectParent;
    uint32_t  hObjectNew;
    uint32_t  hClass;
    uint64_t  pAllocParms __attribute__((aligned(8)));
    uint32_t  paramsSize;
    uint32_t  status;
} NVOS21_PARAMETERS;

/*
 * NVOS02_PARAMETERS: for NV_ESC_RM_ALLOC_MEMORY
 * Layout from nvos.h: 5 x uint32_t, then aligned(8) pMemory and limit.
 * Compiler inserts 4-byte pad after flags automatically.
 */
typedef struct {
    uint32_t  hRoot;           /* offset 0 */
    uint32_t  hObjectParent;   /* offset 4 */
    uint32_t  hObjectNew;      /* offset 8 */
    uint32_t  hClass;          /* offset 12 */
    uint32_t  flags;           /* offset 16 */
    /* compiler pads 4 bytes here to align pMemory to 8 */
    uint64_t  pMemory __attribute__((aligned(8)));  /* offset 24 */
    uint64_t  limit   __attribute__((aligned(8)));  /* offset 32 */
    uint32_t  status;          /* offset 40 */
    /* compiler pads 4 bytes here to align struct to 8 */
} NVOS02_PARAMETERS;           /* sizeof = 48 */

/* Wrapper for NV_ESC_RM_ALLOC_MEMORY (includes fd) */
typedef struct {
    NVOS02_PARAMETERS params;  /* 48 bytes */
    int fd;                    /* offset 48 */
    /* compiler pads 4 bytes to align struct to 8 → sizeof = 56 */
} nv_ioctl_nvos02_parameters_with_fd;

/* NV0080_ALLOC_PARAMETERS: for NV01_DEVICE_0 */
typedef struct {
    uint32_t  deviceId;
    uint32_t  hClientShare;
    uint32_t  hTargetClient;
    uint32_t  hTargetDevice;
    uint32_t  flags;
    /* compiler pads 4 bytes here */
    uint64_t  vaSpaceSize    __attribute__((aligned(8)));
    uint64_t  vaStartInternal __attribute__((aligned(8)));
    uint64_t  vaLimitInternal __attribute__((aligned(8)));
    uint32_t  vaMode;
} NV0080_ALLOC_PARAMETERS;

/* NV2080_ALLOC_PARAMETERS: for NV20_SUBDEVICE_0 */
typedef struct {
    uint32_t  subDeviceId;
} NV2080_ALLOC_PARAMETERS;

/* ── Helpers ──────────────────────────────────────────────────────── */

static int nv_xfer_ioctl(int fd, uint32_t cmd, void *params, uint32_t size)
{
    nv_ioctl_xfer_t xfer;
    xfer.cmd  = cmd;
    xfer.size = size;
    xfer.ptr  = (uint64_t)(uintptr_t)params;

    int ioctl_nr = _IOC(_IOC_READ | _IOC_WRITE, NV_IOCTL_MAGIC,
                        NV_ESC_IOCTL_XFER_CMD, sizeof(nv_ioctl_xfer_t));
    return ioctl(fd, ioctl_nr, &xfer);
}

static const char *nv_status_str(uint32_t status)
{
    switch (status) {
    case 0x00: return "NV_OK";
    case 0x03: return "NV_ERR_NOT_SUPPORTED";
    case 0x06: return "NV_ERR_INVALID_ARGUMENT";
    case 0x0D: return "NV_ERR_INVALID_OBJECT_HANDLE";
    case 0x0E: return "NV_ERR_INVALID_OBJECT_PARENT";
    case 0x0F: return "NV_ERR_INVALID_OBJECT_NEW";
    case 0x13: return "NV_ERR_INVALID_STATE";
    case 0x14: return "NV_ERR_INVALID_FLAGS";
    case 0x16: return "NV_ERR_INVALID_ADDRESS";
    case 0x17: return "NV_ERR_INSUFFICIENT_PERMISSIONS";
    case 0x1A: return "NV_ERR_OBJECT_NOT_FOUND";
    case 0x24: return "NV_ERR_INVALID_CLASS";
    case 0x26: return "NV_ERR_INSUFFICIENT_RESOURCES";
    case 0x34: return "NV_ERR_INVALID_DEVICE";
    case 0x57: return "NV_ERR_INVALID_PARAMETER";
    case 0x65: return "NV_ERR_GENERIC";
    default:   return "(unknown)";
    }
}

/* ── Main ─────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <NVMe_PCI_BDF>\n", argv[0]);
        fprintf(stderr, "Example: %s 0000:0b:00.0\n", argv[0]);
        return 1;
    }

    const char *bdf = argv[1];
    int ret;

    /* ── Step 1: mmap NVMe BAR0 ─────────────────────────────────── */
    char path[256];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource0", bdf);
    int bar_fd = open(path, O_RDWR | O_SYNC);
    if (bar_fd < 0) {
        perror("open BAR0 resource0");
        return 1;
    }
    off_t bar_size = lseek(bar_fd, 0, SEEK_END);
    lseek(bar_fd, 0, SEEK_SET);
    printf("BAR0 size: 0x%lx (%ld KB)\n", (long)bar_size, (long)bar_size / 1024);

    volatile void *bar0 = mmap(NULL, bar_size, PROT_READ | PROT_WRITE,
                                MAP_SHARED, bar_fd, 0);
    if (bar0 == MAP_FAILED) {
        perror("mmap BAR0");
        close(bar_fd);
        return 1;
    }

    /* Debug: verify struct sizes match kernel expectations */
    printf("Struct sizes: NVOS02=%zu, NVOS02+fd=%zu, NVOS21=%zu\n",
           sizeof(NVOS02_PARAMETERS),
           sizeof(nv_ioctl_nvos02_parameters_with_fd),
           sizeof(NVOS21_PARAMETERS));

    /* Quick sanity: read NVMe VS register */
    volatile uint32_t *vs_ptr = (volatile uint32_t *)((char *)bar0 + 0x08);
    uint32_t vs = *vs_ptr;
    printf("CPU read NVMe VS = 0x%08x (NVMe %u.%u)\n\n",
           vs, (vs >> 16) & 0xFFFF, (vs >> 8) & 0xFF);

    if (vs == 0xFFFFFFFF) {
        fprintf(stderr, "ERROR: BAR0 reads all-Fs. Device in D3? Fix power state first.\n");
        munmap((void *)bar0, bar_size);
        close(bar_fd);
        return 1;
    }

    /* ── Step 2: Open NVIDIA control device ──────────────────────── */
    int ctl_fd = open("/dev/nvidiactl", O_RDWR);
    if (ctl_fd < 0) {
        perror("open /dev/nvidiactl");
        goto cleanup_bar;
    }
    printf("Opened /dev/nvidiactl (fd=%d)\n", ctl_fd);

    int gpu_fd = open("/dev/nvidia0", O_RDWR);
    if (gpu_fd < 0) {
        perror("open /dev/nvidia0");
        goto cleanup_ctl;
    }
    printf("Opened /dev/nvidia0 (fd=%d)\n", gpu_fd);

    /* ── Step 3: Register gpu_fd with ctl_fd ─────────────────────── */
    {
        nv_ioctl_register_fd_t reg = { .ctl_fd = ctl_fd };
        int ioctl_nr = _IOC(_IOC_READ | _IOC_WRITE, NV_IOCTL_MAGIC,
                            NV_ESC_REGISTER_FD, sizeof(reg));
        ret = ioctl(gpu_fd, ioctl_nr, &reg);
        if (ret < 0) {
            printf("Register FD: ioctl returned %d (errno=%d: %s)\n",
                   ret, errno, strerror(errno));
            /* Non-fatal, continue */
        } else {
            printf("Registered gpu_fd with ctl_fd: OK\n");
        }
    }

    /* ── Step 4: Allocate RM root client ─────────────────────────── */
    uint32_t hClient = 0;
    {
        NVOS21_PARAMETERS alloc;
        memset(&alloc, 0, sizeof(alloc));
        alloc.hRoot = 0;
        alloc.hObjectParent = 0;
        alloc.hObjectNew = 0;  /* 0 = auto-generate */
        alloc.hClass = NV01_ROOT;  /* Kernel forces to NV01_ROOT_CLIENT */
        alloc.pAllocParms = 0;
        alloc.paramsSize = 0;
        alloc.status = 0xFFFFFFFF;

        ret = nv_xfer_ioctl(ctl_fd, NV_ESC_RM_ALLOC, &alloc, sizeof(alloc));
        printf("\n=== RM_ALLOC Root Client ===\n");
        printf("  ioctl ret: %d (errno=%d)\n", ret, ret < 0 ? errno : 0);
        printf("  status: 0x%02x (%s)\n", alloc.status, nv_status_str(alloc.status));
        printf("  hObjectNew: 0x%08x\n", alloc.hObjectNew);

        if (alloc.status != 0) {
            fprintf(stderr, "Failed to allocate root client\n");
            goto cleanup_gpu;
        }
        hClient = alloc.hObjectNew;
    }

    /* ── Step 5: Attach GPU to FD ────────────────────────────────── */
    {
        /* The attach ioctl takes a GPU ID (0 for first GPU) */
        uint32_t gpu_ids[2] = { 0x00000100, 0 };  /* GPU ID for device 0 */
        int ioctl_nr = _IOC(_IOC_READ | _IOC_WRITE, NV_IOCTL_MAGIC,
                            NV_ESC_ATTACH_GPUS_TO_FD, sizeof(gpu_ids));
        ret = ioctl(gpu_fd, ioctl_nr, gpu_ids);
        printf("\n=== Attach GPUs to FD ===\n");
        printf("  ioctl ret: %d (errno=%d)\n", ret, ret < 0 ? errno : 0);
    }

    /* ── Step 6: Allocate device ─────────────────────────────────── */
    uint32_t hDevice = 0;
    {
        NV0080_ALLOC_PARAMETERS devParams;
        memset(&devParams, 0, sizeof(devParams));
        devParams.deviceId = 0;  /* First GPU */

        NVOS21_PARAMETERS alloc;
        memset(&alloc, 0, sizeof(alloc));
        alloc.hRoot = hClient;
        alloc.hObjectParent = hClient;
        alloc.hObjectNew = 0;  /* auto */
        alloc.hClass = NV01_DEVICE_0;
        alloc.pAllocParms = (uint64_t)(uintptr_t)&devParams;
        alloc.paramsSize = sizeof(devParams);
        alloc.status = 0xFFFFFFFF;

        ret = nv_xfer_ioctl(ctl_fd, NV_ESC_RM_ALLOC, &alloc, sizeof(alloc));
        printf("\n=== RM_ALLOC Device ===\n");
        printf("  ioctl ret: %d (errno=%d)\n", ret, ret < 0 ? errno : 0);
        printf("  status: 0x%02x (%s)\n", alloc.status, nv_status_str(alloc.status));
        printf("  hObjectNew: 0x%08x\n", alloc.hObjectNew);

        if (alloc.status != 0) {
            fprintf(stderr, "Failed to allocate device\n");
            goto cleanup_client;
        }
        hDevice = alloc.hObjectNew;
    }

    /* ── Step 7: Allocate subdevice ──────────────────────────────── */
    uint32_t hSubDevice = 0;
    {
        NV2080_ALLOC_PARAMETERS subParams;
        memset(&subParams, 0, sizeof(subParams));
        subParams.subDeviceId = 0;

        NVOS21_PARAMETERS alloc;
        memset(&alloc, 0, sizeof(alloc));
        alloc.hRoot = hClient;
        alloc.hObjectParent = hDevice;
        alloc.hObjectNew = 0;
        alloc.hClass = NV20_SUBDEVICE_0;
        alloc.pAllocParms = (uint64_t)(uintptr_t)&subParams;
        alloc.paramsSize = sizeof(subParams);
        alloc.status = 0xFFFFFFFF;

        ret = nv_xfer_ioctl(ctl_fd, NV_ESC_RM_ALLOC, &alloc, sizeof(alloc));
        printf("\n=== RM_ALLOC SubDevice ===\n");
        printf("  ioctl ret: %d (errno=%d)\n", ret, ret < 0 ? errno : 0);
        printf("  status: 0x%02x (%s)\n", alloc.status, nv_status_str(alloc.status));
        printf("  hObjectNew: 0x%08x\n", alloc.hObjectNew);

        if (alloc.status != 0) {
            fprintf(stderr, "Failed to allocate subdevice (non-fatal, continuing)\n");
        } else {
            hSubDevice = alloc.hObjectNew;
        }
    }

    /* ── Step 8: Register BAR0 as IO memory ──────────────────────── */
    uint32_t hMemory = 0;
    {
        nv_ioctl_nvos02_parameters_with_fd api;
        memset(&api, 0, sizeof(api));

        api.params.hRoot = hClient;
        api.params.hObjectParent = hDevice;
        api.params.hObjectNew = 0;  /* auto-generate handle */
        api.params.hClass = NV01_MEMORY_SYSTEM_OS_DESCRIPTOR;
        api.params.flags = NVOS02_FLAGS_PHYSICALITY_CONTIGUOUS
                         | NVOS02_FLAGS_LOCATION_PCI
                         | NVOS02_FLAGS_COHERENCY_UNCACHED
                         | NVOS02_FLAGS_ALLOC_NONE
                         | NVOS02_FLAGS_GPU_CACHEABLE_NO
                         | NVOS02_FLAGS_MAPPING_NO_MAP;
        api.params.pMemory = (uint64_t)(uintptr_t)bar0;
        api.params.limit = (uint64_t)bar_size - 1;
        api.params.status = 0xFFFFFFFF;
        api.fd = bar_fd;  /* FD for mmap context */

        printf("\n=== RM_ALLOC_MEMORY (IO Memory Registration) ===\n");
        printf("  hRoot: 0x%08x\n", api.params.hRoot);
        printf("  hObjectParent: 0x%08x\n", api.params.hObjectParent);
        printf("  hClass: 0x%08x (NV01_MEMORY_SYSTEM_OS_DESCRIPTOR)\n", api.params.hClass);
        printf("  flags: 0x%08x\n", api.params.flags);
        printf("  pMemory: %p (mmap'd BAR0)\n", (void *)(uintptr_t)api.params.pMemory);
        printf("  limit: 0x%lx\n", (unsigned long)api.params.limit);

        ret = nv_xfer_ioctl(gpu_fd, NV_ESC_RM_ALLOC_MEMORY, &api, sizeof(api));
        printf("  ioctl ret: %d (errno=%d: %s)\n", ret, ret < 0 ? errno : 0,
               ret < 0 ? strerror(errno) : "OK");
        printf("  RM status: 0x%02x (%s)\n", api.params.status, nv_status_str(api.params.status));
        printf("  hObjectNew: 0x%08x\n", api.params.hObjectNew);

        if (api.params.status == 0) {
            hMemory = api.params.hObjectNew;
            printf("\n  *** SUCCESS: BAR0 registered with GPU RM as handle 0x%08x ***\n", hMemory);
        } else {
            fprintf(stderr, "\n  FAILED to register BAR0.\n");

            /* Try again with PEER_MAP_OVERRIDE flag */
            printf("\n  Retrying with PEER_MAP_OVERRIDE flag...\n");
            api.params.hObjectNew = 0;
            api.params.flags |= NVOS02_FLAGS_PEER_MAP_OVERRIDE_REQUIRED;
            api.params.status = 0xFFFFFFFF;

            ret = nv_xfer_ioctl(gpu_fd, NV_ESC_RM_ALLOC_MEMORY, &api, sizeof(api));
            printf("  ioctl ret: %d (errno=%d)\n", ret, ret < 0 ? errno : 0);
            printf("  RM status: 0x%02x (%s)\n", api.params.status, nv_status_str(api.params.status));
            printf("  hObjectNew: 0x%08x\n", api.params.hObjectNew);

            if (api.params.status == 0) {
                hMemory = api.params.hObjectNew;
                printf("\n  *** SUCCESS (with PEER_MAP_OVERRIDE): handle 0x%08x ***\n", hMemory);
            } else {
                /* Try with no special flags at all */
                printf("\n  Retrying with minimal flags...\n");
                api.params.hObjectNew = 0;
                api.params.flags = 0;  /* Let kernel figure it out */
                api.params.status = 0xFFFFFFFF;

                ret = nv_xfer_ioctl(gpu_fd, NV_ESC_RM_ALLOC_MEMORY, &api, sizeof(api));
                printf("  ioctl ret: %d (errno=%d)\n", ret, ret < 0 ? errno : 0);
                printf("  RM status: 0x%02x (%s)\n", api.params.status, nv_status_str(api.params.status));
                printf("  hObjectNew: 0x%08x\n", api.params.hObjectNew);

                if (api.params.status == 0) {
                    hMemory = api.params.hObjectNew;
                    printf("\n  *** SUCCESS (minimal flags): handle 0x%08x ***\n", hMemory);
                }
            }
        }
    }

    /* ── Step 9: If registered, try to map to GPU VA ─────────────── */
    if (hMemory) {
        printf("\n=== Next Step: Map to GPU VA ===\n");
        printf("  IO memory registered as RM handle 0x%08x\n", hMemory);
        printf("  TODO: Use NV_ESC_RM_MAP_MEMORY_DMA (0x57) to create GPU VA mapping\n");
        printf("  TODO: Then use GPU VA in CUDA kernel for MMIO reads/writes\n");
    }

    /* ── Cleanup ──────────────────────────────────────────────────── */
    if (hMemory) {
        /* Free memory object */
        /* NV_ESC_RM_FREE = 0x29, uses NVOS00_PARAMETERS */
        /* TODO: proper cleanup */
    }

cleanup_client:
    if (hClient) {
        /* Free root client */
        /* TODO: proper RM_FREE call */
    }

cleanup_gpu:
    close(gpu_fd);
cleanup_ctl:
    close(ctl_fd);
cleanup_bar:
    munmap((void *)bar0, bar_size);
    close(bar_fd);

    return (hMemory != 0) ? 0 : 1;
}
