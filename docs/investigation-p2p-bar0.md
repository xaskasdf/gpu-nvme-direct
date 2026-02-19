# Investigation: GPU Access to NVMe BAR0 on RTX 3090

## Date: 2026-02-19
## Hardware: RTX 3090 (GA102), AMD B550, NVMe at 0000:0b:00.0

---

## Summary

Making the GPU read/write NVMe BAR0 registers is the critical step for
gpu-nvme-direct. After extensive investigation and a kernel module patch:

- **cudaHostRegisterIoMemory**: NOW WORKS (after patching nvidia DKMS module)
- **GPU MMIO Writes to NVMe BAR0**: WORK (PCIe posted writes succeed)
- **GPU MMIO Reads from NVMe BAR0**: FAIL (CmpltTO — AMD root complex drops P2P read TLPs)
- **Tier 1 is VIABLE**: Only writes are needed (doorbells). CQ polling is from host pinned memory.

---

## Environment

| Component | Detail |
|-----------|--------|
| GPU | NVIDIA RTX 3090 (GA102, sm_86) at 0000:0a:00.0 |
| NVMe (test) | WD SN530 1TB at 0000:0b:00.0 (NVMe 1.4.0) |
| Driver | 590.48.01 (open kernel modules via DKMS) |
| CUDA | 13.1 |
| OS | Ubuntu 25.10 (kernel 6.17.0-14-generic) |
| IOMMU | OFF (`amd_iommu=off` in GRUB) |

PCIe Topology (both devices share root complex):
```
Root Complex (AMD Matisse/Vermeer)
├── Root Port 03.1 → GPU 0a:00.0
└── Root Port 03.4 → NVMe 0b:00.0
```

---

## What Works

### Phase 0: Simulator
- All simulator tests pass (test_nvme_structs, test_sim_basic)
- GPU kernels correctly build SQ entries, ring doorbells, poll CQ
- Phase bit protocol works correctly

### CPU reads NVMe BAR0 (dump_bar0)
- mmap of `/sys/bus/pci/devices/0000:0b:00.0/resource0` works
- CPU can read all NVMe registers correctly
- Identified: NVMe 1.4.0, SanDisk SN530, MQES=1023, DSTRD=0

### VFIO-NOIOMMU bind
- NVMe successfully unbound from nvme driver and bound to vfio-pci
- Requires power management fixup after bind (device falls to D3):
  ```bash
  echo on > /sys/bus/pci/devices/0000:0b:00.0/power/control
  setpci -s 0000:0b:00.0 0x84.W=0x0008   # Force D0
  setpci -s 0000:0b:00.0 COMMAND=0x0006   # Memory + BusMaster enable
  ```

---

## cudaHostRegisterIoMemory: Timeline

### Before kernel module patch (broken nv_follow_pfn)

| Approach | API | Flags | Result |
|----------|-----|-------|--------|
| Strategy 1 | cudaHostRegister | cudaHostRegisterIoMemory \| cudaHostRegisterMapped | **invalid argument** |
| Strategy 2 | cudaHostRegister | cudaHostRegisterMapped (no IoMemory) | **invalid argument** |
| Strategy 3 | cudaHostRegister | cudaHostRegisterPortable \| cudaHostRegisterMapped | **invalid argument** |
| Strategy 4 | cudaHostRegister | cudaHostRegisterDefault | **invalid argument** |
| Strategy 5 | cuMemHostRegister (Driver API) | CU_MEMHOSTREGISTER_IOMEMORY \| CU_MEMHOSTREGISTER_DEVICEMAP | **CUDA_ERROR_INVALID_VALUE** |
| Strategy 6 | cuMemHostRegister (Driver API) | CU_MEMHOSTREGISTER_DEVICEMAP only | **CUDA_ERROR_INVALID_VALUE** |
| Strategy 7 | cuMemHostRegister (Driver API) | CU_MEMHOSTREGISTER_PORTABLE \| CU_MEMHOSTREGISTER_DEVICEMAP | **CUDA_ERROR_INVALID_VALUE** |
| /dev/mem mapping | cudaHostRegister | All variants | **invalid argument** |
| Direct RM ioctl | NV_ESC_RM_ALLOC_MEMORY | NV01_MEMORY_SYSTEM_OS_DESCRIPTOR | **NV_ERR_INVALID_ADDRESS (0x1E)** |

### After kernel module patch (vm_pgoff PFN fix)

| Approach | Result |
|----------|--------|
| cudaHostRegister(IoMemory \| Mapped) | **SUCCESS** |
| Direct RM ioctl (rm_bar0_register) | **NV_OK (0x00)** |
| GPU MMIO read (ld.relaxed.mmio.sys) | 0xffffffff (CmpltTO) |
| GPU MMIO write (st.relaxed.mmio.sys) | **SUCCESS** (verified CC register change) |

All pre-patch strategies tested in tools/: test_gpu_write.cu, test_gpu_write2.cu,
test_devmem.cu, test_driver_api.cu

### strace analysis

Full strace of cudaHostRegister with IoMemory flag shows:
- All ioctls to /dev/nvidiactl and /dev/nvidia0 return 0 (success)
- The "invalid argument" error is generated **inside libcuda.so userspace code**
- libcuda.so performs its own validation before issuing the actual mapping ioctl
- The check appears to be based on GPU class/family (GeForce vs Tesla/Quadro)

### Critical observation: First-boot success

On the **very first boot** after installing Ubuntu, `cudaHostRegisterIoMemory`
**succeeded**. The GPU read returned 0xFFFFFFFF (P2P read blocked at PCIe level),
but the registration itself worked. After a reboot (which included nvidia DKMS
module rebuild from `apt --fix-broken install` for nvidia-firmware), the
registration started failing.

This suggests: the nvidia module version or build configuration changed during
the package fix, and the new version has a stricter check in libcuda.so or
reports different capabilities to it.

---

## NVIDIA Module Configuration Attempts

### Registry keys (/etc/modprobe.d/nvidia-p2p.conf)
```
options nvidia NVreg_RegistryDwords="PeerMappingOverride=1;ForceP2P=0x11;RMForceP2PType=1"
```
- Confirmed loaded: /proc/driver/nvidia/params shows RegistryDwords set
- initramfs updated with `update-initramfs -u`
- **Result: No effect** — the block is in libcuda.so, not in the kernel module

### Key insight
The kernel module's `peerMappingOverride` flag (in kernel_bif.c) controls
whether the kernel allows P2P mapping. But libcuda.so checks GPU capabilities
**before** sending the ioctl to the kernel. So even with the kernel module
configured to allow P2P, libcuda.so blocks the call.

---

## Open-Source Kernel Module Analysis

### Code path for IoMemory registration

File: `src/nvidia/arch/nvalloc/unix/src/osmemdesc.c`

```c
// Lines 448-467: osCreateOsDescriptorFromIoMemory
// Checks PEER_MAP_OVERRIDE flag — if set, allows IoMemory
// This code IS reached when registry key is set
// But libcuda.so never sends the ioctl in the first place
```

File: `src/nvidia/src/kernel/gpu/bif/kernel_bif.c`
```c
// Line ~1189: peerMappingOverride defaults to 0
// Registry key "PeerMappingOverride=1" sets it to 1
// This controls kernel-side P2P permission
```

File: `src/nvidia/src/kernel/mem_mgr/os_desc_mem.c`
```c
// Lines 94-109: Forces PEER_MAP_OVERRIDE flag on memory descriptor
// This is the kernel-side allow for IoMemory mapping
```

### tinygrad P2P patch analysis

The tinygrad patch (for open-gpu-kernel-modules) enables GPU-to-GPU P2P on
consumer GPUs. Key changes:
- Forces `p2pOverride = 0x11` and `forceP2PType = PCIEP2P`
- Modifies `kbifGetP2PCapability` to always report P2P capable
- Does NOT address the IoMemory/BAR mapping use case specifically
- The patch is for GPU↔GPU P2P, not GPU↔NVMe BAR0

---

## ROOT CAUSE FOUND: follow_pfn() removed in kernel 6.17

**This is the actual root cause**, not libcuda.so. On kernel 6.17.0:

1. `follow_pfn()` was removed (commit 233eb0bf3b94)
2. The nvidia DKMS module's `nv_follow_pfn()` falls back to `nv_follow_flavors()`
   which returns `-1` (always fails)
3. `os_lookup_user_io_memory()` calls `nv_follow_pfn()` to resolve IO memory PFNs
4. Since it always fails, **no IO memory can be registered with the GPU**
5. This affects both `cudaHostRegisterIoMemory` (via libcuda.so → kernel ioctl)
   and direct RM ioctls

### Evidence

The direct RM ioctl tool (`rm_bar0_register`) bypasses libcuda.so entirely and
calls `NV_ESC_RM_ALLOC_MEMORY` with `NV01_MEMORY_SYSTEM_OS_DESCRIPTOR` directly.
Result: `NV_ERR_INVALID_ADDRESS` (0x1E) — the kernel module itself cannot resolve
the IO memory address.

### Fix Applied

Patched `/usr/src/nvidia-590.48.01/nvidia/os-mlock.c`:

```c
// Old fallback (always fails):
static inline int nv_follow_flavors(...) { return -1; }

// New: compute PFN from VMA for VM_PFNMAP mappings
if (vma && (vma->vm_flags & VM_PFNMAP)) {
    *pfn = vma->vm_pgoff + ((address - vma->vm_start) >> PAGE_SHIFT);
    return 0;
}
```

Note: `follow_pfnmap_start()`/`follow_pfnmap_end()` are GPL-only symbols and
cannot be used by the nvidia module (MIT license). The `vm_pgoff` approach works
because `io_remap_pfn_range()` (used by PCI BAR mmap) stores the starting PFN in
`vm_pgoff`, and the mapping is contiguous.

DKMS module rebuilt and installed. After reboot, both `rm_bar0_register` (direct
RM ioctl) and `cudaHostRegisterIoMemory` succeed — confirming the fix works.

### Why first-boot worked

On the first boot, the ubuntu-shipped `nvidia-590-open` kernel module (not DKMS)
was loaded. That module was likely compiled against an older kernel header
where `follow_pfn()` still existed. After `apt --fix-broken install` rebuilt
DKMS, the new module was compiled against kernel 6.17 where `follow_pfn()` is
absent, breaking IO memory support.

---

## POST-PATCH RESULTS: Writes Work, Reads Don't

After reboot with patched nvidia module:

### cudaHostRegisterIoMemory: SUCCESS
```
cudaHostRegisterIoMemory SUCCEEDED!
GPU device pointer: 0x7698b85c3000
```

### GPU MMIO Reads: FAIL (CmpltTO)
```
GPU read NVMe Version: 0xffffffff (expected 0x00010400)
GPU AER UESta: 0x00004000 (Completion Timeout)
NVMe AER UESta: 0x00000000 (clean — never received the TLP)
```

The GPU sends a PCIe MemRd TLP but the AMD Matisse root complex silently drops
it. The TLP never reaches the NVMe (no errors on NVMe side). The GPU times out
waiting for the completion. Repeated read attempts eventually crash the NVMe's
PCIe link (device disappears from bus, requires full power cycle to recover).

### GPU MMIO Writes: SUCCESS
```
=== GPU Write to NVMe BAR0 Test ===
TEST 1: GPU Write CC.EN=0 (disable controller)
  Writing CC=0x00464000 (EN=0) from GPU...
  CPU after write: CC=0x00000000 CSTS=0x00000000
  CC.EN=0  CSTS.RDY=0
  *** WRITE SUCCESS: CC.EN changed to 0! GPU writes reach NVMe! ***

TEST 2: GPU Write to Admin SQ Tail Doorbell (0x1000)
  Writing doorbell value 0 from GPU...
  NVMe still responding (VS=0x00010400)
  *** DOORBELL WRITE TEST PASSED ***
```

PCIe posted writes (Memory Write TLPs) don't need completions, so they succeed
through the AMD data fabric. The GPU can write to any NVMe BAR0 register,
including doorbells.

### Why reads fail but writes work

| Operation | PCIe TLP Type | Needs Completion? | AMD Root Complex | Result |
|-----------|--------------|-------------------|-----------------|--------|
| GPU→NVMe Read | MemRd (non-posted) | Yes | Drops TLP | CmpltTO, 0xffffffff |
| GPU→NVMe Write | MemWr (posted) | No | Forwards | SUCCESS |

The AMD Matisse/Vermeer data fabric forwards posted writes between root ports
but does not generate completions for non-posted reads directed at peer devices.
This is a known limitation of AMD consumer platforms for PCIe P2P.

### Implications for Tier 1

This is **sufficient for Tier 1 operation**:
- CPU handles all init (reads CAP, VS, CSTS, configures CC — all BAR0 reads)
- GPU writes SQ entries to **host pinned memory** (cudaMallocHost)
- GPU writes doorbell to **NVMe BAR0** (posted MMIO write — WORKS)
- NVMe DMAs completion to **CQ in host pinned memory**
- GPU polls CQ from **host pinned memory** (not from BAR0 — no P2P read needed)
- GPU reads data from **host pinned memory** (NVMe DMA target)

The only GPU→BAR0 operation in steady-state is the doorbell write, which works.

---

## Viable Approaches Forward

### Option A: Bypass libcuda.so with direct RM ioctls

The NVIDIA kernel module accepts IO memory registrations via RM (Resource
Manager) ioctls. libcuda.so is just a userspace client. We can call the
kernel module directly:

1. Open `/dev/nvidiactl` and `/dev/nvidia0`
2. Use `NV_ESC_RM_ALLOC_MEMORY` ioctl with `NVOS32_DESCRIPTOR_TYPE_OS_IO_MEMORY`
3. Pass the physical address of NVMe BAR0
4. Get back a GPU virtual address
5. Use that VA in CUDA kernels

This is essentially what libnvm/ssd-gpu-dma does.

**Pros**: Bypasses all libcuda.so checks, kernel module already supports it
**Cons**: Complex RM ioctl protocol, version-specific struct layouts, fragile

### Option B: Custom kernel module with nvidia_p2p APIs

Write a Linux kernel module that:
1. Maps NVMe BAR0 physical addresses
2. Uses `nvidia_p2p_get_pages()` / `nvidia_p2p_dma_map_pages()` to register
   with the NVIDIA driver
3. Exposes a char device for userspace to get the GPU pointer

**Pros**: Uses supported kernel API, more stable across driver versions
**Cons**: nvidia_p2p is designed for GPU VRAM export, not device BAR import;
may need adaptation

### Option C: Patch libcuda.so / use older CUDA version

Find and patch the check in libcuda.so, or use a CUDA version where the
check doesn't exist (possibly CUDA 12.x era).

**Pros**: Simplest if it works
**Cons**: Closed-source binary patching is fragile and legally questionable

### Option D: Build nvidia open modules with additional patches

Even though the block is in libcuda.so, the kernel module reports GPU
capabilities that libcuda.so queries. If we make the kernel module report
"Tesla-like" P2P capabilities, libcuda.so might not block the call.

**Pros**: Changes are in open-source code we can modify
**Cons**: Must reverse-engineer which capability bits libcuda.so checks

### Option E: First-boot state recovery

The first boot showed cudaHostRegisterIoMemory working. Investigate:
- What nvidia module was loaded initially (before DKMS rebuild)?
- Can we reproduce that module version/configuration?
- Was it a different libcuda.so?

---

## Build Fixes Applied (Ubuntu 25.10 + CUDA 13.1)

### 1. glibc 2.42 rsqrt conflict
**File**: `/usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.h`
**Problem**: glibc 2.42 declares `rsqrt()`/`rsqrtf()` with `noexcept`, CUDA 13.1 without
**Fix**: Added conditional `noexcept` guard to CUDA declarations

### 2. gcc 15.2.0 incompatible
**Fix**: Installed gcc-14, set `CMAKE_C_COMPILER=gcc-14`, `CMAKE_CXX_COMPILER=g++-14`

### 3. _Static_assert in C++ context
**File**: `include/gpunvme/nvme_regs.h`
**Fix**: `#define _Static_assert static_assert` under `#ifdef __cplusplus`

### 4. __attribute__((packed)) GPU misaligned access
**File**: `include/gpunvme/nvme_regs.h`
**Fix**: Removed `packed` from SQ/CQ entry structs (naturally aligned anyway)

### 5. cudaDeviceProp.clockRate removed in CUDA 13.1
**File**: `bench/bench_gpu_direct.cu`
**Fix**: Replaced with `cudaDeviceGetAttribute(&rate, cudaDevAttrClockRate, dev)`

### 6. CQ poll misaligned reads
**File**: `src/device/cq_poll.cuh`
**Fix**: Read DW3 (offset 12, aligned) and extract uint16_t fields via bit shifts

### 7. cuCtxCreate API change in CUDA 13.1
**Fix**: Added `CUctxCreateParams ctxParams = {};` parameter

---

## NVMe Setup Procedure (after every reboot)

```bash
# Load VFIO
sudo modprobe vfio enable_unsafe_noiommu_mode=1
sudo modprobe vfio-pci

# Unbind from nvme driver, bind to vfio-pci
sudo bash scripts/setup_vfio.sh 0000:0b:00.0

# Fix power state (device falls to D3 after unbind)
sudo sh -c 'echo on > /sys/bus/pci/devices/0000:0b:00.0/power/control'
sudo setpci -s 0000:0b:00.0 0x84.W=0x0008   # Force D0
sudo setpci -s 0000:0b:00.0 COMMAND=0x0006   # Memory + BusMaster enable

# Verify: should NOT read all-Fs
sudo ./build/dump_bar0 0000:0b:00.0
```

If all registers read 0xFFFFFFFF, do a PCI remove+rescan:
```bash
sudo sh -c 'echo 1 > /sys/bus/pci/devices/0000:0b:00.0/remove'
sudo sh -c 'echo 1 > /sys/bus/pci/rescan'
# Then re-bind to vfio-pci and fix power state again
```
