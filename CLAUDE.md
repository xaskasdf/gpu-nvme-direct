# CLAUDE.md — gpu-nvme-direct

## What is this project

GPU-initiated NVMe I/O via PCIe BAR MMIO. The CUDA kernels on the GPU write
directly to the NVMe controller's BAR0 registers to issue read/write commands,
eliminating the CPU from the storage data path.

**Motivation**: GPU inference pipelines waste time on CPU-mediated `memcpy`
(NVMe → host RAM → GPU VRAM). We want the GPU to talk directly to the NVMe.

## Hardware

| Component | Detail |
|---|---|
| GPU | NVIDIA RTX 3090 (GA102, sm_86, 24GB, Ampere) at 0000:0a:00.0 |
| CPU | AMD Ryzen 7 5800X (Zen 3, AM4) |
| Motherboard | ASUS ROG STRIX B450-F GAMING II (B450, Rev 1.xx) |
| RAM | 48GB DDR4 |
| NVMe test | WD SN740 512GB at 0000:01:00.0 (Gen4 x4 device, runs Gen3 on B450, MDTS=1024KB) |
| NVMe boot | WD SN530 1TB at 0000:0b:00.0 (Gen3 x4, MDTS=512KB) |
| OS | Ubuntu 25.10 (kernel 6.17.0-14-generic) |
| CUDA | 13.1 |
| Driver | 590.48.01 (open kernel modules via DKMS, patched) |
| Compiler | gcc-14 (gcc-15 incompatible with CUDA 13.1) |

### PCIe Topology (B450 — all Gen3, no Gen4)
```
CPU Ryzen 5800X (24 PCIe lanes — all Gen3 on B450)
├── 00:01.1 (x4 Gen3) → SN740 01:00.0  [M.2_1, CPU-direct, shares lanes with GPU]
├── 00:01.3 (x4 Gen3) → B450 chipset   [USB, SATA, Ethernet → bus 02-09]
├── 00:03.1 (x8 Gen3) → GPU   0a:00.0  [PCIEX16_1, CPU-direct, x8 due to M.2 sharing]
└── 00:03.4 (x4 Gen3) → SN530 0b:00.0  [M.2_2, CPU-direct]

Free slots (all chipset, Gen2):
  PCIEX16_2 (x4 Gen2), PCIEX16_3 (x4 Gen2), PCIEX1_1/2/3 (x1 Gen2)
```

**Note**: B450 does NOT support Gen4. SN740 capped at 8GT/s (Gen3) by the board.
GPU at x8 because M.2_1 shares lanes with PCIEX16_1. 3.35 GB/s = 96% of the Gen3 x4 ceiling.
Upgrading to B550/X570 would provide Gen4 x4 (~6-7 GB/s) and GPU x16.

### P2P: Writes YES, Reads NO
- GPU→NVMe **posted writes** (MemWr): work through the AMD data fabric
- GPU→NVMe **non-posted reads** (MemRd): FAIL (CmpltTO, AMD root complex drops them)
- **Tier 1 works**: only doorbell writes are needed, CQ is polled from host pinned memory

### Required BIOS settings
- Above 4G Decoding: **ON**
- IOMMU: **OFF** (`amd_iommu=off` in GRUB)
- Secure Boot: **OFF**

## Current status (2026-02-21)

### Completed milestones

1. ✅ Code complete (all phases)
2. ✅ Phase 0 tests pass (simulator)
3. ✅ dump_bar0 reads NVMe registers from CPU
4. ✅ **cudaHostRegisterIoMemory works** (after patching nvidia DKMS)
5. ✅ **GPU MMIO writes to BAR0 work** (tested with CC and doorbells)
6. ✅ **GPU reads a block from NVMe autonomously** (test_single_block)
7. ✅ **Multi-block reads** (PRP lists up to MDTS, 6/6 tests)
8. ✅ **Large sequential reads** (669MB @ 2.1 GB/s on SN530, pipeline depth 32)
9. ✅ **Layer Loader API** (`gpunvme_layer_loader_init/load_layer/destroy`)
10. ✅ **SN740 validated** (8.6GB @ 3.35 GB/s sustained, 3/3 tests)
11. ✅ **ntransformer integration** — gpu-nvme-direct as tier C NVMe backend

### ntransformer integration (COMPLETED)

gpu-nvme-direct was integrated as the I/O backend for ntransformer (`../ntransformer`),
an inference engine that runs 70B parameter models on 24GB VRAM via layer streaming
(SLEP — Streaming Layer Execution Pipeline).

**Previous pipeline (with CPU bottleneck):**
```
NVMe → page cache → CPU memcpy → pinned staging → H2D DMA → GPU compute
        (mmap)      (worker thread)   (1.3 GB×2)     (PCIe)
```

**Current pipeline (GPU-autonomous for tier C):**
```
GPU doorbell write → NVMe DMA → nvme_read_buf → scatter-copy → staging → H2D → GPU
  (MMIO to BAR0)     (3.3 GB/s)   (pinned)       (reorder)     (pinned)
```

**Measured results:**
- 8B Q8_0: 16 VRAM + 16 NVMe → output identical to baseline (temp=0), 1.8 tok/s decode
- 70B Q6_K: 20 VRAM + 30 RAM + 30 NVMe → correct output ("Paris"), 0.06 tok/s decode
- NVMe read: 670 MB/layer @ 3.3 GB/s sustained (SN740, Gen3 x4)

**Critical bugs resolved during integration:**
1. **Tensor order mismatch**: GGUF stores tensors in header order (attn_norm→ffn_down→...→attn_q→attn_v),
   but the GPU buffer layout expects (attn_q→attn_k→attn_v→attn_output→ffn_gate→ffn_up→ffn_down).
   Fix: scatter-copy with NvmeTensorMap per tensor.
2. **Sub-LBA offset**: Tensors do not start at 512-byte boundaries.
   Fix: LBA-aligned NVMe read span, with `read_offset` per tensor.

**Env vars for testing:**
```bash
GPUNVME_PCI_BDF=0000:01:00.0   # NVMe PCI address
GPUNVME_GGUF_LBA=0              # LBA where GGUF starts on raw device
GPUNVME_MAX_VRAM_LAYERS=N       # Cap tier A (force layers to tier C)
GPUNVME_MAX_RAM_LAYERS=M        # Cap tier B (force layers to tier C)
```

**Roadmap:**

| # | Step | Status | Description |
|---|------|--------|-------------|
| 1 | Multi-block read | ✅ | PRP lists up to MDTS (6/6 tests pass) |
| 2 | Large sequential read | ✅ | 669MB @ 2.1 GB/s (SN530), pipeline depth 32 |
| 3 | Layer loader API | ✅ | `gpunvme_layer_loader_init/load_layer/destroy` — 3-call reusable API |
| 4 | SN740 validation | ✅ | 8.6GB @ 3.35 GB/s sustained (PCIe 4.0, MDTS=1024KB) |
| 5 | ntransformer integration | ✅ | Tier C NVMe backend with scatter-copy (8B and 70B verified) |
| 6 | Port ntransformer to Linux | ✅ | Previously completed |

### Measured throughput (gpu-nvme-direct Layer Loader)

| NVMe | Interface | MDTS | 4 MB | 128 MB | 669 MB | 8.6 GB |
|------|----------|------|------|--------|--------|--------|
| **SN740** | PCIe 4.0 x4 (via B550, downgraded 8GT/s) | 1024 KB | — | — | ~3.1 GB/s | **3.35 GB/s** |
| SN530 | PCIe 3.0 x4 | 512 KB | 2.1 GB/s | 2.7 GB/s | 2.1 GB/s | — |

**Note**: SN740 LnkCap=16GT/s but LnkSta=8GT/s (B450 does not support Gen4).
3.35 GB/s = 96% of the Gen3 x4 ceiling. Larger MDTS (1024K vs 512K) gives ~60% more throughput than SN530.
With B550/X570 (real Gen4), SN740 would deliver ~6-7 GB/s.

### Data path comparison

| Data path | Measured/estimated BW | 1 layer (669MB) | 80 layers | tok/s |
|---------------|-------------------|-----------------|-----------|-------|
| mmap+memcpy+H2D (original) | ~1.5-2 GB/s | ~400ms | 32s | 0.03 |
| 3-tier VRAM+RAM (29+51+0) | ~6.5 GB/s (H2D) | ~100ms | 5.3s | 0.2 |
| **gpu-nvme-direct (20V+30R+30N)** | **3.3 GB/s NVMe** | **~203ms** | **~16s** | **0.06 measured** |
| **gpu-nvme-direct (SN740 + B550 Gen4)** | **~6-7 GB/s estimated** | **~100ms** | **~8s** | **0.12** |
| Warm page cache + H2D | ~13 GB/s | ~52ms | 4.1s | 0.24 |

The real benefit of the NVMe path is for models that **do not fit in RAM** (70B Q8_0 = 70GB > 48GB RAM).
For 70B Q6_K (56GB): 20 VRAM + 30 RAM + 30 NVMe = 0.06 tok/s measured (vs 0.02 with mmap baseline).

### GGUF on NVMe (SN740 = /dev/nvme1n1)
- **llama-3.1-70b-instruct-q6_k.gguf** (56GB) written with `dd` to LBA 0
- Previously: llama-3.1-8b-instruct-q8_0.gguf (8.5GB) — replaced by 70B
- Validated: ntransformer 70B with 30 NVMe layers, correct output
- **NOTE**: SN740 is `/dev/nvme1n1` (NOT nvme0n1, which is the boot SN530)

## Project structure

```
include/gpunvme/    Public headers (nvme_regs.h, nvme_cmds.h, controller.h, queue.h, layer_loader.h)
src/device/         GPU-side CUDA (mmio_ops.cuh, sq_submit.cuh, cq_poll.cuh, block_io.cu)
src/host/           CPU-side C/CUDA (controller.c, admin.c, io_queue.c, layer_loader.cu)
src/sim/            Software NVMe simulator (nvme_sim.c/h)
kmod/               Linux kernel module (PCI probe, char device, nvidia_p2p DMA)
bench/              Benchmarks (gpu-direct, cuFile, cpu-memcpy, cpu-pinned, sweep, plots)
tests/              Tests (struct sizes, simulator, BAR read, single block read)
tools/              Diagnostics (dump_bar0, check_p2p, nvme_identify, pcie_topology)
scripts/            Setup/teardown (VFIO, prereqs, kernel module)
docs/               Design, NVMe reference, safety, benchmark methodology
```

## Build

```bash
# Phase 0 (simulator)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DGPUNVME_USE_SIM=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-14 \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build . -j$(nproc)
ctest --output-on-failure

# Phase 1+ (real hardware)
mkdir build-hw && cd build-hw
cmake .. -DCMAKE_BUILD_TYPE=Release -DGPUNVME_USE_SIM=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-14 \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build . -j$(nproc)

# Run hardware tests (requires VFIO setup first)
sudo ./test_single_block 0000:01:00.0
sudo ./test_layer_loader 0000:01:00.0        # Layer Loader API (3 tests)
sudo ./test_layer_loader 0000:01:00.0 669    # Full layer size
```

### NVMe Setup (after each reboot)
```bash
sudo modprobe vfio enable_unsafe_noiommu_mode=1
sudo modprobe vfio-pci
sudo bash scripts/setup_vfio.sh 0000:01:00.0
sudo sh -c 'echo on > /sys/bus/pci/devices/0000:01:00.0/power/control'
sudo setpci -s 0000:01:00.0 0x84.W=0x0008   # Force D0
sudo setpci -s 0000:01:00.0 COMMAND=0x0006   # Memory + BusMaster enable
```

## Key technical architecture

### MMIO from GPU (PTX inline asm)

Normal CUDA `volatile` does NOT work for MMIO. PTX instructions are used:

```cuda
// src/device/mmio_ops.cuh
__device__ void mmio_write32(volatile uint32_t* addr, uint32_t val) {
    asm volatile("st.relaxed.mmio.sys.u32 [%0], %1;" :: "l"(addr), "r"(val) : "memory");
}
__device__ uint32_t mmio_read32(volatile uint32_t* addr) {
    uint32_t val;
    asm volatile("ld.relaxed.mmio.sys.u32 %0, [%1];" : "=r"(val) : "l"(addr) : "memory");
    return val;
}
```

When `GPUNVME_USE_SIM=1`, normal `volatile` is used (the simulator runs in host memory).

### Memory ordering for NVMe

Each SQ write requires:
1. Write 64 bytes of the SQ entry
2. `__threadfence_system()` — SQ visible via PCIe
3. `mmio_write32(doorbell, tail)` — PTX st.mmio.sys
4. `__threadfence_system()` — doorbell reaches the NVMe

### Phase bit protocol

- Controller starts with phase=1 on the first CQ pass
- Host expects phase=1 initially
- New completion: `(cqe.status_phase & 1) == expected_phase`
- Phase inverts when CQ head wraps back to 0

### Doorbell offsets

```
SQ Y Tail = 0x1000 + (2*Y * (4 << CAP.DSTRD))
CQ Y Head = 0x1000 + ((2*Y+1) * (4 << CAP.DSTRD))

With DSTRD=0 (common):
  Admin SQ Tail = 0x1000    Admin CQ Head = 0x1004
  IO SQ1 Tail   = 0x1008    IO CQ1 Head   = 0x100C
```

### NVMe controller init sequence (CPU-side)

```
Read CAP → CC.EN=0 → Wait CSTS.RDY=0 →
Alloc admin SQ/CQ (cudaMallocHost) →
Set AQA, ASQ, ACQ → Configure CC (MPS, IOSQES=6, IOCQES=4) →
CC.EN=1 → Wait CSTS.RDY=1 → Identify Controller
```

### Tiered approach (P2P on consumer GPU)

NVIDIA disables PCIe P2P DMA on GeForce GPUs. 3-tier strategy:

| Tier | Description | Needs P2P? |
|------|-------------|---------------|
| 1 | GPU writes doorbells + polls CQ via MMIO. Queues + data in host pinned | No |
| 2 | Same + data buffers in GPU VRAM via patched nvidia open-source kernel modules | Yes (patched) |
| 3 | Full BaM: queues AND data in GPU VRAM | Yes (native, Tesla/A-series) |

**Tier 1 alone is already a publishable result** — it demonstrates that the GPU can act as an autonomous I/O processor.

### NVMe simulator (src/sim/)

- Allocates SQ/CQ/data in `cudaMallocHost` (pinned, accessible from GPU)
- CPU thread polls the SQ and processes READ/WRITE/FLUSH commands
- Simulates configurable latency (default 50us)
- Phase bit tracking identical to the NVMe spec
- Allows developing and testing GPU kernels without real NVMe hardware

## Challenge solved: cudaHostRegisterIoMemory on RTX 3090

`cudaHostRegisterIoMemory` **DOES work** on GeForce RTX 3090, but requires
patching the nvidia DKMS module (`os-mlock.c`) because `follow_pfn()` was removed
in kernel 6.12+. See `docs/investigation-p2p-bar0.md` for full details.

**Result**: GPU can write MMIO to NVMe BAR0 (doorbells, CC, etc.)
and the NVMe responds correctly. GPU reads fail (CmpltTO on AMD platform)
but are not needed for Tier 1.

## Completed milestones

1. ✅ Code complete
2. ✅ Phase 0 tests pass (simulator)
3. ✅ dump_bar0 reads NVMe registers (VS=1.4.0, MQES=1024)
4. ✅ cudaHostRegisterIoMemory works (after patching nvidia DKMS)
5. ✅ GPU MMIO writes to BAR0 work (CC.EN and doorbells verified)
6. ✅ **test_single_block: GPU reads a block from NVMe autonomously**
7. ✅ **Multi-block reads** (PRP lists up to MDTS, 6/6 tests)
8. ✅ **Large sequential reads**: 669MB @ 2.1 GB/s (SN530), pipeline depth 32
9. ✅ **Layer Loader API**: 3-call reusable interface (`init/load_layer/destroy`)
10. ✅ **SN740 validated**: 8.6GB @ 3.35 GB/s sustained (PCIe 4.0, MDTS=1024KB)
11. ✅ **ntransformer integration**: tier C NVMe backend, 8B+70B verified @ 3.3 GB/s

## NVMe quick reference

### BAR0 registers

| Offset | Size | Name | Description |
|--------|------|------|-------------|
| 0x00 | 8B | CAP | Controller Capabilities |
| 0x08 | 4B | VS | Version |
| 0x14 | 4B | CC | Controller Configuration |
| 0x1C | 4B | CSTS | Controller Status |
| 0x24 | 4B | AQA | Admin Queue Attributes |
| 0x28 | 8B | ASQ | Admin SQ Base Address |
| 0x30 | 8B | ACQ | Admin CQ Base Address |
| 0x1000+ | 4B | Doorbells | SQ Tail / CQ Head |

### NVMe READ command (opcode 0x02)

- CDW0: opcode=0x02, CID=unique_id
- NSID: 1
- PRP1: physical addr of the data buffer
- PRP2: second page or PRP list
- CDW10/11: Starting LBA (64-bit)
- CDW12: Number of Logical Blocks - 1

### SQ entry = 64 bytes, CQ entry = 16 bytes

## Key papers and references

- **BaM (ASPLOS 2023)**: https://arxiv.org/abs/2203.04910 — GPU-initiated on-demand storage
- **libnvm/ssd-gpu-dma**: https://github.com/enfiskutensykkel/ssd-gpu-dma — closest implementation
- **GPUDirect RDMA**: https://docs.nvidia.com/cuda/gpudirect-rdma/
- **GPUDirect Storage**: https://docs.nvidia.com/gpudirect-storage/overview-guide/
- **NVMe spec**: https://nvmexpress.org/specifications/
- **SPDK NVMe driver**: https://spdk.io/doc/nvme.html — reference for init sequence
- **tinygrad P2P patch**: https://github.com/tinygrad/open-gpu-kernel-modules — patch for P2P on consumer GPUs
- **NVIDIA open-gpu-kernel-modules**: https://github.com/NVIDIA/open-gpu-kernel-modules

## Active roadmap: ntransformer integration

### ntransformer (`../ntransformer`)
Custom C++/CUDA inference engine that runs Llama 70B on 24GB VRAM using SLEP
(Streaming Layer Execution Pipeline). Reads layers from disk one by one, executes
on GPU with double buffering. Currently uses mmap+memcpy+H2D (0.02 tok/s on 70B).

### Goal
Replace ntransformer's streaming backend with gpu-nvme-direct:
- GPU initiates layer reads directly to NVMe
- NVMe DMA to host pinned memory (Tier 1)
- GPU reads data from pinned without CPU intervention
- Eliminate worker thread and staging buffers

### Technical steps
1. **Multi-block read**: PRP lists to read >4KB (one layer = ~669MB Q6_K)
2. **Queue depth pipelining**: Submit N reads, poll completions, overlap with compute
3. **Layer loader API**: Wrapper that takes offset+size in the GGUF file, maps to LBAs
4. **ntransformer backend**: Implement `LayerStreamer` interface with gpunvme

### For Q8_0 (70B, ~70GB)
- Each layer: ~875MB
- 80 layers x 875MB = 70GB (does not fit in 48GB RAM → MUST stream from NVMe)
- With Q8 there is no warm page cache possible → gpu-nvme-direct is the way

## Future ideas (post-integration)

- **GPU-native unikernel OS**: GPU as autonomous I/O processor
- **Multi-queue parallel reads**: Multiple GPU threads with separate queues
- **Warp-cooperative submission**: An entire warp collaborates on SQ entries
- **Write support**: Writes follow the same logic
- **Filesystem-aware reads**: Parse metadata from the GPU
- **Tier 2**: NVMe DMA direct to GPU VRAM (requires patched nvidia_p2p)

## Code conventions

- **C11** for host code, **C++17/CUDA 17** for GPU code
- **gcc-14** required (gcc-15 incompatible with CUDA 13.1)
- **sm_86** target (RTX 3090 Ampere)
- Headers in `include/gpunvme/`, implementation in `src/`
- GPU device code uses `.cuh` for headers, `.cu` for implementation
- `GPUNVME_USE_SIM=1` macro controls whether the simulator or real hardware is used
- NVMe structs without `packed` (removed — causes GPU misaligned access), with static asserts
- Queue allocations must be >= 4096 bytes (page alignment for NVMe)
- host_mmio_write64: two 32-bit writes (NVMe spec 3.1.1.1)
- Error handling via `gpunvme_err_t` enum (include/gpunvme/error.h)
- Scripts assume bash, Linux paths, require sudo for hardware

## Important bugs resolved

| Bug | Root cause | Fix |
|-----|-----------|-----|
| cudaHostRegisterIoMemory fails | `follow_pfn()` removed in kernel 6.12+ | Patch `os-mlock.c`: PFN from `vm_pgoff` |
| GPU reads → 0xffffffff | AMD root complex drops non-posted P2P TLPs | No fix possible; Tier 1 only uses writes |
| Admin commands hang | Admin CQ not page-aligned (0x...800) | Allocations >= 4096 bytes |
| host_mmio_write64 corrupts data | 64-bit write not atomic on NVMe | Two 32-bit writes (low first) |
| GPU reads crash NVMe link | Accumulated PCIe link error | Avoid GPU reads to BAR0; power cycle if it occurs |
| Pipeline depth ≥4 timeout | NVMe completions out-of-order; `cq_poll_for_cid` discarded CQEs | Use `cq_poll_completion` (accepts any CID) |
| I/O SQ/CQ not page-aligned | cudaMallocHost suballocator (after many allocs) | posix_memalign + mlock + cudaHostRegister |
| PRP lists not page-aligned | Same suballocator issue | Pool allocation with posix_memalign |
