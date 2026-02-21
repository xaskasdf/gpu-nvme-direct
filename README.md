# gpu-nvme-direct

GPU-initiated NVMe I/O via PCIe BAR MMIO. CUDA kernels on the GPU directly
issue NVMe read/write commands by writing to the controller's BAR0 registers,
eliminating the CPU from the storage data path entirely.

## Why

Modern GPU inference pipelines waste time on CPU-mediated `memcpy`:
NVMe → host RAM → GPU VRAM (double hop, CPU overhead). This project lets the
GPU talk directly to the NVMe controller — no CPU in the I/O hot path.

Inspired by [BaM (ASPLOS 2023)](https://arxiv.org/abs/2203.04910) and
[libnvm/ssd-gpu-dma](https://github.com/enfiskutensykkel/ssd-gpu-dma),
designed for consumer hardware (RTX 3090).

## Hardware

| Component | Detail |
|-----------|--------|
| GPU | NVIDIA RTX 3090 (GA102, sm_86, 24GB) at 0000:0a:00.0 |
| CPU | AMD Ryzen 7 5800X (Zen 3, AM4) |
| Motherboard | ASUS ROG STRIX B450-F GAMING II (B450 — all Gen3) |
| NVMe test | WD SN740 512GB at 0000:01:00.0 (Gen4 device, runs Gen3 on B450) |
| NVMe boot | WD SN530 1TB at 0000:0b:00.0 (Gen3 x4) |
| OS | Ubuntu 25.10 (kernel 6.17, bare metal) |
| CUDA | 13.1 |
| Driver | 590.48.01 (open kernel modules, patched for `cudaHostRegisterIoMemory`) |

## Architecture

```
GPU Kernel (CUDA)
  ├── Build NVMe SQ entry (READ command)
  ├── __threadfence_system()
  ├── Write SQ tail doorbell (PTX st.mmio.sys)
  ├── Poll CQ for completion  (PTX ld.mmio.sys)
  └── Data arrives in buffer via NVMe DMA
         ↕ PCIe BAR0 MMIO
NVMe Controller
  ├── Submission Queue → processes command
  ├── DMA engine → writes data to PRP address
  └── Completion Queue → signals done with phase bit
```

## Tiered Approach (Consumer GPU P2P Limitations)

NVIDIA disables PCIe P2P DMA on GeForce GPUs. We degrade gracefully:

| Tier | Description | P2P needed? |
|------|-------------|-------------|
| **1** | GPU drives NVMe (doorbells + CQ poll via MMIO). Queues + data in host pinned memory | No |
| **2** | Same + data buffers in GPU VRAM via patched NVIDIA open-source kernel modules | Yes (patched) |
| **3** | Full BaM: queues AND data in GPU VRAM | Yes (native) |

Tier 1 alone proves the GPU can act as an autonomous I/O processor.

## Project Structure

```
include/gpunvme/    NVMe register structs, command builders, public API
src/device/         GPU-side CUDA code (MMIO ops, SQ submit, CQ poll, block I/O)
src/host/           CPU-side init (controller, admin queues, BAR0 mapping, DMA)
src/sim/            Software NVMe simulator (dev/test without real hardware)
kmod/               Linux kernel module (BAR0 mmap, GPU DMA via nvidia_p2p)
bench/              Benchmarks (gpu-direct, cuFile, cpu-memcpy, cpu-pinned)
tests/              Struct tests, simulator tests, hardware milestone tests
tools/              Diagnostics (BAR0 dump, P2P probe, PCIe topology)
scripts/            Setup/teardown scripts (VFIO, prereqs, kernel module)
docs/               Architecture design, NVMe reference, safety, benchmarks
```

## Measured Throughput

| NVMe | Interface | MDTS | Sustained | Notes |
|------|-----------|------|-----------|-------|
| **SN740** | Gen4 x4 (Gen3 on B450) | 1024K | **3.35 GB/s** | 96% of Gen3 x4 max |
| SN530 | Gen3 x4 | 512K | 2.1 GB/s | Boot disk |

## Quick Start

See [BUILD.md](BUILD.md) for full instructions.

```bash
# Phase 0: Simulator (verify logic without real NVMe hardware)
mkdir build && cd build
cmake .. -DGPUNVME_USE_SIM=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-14 \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build . -j$(nproc)
ctest --output-on-failure

# Real hardware (requires VFIO setup first)
mkdir ../build-hw && cd ../build-hw
cmake .. -DGPUNVME_USE_SIM=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-14 \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build . -j$(nproc)
sudo ../scripts/setup_vfio.sh 0000:01:00.0
sudo ./test_single_block 0000:01:00.0
sudo ./test_layer_loader 0000:01:00.0
```

## Milestones

1. ✅ **Phase 0** — GPU reads data through software NVMe simulator
2. ✅ **cudaHostRegisterIoMemory** — GPU MMIO to NVMe BAR0 (after nvidia DKMS patch)
3. ✅ **Single block read** — GPU reads one NVMe block autonomously
4. ✅ **Multi-block reads** — PRP lists up to MDTS (1024K), 6/6 tests
5. ✅ **Large sequential reads** — 669 MB @ 2.1 GB/s (SN530), pipeline depth 32
6. ✅ **Layer Loader API** — `gpunvme_layer_loader_init/load_layer/destroy`
7. ✅ **SN740 validated** — 8.6 GB @ 3.35 GB/s sustained (96% of Gen3 x4 max)
8. ✅ **ntransformer integrated** — 70B Q6_K streaming at 0.2 tok/s (33x over mmap)

## References

- [BaM: GPU-Initiated On-Demand High-Throughput Storage (ASPLOS 2023)](https://arxiv.org/abs/2203.04910)
- [ssd-gpu-dma / libnvm](https://github.com/enfiskutensykkel/ssd-gpu-dma)
- [NVIDIA GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [NVIDIA GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/overview-guide/)
- [NVMe Specification](https://nvmexpress.org/specifications/)
- [SPDK NVMe Driver](https://spdk.io/doc/nvme.html)

## License

BSD-2-Clause
