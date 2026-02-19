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
| GPU | NVIDIA RTX 3090 (GA102, sm_86, 24GB) |
| CPU | AMD Ryzen 7 5800X (Zen 3) |
| NVMe | Dedicated test SSD, PCIe 4.0 (**not** the boot drive) |
| OS | Ubuntu 24.04 (bare metal) |
| CUDA | 12.4 |

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

## Quick Start

See [BUILD.md](BUILD.md) for full instructions.

```bash
# Install prerequisites
sudo apt install -y build-essential cmake nvidia-driver-555 cuda-toolkit-12-4

# Phase 0: Simulator (verify logic without real NVMe hardware)
mkdir build && cd build
cmake .. -DGPUNVME_USE_SIM=ON && cmake --build . -j$(nproc)
ctest --output-on-failure

# Phase 1+: Real hardware
mkdir ../build-hw && cd ../build-hw
cmake .. -DGPUNVME_USE_SIM=OFF && cmake --build . -j$(nproc)
sudo ../scripts/setup_vfio.sh 0000:XX:00.0
sudo ./check_p2p 0000:XX:00.0       # THE milestone
```

## Milestones

1. **Phase 0** — GPU reads data through software NVMe simulator
2. **Phase 1** — GPU reads NVMe Version register via BAR0 MMIO
3. **Phase 2** — CPU initializes NVMe controller, creates I/O queues
4. **Phase 3** — GPU reads a block from NVMe autonomously (zero CPU in data path)
5. **Phase 4** — NVMe DMA writes directly to GPU VRAM (if P2P works)
6. **Phase 5** — Benchmarks vs cuFile, cpu-memcpy, cpu-pinned

## References

- [BaM: GPU-Initiated On-Demand High-Throughput Storage (ASPLOS 2023)](https://arxiv.org/abs/2203.04910)
- [ssd-gpu-dma / libnvm](https://github.com/enfiskutensykkel/ssd-gpu-dma)
- [NVIDIA GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [NVIDIA GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/overview-guide/)
- [NVMe Specification](https://nvmexpress.org/specifications/)
- [SPDK NVMe Driver](https://spdk.io/doc/nvme.html)

## License

BSD-2-Clause
