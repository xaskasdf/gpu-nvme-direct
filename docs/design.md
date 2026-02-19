# gpu-nvme-direct: Architecture Design

## Overview

gpu-nvme-direct enables CUDA kernels running on an NVIDIA GPU to directly
issue NVMe I/O commands by writing to the NVMe controller's PCIe BAR0
registers via MMIO. This eliminates the CPU from the storage I/O data path.

## Architecture Layers

```
┌─────────────────────────────────────────────────────┐
│  GPU Kernel (CUDA)                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ sq_submit │→ │ doorbell │→ │ cq_poll          │  │
│  │ (build    │  │ (MMIO    │  │ (poll phase bit  │  │
│  │  SQE)     │  │  write)  │  │  via MMIO read)  │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
└────────────┬──────────┬──────────────┬──────────────┘
             │ PCIe     │ PCIe         │ PCIe
             ▼          ▼              ▼
┌─────────────────────────────────────────────────────┐
│  NVMe Controller (BAR0 Registers)                    │
│  ┌──────┐  ┌──────────┐  ┌──────┐  ┌────────────┐  │
│  │ SQ   │  │ Doorbells│  │ CQ   │  │ DMA Engine │  │
│  └──────┘  └──────────┘  └──────┘  └────────────┘  │
└────────────────────────────┬────────────────────────┘
                             │ DMA
                             ▼
┌─────────────────────────────────────────────────────┐
│  Memory (Host Pinned / GPU VRAM)                     │
│  ┌─────────┐  ┌─────────┐  ┌───────────────────┐   │
│  │ SQ Ring │  │ CQ Ring │  │ Data Buffer       │   │
│  └─────────┘  └─────────┘  └───────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## Tier Model

| Tier | Queues | Data | P2P Required | Status |
|------|--------|------|-------------|--------|
| 1 | Host pinned | Host pinned | No | Primary target |
| 2 | Host pinned | GPU VRAM | Yes (patched) | If P2P works |
| 3 | GPU VRAM | GPU VRAM | Yes (native) | Tesla/A-series |

## Key Components

### GPU-Side (CUDA)
- **mmio_ops.cuh**: PTX `ld.mmio.sys` / `st.mmio.sys` for PCIe BAR MMIO
- **sq_submit.cuh**: Build NVMe READ/WRITE commands in SQ
- **cq_poll.cuh**: Poll CQ with phase bit protocol
- **block_io.cu**: High-level read/write kernels

### Host-Side (C)
- **controller.c**: NVMe controller init/shutdown sequence
- **admin.c**: Admin command submission (Identify, Create/Delete queues)
- **io_queue.c**: I/O queue pair creation with GPU-visible state
- **bar_map.c**: BAR0 mmap + cudaHostRegisterIoMemory

### Kernel Module (kmod/)
- PCI device management
- Character device for mmap/ioctl
- nvidia_p2p DMA mapping (Tier 2+)

## Comparison with Related Work

| | BaM (ASPLOS'23) | libnvm | GPUDirect Storage | gpu-nvme-direct |
|---|---|---|---|---|
| GPU initiates I/O | Yes | Yes | No (CPU) | Yes |
| Consumer GPU | No (Tesla) | No (Tesla) | No | Yes (tiered) |
| Requires kmod | Yes | Yes | Yes (nvidia-fs) | Optional |
| Open source | Yes | Yes | No | Yes |
