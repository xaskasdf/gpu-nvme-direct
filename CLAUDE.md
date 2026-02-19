# CLAUDE.md — gpu-nvme-direct

## Qué es este proyecto

GPU-initiated NVMe I/O via PCIe BAR MMIO. Los CUDA kernels en la GPU escriben
directamente a los registros BAR0 del controlador NVMe para emitir comandos de
lectura/escritura, eliminando al CPU del data path de storage.

**Motivación**: Los pipelines de inferencia GPU pierden tiempo en `memcpy`
mediado por CPU (NVMe → host RAM → GPU VRAM). Queremos que la GPU hable
directamente con el NVMe.

## Hardware

| Componente | Detalle |
|---|---|
| GPU | NVIDIA RTX 3090 (GA102, sm_86, 24GB, Ampere) at 0000:0a:00.0 |
| CPU | AMD Ryzen 7 5800X (Zen 3, AM4) |
| Plataforma | AMD B550/X570, PCIe 4.0 |
| RAM | 48GB DDR4 |
| NVMe test | WD SN530 1TB at 0000:0b:00.0 (PCIe 3.0 x4, NVMe 1.4.0, ~3.5 GB/s seq) |
| OS | Ubuntu 25.10 (kernel 6.17.0-14-generic) |
| CUDA | 13.1 |
| Driver | 590.48.01 (open kernel modules via DKMS, patched) |
| Compiler | gcc-14 (gcc-15 incompatible con CUDA 13.1) |

### PCIe Topology
```
Root Complex (AMD Matisse/Vermeer)
├── Root Port 03.1 → GPU 0a:00.0   (PCIe 4.0 x16, CPU direct)
└── Root Port 03.4 → NVMe 0b:00.0  (PCIe 3.0 x4, via B550 chipset)
```

### P2P: Writes SI, Reads NO
- GPU→NVMe **posted writes** (MemWr): funcionan a través del AMD data fabric
- GPU→NVMe **non-posted reads** (MemRd): FAIL (CmpltTO, AMD root complex las droppea)
- **Tier 1 funciona**: solo se necesitan doorbell writes, CQ se polea desde host pinned

### BIOS settings necesarios
- Above 4G Decoding: **ON**
- IOMMU: **OFF** (`amd_iommu=off` en GRUB)
- Secure Boot: **OFF**

## Estado actual (2026-02-19)

### Milestones completados

1. ✅ Código completo (todas las fases)
2. ✅ Phase 0 tests pasan (simulador)
3. ✅ dump_bar0 lee registros NVMe desde CPU
4. ✅ **cudaHostRegisterIoMemory funciona** (tras patchear nvidia DKMS)
5. ✅ **GPU MMIO writes a BAR0 funcionan** (probado con CC y doorbells)
6. ✅ **GPU lee un bloque del NVMe autónomamente** (test_single_block)

### Lo que sigue: Multi-block reads → Layer loader para ntransformer

El objetivo aplicado es **servir como backend de I/O para ntransformer** (`../ntransformer`),
un inference engine que corre modelos de 70B parámetros en 24GB VRAM via layer streaming
(SLEP — Streaming Layer Execution Pipeline).

**Pipeline actual de ntransformer (con CPU bottleneck):**
```
NVMe → page cache → CPU memcpy → pinned staging → H2D DMA → GPU compute
        (mmap)      (worker thread)   (1.3 GB×2)     (PCIe)
```
Resultado: 0.02 tok/s en 70B. El memcpy del CPU worker thread es el bottleneck.

**Pipeline objetivo (GPU-autónomo, sin CPU en el data path):**
```
GPU doorbell write → NVMe DMA → host pinned buffer → GPU compute
  (MMIO a BAR0)      (autónomo)   (sin CPU memcpy)    (lee directo)
```

**Roadmap inmediato:**

| # | Paso | Estado | Descripción |
|---|------|--------|-------------|
| 1 | Multi-block read | ✅ | PRP lists hasta MDTS=512KB (6/6 tests pass) |
| 2 | Large sequential read | ✅ | 669MB (1 layer Q6_K) @ 2.1 GB/s, pipeline depth 32 |
| 3 | Benchmarks | ⬜ | Latencia y throughput: 1-bloque, N-bloques, secuencial grande |
| 4 | Layer loader API | ⬜ | `gpunvme_load_layer(ctrl, lba_offset, size, dest_pinned)` |
| 5 | ntransformer integración | ⬜ | Reemplazar `LayerStreamer` con gpu-nvme-direct backend |
| 6 | Port ntransformer a Linux | ⬜ | Actualmente solo testeado en Windows/MSVC |

### Números de referencia

El SN530 es **PCIe 3.0 x4** (~3.5 GB/s seq read). El slot va por el chipset B550.

| Ruta de datos | BW medido/estimado | 1 layer (669MB) | 80 layers | tok/s |
|---------------|-------------------|-----------------|-----------|-------|
| mmap+memcpy+H2D (actual ntransformer) | ~1.5-2 GB/s | ~400ms | 32s | 0.03 |
| **gpu-nvme-direct Tier 1 (SN530 Gen3)** | **2.1 GB/s medido** | **315ms** | **25s** | **0.04** |
| gpu-nvme-direct pico (128MB chunks) | 2.7 GB/s medido | ~248ms | ~20s | 0.05 |
| Tier 1 + compute overlap | ~2.1-2.7 GB/s | ~250ms oculto | ~20s | 0.05 |
| Warm page cache + H2D | ~13 GB/s | ~52ms | 4.1s | 0.24 |
| **Con NVMe Gen4 x4 (futuro upgrade)** | ~4-6 GB/s | ~130ms | 10s | 0.10 |

La ganancia real es **eliminar el CPU del data path** y el memcpy sincrónico.
Un upgrade a NVMe Gen4 duplicaría el throughput.
Para Q8_0 (70B = ~70GB, no cabe en 48GB RAM): el streaming desde NVMe es obligatorio.

## Estructura del proyecto

```
include/gpunvme/    Headers públicos (nvme_regs.h, nvme_cmds.h, controller.h, queue.h, etc.)
src/device/         GPU-side CUDA (mmio_ops.cuh, sq_submit.cuh, cq_poll.cuh, block_io.cu)
src/host/           CPU-side C (controller.c, admin.c, io_queue.c, bar_map.c, dma_alloc.c)
src/sim/            Simulador NVMe en software (nvme_sim.c/h)
kmod/               Kernel module Linux (PCI probe, char device, nvidia_p2p DMA)
bench/              Benchmarks (gpu-direct, cuFile, cpu-memcpy, cpu-pinned, sweep, plots)
tests/              Tests (struct sizes, simulador, BAR read, single block read)
tools/              Diagnóstico (dump_bar0, check_p2p, nvme_identify, pcie_topology)
scripts/            Setup/teardown (VFIO, prereqs, kernel module)
docs/               Diseño, referencia NVMe, safety, metodología benchmarks
```

## Build

```bash
# Phase 0 (simulador)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DGPUNVME_USE_SIM=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-14 \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build . -j$(nproc)
ctest --output-on-failure

# Phase 1+ (hardware real)
mkdir build-hw && cd build-hw
cmake .. -DCMAKE_BUILD_TYPE=Release -DGPUNVME_USE_SIM=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-14 \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build . -j$(nproc)

# Correr test de hardware (necesita VFIO setup primero)
sudo ./test_single_block 0000:0b:00.0
```

### Setup NVMe (después de cada reboot)
```bash
sudo modprobe vfio enable_unsafe_noiommu_mode=1
sudo modprobe vfio-pci
sudo bash scripts/setup_vfio.sh 0000:0b:00.0
sudo sh -c 'echo on > /sys/bus/pci/devices/0000:0b:00.0/power/control'
sudo setpci -s 0000:0b:00.0 0x84.W=0x0008   # Force D0
sudo setpci -s 0000:0b:00.0 COMMAND=0x0006   # Memory + BusMaster enable
```

## Arquitectura técnica clave

### MMIO desde GPU (PTX inline asm)

El `volatile` normal de CUDA NO sirve para MMIO. Se usan instrucciones PTX:

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

Cuando `GPUNVME_USE_SIM=1`, se usa `volatile` normal (el simulador está en host memory).

### Memory ordering para NVMe

Cada escritura al SQ requiere:
1. Escribir 64 bytes del SQ entry
2. `__threadfence_system()` — SQ visible via PCIe
3. `mmio_write32(doorbell, tail)` — PTX st.mmio.sys
4. `__threadfence_system()` — doorbell llega al NVMe

### Phase bit protocol

- Controller empieza con phase=1 en la primera pasada del CQ
- Host espera phase=1 inicialmente
- Nueva completion: `(cqe.status_phase & 1) == expected_phase`
- Phase se invierte cuando CQ head vuelve a 0 (wrap)

### Doorbell offsets

```
SQ Y Tail = 0x1000 + (2*Y * (4 << CAP.DSTRD))
CQ Y Head = 0x1000 + ((2*Y+1) * (4 << CAP.DSTRD))

Con DSTRD=0 (común):
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

### Tiered approach (P2P en consumer GPU)

NVIDIA deshabilita PCIe P2P DMA en GPUs GeForce. Estrategia de 3 niveles:

| Tier | Descripción | Necesita P2P? |
|------|-------------|---------------|
| 1 | GPU escribe doorbells + poll CQ via MMIO. Queues + data en host pinned | No |
| 2 | Igual + data buffers en GPU VRAM via nvidia open-source kernel modules patcheados | Sí (patcheado) |
| 3 | Full BaM: queues Y data en GPU VRAM | Sí (nativo, Tesla/A-series) |

**Tier 1 solo ya es un resultado publicable** — demuestra que la GPU puede actuar como procesador autónomo de I/O.

### Simulador NVMe (src/sim/)

- Aloca SQ/CQ/data en `cudaMallocHost` (pinned, accesible desde GPU)
- Thread de CPU polea el SQ y procesa comandos READ/WRITE/FLUSH
- Simula latencia configurable (default 50us)
- Phase bit tracking igual que el spec NVMe
- Permite desarrollar y testear los GPU kernels sin hardware NVMe real

## Desafío resuelto: cudaHostRegisterIoMemory en RTX 3090

`cudaHostRegisterIoMemory` **SÍ funciona** en GeForce RTX 3090, pero requiere
parchear el nvidia DKMS module (`os-mlock.c`) porque `follow_pfn()` fue removido
en kernel 6.12+. Ver `docs/investigation-p2p-bar0.md` para detalles completos.

**Resultado**: GPU puede escribir MMIO a NVMe BAR0 (doorbells, CC, etc.)
y el NVMe responde correctamente. GPU reads fallan (CmpltTO en AMD platform)
pero no se necesitan para Tier 1.

## Milestones completados

1. ✅ Código completo
2. ✅ Phase 0 tests pasan (simulador)
3. ✅ dump_bar0 lee registros NVMe (VS=1.4.0, MQES=1024)
4. ✅ cudaHostRegisterIoMemory funciona (tras parchear nvidia DKMS)
5. ✅ GPU MMIO writes a BAR0 funcionan (CC.EN y doorbells verificados)
6. ✅ **test_single_block: GPU lee un bloque del NVMe autónomamente**
7. ✅ **Multi-block reads (PRP lists hasta MDTS=512KB, 6/6 tests)**
8. ✅ **Large sequential reads: 669MB (1 layer) @ 2.1 GB/s, pipeline depth 32**
9. ⬜ Benchmarks vs cuFile, cpu-memcpy, cpu-pinned
10. ⬜ Layer loader API para ntransformer

## Referencia rápida NVMe

### Registros BAR0

| Offset | Size | Name | Descripción |
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
- PRP1: physical addr del data buffer
- PRP2: second page o PRP list
- CDW10/11: Starting LBA (64-bit)
- CDW12: Number of Logical Blocks - 1

### SQ entry = 64 bytes, CQ entry = 16 bytes

## Papers y referencias clave

- **BaM (ASPLOS 2023)**: https://arxiv.org/abs/2203.04910 — GPU-initiated on-demand storage
- **libnvm/ssd-gpu-dma**: https://github.com/enfiskutensykkel/ssd-gpu-dma — implementación más cercana
- **GPUDirect RDMA**: https://docs.nvidia.com/cuda/gpudirect-rdma/
- **GPUDirect Storage**: https://docs.nvidia.com/gpudirect-storage/overview-guide/
- **NVMe spec**: https://nvmexpress.org/specifications/
- **SPDK NVMe driver**: https://spdk.io/doc/nvme.html — referencia para init sequence
- **tinygrad P2P patch**: https://github.com/tinygrad/open-gpu-kernel-modules — parche para P2P en consumer GPUs
- **NVIDIA open-gpu-kernel-modules**: https://github.com/NVIDIA/open-gpu-kernel-modules

## Roadmap activo: ntransformer integration

### ntransformer (`../ntransformer`)
Inference engine custom C++/CUDA que corre Llama 70B en 24GB VRAM usando SLEP
(Streaming Layer Execution Pipeline). Lee layers del disco uno a uno, ejecuta
en GPU con double buffering. Actualmente usa mmap+memcpy+H2D (0.02 tok/s en 70B).

### Objetivo
Reemplazar el streaming backend de ntransformer con gpu-nvme-direct:
- GPU inicia reads de layers directamente al NVMe
- NVMe DMA a host pinned memory (Tier 1)
- GPU lee data desde pinned sin intervención del CPU
- Eliminar worker thread y staging buffers

### Pasos técnicos
1. **Multi-block read**: PRP lists para leer >4KB (una layer = ~669MB Q6_K)
2. **Queue depth pipelining**: Submit N reads, poll completions, overlap con compute
3. **Layer loader API**: Wrapper que toma offset+size en el GGUF file, mapea a LBAs
4. **ntransformer backend**: Implementar `LayerStreamer` interface con gpunvme

### Para Q8_0 (70B, ~70GB)
- Cada layer: ~875MB
- 80 layers × 875MB = 70GB (no cabe en 48GB RAM → DEBE streamear desde NVMe)
- Con Q8 no hay warm page cache posible → gpu-nvme-direct es el camino

## Ideas futuras (post-integración)

- **GPU-native unikernel OS**: GPU como procesador de I/O autónomo
- **Multi-queue parallel reads**: Múltiples GPU threads con queues separadas
- **Warp-cooperative submission**: Un warp entero colabora en SQ entries
- **Write support**: Writes siguen la misma lógica
- **Filesystem-aware reads**: Parsear metadata desde la GPU
- **Tier 2**: NVMe DMA directo a GPU VRAM (requiere nvidia_p2p patcheado)

## Convenciones del código

- **C11** para host code, **C++17/CUDA 17** para GPU code
- **gcc-14** requerido (gcc-15 incompatible con CUDA 13.1)
- **sm_86** target (RTX 3090 Ampere)
- Headers en `include/gpunvme/`, implementación en `src/`
- GPU device code usa `.cuh` para headers, `.cu` para implementación
- `GPUNVME_USE_SIM=1` macro controla si se usa simulador o hardware real
- Structs NVMe sin `packed` (removed — causa GPU misaligned access), con static asserts
- Queue allocations deben ser >= 4096 bytes (page alignment para NVMe)
- host_mmio_write64: dos writes de 32 bits (NVMe spec 3.1.1.1)
- Error handling via `gpunvme_err_t` enum (include/gpunvme/error.h)
- Scripts asumen bash, paths Linux, necesitan sudo para hardware

## Bugs importantes resueltos

| Bug | Causa raíz | Fix |
|-----|-----------|-----|
| cudaHostRegisterIoMemory falla | `follow_pfn()` removed en kernel 6.12+ | Parche `os-mlock.c`: PFN desde `vm_pgoff` |
| GPU reads → 0xffffffff | AMD root complex droppea non-posted P2P TLPs | No fix posible; Tier 1 solo usa writes |
| Admin commands cuelgan | Admin CQ no page-aligned (0x...800) | Allocations >= 4096 bytes |
| host_mmio_write64 corrupta | 64-bit write no atómica en NVMe | Dos 32-bit writes (low first) |
| GPU reads crashean NVMe link | PCIe link error acumulado | Evitar GPU reads a BAR0; power cycle si ocurre |
| Pipeline depth ≥4 timeout | NVMe completions out-of-order; `cq_poll_for_cid` descartaba CQEs | Usar `cq_poll_completion` (acepta cualquier CID) |
| I/O SQ/CQ no page-aligned | cudaMallocHost suballocator (después de muchas allocs) | posix_memalign + mlock + cudaHostRegister |
| PRP lists no page-aligned | Mismo suballocator issue | Pool allocation con posix_memalign |
