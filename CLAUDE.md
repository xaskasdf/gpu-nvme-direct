# CLAUDE.md — gpu-nvme-direct

## Qué es este proyecto

GPU-initiated NVMe I/O via PCIe BAR MMIO. Los CUDA kernels en la GPU escriben
directamente a los registros BAR0 del controlador NVMe para emitir comandos de
lectura/escritura, eliminando al CPU del data path de storage.

**Motivación**: Los pipelines de inferencia GPU pierden tiempo en `memcpy`
mediado por CPU (NVMe → host RAM → GPU VRAM). Queremos que la GPU hable
directamente con el NVMe.

## Hardware del usuario

| Componente | Detalle |
|---|---|
| GPU | NVIDIA RTX 3090 (GA102, sm_86, 24GB, Ampere) |
| CPU | AMD Ryzen 7 5800X (Zen 3, AM4) |
| Plataforma | AMD B550/X570, PCIe 4.0 |
| RAM | 48GB DDR4 |
| NVMe test | PCIe 4.0 SSD dedicado (NO es el boot drive) |
| OS | Ubuntu 24.04 LTS (bare metal, instalado en la NVMe) |
| CUDA target | 12.4 |
| Driver | 581.80 (o el que venga con Ubuntu) |

## Estado actual

### Código: COMPLETO (todas las fases implementadas)

- **61 archivos**, ~10,200 líneas de código
- Todo el código fue escrito pero **nunca compilado ni testeado**
- El usuario va a instalar Ubuntu bare metal en la NVMe nueva y buildear desde ahí

### Lo que falta hacer

1. **Instalar Ubuntu 24.04** en la NVMe nueva (dual-boot con Windows)
2. **Instalar CUDA toolkit 12.4** + build-essential + cmake
3. **Build Phase 0** (simulador) — `cmake .. -DGPUNVME_USE_SIM=ON` — verificar que compila y pasan los tests
4. **Iterar bugs de compilación** — es probable que haya errores ya que nunca se compiló
5. **Build Phase 1+** (hardware real) — `cmake .. -DGPUNVME_USE_SIM=OFF`
6. **Configurar VFIO** para la NVMe de test
7. **Correr milestones de hardware** (check_p2p, test_bar_read, test_single_block)
8. **Benchmarks** si los milestones pasan

### BIOS settings necesarios

- Above 4G Decoding: **ON**
- IOMMU: **OFF** (o `amd_iommu=off` en GRUB)
- Secure Boot: **OFF** (para el kernel module)

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
# Phase 0 (simulador, para verificar lógica sin hardware NVMe)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DGPUNVME_USE_SIM=ON
cmake --build . -j$(nproc)
ctest --output-on-failure

# Phase 1+ (hardware real)
mkdir build-hw && cd build-hw
cmake .. -DCMAKE_BUILD_TYPE=Release -DGPUNVME_USE_SIM=OFF
cmake --build . -j$(nproc)
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

## Desafío crítico: cudaHostRegisterIoMemory en RTX 3090

El paso más incierto es si `cudaHostRegisterIoMemory` funciona para mapear
BAR0 del NVMe en el address space del GPU en una GeForce consumer.

- **Si funciona** → Tier 1 completo, GPU lee/escribe NVMe de forma autónoma
- **Si falla** → Intentar con NVIDIA open-source kernel modules, o kernel module custom
- **En cualquier caso** → Es un resultado de investigación válido

El test es: `sudo ./build-hw/check_p2p 0000:XX:00.0`

## Milestones en orden

1. ✅ Código completo (todo escrito, falta compilar)
2. ⬜ Phase 0 tests pasan (test_nvme_structs + test_sim_basic)
3. ⬜ dump_bar0 lee registros NVMe desde CPU
4. ⬜ check_p2p — `cudaHostRegisterIoMemory` en BAR0 (MILESTONE 1)
5. ⬜ test_bar_read — GPU lee NVMe Version register via MMIO
6. ⬜ test_single_block — GPU lee un bloque del NVMe autónomamente (MILESTONE 2)
7. ⬜ Benchmarks vs cuFile, cpu-memcpy, cpu-pinned

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

## Ideas futuras (post-milestones)

- **GPU-native unikernel OS**: Si los milestones funcionan, la GPU puede ser un
  procesador de I/O autónomo — base para un OS que corre enteramente en la GPU
- **Multi-queue parallel reads**: Múltiples GPU threads con queues separadas para
  maximizar throughput
- **Warp-cooperative submission**: Un warp entero colabora para llenar y submitir SQ entries
- **Write support**: Actualmente enfocado en reads, writes siguen la misma lógica
- **Filesystem-aware reads**: Parsear FAT32/ext4 metadata desde la GPU para lectura
  directa de archivos (no solo bloques raw)
- **Model weight loader**: Aplicación práctica — cargar pesos de modelos de ML
  directamente del NVMe al GPU VRAM sin CPU

## Convenciones del código

- **C11** para host code, **C++17/CUDA 17** para GPU code
- **sm_86** target (RTX 3090 Ampere)
- Headers en `include/gpunvme/`, implementación en `src/`
- GPU device code usa `.cuh` para headers, `.cu` para implementación
- `GPUNVME_USE_SIM=1` macro controla si se usa simulador o hardware real
- Todos los structs NVMe tienen `__attribute__((packed))` y static asserts de tamaño
- Error handling via `gpunvme_err_t` enum (include/gpunvme/error.h)
- Scripts asumen bash, paths Linux, necesitan sudo para hardware

## Notas sobre el entorno previo (WSL — ABANDONADO)

WSL fue abandonado por estas razones:
- No soporta PCIe passthrough (imposible para Phase 1+)
- El disco virtual se llenó y corrompió al instalar CUDA toolkit
- MSYS2/Git Bash en Windows manglea rutas Linux al pasar a wsl.exe

Todo el desarrollo futuro es en Ubuntu bare metal.
