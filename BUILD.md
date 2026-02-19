# gpu-nvme-direct — Build & Test Guide

## Hardware

| Componente | Detalle |
|---|---|
| GPU | NVIDIA RTX 3090 (sm_86) |
| CPU | AMD Ryzen 7 5800X |
| OS | Ubuntu 24.04 LTS (bare metal, instalado en NVMe) |
| NVMe test | PCIe 4.0 (SSD dedicado para testing, **NO** el boot) |
| CUDA | 12.4 |

---

## Paso 0: Instalar Ubuntu y dependencias

### BIOS (configurar antes de instalar)

- **Above 4G Decoding**: ON (necesario para mapear BAR0)
- **IOMMU**: OFF (o agregar `amd_iommu=off` a GRUB después)
- **Secure Boot**: OFF (para el kernel module custom)

### Instalar Ubuntu

1. Descargar [Ubuntu 24.04 LTS](https://ubuntu.com/download/desktop)
2. Grabar en USB con [Rufus](https://rufus.ie) o [Ventoy](https://ventoy.net)
3. Instalar en la NVMe nueva (NO en el disco de Windows)
4. Verificar que arranca correctamente con dual-boot

### Instalar dependencias

```bash
# Build essentials
sudo apt update
sudo apt install -y build-essential cmake git

# NVIDIA driver + CUDA toolkit
# Opción 1: Desde repos de Ubuntu (más simple)
sudo apt install -y nvidia-driver-555

# Opción 2: Desde repos de NVIDIA (más control sobre la versión)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Instalar CUDA toolkit (nvcc, headers, libs)
sudo apt install -y cuda-toolkit-12-4

# Agregar al PATH
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verificar
nvcc --version       # debe decir 12.4
nvidia-smi           # debe mostrar RTX 3090

# Para benchmarks (opcional)
sudo apt install -y python3 python3-pip python3-matplotlib python3-numpy
```

### Si cmake < 3.22

```bash
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
    | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main' \
    | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
sudo apt update && sudo apt install -y cmake
```

---

## Paso 1: Build Phase 0 (Simulador)

Compila y testea la lógica NVMe completa usando un simulador en software.
No necesita hardware NVMe real. Sirve para verificar que el código GPU funciona.

```bash
cd ~/gpu-nvme-direct
mkdir -p build && cd build

# Configurar con simulador habilitado
cmake .. -DCMAKE_BUILD_TYPE=Debug -DGPUNVME_USE_SIM=ON

# Compilar
cmake --build . -j$(nproc)

# Correr tests
ctest --output-on-failure
```

### Qué esperar

```
=== NVMe Struct Tests ===

-- Size Tests --
  SQ entry size == 64 bytes                            [PASS]
  CQ entry size == 16 bytes                            [PASS]
  CAP register size == 8 bytes                         [PASS]
  ...
=== Results: 25/25 passed ===
```

```
=== GPU NVMe Simulator Tests ===

Using GPU: NVIDIA GeForce RTX 3090 (SM 8.6)

  TEST: Single block read through simulator... [PASS]
  TEST: Multi-block read (8 blocks, 4KB)...    [PASS]
  TEST: 4KB block size read...                 [PASS]

=== Results: 3/3 passed ===
```

Si ambos pasan, la lógica del GPU kernel (SQ submit -> doorbell -> CQ poll) funciona.

---

## Paso 2: Identificar y preparar la NVMe de test

```bash
# 1. Verificar prerequisitos del sistema
sudo ./scripts/check_prereqs.sh

# 2. Identificar la NVMe de test (anotar el BDF, ej: 0000:03:00.0)
lspci -nn | grep NVMe

# 3. VERIFICAR que NO es el boot drive
findmnt /
# Si dice /dev/nvme0n1p2, tu boot está en nvme0.
# La NVMe de test debería ser nvme1 o similar.
# NUNCA usar el boot drive.

# 4. Capturar baseline para verificación posterior
sudo dd if=/dev/nvmeXn1 bs=512 count=1 of=/tmp/baseline.bin

# 5. Verificar topología PCIe (GPU y NVMe en el mismo root complex = mejor)
sudo ./tools/pcie_topology.sh

# 6. Unbind del kernel y bind a vfio-pci
sudo ./scripts/setup_vfio.sh 0000:XX:00.0   # <-- tu BDF
```

---

## Paso 3: Build Phase 1+ (Hardware real)

```bash
cd ~/gpu-nvme-direct
mkdir -p build-hw && cd build-hw

# Configurar SIN simulador
cmake .. -DCMAKE_BUILD_TYPE=Release -DGPUNVME_USE_SIM=OFF

# Compilar
cmake --build . -j$(nproc)
```

---

## Paso 4: Tests de hardware

### Test 1: Dump de registros BAR0 (CPU-side)

```bash
sudo ./build-hw/dump_bar0 0000:XX:00.0
```

Debe imprimir: CAP, VS (versión NVMe), CC, CSTS, etc.

### Test 2: GPU lee registro NVMe via MMIO (MILESTONE 1)

```bash
sudo ./build-hw/check_p2p 0000:XX:00.0
```

Si `cudaHostRegisterIoMemory` **funciona**:

```
RESULT: cudaHostRegisterIoMemory SUCCEEDED!
GPU read NVMe Version: 1.4.0 (raw: 0x00010400)
RESULT: *** SUCCESS *** GPU and CPU read matching Version registers!
```

Si **falla** (probable en GeForce):

```
RESULT: cudaHostRegisterIoMemory FAILED: invalid argument
```

Esto es un resultado de investigación válido. Documentarlo.

### Test 3: Lectura BAR0 completa desde GPU

```bash
sudo ./build-hw/test_bar_read 0000:XX:00.0
```

Compara 5 registros leídos por CPU vs GPU. Todos deben coincidir.

### Test 4: GPU lee un bloque del NVMe (MILESTONE 2 — EL GRANDE)

```bash
sudo ./build-hw/test_single_block 0000:XX:00.0 /tmp/baseline.bin
```

Si dice `*** MILESTONE: Data matches baseline! ***` -> **GPU leyó un bloque
del NVMe de forma autónoma, sin CPU en el data path.**

---

## Paso 5: Benchmarks

```bash
cd build-hw/bench

# Benchmark GPU-direct
./bench_gpu_direct --block-size 4K --num-ops 1000 --output results_gpu.csv

# Baselines CPU
./bench_cpu_memcpy --device /dev/nvmeXn1 --block-size 4K --num-ops 1000
./bench_cpu_pinned --device /dev/nvmeXn1 --block-size 4K --num-ops 1000

# Sweep completo (~30 min)
python3 bench_sweep.py --device /dev/nvmeXn1

# Generar plots
python3 plot_results.py aggregate_results.csv
```

---

## Paso 6: Kernel module (si Phase 1 funciona)

```bash
# Build
cd ~/gpu-nvme-direct/kmod
make

# Load
sudo ../scripts/setup_kmod.sh 0000:XX:00.0

# Verificar
ls -la /dev/gpunvme0
dmesg | tail -20
```

---

## Resumen de comandos rápidos

```bash
# === SIMULADOR (verificar lógica) ===
cd ~/gpu-nvme-direct && mkdir build && cd build
cmake .. -DGPUNVME_USE_SIM=ON && cmake --build . -j$(nproc) && ctest

# === HARDWARE REAL ===
cd ~/gpu-nvme-direct && mkdir build-hw && cd build-hw
cmake .. -DGPUNVME_USE_SIM=OFF && cmake --build . -j$(nproc)

# Setup NVMe:
sudo ../scripts/check_prereqs.sh
sudo ../scripts/setup_vfio.sh 0000:XX:00.0

# Milestones:
sudo ./check_p2p 0000:XX:00.0
sudo ./test_bar_read 0000:XX:00.0
sudo ./test_single_block 0000:XX:00.0 /tmp/baseline.bin

# Cuando termines, devolver NVMe al kernel:
sudo ../scripts/teardown.sh 0000:XX:00.0
```

---

## Troubleshooting

| Problema | Solución |
|---|---|
| `nvcc: command not found` | Instalar cuda-toolkit-12-4, verificar PATH |
| `cmake version too old` | Actualizar cmake desde Kitware PPA |
| `cudaHostRegisterIoMemory: invalid argument` | Esperado en GeForce. Intentar con NVIDIA open-source kernel modules |
| `No CUDA devices found` | Verificar `nvidia-smi`. Reinstalar driver si no funciona |
| `test_sim_basic` timeout | Aumentar timeout en `gpu_nvme_queue.poll_timeout_cycles` |
| `resource0: Permission denied` | Necesitas `sudo` + device bound a vfio-pci |
| `CSTS.CFS = 1` (fatal) | Controller crasheó. Rebind con `teardown.sh`, reboot si es necesario |
| NVMe no aparece después de unbind | `sudo ../scripts/teardown.sh`, luego re-setup |
| `vfio-pci: probe failed` | Verificar que IOMMU está OFF en GRUB |
