# gpu-nvme-direct — Build & Test Guide

## Hardware

| Component | Detail |
|---|---|
| GPU | NVIDIA RTX 3090 (sm_86) |
| CPU | AMD Ryzen 7 5800X |
| OS | Ubuntu 24.04 LTS (bare metal, installed on NVMe) |
| NVMe test | PCIe 4.0 (dedicated SSD for testing, **NOT** the boot drive) |
| CUDA | 12.4 |

---

## Step 0: Install Ubuntu and dependencies

### BIOS (configure before installing)

- **Above 4G Decoding**: ON (required for BAR0 mapping)
- **IOMMU**: OFF (or add `amd_iommu=off` to GRUB later)
- **Secure Boot**: OFF (for the custom kernel module)

### Install Ubuntu

1. Download [Ubuntu 24.04 LTS](https://ubuntu.com/download/desktop)
2. Flash to USB with [Rufus](https://rufus.ie) or [Ventoy](https://ventoy.net)
3. Install on the new NVMe (NOT on the Windows disk)
4. Verify it boots correctly with dual-boot

### Install dependencies

```bash
# Build essentials
sudo apt update
sudo apt install -y build-essential cmake git

# NVIDIA driver + CUDA toolkit
# Option 1: From Ubuntu repos (simpler)
sudo apt install -y nvidia-driver-555

# Option 2: From NVIDIA repos (more control over the version)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install CUDA toolkit (nvcc, headers, libs)
sudo apt install -y cuda-toolkit-12-4

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version       # should say 12.4
nvidia-smi           # should show RTX 3090

# For benchmarks (optional)
sudo apt install -y python3 python3-pip python3-matplotlib python3-numpy
```

### If cmake < 3.22

```bash
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
    | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main' \
    | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
sudo apt update && sudo apt install -y cmake
```

---

## Step 1: Build Phase 0 (Simulator)

Compiles and tests the full NVMe logic using a software simulator.
No real NVMe hardware needed. Useful for verifying that the GPU code works.

```bash
cd ~/gpu-nvme-direct
mkdir -p build && cd build

# Configure with simulator enabled
cmake .. -DCMAKE_BUILD_TYPE=Debug -DGPUNVME_USE_SIM=ON

# Build
cmake --build . -j$(nproc)

# Run tests
ctest --output-on-failure
```

### What to expect

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

If both pass, the GPU kernel logic (SQ submit -> doorbell -> CQ poll) works.

---

## Step 2: Identify and prepare the test NVMe

```bash
# 1. Verify system prerequisites
sudo ./scripts/check_prereqs.sh

# 2. Identify the test NVMe (note the BDF, e.g.: 0000:03:00.0)
lspci -nn | grep NVMe

# 3. VERIFY that it is NOT the boot drive
findmnt /
# If it says /dev/nvme0n1p2, your boot is on nvme0.
# The test NVMe should be nvme1 or similar.
# NEVER use the boot drive.

# 4. Capture baseline for later verification
sudo dd if=/dev/nvmeXn1 bs=512 count=1 of=/tmp/baseline.bin

# 5. Verify PCIe topology (GPU and NVMe on the same root complex = best)
sudo ./tools/pcie_topology.sh

# 6. Unbind from the kernel and bind to vfio-pci
sudo ./scripts/setup_vfio.sh 0000:XX:00.0   # <-- your BDF
```

---

## Step 3: Build Phase 1+ (Real hardware)

```bash
cd ~/gpu-nvme-direct
mkdir -p build-hw && cd build-hw

# Configure WITHOUT simulator
cmake .. -DCMAKE_BUILD_TYPE=Release -DGPUNVME_USE_SIM=OFF

# Build
cmake --build . -j$(nproc)
```

---

## Step 4: Hardware tests

### Test 1: BAR0 register dump (CPU-side)

```bash
sudo ./build-hw/dump_bar0 0000:XX:00.0
```

Should print: CAP, VS (NVMe version), CC, CSTS, etc.

### Test 2: GPU reads NVMe register via MMIO (MILESTONE 1)

```bash
sudo ./build-hw/check_p2p 0000:XX:00.0
```

If `cudaHostRegisterIoMemory` **succeeds**:

```
RESULT: cudaHostRegisterIoMemory SUCCEEDED!
GPU read NVMe Version: 1.4.0 (raw: 0x00010400)
RESULT: *** SUCCESS *** GPU and CPU read matching Version registers!
```

If it **fails** (likely on GeForce):

```
RESULT: cudaHostRegisterIoMemory FAILED: invalid argument
```

This is a valid research result. Document it.

### Test 3: Full BAR0 read from GPU

```bash
sudo ./build-hw/test_bar_read 0000:XX:00.0
```

Compares 5 registers read by CPU vs GPU. All should match.

### Test 4: GPU reads a block from NVMe (MILESTONE 2 — THE BIG ONE)

```bash
sudo ./build-hw/test_single_block 0000:XX:00.0 /tmp/baseline.bin
```

If it says `*** MILESTONE: Data matches baseline! ***` -> **GPU read a block
from the NVMe autonomously, without CPU in the data path.**

---

## Step 5: Benchmarks

```bash
cd build-hw/bench

# GPU-direct benchmark
./bench_gpu_direct --block-size 4K --num-ops 1000 --output results_gpu.csv

# CPU baselines
./bench_cpu_memcpy --device /dev/nvmeXn1 --block-size 4K --num-ops 1000
./bench_cpu_pinned --device /dev/nvmeXn1 --block-size 4K --num-ops 1000

# Full sweep (~30 min)
python3 bench_sweep.py --device /dev/nvmeXn1

# Generate plots
python3 plot_results.py aggregate_results.csv
```

---

## Step 6: Kernel module (if Phase 1 works)

```bash
# Build
cd ~/gpu-nvme-direct/kmod
make

# Load
sudo ../scripts/setup_kmod.sh 0000:XX:00.0

# Verify
ls -la /dev/gpunvme0
dmesg | tail -20
```

---

## Quick command reference

```bash
# === SIMULATOR (verify logic) ===
cd ~/gpu-nvme-direct && mkdir build && cd build
cmake .. -DGPUNVME_USE_SIM=ON && cmake --build . -j$(nproc) && ctest

# === REAL HARDWARE ===
cd ~/gpu-nvme-direct && mkdir build-hw && cd build-hw
cmake .. -DGPUNVME_USE_SIM=OFF && cmake --build . -j$(nproc)

# NVMe setup:
sudo ../scripts/check_prereqs.sh
sudo ../scripts/setup_vfio.sh 0000:XX:00.0

# Milestones:
sudo ./check_p2p 0000:XX:00.0
sudo ./test_bar_read 0000:XX:00.0
sudo ./test_single_block 0000:XX:00.0 /tmp/baseline.bin

# When done, return NVMe to kernel:
sudo ../scripts/teardown.sh 0000:XX:00.0
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `nvcc: command not found` | Install cuda-toolkit-12-4, verify PATH |
| `cmake version too old` | Update cmake from Kitware PPA |
| `cudaHostRegisterIoMemory: invalid argument` | Expected on GeForce. Try with NVIDIA open-source kernel modules |
| `No CUDA devices found` | Verify `nvidia-smi`. Reinstall driver if it doesn't work |
| `test_sim_basic` timeout | Increase timeout in `gpu_nvme_queue.poll_timeout_cycles` |
| `resource0: Permission denied` | You need `sudo` + device bound to vfio-pci |
| `CSTS.CFS = 1` (fatal) | Controller crashed. Rebind with `teardown.sh`, reboot if necessary |
| NVMe doesn't appear after unbind | `sudo ../scripts/teardown.sh`, then re-setup |
| `vfio-pci: probe failed` | Verify that IOMMU is OFF in GRUB |
