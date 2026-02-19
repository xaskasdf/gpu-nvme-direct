#!/bin/bash
# Post-reboot test script for patched nvidia module
# Run as: sudo bash scripts/post_reboot_test.sh

set -e

echo "=== Step 1: Verify patched nvidia module is loaded ==="
echo "Kernel: $(uname -r)"
echo "NVIDIA: $(cat /proc/driver/nvidia/version | head -1)"
echo ""

echo "=== Step 2: Setup VFIO for NVMe ==="
modprobe vfio enable_unsafe_noiommu_mode=1
modprobe vfio-pci

# Check if NVMe is already bound to vfio-pci
DRIVER=$(readlink /sys/bus/pci/devices/0000:0b:00.0/driver 2>/dev/null | xargs basename 2>/dev/null || echo "none")
echo "Current NVMe driver: $DRIVER"

if [ "$DRIVER" != "vfio-pci" ]; then
    bash scripts/setup_vfio.sh 0000:0b:00.0
fi

echo ""
echo "=== Step 3: Fix NVMe power state ==="
echo on > /sys/bus/pci/devices/0000:0b:00.0/power/control
setpci -s 0000:0b:00.0 0x84.W=0x0008   # Force D0
setpci -s 0000:0b:00.0 COMMAND=0x0006   # Memory + BusMaster

echo "Power state: $(setpci -s 0000:0b:00.0 0x84.W)"
echo ""

echo "=== Step 4: Verify BAR0 is readable ==="
./build/dump_bar0 0000:0b:00.0 | head -8
echo ""

echo "=== Step 5: Test direct RM ioctl (bypasses libcuda.so) ==="
./build/rm_bar0_register 0000:0b:00.0
echo ""

echo "=== Step 6: Test cudaHostRegisterIoMemory ==="
./build/check_p2p 0000:0b:00.0
echo ""

echo "=== Done ==="
