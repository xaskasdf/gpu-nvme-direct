#!/bin/bash
# gpu-nvme-direct: Bind NVMe device to vfio-pci
#
# Unbinds the NVMe from the kernel nvme driver and binds it to vfio-pci
# so userspace can directly access BAR0 registers.
#
# Usage: sudo ./setup_vfio.sh <PCI_BDF>
#   e.g.: sudo ./setup_vfio.sh 0000:03:00.0
#
# WARNING: Do NOT run this on your boot NVMe!
#
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <PCI_BDF>"
    echo "  e.g.: $0 0000:03:00.0"
    echo ""
    echo "Available NVMe devices:"
    lspci -nn | grep -i "NVMe\|Non-Volatile"
    exit 1
fi

BDF="$1"
SYSFS="/sys/bus/pci/devices/$BDF"

if [ ! -d "$SYSFS" ]; then
    echo "ERROR: Device $BDF not found in sysfs"
    exit 1
fi

# Safety: check this isn't the boot drive
ROOT_DEV=$(findmnt -n -o SOURCE / 2>/dev/null || echo "")
if echo "$ROOT_DEV" | grep -q "nvme"; then
    # Find which NVMe the root fs is on
    ROOT_NVME=$(readlink -f /sys/block/$(echo "$ROOT_DEV" | sed 's|/dev/||;s|p[0-9]*||')/device/device 2>/dev/null || echo "")
    if echo "$ROOT_NVME" | grep -q "$BDF"; then
        echo "ERROR: $BDF appears to be your boot NVMe! Refusing to unbind."
        exit 1
    fi
fi

echo "=== Binding $BDF to vfio-pci ==="

# Get vendor:device IDs
VENDOR=$(cat "$SYSFS/vendor" 2>/dev/null | sed 's/0x//')
DEVICE=$(cat "$SYSFS/device" 2>/dev/null | sed 's/0x//')
echo "  Vendor:Device = $VENDOR:$DEVICE"

# Show current driver
CURRENT_DRIVER=$(basename "$(readlink "$SYSFS/driver" 2>/dev/null)" 2>/dev/null || echo "none")
echo "  Current driver: $CURRENT_DRIVER"

# Load vfio modules
echo "  Loading vfio modules..."
modprobe vfio
modprobe vfio-pci

# Unbind from current driver
if [ "$CURRENT_DRIVER" != "none" ] && [ "$CURRENT_DRIVER" != "vfio-pci" ]; then
    echo "  Unbinding from $CURRENT_DRIVER..."
    echo "$BDF" > "$SYSFS/driver/unbind" 2>/dev/null || true
    sleep 0.5
fi

# Bind to vfio-pci
if [ "$CURRENT_DRIVER" != "vfio-pci" ]; then
    echo "  Binding to vfio-pci..."
    echo "$VENDOR $DEVICE" > /sys/bus/pci/drivers/vfio-pci/new_id 2>/dev/null || true
    echo "$BDF" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true
    sleep 0.5
fi

# Verify
NEW_DRIVER=$(basename "$(readlink "$SYSFS/driver" 2>/dev/null)" 2>/dev/null || echo "none")
if [ "$NEW_DRIVER" = "vfio-pci" ]; then
    echo "  SUCCESS: $BDF now bound to vfio-pci"
else
    echo "  WARNING: Driver is now '$NEW_DRIVER', expected 'vfio-pci'"
    echo "  Try: echo $BDF > /sys/bus/pci/drivers/vfio-pci/bind"
fi

# Show BAR0 info
echo ""
echo "  BAR0 info:"
lspci -v -s "$BDF" 2>/dev/null | grep "Region 0" || echo "    (not available via lspci -v)"

# Show resource file
if [ -f "$SYSFS/resource0" ]; then
    BAR0_SIZE=$(wc -c < "$SYSFS/resource0" 2>/dev/null || echo "unknown")
    echo "  resource0 exists (size: $BAR0_SIZE bytes)"
    echo "  Ready for mmap at: $SYSFS/resource0"
fi

echo ""
echo "To undo: sudo ./teardown.sh $BDF"
