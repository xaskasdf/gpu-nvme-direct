#!/bin/bash
# gpu-nvme-direct: Unbind from vfio-pci and rebind to NVMe driver
#
# Usage: sudo ./teardown.sh <PCI_BDF>
#   e.g.: sudo ./teardown.sh 0000:03:00.0
#
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <PCI_BDF>"
    exit 1
fi

BDF="$1"
SYSFS="/sys/bus/pci/devices/$BDF"

if [ ! -d "$SYSFS" ]; then
    echo "ERROR: Device $BDF not found"
    exit 1
fi

echo "=== Rebinding $BDF to nvme driver ==="

CURRENT_DRIVER=$(basename "$(readlink "$SYSFS/driver" 2>/dev/null)" 2>/dev/null || echo "none")
echo "  Current driver: $CURRENT_DRIVER"

# Unbind from vfio-pci
if [ "$CURRENT_DRIVER" = "vfio-pci" ]; then
    echo "  Unbinding from vfio-pci..."
    echo "$BDF" > /sys/bus/pci/drivers/vfio-pci/unbind 2>/dev/null || true
    sleep 0.5
fi

# Trigger driver probe (let kernel find the right driver)
echo "  Triggering driver probe..."
echo 1 > "$SYSFS/driver_override" 2>/dev/null || true
echo "" > "$SYSFS/driver_override" 2>/dev/null || true
echo "$BDF" > /sys/bus/pci/drivers_probe 2>/dev/null || true
sleep 1

# Verify
NEW_DRIVER=$(basename "$(readlink "$SYSFS/driver" 2>/dev/null)" 2>/dev/null || echo "none")
echo "  New driver: $NEW_DRIVER"

if [ "$NEW_DRIVER" = "nvme" ]; then
    echo "  SUCCESS: $BDF rebound to nvme driver"
else
    echo "  Manual rebind: echo $BDF > /sys/bus/pci/drivers/nvme/bind"
fi
