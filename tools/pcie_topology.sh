#!/bin/bash
# gpu-nvme-direct: PCIe Topology and ACS Check
#
# Shows PCIe device tree and checks for Access Control Services (ACS)
# which can block peer-to-peer DMA between GPU and NVMe.
#
# Usage: sudo ./pcie_topology.sh
#
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

echo "=== PCIe Topology ==="
echo ""
lspci -tv 2>/dev/null || echo "(lspci -tv not available)"

echo ""
echo "=== GPU Devices ==="
lspci -nn 2>/dev/null | grep -i "VGA\|3D\|Display" || echo "No GPU found"

echo ""
echo "=== NVMe Devices ==="
lspci -nn 2>/dev/null | grep -i "NVMe\|Non-Volatile" || echo "No NVMe found"

echo ""
echo "=== ACS Check ==="
echo "Checking PCIe bridges for Access Control Services..."
echo "(ACS can block GPU-NVMe P2P DMA — disable if possible)"
echo ""

for bridge in /sys/bus/pci/devices/*/; do
    bdf=$(basename "$bridge")
    class=$(cat "$bridge/class" 2>/dev/null || echo "0x000000")

    # PCI bridge class = 0x0604xx
    if [[ "$class" == 0x0604* ]]; then
        # Check ACS capability
        acs=$(setpci -s "$bdf" ECAP_ACS+0x6.w 2>/dev/null || echo "N/A")
        if [ "$acs" != "N/A" ] && [ "$acs" != "0000" ]; then
            echo "  $bdf: ACS ENABLED (ctrl=0x$acs) — may block P2P"
            echo "    Disable with: setpci -s $bdf ECAP_ACS+0x6.w=0000"
        fi
    fi
done

echo ""
echo "=== IOMMU Groups ==="
if [ -d /sys/kernel/iommu_groups ]; then
    for group in /sys/kernel/iommu_groups/*/devices/*; do
        if [ -e "$group" ]; then
            bdf=$(basename "$group")
            desc=$(lspci -s "$bdf" 2>/dev/null | head -1 || echo "unknown")
            grp=$(echo "$group" | grep -o 'iommu_groups/[0-9]*' | cut -d/ -f2)
            echo "  Group $grp: $desc"
        fi
    done
else
    echo "  No IOMMU groups found (IOMMU likely disabled — good for P2P)"
fi

echo ""
echo "=== P2P Distance Check ==="
echo "For optimal P2P, GPU and NVMe should be under the same PCIe root complex."
echo "Check the topology tree above for their placement."
