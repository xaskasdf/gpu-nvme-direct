#!/bin/bash
# gpu-nvme-direct: Check prerequisites for bare-metal NVMe access
#
# Verifies BIOS settings, kernel modules, GPU driver, and NVMe devices.
# Run this BEFORE attempting Phase 1+ tests on real hardware.
#
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass=0
warn=0
fail=0

check_pass() { echo -e "  ${GREEN}[PASS]${NC} $1"; ((pass++)); }
check_warn() { echo -e "  ${YELLOW}[WARN]${NC} $1"; ((warn++)); }
check_fail() { echo -e "  ${RED}[FAIL]${NC} $1"; ((fail++)); }

echo "=== gpu-nvme-direct: Prerequisite Check ==="
echo ""

# ---- IOMMU ----
echo "-- IOMMU --"
if dmesg 2>/dev/null | grep -qi "iommu.*enabled\|DMAR.*IOMMU enabled"; then
    check_warn "IOMMU appears ENABLED. For direct BAR access, disable with amd_iommu=off or intel_iommu=off in GRUB."
elif dmesg 2>/dev/null | grep -qi "iommu.*disabled\|AMD-Vi.*disabled"; then
    check_pass "IOMMU appears disabled (good for direct BAR access)"
else
    check_warn "Could not determine IOMMU status. Check: dmesg | grep -i iommu"
fi

# ---- Above 4G Decoding ----
echo ""
echo "-- Above 4G Decoding --"
if dmesg 2>/dev/null | grep -q "BAR.*above 4G\|PCI.*above 4G\|64bit.*BAR"; then
    check_pass "Above 4G Decoding appears enabled"
else
    large_bar=$(lspci -v 2>/dev/null | grep -c "Memory.*64-bit.*prefetchable" || true)
    if [ "$large_bar" -gt 0 ]; then
        check_pass "Found 64-bit BAR mappings (Above 4G Decoding likely enabled)"
    else
        check_warn "Could not confirm Above 4G Decoding. Enable in BIOS if BAR mapping fails."
    fi
fi

# ---- NVIDIA Driver ----
echo ""
echo "-- NVIDIA Driver --"
if lsmod 2>/dev/null | grep -q "^nvidia "; then
    check_pass "NVIDIA kernel module loaded"
    nvidia_ver=$(cat /proc/driver/nvidia/version 2>/dev/null | head -1 || echo "unknown")
    echo "         Driver: $nvidia_ver"
else
    check_fail "NVIDIA kernel module not loaded. Install NVIDIA driver."
fi

if command -v nvidia-smi &>/dev/null; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    check_pass "nvidia-smi available. GPU: $gpu_name"
else
    check_warn "nvidia-smi not found"
fi

# ---- CUDA ----
echo ""
echo "-- CUDA --"
if command -v nvcc &>/dev/null; then
    cuda_ver=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    check_pass "CUDA toolkit found: $cuda_ver"
else
    check_fail "nvcc not found. Install CUDA toolkit."
fi

# ---- NVMe Devices ----
echo ""
echo "-- NVMe Devices --"
nvme_devs=$(lspci -nn 2>/dev/null | grep -i "Non-Volatile\|NVMe" || true)
if [ -z "$nvme_devs" ]; then
    check_fail "No NVMe devices found in lspci"
else
    nvme_count=$(echo "$nvme_devs" | wc -l)
    check_pass "Found $nvme_count NVMe device(s):"
    echo "$nvme_devs" | while read -r line; do
        echo "         $line"
    done

    if [ "$nvme_count" -lt 2 ]; then
        check_warn "Only 1 NVMe device found. Make sure it's NOT your boot drive!"
        echo "         Boot drive check:"
        root_dev=$(findmnt -n -o SOURCE / 2>/dev/null || echo "unknown")
        echo "         Root filesystem: $root_dev"
        if echo "$root_dev" | grep -q "nvme"; then
            check_warn "Root FS is on NVMe. You NEED a second NVMe for testing!"
        fi
    fi
fi

# ---- PCIe Topology ----
echo ""
echo "-- PCIe Topology --"
if command -v lspci &>/dev/null; then
    check_pass "PCIe topology:"
    lspci -tv 2>/dev/null | head -30
    echo "         (truncated)"
else
    check_warn "lspci not available"
fi

# ---- VFIO ----
echo ""
echo "-- VFIO Module --"
if modprobe -n vfio-pci 2>/dev/null; then
    check_pass "vfio-pci module available"
else
    check_warn "vfio-pci module not available. May need: modprobe vfio-pci"
fi

# ---- Kernel Headers ----
echo ""
echo "-- Kernel Headers --"
if [ -d "/lib/modules/$(uname -r)/build" ]; then
    check_pass "Kernel headers found for $(uname -r)"
else
    check_warn "Kernel headers not found. Install: apt install linux-headers-$(uname -r)"
fi

# ---- Summary ----
echo ""
echo "=== Summary ==="
echo -e "  ${GREEN}Pass: $pass${NC}  ${YELLOW}Warn: $warn${NC}  ${RED}Fail: $fail${NC}"

if [ $fail -gt 0 ]; then
    echo ""
    echo "Fix FAIL items before proceeding with Phase 1."
    exit 1
fi

if [ $warn -gt 0 ]; then
    echo ""
    echo "Review WARN items. Some may need BIOS changes."
fi
