#!/bin/bash
# gpu-nvme-direct: Build and load the gpunvme kernel module
#
# Usage: sudo ./setup_kmod.sh [PCI_BDF]
#   e.g.: sudo ./setup_kmod.sh 0000:03:00.0
#
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KMOD_DIR="$SCRIPT_DIR/../kmod"

echo "=== Building gpunvme kernel module ==="

# Build
cd "$KMOD_DIR"
make clean 2>/dev/null || true
make

echo ""
echo "=== Loading module ==="

# Remove if already loaded
rmmod gpunvme 2>/dev/null || true

# Load with optional target BDF
if [ $# -ge 1 ]; then
    insmod gpunvme.ko target_bdf="$1"
    echo "  Loaded with target_bdf=$1"
else
    insmod gpunvme.ko
    echo "  Loaded (will claim first unbound NVMe)"
fi

# Verify
if lsmod | grep -q gpunvme; then
    echo "  Module loaded successfully"
else
    echo "  ERROR: Module failed to load"
    dmesg | tail -5
    exit 1
fi

# Check for char device
if [ -c /dev/gpunvme0 ]; then
    echo "  Character device: /dev/gpunvme0"
    ls -la /dev/gpunvme0
else
    echo "  WARNING: /dev/gpunvme0 not created"
fi

echo ""
echo "Done. Check dmesg for details."
