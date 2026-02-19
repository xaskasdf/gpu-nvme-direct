# NVMe Test Drive Purchase Guide

## Requirements

You need a **dedicated test NVMe** that is NOT your boot drive.
This SSD will be taken over by VFIO (detached from the OS) and
directly controlled by our code. Data on it WILL be destroyed.

## Recommendations

| Drive | Gen | Size | Price (2024) | Notes |
|-------|-----|------|-------------|-------|
| Samsung 980 (non-Pro) | 3 | 250GB | ~$25 used | Well-tested firmware, good for research |
| WD SN570 | 3 | 250GB | ~$25 new | Budget, reliable |
| WD SN770 | 4 | 250GB | ~$30 new | Good Gen4 option |
| Kingston NV2 | 4 | 250GB | ~$20 new | Cheapest option |
| SK Hynix P31 Gold | 3 | 500GB | ~$35 used | Good sustained perf |

## Avoid

- **Intel Optane**: Unusual firmware, non-standard NVMe extensions
- **No-name brands**: Buggy firmware, may not handle raw register access well
- **Your boot drive**: Obviously

## What matters

- **Gen 3 or 4**: Both work fine. Gen4 for max throughput benchmarks.
- **Size**: 128-256GB is plenty. We're testing I/O path, not capacity.
- **Used is fine**: We don't care about endurance for a test SSD.
- **Standard NVMe**: Must be a standard NVMe drive (class 0x0108), not RAID or proprietary.

## Physical installation

1. Install in an M.2 slot or use an M.2-to-PCIe adapter card
2. **Preferred**: Same PCIe root complex as your GPU (check `lspci -tv`)
3. If using an adapter card, use a x4 slot close to the GPU slot
4. BIOS: Ensure the M.2 slot is enabled and visible

## After installation

```bash
# Verify the drive appears
lspci -nn | grep NVMe

# Check it's NOT your boot drive
findmnt /

# Note the BDF (e.g., 0000:03:00.0) for use with setup_vfio.sh
```
