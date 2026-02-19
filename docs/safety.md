# Safety and Error Handling

## Data Corruption Vectors

### 1. Wrong LBA Writes
- **Risk**: GPU writes to wrong LBA, corrupting filesystem data
- **Prevention**: Use a dedicated test SSD with NO important data
- **Mitigation**: Verify LBA range before each write, bounds checking in kernel

### 2. Doorbell Race Conditions
- **Risk**: Multiple GPU threads writing same doorbell simultaneously
- **Prevention**: Single-thread-per-queue model in Phase 0
- **Future**: Per-thread queue assignment or warp-cooperative submission

### 3. Command ID (CID) Reuse
- **Risk**: Reusing a CID before previous command completes causes confusion
- **Prevention**: Sequential CID counter, CQ polling before new submission
- **Mitigation**: CID wraps at 65535 — bounded queue depth prevents collision

### 4. SQ/CQ Memory Corruption
- **Risk**: Bug in SQ entry construction writes invalid command
- **Prevention**: Extensive struct size tests, command builder validation

## Memory Ordering Requirements

Every SQ submission MUST follow this sequence:

```
1. Write 64 bytes of SQ entry (regular stores)
2. __threadfence_system()        ← SQ entry visible across PCIe
3. mmio_write32(doorbell, tail)  ← PTX st.mmio.sys
4. __threadfence_system()        ← doorbell reaches NVMe
```

Violation causes: NVMe reads stale SQ data, DMA to wrong address.

## Controller Fatal Status (CFS)

If CSTS.CFS=1, the controller has encountered an unrecoverable error.

- **Detection**: Monitor thread polls CSTS every 1ms
- **Response**: Stop GPU kernels, log error, attempt controller reset
- **Reset sequence**: CC.EN=0 → wait RDY=0 → reinitialize

## Timeout Handling

All polling loops have bounded timeouts:
- Admin commands: CAP.TO * 500ms (typically 5 seconds)
- I/O commands: 100ms default (configurable in gpu_nvme_queue)
- GPU kernel timeout: Based on GPU clock cycles

## Safe Usage Checklist

1. NEVER use your boot NVMe as the test drive
2. Always run `check_prereqs.sh` before first use
3. Monitor `dmesg` for PCIe errors during operation
4. Use `teardown.sh` to rebind NVMe to kernel driver when done
5. Back up any data before experiments
