# NVMe Quick Reference

## Register Map (BAR0)

| Offset | Size | Name | Description |
|--------|------|------|-------------|
| 0x00 | 8B | CAP | Controller Capabilities |
| 0x08 | 4B | VS | Version |
| 0x0C | 4B | INTMS | Interrupt Mask Set |
| 0x10 | 4B | INTMC | Interrupt Mask Clear |
| 0x14 | 4B | CC | Controller Configuration |
| 0x1C | 4B | CSTS | Controller Status |
| 0x20 | 4B | NSSR | NVM Subsystem Reset |
| 0x24 | 4B | AQA | Admin Queue Attributes |
| 0x28 | 8B | ASQ | Admin SQ Base Address |
| 0x30 | 8B | ACQ | Admin CQ Base Address |
| 0x1000+ | 4B each | Doorbells | SQ Tail / CQ Head doorbells |

## Doorbell Offsets (DSTRD=0)

```
SQ Y Tail = 0x1000 + (2*Y * 4)
CQ Y Head = 0x1000 + ((2*Y+1) * 4)

Admin SQ Tail = 0x1000    Admin CQ Head = 0x1004
IO SQ1 Tail   = 0x1008    IO CQ1 Head   = 0x100C
IO SQ2 Tail   = 0x1010    IO CQ2 Head   = 0x1014
```

## Controller Init Sequence

```
1. Read CAP
2. CC.EN = 0
3. Wait CSTS.RDY = 0
4. Allocate Admin SQ/CQ (page-aligned, DMA-able)
5. Set AQA (queue sizes)
6. Set ASQ, ACQ (physical addresses)
7. Set CC (MPS, IOSQES=6, IOCQES=4, CSS=0)
8. CC.EN = 1
9. Wait CSTS.RDY = 1
```

## I/O Command: READ (opcode 0x02)

| Dword | Field | Description |
|-------|-------|-------------|
| CDW0 | OPC=0x02, CID | Opcode and Command ID |
| CDW1 | NSID | Namespace ID (usually 1) |
| PRP1 | | Physical addr of data buffer |
| PRP2 | | Second PRP or PRP list addr |
| CDW10 | SLBA[31:0] | Starting LBA low |
| CDW11 | SLBA[63:32] | Starting LBA high |
| CDW12 | NLB[15:0] | Number of Logical Blocks - 1 |

## Completion Queue Entry (16 bytes)

| Offset | Field | Description |
|--------|-------|-------------|
| 0 | CDW0 | Command-specific result |
| 4 | CDW1 | Reserved |
| 8 | SQHD | SQ Head Pointer |
| 10 | SQID | SQ Identifier |
| 12 | CID | Command Identifier |
| 14 | Status+Phase | Status[15:1], Phase[0] |

## Phase Bit Protocol

- Controller starts with phase = 1
- Host starts expecting phase = 1
- New completion: `(cqe.status_phase & 1) == expected_phase`
- Phase flips when CQ head wraps around to 0
