/*
 * gpu-nvme-direct: NVMe Struct Size and Layout Tests
 *
 * Verifies that all NVMe data structures match the specification:
 *   - SQ entry = 64 bytes
 *   - CQ entry = 16 bytes
 *   - Register unions = correct sizes
 *   - Bitfield offsets are correct
 *   - Doorbell offset calculations are correct
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <gpunvme/nvme_regs.h>
#include <gpunvme/nvme_cmds.h>
#include <gpunvme/error.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    do { \
        tests_run++; \
        printf("  %-50s ", name); \
    } while(0)

#define PASS() do { tests_passed++; printf("[PASS]\n"); } while(0)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); } while(0)

#define ASSERT_EQ(a, b, msg) \
    do { \
        if ((a) != (b)) { \
            FAIL(msg); \
            printf("    expected: %llu, got: %llu\n", \
                   (unsigned long long)(b), (unsigned long long)(a)); \
            return; \
        } \
    } while(0)

/* ---- Struct Size Tests ---- */

void test_sq_entry_size() {
    TEST("SQ entry size == 64 bytes");
    ASSERT_EQ(sizeof(nvme_sq_entry_t), 64u, "wrong size");
    PASS();
}

void test_cq_entry_size() {
    TEST("CQ entry size == 16 bytes");
    ASSERT_EQ(sizeof(nvme_cq_entry_t), 16u, "wrong size");
    PASS();
}

void test_cap_size() {
    TEST("CAP register size == 8 bytes");
    ASSERT_EQ(sizeof(nvme_cap_t), 8u, "wrong size");
    PASS();
}

void test_vs_size() {
    TEST("VS register size == 4 bytes");
    ASSERT_EQ(sizeof(nvme_vs_t), 4u, "wrong size");
    PASS();
}

void test_cc_size() {
    TEST("CC register size == 4 bytes");
    ASSERT_EQ(sizeof(nvme_cc_t), 4u, "wrong size");
    PASS();
}

void test_csts_size() {
    TEST("CSTS register size == 4 bytes");
    ASSERT_EQ(sizeof(nvme_csts_t), 4u, "wrong size");
    PASS();
}

void test_aqa_size() {
    TEST("AQA register size == 4 bytes");
    ASSERT_EQ(sizeof(nvme_aqa_t), 4u, "wrong size");
    PASS();
}

/* ---- Bitfield Tests ---- */

void test_cap_mqes() {
    TEST("CAP.MQES field at bits 15:0");
    nvme_cap_t cap;
    cap.raw = 0;
    cap.bits.mqes = 0xFFFF;
    ASSERT_EQ(cap.raw & 0xFFFF, 0xFFFFu, "MQES not in bits 15:0");
    PASS();
}

void test_cap_dstrd() {
    TEST("CAP.DSTRD field at bits 35:32");
    nvme_cap_t cap;
    cap.raw = 0;
    cap.bits.dstrd = 0xF;
    ASSERT_EQ((cap.raw >> 32) & 0xF, 0xFu, "DSTRD not in bits 35:32");
    PASS();
}

void test_cc_enable() {
    TEST("CC.EN field at bit 0");
    nvme_cc_t cc;
    cc.raw = 0;
    cc.bits.en = 1;
    ASSERT_EQ(cc.raw & 1, 1u, "CC.EN not at bit 0");
    PASS();
}

void test_cc_shn() {
    TEST("CC.SHN field at bits 15:14");
    nvme_cc_t cc;
    cc.raw = 0;
    cc.bits.shn = 3;
    ASSERT_EQ((cc.raw >> 14) & 0x3, 3u, "CC.SHN not at bits 15:14");
    PASS();
}

void test_cc_iosqes() {
    TEST("CC.IOSQES field at bits 19:16");
    nvme_cc_t cc;
    cc.raw = 0;
    cc.bits.iosqes = 6;  /* 2^6 = 64 bytes */
    ASSERT_EQ((cc.raw >> 16) & 0xF, 6u, "CC.IOSQES not at bits 19:16");
    PASS();
}

void test_cc_iocqes() {
    TEST("CC.IOCQES field at bits 23:20");
    nvme_cc_t cc;
    cc.raw = 0;
    cc.bits.iocqes = 4;  /* 2^4 = 16 bytes */
    ASSERT_EQ((cc.raw >> 20) & 0xF, 4u, "CC.IOCQES not at bits 23:20");
    PASS();
}

void test_csts_rdy() {
    TEST("CSTS.RDY field at bit 0");
    nvme_csts_t csts;
    csts.raw = 0;
    csts.bits.rdy = 1;
    ASSERT_EQ(csts.raw & 1, 1u, "CSTS.RDY not at bit 0");
    PASS();
}

void test_csts_cfs() {
    TEST("CSTS.CFS field at bit 1");
    nvme_csts_t csts;
    csts.raw = 0;
    csts.bits.cfs = 1;
    ASSERT_EQ((csts.raw >> 1) & 1, 1u, "CSTS.CFS not at bit 1");
    PASS();
}

void test_vs_fields() {
    TEST("VS register: NVMe 1.4.0");
    nvme_vs_t vs;
    vs.raw = 0;
    vs.bits.mjr = 1;
    vs.bits.mnr = 4;
    vs.bits.ter = 0;
    /* NVMe 1.4.0 = 0x00010400 */
    ASSERT_EQ(vs.raw, 0x00010400u, "VS encoding wrong");
    PASS();
}

/* ---- CQE Status Macros ---- */

void test_cqe_phase_extract() {
    TEST("CQE phase bit extraction");
    uint16_t sp = 0x0001;  /* phase=1, status=0 */
    ASSERT_EQ(NVME_CQE_PHASE(sp), 1u, "phase should be 1");
    sp = 0x0000;  /* phase=0, status=0 */
    ASSERT_EQ(NVME_CQE_PHASE(sp), 0u, "phase should be 0");
    PASS();
}

void test_cqe_status_extract() {
    TEST("CQE status code extraction");
    /* Status code 0x02 (Invalid Field), phase=1 */
    uint16_t sp = (0x02 << 1) | 1;
    ASSERT_EQ(NVME_CQE_SC(sp), 0x02u, "status code should be 0x02");
    ASSERT_EQ(NVME_CQE_PHASE(sp), 1u, "phase should be 1");
    PASS();
}

/* ---- Doorbell Offset Tests ---- */

void test_doorbell_offsets_dstrd0() {
    TEST("Doorbell offsets with DSTRD=0");
    /* Admin SQ tail = 0x1000 */
    ASSERT_EQ(nvme_sq_doorbell_offset(0, 0), 0x1000u, "admin SQ tail");
    /* Admin CQ head = 0x1004 */
    ASSERT_EQ(nvme_cq_doorbell_offset(0, 0), 0x1004u, "admin CQ head");
    /* I/O SQ1 tail = 0x1008 */
    ASSERT_EQ(nvme_sq_doorbell_offset(1, 0), 0x1008u, "IO SQ1 tail");
    /* I/O CQ1 head = 0x100C */
    ASSERT_EQ(nvme_cq_doorbell_offset(1, 0), 0x100Cu, "IO CQ1 head");
    /* I/O SQ2 tail = 0x1010 */
    ASSERT_EQ(nvme_sq_doorbell_offset(2, 0), 0x1010u, "IO SQ2 tail");
    PASS();
}

void test_doorbell_offsets_dstrd1() {
    TEST("Doorbell offsets with DSTRD=1");
    /* With DSTRD=1, stride = 4 << 1 = 8 bytes */
    ASSERT_EQ(nvme_sq_doorbell_offset(0, 1), 0x1000u, "admin SQ tail");
    ASSERT_EQ(nvme_cq_doorbell_offset(0, 1), 0x1008u, "admin CQ head");
    ASSERT_EQ(nvme_sq_doorbell_offset(1, 1), 0x1010u, "IO SQ1 tail");
    ASSERT_EQ(nvme_cq_doorbell_offset(1, 1), 0x1018u, "IO CQ1 head");
    PASS();
}

/* ---- Command Builder Tests ---- */

void test_cmd_read_builder() {
    TEST("Read command builder");
    nvme_sq_entry_t cmd;
    nvme_cmd_read(&cmd, 42, 1, 0xDEADBEEF000, 0, 100, 7);

    ASSERT_EQ(cmd.opc, NVME_IO_OPC_READ, "opcode should be 0x02");
    ASSERT_EQ(cmd.cid, 42u, "CID should be 42");
    ASSERT_EQ(cmd.nsid, 1u, "NSID should be 1");
    ASSERT_EQ(cmd.prp1, 0xDEADBEEF000ull, "PRP1 mismatch");
    ASSERT_EQ(cmd.prp2, 0ull, "PRP2 should be 0");
    ASSERT_EQ(cmd.cdw10, 100u, "SLBA low should be 100");
    ASSERT_EQ(cmd.cdw11, 0u, "SLBA high should be 0");
    ASSERT_EQ(cmd.cdw12, 7u, "NLB should be 7 (0-based)");
    PASS();
}

void test_cmd_identify_builder() {
    TEST("Identify controller command builder");
    nvme_sq_entry_t cmd;
    nvme_cmd_identify_controller(&cmd, 1, 0x1000);

    ASSERT_EQ(cmd.opc, NVME_ADMIN_OPC_IDENTIFY, "opcode should be 0x06");
    ASSERT_EQ(cmd.cid, 1u, "CID should be 1");
    ASSERT_EQ(cmd.prp1, 0x1000ull, "PRP1 mismatch");
    ASSERT_EQ(cmd.cdw10, (uint32_t)NVME_IDENTIFY_CNS_CONTROLLER, "CNS should be 1");
    PASS();
}

void test_cmd_create_io_cq_builder() {
    TEST("Create I/O CQ command builder");
    nvme_sq_entry_t cmd;
    nvme_cmd_create_io_cq(&cmd, 5, 1, 63, 0xABCD0000, 0);

    ASSERT_EQ(cmd.opc, NVME_ADMIN_OPC_CREATE_IO_CQ, "opcode should be 0x05");
    ASSERT_EQ(cmd.cid, 5u, "CID should be 5");
    ASSERT_EQ(cmd.prp1, 0xABCD0000ull, "PRP1 mismatch");
    /* CDW10: QID=1 in bits 15:0, QSIZE=63 in bits 31:16 */
    ASSERT_EQ(cmd.cdw10 & 0xFFFF, 1u, "QID should be 1");
    ASSERT_EQ(cmd.cdw10 >> 16, 63u, "QSIZE should be 63");
    /* CDW11: PC=1 */
    ASSERT_EQ(cmd.cdw11 & 1, 1u, "PC should be 1");
    PASS();
}

void test_cmd_create_io_sq_builder() {
    TEST("Create I/O SQ command builder");
    nvme_sq_entry_t cmd;
    nvme_cmd_create_io_sq(&cmd, 6, 1, 63, 0xBBBB0000, 1);

    ASSERT_EQ(cmd.opc, NVME_ADMIN_OPC_CREATE_IO_SQ, "opcode should be 0x01");
    ASSERT_EQ(cmd.cdw10 & 0xFFFF, 1u, "QID should be 1");
    ASSERT_EQ(cmd.cdw10 >> 16, 63u, "QSIZE should be 63");
    ASSERT_EQ(cmd.cdw11 & 1, 1u, "PC should be 1");
    ASSERT_EQ(cmd.cdw11 >> 16, 1u, "CQID should be 1");
    PASS();
}

/* ---- SQ Entry Field Offset Tests ---- */

void test_sq_entry_offsets() {
    TEST("SQ entry field byte offsets");
    nvme_sq_entry_t cmd;
    memset(&cmd, 0, sizeof(cmd));

    uint8_t *base = (uint8_t *)&cmd;

    /* NSID should be at offset 4 */
    ASSERT_EQ((uint8_t *)&cmd.nsid - base, 4, "NSID offset");
    /* MPTR should be at offset 16 */
    ASSERT_EQ((uint8_t *)&cmd.mptr - base, 16, "MPTR offset");
    /* PRP1 should be at offset 24 */
    ASSERT_EQ((uint8_t *)&cmd.prp1 - base, 24, "PRP1 offset");
    /* PRP2 should be at offset 32 */
    ASSERT_EQ((uint8_t *)&cmd.prp2 - base, 32, "PRP2 offset");
    /* CDW10 should be at offset 40 */
    ASSERT_EQ((uint8_t *)&cmd.cdw10 - base, 40, "CDW10 offset");
    /* CDW15 should be at offset 60 */
    ASSERT_EQ((uint8_t *)&cmd.cdw15 - base, 60, "CDW15 offset");
    PASS();
}

/* ---- Error Code Tests ---- */

void test_error_strings() {
    TEST("Error code to string conversion");
    if (strcmp(gpunvme_err_str(GPUNVME_OK), "success") != 0) {
        FAIL("GPUNVME_OK string wrong");
        return;
    }
    if (strcmp(gpunvme_err_str(GPUNVME_ERR_TIMEOUT), "timeout") != 0) {
        FAIL("GPUNVME_ERR_TIMEOUT string wrong");
        return;
    }
    PASS();
}

/* ---- Main ---- */

int main() {
    printf("=== NVMe Struct Tests ===\n\n");

    printf("-- Size Tests --\n");
    test_sq_entry_size();
    test_cq_entry_size();
    test_cap_size();
    test_vs_size();
    test_cc_size();
    test_csts_size();
    test_aqa_size();

    printf("\n-- Bitfield Tests --\n");
    test_cap_mqes();
    test_cap_dstrd();
    test_cc_enable();
    test_cc_shn();
    test_cc_iosqes();
    test_cc_iocqes();
    test_csts_rdy();
    test_csts_cfs();
    test_vs_fields();

    printf("\n-- CQE Status Macro Tests --\n");
    test_cqe_phase_extract();
    test_cqe_status_extract();

    printf("\n-- Doorbell Offset Tests --\n");
    test_doorbell_offsets_dstrd0();
    test_doorbell_offsets_dstrd1();

    printf("\n-- Command Builder Tests --\n");
    test_cmd_read_builder();
    test_cmd_identify_builder();
    test_cmd_create_io_cq_builder();
    test_cmd_create_io_sq_builder();

    printf("\n-- SQ Entry Layout Tests --\n");
    test_sq_entry_offsets();

    printf("\n-- Error Code Tests --\n");
    test_error_strings();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
