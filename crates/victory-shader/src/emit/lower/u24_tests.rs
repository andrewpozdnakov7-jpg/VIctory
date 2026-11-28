use super::tests::lower_valu;

#[test]
fn unsigned_24_bit_operands_use_one_mask_each() {
    // Inputs come from root constants, and a UAV store keeps the result live.
    // Counts include input/output setup and terminators. Both encodings use
    // v1/v2; v_mad also reads v3, whose full 32 bits must survive.
    let mut counts = Vec::new();
    for (op, expected) in [(0x0Bu32, 19), (0x0C, 23), (0x09, 21), (0x0A, 25)] {
        let vop2 = [op << 25 | 2 << 9 | 0x101];
        let vop3 = [0xD400_0000 | (0x100 + op) << 16, 0x101 | 0x102 << 9];
        for body in [&vop2[..], &vop3[..]] {
            counts.push((lower_valu(body).unwrap().instructions, expected));
        }
    }
    for (op, expected) in [(0x143u32, 21), (0x142, 23)] {
        let body = [0xD400_0000 | op << 16, 0x101 | 0x102 << 9 | 0x103 << 18];
        counts.push((lower_valu(&body).unwrap().instructions, expected));
    }
    eprintln!("DXIL instruction counts and budgets: {counts:?}");
    assert!(
        counts.iter().all(|&(actual, budget)| actual <= budget),
        "DXIL instruction counts and budgets: {counts:?}"
    );
}
