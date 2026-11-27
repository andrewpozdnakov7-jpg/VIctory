use super::{S_ENDPGM, lowers};

#[test]
fn sdwa_16bit_reserved_dst_unused_is_refused() {
    for dst_sel in [4u32, 5] {
        // v_cvt_i16_f16_sdwa v1, v0 dst_sel:{WORD_0,WORD_1} dst_unused:3 (reserved);
        // ds_write_b32 v0, v1 to keep v1 live.
        let sdwa_ext = (dst_sel << 8) | (3 << 11) | (4 << 16);
        let words = [
            0x7E02_A6F9, // v_cvt_i16_f16 v1, sdwa
            sdwa_ext,
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        let err = lowers(&words, 1 << 15).expect_err("reserved dst_unused must fail");
        assert!(
            matches!(
                err,
                crate::ShaderError::Unsupported {
                    what: "an SDWA destination of a 16-bit operation",
                    ..
                }
            ),
            "unexpected error for dst_sel={dst_sel}: {err}"
        );
    }
}
