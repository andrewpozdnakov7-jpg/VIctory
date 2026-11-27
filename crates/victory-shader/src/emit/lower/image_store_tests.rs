//! Unit and DXIL validation tests for partial-channel and `d16` `image_store`.

use super::tests::{S_ENDPGM, lower_with};
use crate::ShaderError;
use crate::native::{ResourceType, RootParam, RootType, UsedResource};

fn table_image_uav_layout() -> Vec<RootParam> {
    vec![RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data: 0,
        slot: 0,
    }]
}

fn table_image_uav_used() -> Vec<UsedResource> {
    vec![UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 8,
        user_data: 0,
        slot: 0,
    }]
}

#[test]
fn partial_dmask_and_d16_image_stores_lower_and_validate() {
    let layout = table_image_uav_layout();
    let used = table_image_uav_used();
    // Every non-zero 4-bit dmask in both 32-bit and d16 modes on 2D UAVs.
    for d16 in [false, true] {
        for dmask in 1u32..=0xF {
            let w0 = 0xF020_1008 | (dmask << 8); // image_store dim:2D unorm dmask
            let w1 = (u32::from(d16) << 31) | 0x0002_0105; // vdata=v1, vaddr=v[5:6], srsrc=s[8:15]
            let words = [
                0xBE84_0300, // s_mov_b32 s4, s0
                0xBE85_0380, // s_mov_b32 s5, 0
                0xF40C_0202,
                0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0x0
                0x7E02_02F2, // v_mov_b32 v1, 1.0
                0x7E04_02F0, // v_mov_b32 v2, 0.5
                0x7E06_0280, // v_mov_b32 v3, 0
                0x7E08_02F2, // v_mov_b32 v4, 1.0
                0x7E0A_0300, // v_mov_b32 v5, v0 (x)
                0x7E0C_0280, // v_mov_b32 v6, 0  (y)
                0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
                w0,
                w1,
                S_ENDPGM,
            ];
            let emitted = lower_with(&words, layout.clone(), used.clone())
                .unwrap_or_else(|e| panic!("image_store dmask={dmask:#x} d16={d16} failed: {e}"));
            assert!(!emitted.dxil.is_empty());
            assert_eq!(
                emitted.typed_texture_annotations,
                [true, d16],
                "2D image_store dmask={dmask:#x} d16={d16}",
            );
            #[cfg(windows)]
            if let Ok(mut v) = crate::emit::validator::Validator::load(
                crate::emit::validator::Validator::default_path().as_deref(),
            ) {
                v.validate(&mut emitted.dxil.clone()).unwrap_or_else(|e| {
                    panic!("image_store dmask={dmask:#x} d16={d16} invalid DXIL: {e}")
                });
            }
        }
    }
    // Partial-channel stores across 1D, 3D, 1DArray, and 2DArray dimensions.
    for dim in [0u32, 2, 4, 5] {
        for dmask in [0x1u32, 0x3, 0x7] {
            let w0 = 0xF020_1000 | (dmask << 8) | (dim << 3);
            let w1 = 0x0002_0104; // vdata=v1, vaddr=v[4..], srsrc=s[8:15]
            let words = [
                0xBE84_0300, // s_mov_b32 s4, s0
                0xBE85_0380, // s_mov_b32 s5, 0
                0xF40C_0202,
                0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0x0
                0x7E02_02F2, // v_mov_b32 v1, 1.0
                0x7E04_02F0, // v_mov_b32 v2, 0.5
                0x7E06_0280, // v_mov_b32 v3, 0
                0x7E08_0300, // v_mov_b32 v4, v0 (x)
                0x7E0A_0280, // v_mov_b32 v5, 0  (y/slice)
                0x7E0C_0280, // v_mov_b32 v6, 0  (z/slice)
                0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
                w0,
                w1,
                S_ENDPGM,
            ];
            let emitted = lower_with(&words, layout.clone(), used.clone())
                .unwrap_or_else(|e| panic!("image_store dim={dim} dmask={dmask:#x} failed: {e}"));
            assert!(!emitted.dxil.is_empty());
            #[cfg(windows)]
            if let Ok(mut v) = crate::emit::validator::Validator::load(
                crate::emit::validator::Validator::default_path().as_deref(),
            ) {
                v.validate(&mut emitted.dxil.clone()).unwrap_or_else(|e| {
                    panic!("image_store dim={dim} dmask={dmask:#x} invalid DXIL: {e}")
                });
            }
        }
    }
}

#[test]
fn image_store_with_zero_dmask_is_noop() {
    let layout = table_image_uav_layout();
    let used = table_image_uav_used();
    for d16 in [false, true] {
        let w0 = 0xF020_1008; // image_store dim:2D unorm dmask:0x0
        let w1 = (u32::from(d16) << 31) | 0x0002_0104;
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF40C_0202,
            0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0x0
            0x7E02_02F2, // v_mov_b32 v1, 1.0
            0x7E08_0300, // v_mov_b32 v4, v0
            0x7E0A_0280, // v_mov_b32 v5, 0
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            w0,
            w1,
            S_ENDPGM,
        ];
        let emitted = lower_with(&words, layout.clone(), used.clone())
            .unwrap_or_else(|e| panic!("image_store dmask=0 d16={d16} failed: {e}"));
        assert!(!emitted.dxil.is_empty());
        #[cfg(windows)]
        if let Ok(mut v) = crate::emit::validator::Validator::load(
            crate::emit::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut emitted.dxil.clone())
                .unwrap_or_else(|e| panic!("image_store dmask=0 d16={d16} invalid DXIL: {e}"));
        }
    }
}

#[test]
fn image_store_mip_and_pck_are_still_refused() {
    let layout = table_image_uav_layout();
    let used = table_image_uav_used();
    let cases = [
        (0xF024_1008u32, "an image store to a mip level"), // image_store_mip (0x009)
        (0xF028_1008u32, "a packed image store"),          // image_store_pck (0x00a)
    ];
    for (base, expected) in cases {
        for dmask in [0x0u32, 0x1, 0xF] {
            let w0 = base | (dmask << 8);
            let w1 = 0x0002_0104;
            let words = [
                0xBE84_0300, // s_mov_b32 s4, s0
                0xBE85_0380, // s_mov_b32 s5, 0
                0xF40C_0202,
                0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0x0
                0x7E02_02F2, // v_mov_b32 v1, 1.0
                0x7E04_02F0, // v_mov_b32 v2, 0.5
                0x7E06_0280, // v_mov_b32 v3, 0
                0x7E08_0300, // v_mov_b32 v4, v0
                0x7E0A_0280, // v_mov_b32 v5, 0
                0x7E0C_0280, // v_mov_b32 v6, 0 (mip)
                0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
                w0,
                w1,
                S_ENDPGM,
            ];
            assert!(matches!(
                lower_with(&words, layout.clone(), used.clone()),
                Err(ShaderError::Unsupported { what, .. }) if what == expected
            ));
        }
    }
}
