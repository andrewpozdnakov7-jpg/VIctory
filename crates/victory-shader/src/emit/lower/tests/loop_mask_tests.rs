use super::*;

/// Lowers a multi-thread compute shader storing `v2` at `v0 * 4` in a raw
/// UAV at heap index 0 (table at s0).
fn uav_cs_parts(
    body: &[u32],
    wave64: bool,
) -> Result<(ir::Program, structure::Structured, Lowered), crate::ShaderError> {
    let mut words = vec![
        0xBE81_0380, // s_mov_b32 s1, 0
        0xF408_0100,
        0xFA00_0000, // s_load_dwordx4 s[4:7], s[0:1], 0
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
    ];
    words.extend_from_slice(body);
    words.extend_from_slice(&[
        0xE070_1000,
        0x8001_0203, // buffer_store_dword v2, v3, s[4:7], 0 offen
        S_ENDPGM,
    ]);
    let layout = vec![root(RootType::Table, 0, 0)];
    let used = vec![UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 0,
        slot: 0,
    }];
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let threads = if wave64 { [64, 1, 1] } else { [32, 1, 1] };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 8,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads,
        }),
        code,
    };
    let p = ir::build(&native.code, wave64, 8).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native).unwrap();
    let lowered = lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
        view_marks: false,
    })?;
    Ok((p, st, lowered))
}

fn lower_uav_cs(body: &[u32], wave64: bool) -> Result<Lowered, crate::ShaderError> {
    uav_cs_parts(body, wave64).map(|(_, _, l)| l)
}

#[test]
fn issue_1498_wave64_divergent_loop_mask_ballots_after_loop() {
    // Exact #1498 Wave64 64-thread repro: every lane exits on iteration v0 + 1,
    // reconverges, and counts the exit mask s[10:11].
    let body = [
        0x4A02_0081, // v_add_nc_u32 v1, 1, v0
        0xBE8A_0480, // s_mov_b64 s[10:11], 0
        0x4E02_0281, // loop: v_subrev_nc_u32 v1, 1, v1
        0x7D84_0280, // v_cmp_eq_u32 vcc, 0, v1
        0x888A_0A6A, // s_or_b64 s[10:11], vcc, s[10:11]
        0x8AFE_0A7E, // s_andn2_b64 exec, exec, s[10:11]
        0xBF89_FFFB, // s_cbranch_execnz loop
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        0xBE89_100A, // s_bcnt1_i32_b64 s9, s[10:11]
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    let (p, st, mut lowered) = uav_cs_parts(&body, true).expect("wave64 #1498 lowers");
    assert!(lowered.wave_ops);
    // IR block 0 = preheader, block 1 = loop, block 2 = post-loop reconvergence.
    let post_masks: Vec<_> = p.blocks[2]
        .insts
        .iter()
        .filter(|i| i.name().starts_with("s_bcnt1"))
        .flat_map(|i| i.reads.iter().map(|r| r.1))
        .filter(|v| st.masks.contains_key(v))
        .collect();
    assert!(!post_masks.is_empty());
    for lo in post_masks {
        assert_eq!(
            st.ballot_site(lo, Some(1), 2),
            Some(structure::BallotSite::Read),
            "ballot must be placed after the loop (block 2), not inside (block 1)"
        );
    }
    #[cfg(windows)]
    if let Ok(mut v) = crate::emit::validator::Validator::load(
        crate::emit::validator::Validator::default_path().as_deref(),
    ) {
        v.validate(&mut lowered.dxil)
            .expect("DXIL validation failed for wave64 #1498");
    }
}

#[test]
fn divergent_loop_mask_read_inside_loop_or_narrowed_branch_is_refused() {
    // 1. Reading the divergent-loop accumulated mask s10 INSIDE the loop.
    let read_inside_loop = [
        0x4A02_0081, // v_add_nc_u32 v1, 1, v0
        0xBE8A_0380, // s_mov_b32 s10, 0
        0x4E02_0281, // loop: v_subrev_nc_u32 v1, 1, v1
        0x7D84_0280, // v_cmp_eq_u32 vcc_lo, 0, v1
        0x880A_0A6A, // s_or_b32 s10, vcc_lo, s10
        0xBE89_0F0A, // s_bcnt1_i32_b32 s9, s10 (inside divergent loop!)
        0x7E04_0209, // v_mov_b32 v2, s9
        0x8A7E_0A7E, // s_andn2_b32 exec_lo, exec_lo, s10
        0xBF89_FFF9, // s_cbranch_execnz loop
        0x887E_0A7E, // s_or_b32 exec_lo, exec_lo, s10
    ];
    match lower_uav_cs(&read_inside_loop, false) {
        Err(crate::ShaderError::Unsupported { what, .. }) => {
            assert_eq!(what, "a lane mask holding lanes elsewhere read as data");
        }
        Ok(_) => panic!("must refuse mask read inside divergent loop"),
        Err(e) => panic!("unexpected error: {e}"),
    }

    // 2. Reading the divergent-loop accumulated mask s10 under a narrowed
    // EXEC subset after the loop.
    let read_under_narrowed_exec = [
        0x4A02_0081, // v_add_nc_u32 v1, 1, v0
        0xBE8A_0380, // s_mov_b32 s10, 0
        0x4E02_0281, // loop: v_subrev_nc_u32 v1, 1, v1
        0x7D84_0280, // v_cmp_eq_u32 vcc_lo, 0, v1
        0x880A_0A6A, // s_or_b32 s10, vcc_lo, s10
        0x8A7E_0A7E, // s_andn2_b32 exec_lo, exec_lo, s10
        0xBF89_FFFB, // s_cbranch_execnz loop
        0x887E_0A7E, // s_or_b32 exec_lo, exec_lo, s10
        0x7E04_0280, // v_mov_b32 v2, 0
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0 (v0 < 16)
        0xBE8C_3C6A, // s_and_saveexec_b32 s12, vcc_lo
        0xBF88_0002, // s_cbranch_execz -> skip (+2 dwords)
        0xBE89_0F0A, // s_bcnt1_i32_b32 s9, s10 (under narrowed EXEC!)
        0x7E04_0209, // v_mov_b32 v2, s9
        0x887E_0C7E, // skip: s_or_b32 exec_lo, exec_lo, s12
    ];
    match lower_uav_cs(&read_under_narrowed_exec, false) {
        Err(crate::ShaderError::Unsupported { what, .. }) => {
            assert_eq!(what, "a lane mask holding lanes elsewhere read as data");
        }
        Ok(_) => panic!("must refuse mask read under narrowed EXEC"),
        Err(e) => panic!("unexpected error: {e}"),
    }

    // 3. Non-loop `s_not` mask in a narrowed branch now lowers cleanly via
    // whole-wave `mask_number` (upstream `8af0472`) rather than failing.
    let read_s_not_narrowed = [
        0x7E04_0280, // v_mov_b32 v2, 0
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0 (v0 < 16)
        0xBE8C_3C6A, // s_and_saveexec_b32 s12, vcc_lo
        0xBF88_0003, // s_cbranch_execz -> skip (+3 dwords)
        0xBE8A_076A, // s_not_b32 s10, vcc_lo (sets bits >= 16!)
        0xBE89_0F0A, // s_bcnt1_i32_b32 s9, s10
        0x7E04_0209, // v_mov_b32 v2, s9
        0x887E_0C7E, // skip: s_or_b32 exec_lo, exec_lo, s12
    ];
    let mut s_not_lowered =
        lower_uav_cs(&read_s_not_narrowed, false).expect("8af0472 lowers s_not mask");
    assert!(s_not_lowered.wave_ops);
    #[cfg(windows)]
    if let Ok(mut v) = crate::emit::validator::Validator::load(
        crate::emit::validator::Validator::default_path().as_deref(),
    ) {
        v.validate(&mut s_not_lowered.dxil)
            .expect("DXIL validation failed for s_not mask");
    }

    // 4. Loop inside `if (v0 < 16)` where s10 was initialized to 1 BEFORE
    // the `if` (holding bit 0, and could hold bits from outside), and read
    // inside `if (v0 < 16)` after the loop -> refuses.
    let pre_if_nonzero_init = [
        0xBE8A_03C1, // s_mov_b32 s10, -1
        0x7E04_0280, // v_mov_b32 v2, 0
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0 (v0 < 16)
        0xBE8C_3C6A, // s_and_saveexec_b32 s12, vcc_lo
        0xBF88_0009, // s_cbranch_execz -> after_if (+9 dwords)
        0x4A02_0081, // v_add_nc_u32 v1, 1, v0
        0x4E02_0281, // loop: v_subrev_nc_u32 v1, 1, v1
        0x7D84_0280, // v_cmp_eq_u32 vcc_lo, 0, v1
        0x880A_0A6A, // s_or_b32 s10, vcc_lo, s10
        0x8A7E_0A7E, // s_andn2_b32 exec_lo, exec_lo, s10
        0xBF89_FFFB, // s_cbranch_execnz loop
        0x887E_0A7E, // s_or_b32 exec_lo, exec_lo, s10
        0xBE89_0F0A, // s_bcnt1_i32_b32 s9, s10 (still inside v0 < 16!)
        0x7E04_0209, // v_mov_b32 v2, s9
        0x887E_0C7E, // after_if: s_or_b32 exec_lo, exec_lo, s12
    ];
    match lower_uav_cs(&pre_if_nonzero_init, false) {
        Err(crate::ShaderError::Unsupported { what, .. }) => {
            assert_eq!(what, "a lane mask holding lanes elsewhere read as data");
        }
        Ok(_) => panic!("must refuse mask holding -1 bits from outside narrowed if"),
        Err(e) => panic!("unexpected error: {e}"),
    }

    // 5. Fresh `v_cmp` (a `same_iter` mask) defined inside a divergent
    // loop and read after the loop reconverges: on hardware the SGPR holds
    // only the last iteration's mask, whereas a ballot in the loop gives
    // each lane its own exit iteration and a post-loop ballot sees stale
    // per-lane bits -> must refuse.
    let vcmp_in_div_loop_read_after = [
        0x4A02_0081, // v_add_nc_u32 v1, 1, v0
        0xBE8A_0380, // s_mov_b32 s10, 0
        0x4E02_0281, // loop: v_subrev_nc_u32 v1, 1, v1
        0x7D84_0280, // v_cmp_eq_u32 vcc_lo, 0, v1
        0x880A_0A6A, // s_or_b32 s10, vcc_lo, s10
        0x8A7E_0A7E, // s_andn2_b32 exec_lo, exec_lo, s10
        0xBF89_FFFB, // s_cbranch_execnz loop
        0x887E_0A7E, // s_or_b32 exec_lo, exec_lo, s10
        0xBE89_0F6A, // s_bcnt1_i32_b32 s9, vcc_lo (post-loop read of loop v_cmp!)
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    match lower_uav_cs(&vcmp_in_div_loop_read_after, false) {
        Err(crate::ShaderError::Unsupported { what, .. }) => {
            assert_eq!(what, "a lane mask holding lanes elsewhere read as data");
        }
        Ok(_) => panic!("must refuse divergent-loop v_cmp read after loop"),
        Err(e) => panic!("unexpected error: {e}"),
    }
}

#[test]
fn divergent_loop_with_inner_if_accumulated_mask_ballots_after_loop() {
    // Variant 2A: s12 is initialized to 0, updated inside a divergent `if`
    // inside a divergent loop, and counted after the loop reconverges.
    let body = [
        0x4A02_0081, // v_add_nc_u32 v1, 1, v0
        0xBE8A_0380, // s_mov_b32 s10, 0
        0xBE8C_0380, // s_mov_b32 s12, 0
        // loop (block 1):
        0x7D84_0282, // v_cmp_eq_u32 vcc_lo, 2, v1
        0x880C_0C6A, // s_or_b32 s12, vcc_lo, s12
        0xBE8E_3C6A, // s_and_saveexec_b32 s14, vcc_lo
        0xBF88_0001, // s_cbranch_execz -> merge (+1 dword)
        0x880C_0C7E, // s_or_b32 s12, exec_lo, s12
        0x887E_0E7E, // merge: s_or_b32 exec_lo, exec_lo, s14
        0x4E02_0281, // v_subrev_nc_u32 v1, 1, v1
        0x7D84_0280, // v_cmp_eq_u32 vcc_lo, 0, v1
        0x880A_0A6A, // s_or_b32 s10, vcc_lo, s10
        0x8A7E_0A7E, // s_andn2_b32 exec_lo, exec_lo, s10
        0xBF89_FFF5, // s_cbranch_execnz loop (-11 dwords)
        // after loop:
        0x887E_0A7E, // s_or_b32 exec_lo, exec_lo, s10
        0xBE89_0F0C, // s_bcnt1_i32_b32 s9, s12
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    let (p, st, mut lowered) = uav_cs_parts(&body, false).expect("inner-if loop mask lowers");
    assert!(lowered.wave_ops);
    // Post-loop block is block 4 (0=entry, 1=loop_hdr, 2=inner_then, 3=inner_merge, 4=post_loop).
    let post_masks: Vec<_> = p.blocks[4]
        .insts
        .iter()
        .filter(|i| i.name().starts_with("s_bcnt1"))
        .flat_map(|i| i.reads.iter().map(|r| r.1))
        .filter(|v| st.masks.contains_key(v))
        .collect();
    assert!(!post_masks.is_empty());
    for lo in post_masks {
        assert_eq!(st.ballot_site(lo, Some(1), 4), Some(structure::BallotSite::Read));
    }
    #[cfg(windows)]
    if let Ok(mut v) = crate::emit::validator::Validator::load(
        crate::emit::validator::Validator::default_path().as_deref(),
    ) {
        v.validate(&mut lowered.dxil)
            .expect("DXIL validation failed for inner-if loop mask");
    }
}
