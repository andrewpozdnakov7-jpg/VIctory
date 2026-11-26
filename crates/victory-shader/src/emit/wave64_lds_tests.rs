use super::*;
use crate::io;
use crate::ir;
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::resource;
use crate::structure;

const S_ENDPGM: u32 = 0xBF81_0000;

fn lower_wave_cs(words: &[u32], threads: [u32; 3], lds_dwords: u32) -> Lowered {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let lds_granules = lds_dwords.div_ceil(128).max(1);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 256,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 4,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo {
            // lds_size field in COMPUTE_PGM_RSRC2 is bits [15:23] in 128-dword granules.
            rsrc2: lds_granules << 15 | 5 << 1,
            threads,
        }),
        code,
    };
    let p = ir::build(&native.code, true, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    lower(&Parts {
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
    })
    .unwrap()
}

#[test]
fn wave64_lds_within_32_threads_can_span_host_waves() {
    // A smaller workgroup may still span several host waves.
    let words = [
        0x7E00_0280, // v_mov_b32 v0, 0 (addr)
        0x7E02_0287, // v_mov_b32 v1, 7 (data)
        0xD834_0000, // ds_write_b32 v0, v1
        0x0000_0100, // v1 in src0
        0xD8D8_0000, // ds_read_b32 v2, v0
        0x0200_0000, // vdst = v2
        S_ENDPGM,
    ];
    let lowered = lower_wave_cs(&words, [32, 1, 1], 16);
    assert_eq!(lowered.wave_size, None);
    assert!(lowered.lds_wave32_sync);
}

#[test]
fn wave64_lds_64_threads_uniform_drops_wave_size_and_emits_barriers() {
    // Wave64 CS with 64 threads and uniform ds_write/ds_read uses DXIL barriers instead of WaveSize(64).
    let words = [
        0x7E00_0280, // v_mov_b32 v0, 0 (addr)
        0x7E02_0287, // v_mov_b32 v1, 7 (data)
        0xD834_0000, // ds_write_b32 v0, v1
        0x0000_0100, // v1 in src0
        0xD8D8_0000, // ds_read_b32 v2, v0
        0x0200_0000, // vdst = v2
        S_ENDPGM,
    ];
    let lowered = lower_wave_cs(&words, [64, 1, 1], 16);
    assert_eq!(lowered.wave_size, None);
    assert!(lowered.lds_wave32_sync);
}

#[test]
fn wave64_lds_64_threads_guarded_keeps_wave_size_64() {
    // Divergent EXEC-guarded LDS access cannot safely insert a group barrier; keep WaveSize(64).
    let words = [
        0x7E02_0287, // v_mov_b32 v1, 7
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0 (v0 = local thread id x)
        0xBE94_246A, // s_and_saveexec_b64 s[20:21], vcc
        0xD834_0000, // ds_write_b32 v0, v1
        0x0000_0100,
        0xBEFE_0414, // s_mov_b64 exec, s[20:21]
        S_ENDPGM,
    ];
    let lowered = lower_wave_cs(&words, [64, 1, 1], 16);
    assert_eq!(lowered.wave_size, Some(64));
    assert!(!lowered.lds_wave32_sync);
}

#[test]
fn wave64_lds_lane_divergent_branch_keeps_wave_size_64() {
    // The store is unguarded inside its block, but only some lanes reach it.
    let words = [
        0x7E02_0287, // v_mov_b32 v1, 7
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE94_246A, // s_and_saveexec_b64 s[20:21], vcc
        0xBF88_0002, // s_cbranch_execz past the store
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        0xBEFE_0414, // s_mov_b64 exec, s[20:21]
        S_ENDPGM,
    ];
    let lowered = lower_wave_cs(&words, [64, 1, 1], 16);
    assert_eq!(lowered.wave_size, Some(64));
    assert!(!lowered.lds_wave32_sync);
}

#[cfg(windows)]
#[test]
fn wave64_lds_exchange_orders_reads_before_reuse() {
    use super::super::{
        HostShader,
        host::{Dispatch, Host, Root, View},
        validator,
        valu_tests::ONE_AT_A_TIME,
    };

    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Some(host) = Host::new(false).or_else(|_| Host::new(true)).ok() else {
        eprintln!("no SM 6.6 device; skipped LDS readback");
        return;
    };
    let mut failures = Vec::new();
    for threads in [16u32, 32, 64] {
        // Store this lane number, read the opposite half, and then replace
        // this lane's slot. The next read must see the replacement.
        let half = threads / 2;
        let vop2 =
            |op: u32, dst: u32, src0: u32, src1: u32| op << 25 | dst << 17 | src1 << 9 | src0;
        let words = [
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000,                  // s_load_dwordx4 s[8:11], s[4:5], 0
            0xBF8C_0000,                  // s_waitcnt 0
            vop2(0x1A, 1, 130, 0),        // v1 = local_id * 4
            vop2(0x1D, 2, 128 + half, 0), // v2 = local_id xor half
            vop2(0x1A, 2, 130, 2),        // v2 *= 4
            0xD834_0000,
            0x0000_0001, // ds_write_b32 [v1], v0
            0xBF8C_0000,
            0xD8D8_0000,
            0x0300_0002, // ds_read_b32 v3, [v2]
            0xBF8C_0000,
            0x7E08_02BF, // v_mov_b32 v4, 63
            0xD834_0000,
            0x0000_0401, // ds_write_b32 [v1], v4
            0xBF8C_0000,
            0xD8D8_0000,
            0x0400_0002, // ds_read_b32 v4, [v2]
            0xBF8C_0000,
            0xE070_1000,
            0x8002_0301, // buffer_store_dword v3, [v1]
            0xE070_1100,
            0x8002_0401, // buffer_store_dword v4, [v1 + 256]
            S_ENDPGM,
        ];
        let lowered = lower_wave_cs(&words, [threads, 1, 1], threads);
        let mut shader = HostShader {
            stage: HwStage::Cs,
            dxil: lowered.dxil,
            user_data: lowered.user_data,
            samplers: lowered.samplers,
            static_samplers: Vec::new(),
            root_cbvs: lowered.root_cbvs,
            root_srvs: lowered.root_srvs,
            root_uavs: lowered.root_uavs,
            memory: lowered.memory,
            memory_writes: lowered.memory_writes,
            view_addresses: lowered.view_addresses,
            heap: lowered.heap,
            heap_writes: lowered.heap_writes,
            view_marks: lowered.view_marks,
        };
        if let Ok(mut validator) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            validator
                .validate(&mut shader.dxil)
                .expect("LDS exchange DXIL");
        }
        let (root, pipeline) = host
            .compute_pipeline(&shader)
            .expect("LDS exchange pipeline");
        let out = host
            .dispatch(
                &root,
                &pipeline,
                &Dispatch {
                    roots: vec![(
                        0,
                        Root::Constants(vec![0; shader.user_data_dwords() as usize]),
                    )],
                    views: &[View::RawUav(128)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .expect("LDS exchange readback");
        let before: Vec<_> = (0..threads).map(|lane| lane ^ half).collect();
        let n = threads as usize;
        println!(
            "LDS exchange threads={threads} old_matches={n_matches}/{n} new_matches={new_matches}/{n}",
            n_matches = out[0][..n]
                .iter()
                .zip(&before)
                .filter(|(a, b)| a == b)
                .count(),
            new_matches = out[0][64..64 + n].iter().filter(|&&v| v == 63).count()
        );
        if out[0][..n] != before || out[0][64..64 + n] != vec![63; n] {
            failures.push(threads);
        }
    }
    assert!(
        failures.is_empty(),
        "wrong LDS exchange readback for {failures:?}"
    );
}
