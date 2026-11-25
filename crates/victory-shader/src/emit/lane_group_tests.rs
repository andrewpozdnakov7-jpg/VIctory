//! Compute programs reading other lanes only within aligned groups (quads,
//! DPP8, rows, halves) declare no wave size: run on the hardware adapter
//! and WARP whenever the host's waves hold whole groups, and compared lane
//! by lane with the console's DPP and `ds_swizzle_b32`.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::dxbc::{Container, FourCC};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const THREADS: u32 = 128;

/// One cross-lane read of v1 into `v[dst]`: its words, its SP3 text, and the
/// console lane each console lane reads (`None`: no write, `v[dst]` keeps
/// its old value).
struct Read {
    words: Vec<u32>,
    text: &'static str,
    lane: fn(u32) -> Option<u32>,
}

/// `v_mov_b32_dpp v2, v1` with a DPP16 control and write masks.
fn dpp16(ctrl: u32, row_mask: u32, bank_mask: u32) -> Vec<u32> {
    vec![
        0x7E04_02FA,
        1 | ctrl << 8 | bank_mask << 24 | row_mask << 28,
    ]
}

/// `ds_swizzle_b32 v2, v1` with `offset`.
fn swizzle(offset: u32) -> Vec<u32> {
    vec![0xD8D4_0000 | offset, 0x0200_0001]
}

/// Whether DPP16's write masks let console lane `c` write.
fn masked(c: u32, row_mask: u32, bank_mask: u32) -> bool {
    row_mask >> (c >> 4 & 3) & 1 != 0 && bank_mask >> (c >> 2 & 3) & 1 != 0
}

fn quad_perm() -> Read {
    // [2,3,1,0] in rows 1 and 2, banks 0, 1 and 3.
    Read {
        words: dpp16(0x1E, 0x6, 0xB),
        text: "v_mov_b32 v2, v1 quad_perm:[2,3,1,0] row_mask:0x6 bank_mask:0xb",
        lane: |c| masked(c, 0x6, 0xB).then_some(c & !3 | [2, 3, 1, 0][c as usize & 3]),
    }
}

fn quad_swizzle() -> Read {
    Read {
        words: swizzle(0x8063),
        text: "ds_swizzle_b32 v2, v1 offset:32867",
        lane: |c| Some(c & !3 | [3, 0, 2, 1][c as usize & 3]),
    }
}

fn row_shr() -> Read {
    // Lanes 0-2 of a row read outside it and keep their value.
    Read {
        words: dpp16(0x113, 0x5, 0xD),
        text: "v_mov_b32 v2, v1 row_shr:3 row_mask:0x5 bank_mask:0xd",
        lane: |c| (masked(c, 0x5, 0xD) && c & 15 >= 3).then(|| c - 3),
    }
}

fn row_mirror() -> Read {
    Read {
        words: dpp16(0x140, 0xF, 0xF),
        text: "v_mov_b32 v2, v1 row_mirror",
        lane: |c| Some(c ^ 15),
    }
}

fn dpp8() -> Read {
    Read {
        words: vec![0x7E04_02E9, 0x8F15_C501],
        text: "v_mov_b32 v2, v1 dpp8:[5,0,7,2,1,6,3,4]",
        lane: |c| Some(c & !7 | [5, 0, 7, 2, 1, 6, 3, 4][c as usize & 7]),
    }
}

fn row_swizzle() -> Read {
    // and 0x1f, or 0, xor 0xa: bits 1 and 3.
    Read {
        words: swizzle(0x281F),
        text: "ds_swizzle_b32 v2, v1 offset:10271",
        lane: |c| Some(c ^ 0xA),
    }
}

fn half_swizzle() -> Read {
    // and 0x1f, or 0, xor 0x10: the other row of a half.
    Read {
        words: swizzle(0x401F),
        text: "ds_swizzle_b32 v2, v1 offset:16415",
        lane: |c| Some(c ^ 0x10),
    }
}

/// v1 = 1000 + the thread id; read k of it goes to v[2 + k], stored at
/// dword k × THREADS + thread id. `extra` runs after the reads.
fn program(reads: &[&Read], extra: &[u32]) -> Vec<u32> {
    let mut w = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x4A02_00FF,
        1000, // v_add_nc_u32 v1, 1000, v0
    ];
    for (k, r) in reads.iter().enumerate() {
        let dst = 2 + k as u32;
        w.extend([0x7E00_02FF | dst << 17, 0xBEEF + k as u32]); // v_mov_b32 v[dst], old
        // The read, into v[dst] (the encodings name v2).
        let mut words = r.words.clone();
        match words[0] >> 26 {
            0x36 => words[1] = words[1] & 0x00FF_FFFF | dst << 24,
            _ => words[0] = words[0] & !(0xFF << 17) | dst << 17,
        }
        w.extend(words);
    }
    w.extend(extra);
    w.push(0xBF8C_C07F); // s_waitcnt lgkmcnt(0)
    for k in 0..reads.len() as u32 {
        // buffer_store_dword v[2 + k], v16, s[8:11], 0 offen offset:k × THREADS × 4
        w.extend([0xE070_1000 | (k * THREADS * 4), 0x8002_0010 | (2 + k) << 8]);
    }
    w.push(0xBF81_0000); // s_endpgm
    w
}

fn lowered(words: &[u32], wave64: bool) -> lower::Lowered {
    lowered_with_lds(words, wave64, 0)
}

fn lowered_with_lds(words: &[u32], wave64: bool, lds_blocks: u32) -> lower::Lowered {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1 | lds_blocks << 15,
            threads: [THREADS, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        view_marks: false,
    })
    .expect("lower")
}

fn build(reads: &[&Read], wave64: bool) -> HostShader {
    let words = program(reads, &[]);
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let texts: Vec<String> = crate::gfx10::decode(&code)
        .expect("decode")
        .iter()
        .map(|i| i.text(wave64))
        .collect();
    for (k, r) in reads.iter().enumerate() {
        let text = r.text.replacen("v2", &format!("v{}", 2 + k), 1);
        assert!(texts.contains(&text), "{text} not in {texts:?}");
    }
    let l = lowered(&words, wave64);
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        memory: l.memory,
        memory_writes: l.memory_writes,
        view_addresses: l.view_addresses,
        heap: l.heap,
        heap_writes: l.heap_writes,
        view_marks: l.view_marks,
        min_lanes: l.min_lanes,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    shader
}

/// The wave size PSV0 declares (min = max), if any.
fn wave_size(dxil: &[u8]) -> Option<u32> {
    let c = Container::parse(dxil).expect("container");
    let psv = c.part(FourCC::PSV0).expect("PSV0").data;
    let word = |at: usize| u32::from_le_bytes(psv[4 + at..8 + at].try_into().unwrap());
    match (word(16), word(20)) {
        (0, u32::MAX) => None,
        (min, max) => {
            assert_eq!(min, max);
            Some(min)
        }
    }
}

/// Runs `reads` in a wave64 or wave32 program on each host whose waves
/// hold its lane groups; the labels of those it ran on.
fn run(reads: &[&Read], wave64: bool) -> Vec<&'static str> {
    let shader = build(reads, wave64);
    let wave = if wave64 { 64 } else { 32 };
    assert_eq!(wave_size(&shader.dxil), None, "a wave size declared");
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = Vec::new();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (lanes, _) = host.wave_lanes().expect("wave lanes");
        if lanes < shader.min_lanes {
            eprintln!(
                "{label}: waves of {lanes} lanes split groups of {}; skipped",
                shader.min_lanes
            );
            continue;
        }
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: no pipeline: {e}"));
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(THREADS * reads.len() as u32)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        for (k, r) in reads.iter().enumerate() {
            for t in 0..THREADS {
                let c = t % wave;
                let want = match (r.lane)(c) {
                    Some(src) => 1000 + t - c + src,
                    None => 0xBEEF + k as u32,
                };
                let got = out[0][k * THREADS as usize + t as usize];
                assert_eq!(got, want, "{label}, wave{wave}, {}: thread {t}", r.text);
            }
        }
        ran.push(label);
    }
    ran
}

#[test]
fn quad_reads_run_on_any_host_wave() {
    for wave64 in [false, true] {
        let ran = run(&[&quad_perm(), &quad_swizzle()], wave64);
        // WARP's waves are 4 lanes: a quad each.
        assert!(ran.contains(&"warp"), "wave64 {wave64}: ran on {ran:?}");
    }
}

fn quad_xmask() -> Read {
    Read {
        words: dpp16(0x163, 0x6, 0xB),
        text: "v_mov_b32 v2, v1 row_xmask:3 row_mask:0x6 bank_mask:0xb",
        lane: |c| masked(c, 0x6, 0xB).then_some(c ^ 3),
    }
}

#[test]
fn quad_xmask_runs_on_four_lane_warp() {
    for wave64 in [false, true] {
        let ran = run(&[&quad_xmask()], wave64);
        assert!(ran.contains(&"warp"), "wave64 {wave64}: ran on {ran:?}");
    }
}

#[test]
fn row_reads_run_on_host_waves_of_16_lanes() {
    for wave64 in [false, true] {
        let ran = run(
            &[&row_shr(), &row_mirror(), &dpp8(), &row_swizzle()],
            wave64,
        );
        assert!(!ran.contains(&"warp"), "WARP ran rows on waves of 4");
    }
}

#[test]
fn half_reads_of_wave64_run_on_host_waves_of_32_lanes() {
    let ran = run(&[&half_swizzle(), &quad_perm()], true);
    assert!(!ran.contains(&"warp"), "WARP ran halves on waves of 4");
}

#[test]
fn lane_groups_set_the_fewest_host_lanes() {
    for (reads, wave64, lanes, size) in [
        (vec![quad_perm(), quad_swizzle()], false, 4, None),
        (vec![quad_xmask()], true, 4, None),
        (vec![dpp8()], true, 8, None),
        (vec![row_shr(), quad_perm()], false, 16, None),
        (vec![row_swizzle()], true, 16, None),
        (vec![half_swizzle()], true, 32, None),
        // Half of a wave of 32 is the wave.
        (vec![half_swizzle()], false, 0, Some(32)),
    ] {
        let reads: Vec<&Read> = reads.iter().collect();
        let l = lowered(&program(&reads, &[]), wave64);
        let texts: Vec<_> = reads.iter().map(|r| r.text).collect();
        assert_eq!(l.min_lanes, lanes, "{texts:?}");
        assert_eq!(wave_size(&l.dxil), size, "{texts:?}");
    }
}

#[test]
fn reads_of_the_whole_wave_keep_its_size() {
    // Each with a quad read, into v15 (not stored).
    let cases: [(&str, Vec<u32>); 6] = [
        (
            "v_readlane_b32",
            vec![0xD760_000F, 0x0001_0B01], // v_readlane_b32 s15, v1, 5
        ),
        (
            "v_mbcnt_lo_u32_b32",
            vec![0xD765_000F, 0x0001_01C1], // v_mbcnt_lo_u32_b32 v15, -1, 0
        ),
        (
            "v_readfirstlane_b32",
            vec![0x7E1E_0501], // v_readfirstlane_b32 s15, v1
        ),
        // A host wave whose lanes all left would never reach it.
        ("s_barrier", vec![0xBF8A_0000]),
        // GFX9's row_bcast:15, which GFX10 does not have.
        ("dpp_ctrl:0x142", vec![0x7E1E_02FA, 0xFF01_4201]),
        (
            // A whole-wave section: a scan step in every lane.
            "s_or_saveexec_b32",
            vec![
                0xBE94_3DC1, // s_or_saveexec_b32 s20, -1
                0x7E1E_02FA,
                0xFF01_1101, // v_mov_b32 v15, v1 row_shr:1
                0xBEFE_0314, // s_mov_b32 exec_lo, s20
            ],
        ),
    ];
    for (what, extra) in cases {
        let words = program(&[&quad_perm()], &extra);
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let texts: Vec<String> = crate::gfx10::decode(&code)
            .expect("decode")
            .iter()
            .map(|i| i.text(false))
            .collect();
        assert!(
            texts
                .iter()
                .any(|t| t.starts_with(what) || t.ends_with(what)),
            "{what} not in {texts:?}"
        );
        let l = lowered(&words, false);
        assert_eq!(l.min_lanes, 0, "{what}");
        assert_eq!(wave_size(&l.dxil), Some(32), "{what}");
    }
}

#[test]
fn lds_shared_within_the_wave_keeps_its_size() {
    // The console compiler drops barriers within one wave; narrower host
    // waves would race on it.
    let extra = [
        0xD834_0000,
        0x0000_0110, // ds_write_b32 v16, v1
        0xD8D8_0000,
        0x0F00_0010, // ds_read_b32 v15, v16
    ];
    let words = program(&[&quad_perm()], &extra);
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let texts: Vec<String> = crate::gfx10::decode(&code)
        .expect("decode")
        .iter()
        .map(|i| i.text(false))
        .collect();
    for what in ["ds_write_b32 v16, v1", "ds_read_b32 v15, v16"] {
        assert!(texts.iter().any(|t| t == what), "{what} not in {texts:?}");
    }
    for (wave64, size) in [(false, 32), (true, 64)] {
        let l = lowered_with_lds(&words, wave64, 1);
        assert_eq!(l.min_lanes, 0);
        assert_eq!(wave_size(&l.dxil), Some(size));
    }
}
