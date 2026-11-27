//! End-to-end RTX/D3D12 hardware tests of RDNA2 cross-lane instructions
//! (`DPP16`, `DPP8`/`DPP8FI`, and `ds_swizzle_b32`) under divergent `EXEC`
//! masks, verified against an exact CPU reference model of the AMD RDNA2
//! (GFX10.3) ISA.
//!
//! Covers all control-flow forms produced by [`crate::structure`]:
//! - straight-line `EXEC` narrowing (`v_cmpx_*` without a branch, lowered with
//!   a per-lane `guard`),
//! - branch-structurised divergence (`v_cmp_*` + `s_and_saveexec_b32` +
//!   `s_cbranch_execz`, lowered to a DXIL branch with `guard == None` inside
//!   the taken arm), and
//! - nested divergence (a DXIL branch further narrowed by a per-lane `guard`).

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;

use super::container::SigElement;
use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, container, emit, lower, validator};
use crate::dxbc::{Container, FourCC};
use crate::gfx10::{self, Dpp};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const EXEC_LO: u8 = 0x7E;
const VCC_LO: u8 = 0x6A;
const LITERAL: u32 = 0xFF;

const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

const fn s_and_saveexec_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x3C << 8 | src0 as u32
}

const fn s_or_b32(sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000 | (0x10u32 << 23) | (sdst as u32) << 16 | (src1 as u32) << 8 | src0 as u32
}

const fn s_cbranch_execz(simm16: i16) -> u32 {
    0xBF88_0000 | (simm16 as u16 as u32)
}

const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

const fn v_add_nc_u32_lit(vdst: u8, lit: u32, vsrc1: u8) -> [u32; 2] {
    [
        (0x25 << 25) | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | LITERAL,
        lit,
    ]
}

const fn v_and_b32_imm(vdst: u8, imm: u8, vsrc1: u8) -> u32 {
    (0x1B << 25) | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (0x80 + imm as u32)
}

const fn v_lshlrev_b32_imm(vdst: u8, imm: u8, vsrc1: u8) -> u32 {
    (0x1A << 25) | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (0x80 + imm as u32)
}

const fn v_cmp_eq_u32_imm(imm: u8, vsrc1: u8) -> u32 {
    0x7C00_0000 | (0xC2 << 17) | (vsrc1 as u32) << 9 | (0x80 + imm as u32)
}

const fn v_cmpx_eq_u32_imm(imm: u8, vsrc1: u8) -> u32 {
    0x7C00_0000 | (0xD2 << 17) | (vsrc1 as u32) << 9 | (0x80 + imm as u32)
}

const fn v_cmp_ne_u32_imm(imm: u8, vsrc1: u8) -> u32 {
    0x7C00_0000 | (0xC5 << 17) | (vsrc1 as u32) << 9 | (0x80 + imm as u32)
}

const fn v_cmpx_ne_u32_imm(imm: u8, vsrc1: u8) -> u32 {
    0x7C00_0000 | (0xD5 << 17) | (vsrc1 as u32) << 9 | (0x80 + imm as u32)
}

const fn dpp16_word(
    vsrc0: u8,
    ctrl: u16,
    row_mask: u8,
    bank_mask: u8,
    bound_ctrl: bool,
    fi: bool,
) -> u32 {
    ((row_mask as u32 & 0xF) << 28)
        | ((bank_mask as u32 & 0xF) << 24)
        | ((bound_ctrl as u32) << 19)
        | ((fi as u32) << 18)
        | ((ctrl as u32 & 0x1FF) << 8)
        | vsrc0 as u32
}

const fn dpp8_word(vsrc0: u8, sel: [u8; 8]) -> u32 {
    let mut s = 0u32;
    let mut i = 0;
    while i < 8 {
        s |= (sel[i] as u32 & 7) << (3 * i);
        i += 1;
    }
    (s << 8) | vsrc0 as u32
}

fn encode_dpp16(vdst: u8, vsrc0: u8, vsrc1: Option<u8>, dpp: Dpp) -> [u32; 2] {
    let Dpp::Row {
        ctrl,
        row_mask,
        bank_mask,
        bound_ctrl,
        fi,
    } = dpp
    else {
        panic!("expected Dpp::Row");
    };
    let w0 = match vsrc1 {
        None => 0x7E00_0000 | (vdst as u32) << 17 | (0x01 << 9) | 0xFA,
        Some(v1) => (0x25 << 25) | (vdst as u32) << 17 | (v1 as u32) << 9 | 0xFA,
    };
    [
        w0,
        dpp16_word(vsrc0, ctrl, row_mask, bank_mask, bound_ctrl, fi),
    ]
}

const fn v_mov_b32_dpp8(vdst: u8, vsrc0: u8, sel: [u8; 8], fi: bool) -> [u32; 2] {
    let tag = if fi { 0xEA } else { 0xE9 };
    [
        0x7E00_0000 | (vdst as u32) << 17 | (0x01 << 9) | tag,
        dpp8_word(vsrc0, sel),
    ]
}

const fn v_add_nc_u32_dpp8(vdst: u8, vsrc0: u8, vsrc1: u8, sel: [u8; 8], fi: bool) -> [u32; 2] {
    let tag = if fi { 0xEA } else { 0xE9 };
    [
        (0x25 << 25) | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | tag,
        dpp8_word(vsrc0, sel),
    ]
}

const fn ds_swizzle_b32(vdst: u8, vsrc: u8, offset: u16) -> [u32; 2] {
    [
        (0x36 << 26) | (0x35 << 18) | offset as u32,
        (vdst as u32) << 24 | vsrc as u32,
    ]
}

const fn buffer_store_dword_offen(vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
    [
        (0x38 << 26) | (0x1C << 18) | (1 << 12),
        (0x80 << 24) | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

#[derive(Clone, Copy, Debug)]
enum Divergence {
    /// Full `EXEC = 0xFFFF_FFFF` (no narrowing).
    Full,
    /// Straight-line `v_cmpx_*` narrowing (lowers with a per-lane `guard`).
    Guarded,
    /// `v_cmp_*` + `s_and_saveexec_b32` + `s_cbranch_execz` (lowers to a DXIL
    /// branch with `guard == None` inside the active arm).
    Branched,
    /// Branch on `(lane & 4) == 0` (`0x0F0F_0F0F`), then inside the branch
    /// narrow with `v_cmpx_ne_u32` on `(lane & 3) != 3` (`0x0707_0707`),
    /// exercising active, guard-inactive, branch-inactive, and row-OOB lanes
    /// in a single wave.
    NestedBranchAndGuard,
}

#[derive(Clone, Copy, Debug)]
enum MaskPattern {
    All,
    Even,
    ThreeOfFour,
    Nested0707,
}

impl MaskPattern {
    fn mask(self) -> u32 {
        match self {
            MaskPattern::All => 0xFFFF_FFFF,
            MaskPattern::Even => 0x5555_5555,
            MaskPattern::ThreeOfFour => 0x7777_7777,
            MaskPattern::Nested0707 => 0x0707_0707,
        }
    }
}

fn lane_v1(lane: u32) -> u32 {
    0x1000_0001u32.wrapping_add(lane)
}
fn lane_v2_old(lane: u32) -> u32 {
    0xDEAD_0000u32.wrapping_add(lane)
}
fn lane_v3_rhs(lane: u32) -> u32 {
    0x0001_0000u32.wrapping_add(lane)
}

/// Exact CPU reference model for AMD RDNA2 (GFX10.3) `DPP16` and `DPP8`/`DPP8FI`.
fn ref_dpp(dpp: Dpp, add_rhs: bool, exec: u32) -> [u32; 32] {
    let mut out = [0u32; 32];
    for lane in 0..32u32 {
        let old = lane_v2_old(lane);
        if (exec >> lane) & 1 == 0 {
            out[lane as usize] = old;
            continue;
        }
        let x = match dpp {
            Dpp::Lanes { sel, fi } => {
                let src = (lane & !7) | ((sel >> (3 * (lane & 7))) & 7);
                let src_active = (exec >> src) & 1 != 0;
                if src_active || fi { lane_v1(src) } else { 0 }
            }
            Dpp::Row {
                ctrl,
                row_mask,
                bank_mask,
                bound_ctrl,
                fi,
            } => {
                let row_ok = (row_mask >> (lane >> 4 & 3)) & 1 != 0;
                let bank_ok = (bank_mask >> (lane >> 2 & 3)) & 1 != 0;
                if !(row_ok && bank_ok) {
                    out[lane as usize] = old;
                    continue;
                }
                let row = lane & !15;
                let pos = lane & 15;
                let n = u32::from(ctrl & 15);
                let (src, inside) = match ctrl {
                    0..=0xFF => (
                        (lane & !3) | (u32::from(ctrl) >> (2 * (lane & 3)) & 3),
                        true,
                    ),
                    0x101..=0x10F => (lane.wrapping_add(n), pos + n < 16),
                    0x111..=0x11F => (lane.wrapping_sub(n), pos >= n),
                    0x121..=0x12F => (row | (pos.wrapping_sub(n) & 15), true),
                    0x140 => (lane ^ 15, true),
                    0x141 => (lane ^ 7, true),
                    0x150..=0x15F => (row | n, true),
                    0x160..=0x16F => (lane ^ n, true),
                    _ => panic!("unhandled ctrl {ctrl:#x}"),
                };
                let src_active = inside && ((exec >> src) & 1 != 0);
                if inside && (src_active || fi) {
                    lane_v1(src)
                } else if bound_ctrl {
                    0
                } else {
                    out[lane as usize] = old;
                    continue;
                }
            }
        };
        out[lane as usize] = if add_rhs {
            x.wrapping_add(lane_v3_rhs(lane))
        } else {
            x
        };
    }
    out
}

/// Exact CPU reference model for AMD RDNA2 `ds_swizzle_b32`.
fn ref_swizzle(offset: u16, exec: u32) -> [u32; 32] {
    let mut out = [0u32; 32];
    for lane in 0..32u32 {
        let old = lane_v2_old(lane);
        if (exec >> lane) & 1 == 0 {
            out[lane as usize] = old;
            continue;
        }
        let off = u32::from(offset);
        let src = if off & 0x8000 != 0 {
            let pick = (off >> (2 * (lane & 3))) & 3;
            (lane & !3) | pick
        } else {
            let and = (off & 0x1F) | !0x1F;
            let or = (off >> 5) & 0x1F;
            let xor = (off >> 10) & 0x1F;
            ((lane & and) | or) ^ xor
        };
        out[lane as usize] = if (exec >> src) & 1 != 0 {
            lane_v1(src)
        } else {
            0
        };
    }
    out
}

fn build_wave32_cs(pattern: MaskPattern, div: Divergence, op_words: &[u32]) -> HostShader {
    let mut w = Vec::new();
    w.push(s_mov_b32(1, 0x80)); // s1 = 0
    w.extend_from_slice(&s_load_dwordx4(4, 0));
    w.extend_from_slice(&v_add_nc_u32_lit(1, 0x1000_0001, 0));
    w.extend_from_slice(&v_add_nc_u32_lit(2, 0xDEAD_0000, 0));
    w.extend_from_slice(&v_add_nc_u32_lit(3, 0x0001_0000, 0));
    match (pattern, div) {
        (MaskPattern::All, Divergence::Full) => {
            w.extend_from_slice(op_words);
        }
        (MaskPattern::Even, Divergence::Guarded) => {
            w.push(v_and_b32_imm(4, 1, 0));
            w.push(s_mov_b32(12, EXEC_LO));
            w.push(v_cmpx_eq_u32_imm(0, 4));
            w.extend_from_slice(op_words);
            w.push(s_mov_b32(EXEC_LO, 12));
        }
        (MaskPattern::ThreeOfFour, Divergence::Guarded) => {
            w.push(v_and_b32_imm(4, 3, 0));
            w.push(s_mov_b32(12, EXEC_LO));
            w.push(v_cmpx_ne_u32_imm(3, 4));
            w.extend_from_slice(op_words);
            w.push(s_mov_b32(EXEC_LO, 12));
        }
        (MaskPattern::Even, Divergence::Branched) => {
            w.push(v_and_b32_imm(4, 1, 0));
            w.push(v_cmp_eq_u32_imm(0, 4));
            w.push(s_and_saveexec_b32(12, VCC_LO));
            w.push(s_cbranch_execz(op_words.len() as i16));
            w.extend_from_slice(op_words);
            w.push(s_or_b32(EXEC_LO, EXEC_LO, 12));
        }
        (MaskPattern::ThreeOfFour, Divergence::Branched) => {
            w.push(v_and_b32_imm(4, 3, 0));
            w.push(v_cmp_ne_u32_imm(3, 4));
            w.push(s_and_saveexec_b32(12, VCC_LO));
            w.push(s_cbranch_execz(op_words.len() as i16));
            w.extend_from_slice(op_words);
            w.push(s_or_b32(EXEC_LO, EXEC_LO, 12));
        }
        (MaskPattern::Nested0707, Divergence::NestedBranchAndGuard) => {
            // Outer branch: (lane & 4) == 0 -> 0x0F0F_0F0F.
            w.push(v_and_b32_imm(4, 4, 0));
            w.push(v_cmp_eq_u32_imm(0, 4));
            w.push(s_and_saveexec_b32(12, VCC_LO));
            // Body inside branch: 4 scalar/vector 1-dword words + op_words.
            let body_words = 4 + op_words.len() as i16;
            w.push(s_cbranch_execz(body_words));
            // Inner straight-line guard: (lane & 3) != 3 -> 0x0707_0707.
            w.push(v_and_b32_imm(6, 3, 0));
            w.push(s_mov_b32(13, EXEC_LO));
            w.push(v_cmpx_ne_u32_imm(3, 6));
            w.extend_from_slice(op_words);
            w.push(s_mov_b32(EXEC_LO, 13));
            // Restore outer EXEC.
            w.push(s_or_b32(EXEC_LO, EXEC_LO, 12));
        }
        other => panic!("unsupported (pattern, divergence) pair: {other:?}"),
    }
    w.push(v_lshlrev_b32_imm(5, 2, 0));
    w.extend_from_slice(&buffer_store_dword_offen(2, 5, 4));
    w.push(S_ENDPGM);

    let code: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
    let decoded = gfx10::decode(&code).expect("decode synthetic RDNA2 program");
    assert_eq!(decoded.last().and_then(|i| i.name()), Some("s_endpgm"));

    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
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
            rsrc2: 1 << 1,
            threads: [32, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
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
    .expect("lower");

    let shader = HostShader {
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

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

fn wave32_hardware_info(host: &Host) -> Option<String> {
    // SAFETY: plain D3D12/DXGI capability queries on a live device.
    unsafe {
        let mut opts1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        host.device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS1,
                (&raw mut opts1).cast(),
                size_of_val(&opts1) as u32,
            )
            .ok()?;
        if !opts1.WaveOps.as_bool() || opts1.WaveLaneCountMin > 32 || opts1.WaveLaneCountMax < 32 {
            return None;
        }
        let mut opts5 = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
        let _ = host.device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS5,
            (&raw mut opts5).cast(),
            size_of_val(&opts5) as u32,
        );
        let mut opts7 = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
        let _ = host.device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS7,
            (&raw mut opts7).cast(),
            size_of_val(&opts7) as u32,
        );
        let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL {
            HighestShaderModel: D3D_SHADER_MODEL_6_8,
        };
        let _ = host.device.CheckFeatureSupport(
            D3D12_FEATURE_SHADER_MODEL,
            (&raw mut sm).cast(),
            size_of_val(&sm) as u32,
        );
        let luid = host.device.GetAdapterLuid();
        let mut name = String::from("unknown");
        if let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            && let Ok(adapter) = factory.EnumAdapterByLuid::<IDXGIAdapter1>(luid)
            && let Ok(desc) = adapter.GetDesc1()
        {
            if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0
                || desc.DedicatedVideoMemory == 0
            {
                return None;
            }
            let end = desc
                .Description
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(desc.Description.len());
            name = String::from_utf16_lossy(&desc.Description[..end]);
        }
        Some(format!(
            "{name} | SM={:#x} | Wave={}-{} | RT={:?} | Mesh={:?}",
            sm.HighestShaderModel.0,
            opts1.WaveLaneCountMin,
            opts1.WaveLaneCountMax,
            opts5.RaytracingTier,
            opts7.MeshShaderTier
        ))
    }
}

fn dispatch_wave32(shader: &HostShader) -> Option<(String, [u32; 32])> {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; skipped");
        return None;
    };
    let Some(info) = wave32_hardware_info(&host) else {
        eprintln!("hardware device does not support 32-lane waves; skipped");
        return None;
    };
    let (root, pso) = host
        .compute_pipeline(shader)
        .unwrap_or_else(|e| panic!("compute_pipeline failed on {info}: {e}"));
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        assert_eq!(u.root, RootType::Table);
        ud[u.index as usize] = 0;
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(32)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed on {info}: {e}"));
    let mut got = [0u32; 32];
    got.copy_from_slice(&out[0][..32]);
    Some((info, got))
}

fn check_cases(suite: &str, items: &[(&str, HostShader, [u32; 32])]) {
    let mut failures = Vec::new();
    let mut passed = 0usize;
    let mut skipped = 0usize;
    for (label, shader, expected) in items {
        let Some((info, got)) = dispatch_wave32(shader) else {
            skipped += 1;
            continue;
        };
        let mut bad = Vec::new();
        for lane in 0..32 {
            if got[lane] != expected[lane] {
                bad.push((lane, got[lane], expected[lane]));
            }
        }
        if bad.is_empty() {
            passed += 1;
            eprintln!("[PASS] {label}");
        } else {
            let (l0, g0, e0) = bad[0];
            eprintln!(
                "[FAIL] {label} ({}/32 lanes wrong, first lane {l0}: got {g0:#010x} want {e0:#010x}) on {info}",
                bad.len()
            );
            failures.push(format!(
                "{label}: {}/32 lanes mismatched (lane {l0}: got {g0:#010x}, expected {e0:#010x})",
                bad.len()
            ));
        }
    }
    eprintln!(
        "[SUMMARY] {suite}: {passed} passed, {} failed, {skipped} skipped",
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "{suite} had {} failing GPU case(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn cpu_reference_oracle_boundary_checks() {
    // Bank mask disables writes to bank 1 (lanes 4..7), NOT EXEC of lane 4.
    let bm = ref_dpp(
        Dpp::Row {
            ctrl: 0x101,
            row_mask: 0xF,
            bank_mask: 0x5,
            bound_ctrl: false,
            fi: false,
        },
        false,
        0xFFFF_FFFF,
    );
    assert_eq!(
        bm[3],
        lane_v1(4),
        "lane 3 reads active lane 4 in masked bank 1"
    );
    assert_eq!(
        bm[4],
        lane_v2_old(4),
        "bank 1 destination preserves old vdst"
    );
    assert_eq!(
        bm[15],
        lane_v2_old(15),
        "row-OOB with bound_ctrl:0 preserves old vdst"
    );

    // DPP8 (fi:0) substitutes 0 for inactive source; DPP8FI (fi:1) reads it.
    let sel_bits = dpp8_word(0, [1, 0, 3, 2, 5, 4, 7, 6]) >> 8;
    let d8_0 = ref_dpp(
        Dpp::Lanes {
            sel: sel_bits,
            fi: false,
        },
        false,
        0x7777_7777,
    );
    let d8_1 = ref_dpp(
        Dpp::Lanes {
            sel: sel_bits,
            fi: true,
        },
        false,
        0x7777_7777,
    );
    assert_eq!(d8_0[2], 0);
    assert_eq!(d8_1[2], lane_v1(3));
    assert_eq!(d8_0[3], lane_v2_old(3));
    assert_eq!(d8_1[3], lane_v2_old(3));

    // ds_swizzle_b32 returns 0 for inactive source, keeps old vdst on inactive dest.
    let sw = ref_swizzle(0x041F, 0x7777_7777);
    assert_eq!(sw[0], lane_v1(1));
    assert_eq!(sw[2], 0);
    assert_eq!(sw[3], lane_v2_old(3));
}

struct Dpp16Case {
    label: &'static str,
    pattern: MaskPattern,
    div: Divergence,
    dpp: Dpp,
    add_rhs: bool,
}

#[test]
fn dpp16_inactive_source_and_bound_ctrl_match_rdna2_on_gpu() {
    let row = |ctrl, row_mask, bank_mask, bound_ctrl, fi| Dpp::Row {
        ctrl,
        row_mask,
        bank_mask,
        bound_ctrl,
        fi,
    };
    let specs = [
        Dpp16Case {
            label: "dpp16 row_shr:1 bc:1 fi:0 (guarded, 0x55555555)",
            pattern: MaskPattern::Even,
            div: Divergence::Guarded,
            dpp: row(0x111, 0xF, 0xF, true, false),
            add_rhs: true,
        },
        Dpp16Case {
            label: "dpp16 row_shr:1 bc:1 fi:0 (branched, 0x55555555)",
            pattern: MaskPattern::Even,
            div: Divergence::Branched,
            dpp: row(0x111, 0xF, 0xF, true, false),
            add_rhs: true,
        },
        Dpp16Case {
            label: "dpp16 row_shr:1 bc:0 fi:0 (guarded, 0x55555555)",
            pattern: MaskPattern::Even,
            div: Divergence::Guarded,
            dpp: row(0x111, 0xF, 0xF, false, false),
            add_rhs: true,
        },
        Dpp16Case {
            label: "dpp16 row_shr:1 bc:0 fi:0 (branched, 0x55555555)",
            pattern: MaskPattern::Even,
            div: Divergence::Branched,
            dpp: row(0x111, 0xF, 0xF, false, false),
            add_rhs: true,
        },
        Dpp16Case {
            label: "dpp16 row_shr:1 bc:0 fi:1 (guarded, 0x55555555)",
            pattern: MaskPattern::Even,
            div: Divergence::Guarded,
            dpp: row(0x111, 0xF, 0xF, false, true),
            add_rhs: true,
        },
        Dpp16Case {
            label: "dpp16 row_shr:1 bc:1 fi:1 (guarded, 0x55555555)",
            pattern: MaskPattern::Even,
            div: Divergence::Guarded,
            dpp: row(0x111, 0xF, 0xF, true, true),
            add_rhs: true,
        },
        Dpp16Case {
            label: "dpp16 quad_perm:[1,0,3,2] bc:0 fi:0 (guarded, 0x77777777)",
            pattern: MaskPattern::ThreeOfFour,
            div: Divergence::Guarded,
            dpp: row(0xB1, 0xF, 0xF, false, false),
            add_rhs: false,
        },
        Dpp16Case {
            label: "dpp16 quad_perm:[1,0,3,2] bc:1 fi:0 (branched, 0x77777777)",
            pattern: MaskPattern::ThreeOfFour,
            div: Divergence::Branched,
            dpp: row(0xB1, 0xF, 0xF, true, false),
            add_rhs: false,
        },
        Dpp16Case {
            label: "dpp16 quad_perm:[1,0,3,2] bc:0 fi:1 (guarded, 0x77777777)",
            pattern: MaskPattern::ThreeOfFour,
            div: Divergence::Guarded,
            dpp: row(0xB1, 0xF, 0xF, false, true),
            add_rhs: false,
        },
        Dpp16Case {
            label: "dpp16 row_shl:1 rm:0x1 bm:0x5 bc:1 fi:0 (guarded, 0x77777777)",
            pattern: MaskPattern::ThreeOfFour,
            div: Divergence::Guarded,
            dpp: row(0x101, 0x1, 0x5, true, false),
            add_rhs: true,
        },
        Dpp16Case {
            label: "dpp16 row_shl:1 rm:0xF bm:0x5 bc:0 fi:0 (full EXEC: bank_mask does not deactivate source)",
            pattern: MaskPattern::All,
            div: Divergence::Full,
            dpp: row(0x101, 0xF, 0x5, false, false),
            add_rhs: false,
        },
        Dpp16Case {
            label: "dpp16 row_ror:1 bc:0 fi:0 (guarded, 0x55555555)",
            pattern: MaskPattern::Even,
            div: Divergence::Guarded,
            dpp: row(0x121, 0xF, 0xF, false, false),
            add_rhs: false,
        },
        Dpp16Case {
            label: "dpp16 row_share:3 bc:1 fi:0 (guarded, 0x77777777)",
            pattern: MaskPattern::ThreeOfFour,
            div: Divergence::Guarded,
            dpp: row(0x153, 0xF, 0xF, true, false),
            add_rhs: false,
        },
        Dpp16Case {
            label: "dpp16 row_xmask:1 bc:0 fi:0 (branched, 0x55555555)",
            pattern: MaskPattern::Even,
            div: Divergence::Branched,
            dpp: row(0x161, 0xF, 0xF, false, false),
            add_rhs: false,
        },
        Dpp16Case {
            label: "dpp16 row_mirror bc:1 fi:0 (guarded, 0x55555555)",
            pattern: MaskPattern::Even,
            div: Divergence::Guarded,
            dpp: row(0x140, 0xF, 0xF, true, false),
            add_rhs: false,
        },
        Dpp16Case {
            label: "dpp16 row_half_mirror bc:0 fi:0 (guarded, 0x55555555)",
            pattern: MaskPattern::Even,
            div: Divergence::Guarded,
            dpp: row(0x141, 0xF, 0xF, false, false),
            add_rhs: false,
        },
        Dpp16Case {
            label: "dpp16 row_shr:1 bc:0 fi:0 (nested branch+guard, 0x07070707)",
            pattern: MaskPattern::Nested0707,
            div: Divergence::NestedBranchAndGuard,
            dpp: row(0x111, 0xF, 0xF, false, false),
            add_rhs: true,
        },
        Dpp16Case {
            label: "dpp16 row_shl:1 bc:1 fi:0 (nested branch+guard, 0x07070707)",
            pattern: MaskPattern::Nested0707,
            div: Divergence::NestedBranchAndGuard,
            dpp: row(0x101, 0xF, 0xF, true, false),
            add_rhs: true,
        },
    ];

    let items: Vec<_> = specs
        .iter()
        .map(|s| {
            let words = encode_dpp16(2, 1, s.add_rhs.then_some(3), s.dpp);
            (
                s.label,
                build_wave32_cs(s.pattern, s.div, &words),
                ref_dpp(s.dpp, s.add_rhs, s.pattern.mask()),
            )
        })
        .collect();
    check_cases("DPP16", &items);
}

struct Dpp8Case {
    label: &'static str,
    pattern: MaskPattern,
    div: Divergence,
    fi: bool,
    add_rhs: bool,
}

#[test]
fn dpp8_and_dpp8fi_inactive_source_match_rdna2_on_gpu() {
    let sel = [1, 0, 3, 2, 5, 4, 7, 6];
    let sel_bits = dpp8_word(0, sel) >> 8;
    let specs = [
        Dpp8Case {
            label: "dpp8 (0xE9, fi:0) mov (guarded, 0x77777777)",
            pattern: MaskPattern::ThreeOfFour,
            div: Divergence::Guarded,
            fi: false,
            add_rhs: false,
        },
        Dpp8Case {
            label: "dpp8 (0xE9, fi:0) add (branched, 0x77777777)",
            pattern: MaskPattern::ThreeOfFour,
            div: Divergence::Branched,
            fi: false,
            add_rhs: true,
        },
        Dpp8Case {
            label: "dpp8 (0xE9, fi:0) add (nested branch+guard, 0x07070707)",
            pattern: MaskPattern::Nested0707,
            div: Divergence::NestedBranchAndGuard,
            fi: false,
            add_rhs: true,
        },
        Dpp8Case {
            label: "dpp8fi (0xEA, fi:1) mov (guarded, 0x77777777)",
            pattern: MaskPattern::ThreeOfFour,
            div: Divergence::Guarded,
            fi: true,
            add_rhs: false,
        },
        Dpp8Case {
            label: "dpp8fi (0xEA, fi:1) add (guarded, 0x55555555)",
            pattern: MaskPattern::Even,
            div: Divergence::Guarded,
            fi: true,
            add_rhs: true,
        },
    ];

    let items: Vec<_> = specs
        .iter()
        .map(|s| {
            let words = if s.add_rhs {
                v_add_nc_u32_dpp8(2, 1, 3, sel, s.fi)
            } else {
                v_mov_b32_dpp8(2, 1, sel, s.fi)
            };
            (
                s.label,
                build_wave32_cs(s.pattern, s.div, &words),
                ref_dpp(
                    Dpp::Lanes {
                        sel: sel_bits,
                        fi: s.fi,
                    },
                    s.add_rhs,
                    s.pattern.mask(),
                ),
            )
        })
        .collect();
    check_cases("DPP8/DPP8FI", &items);
}

#[test]
fn ds_swizzle_b32_inactive_source_returns_zero_on_gpu() {
    let specs: &[(&str, MaskPattern, Divergence, u16)] = &[
        (
            "ds_swizzle_b32 bitmode SWAP(1) (guarded, 0x55555555)",
            MaskPattern::Even,
            Divergence::Guarded,
            0x041F,
        ),
        (
            "ds_swizzle_b32 bitmode SWAP(1) (branched, 0x55555555)",
            MaskPattern::Even,
            Divergence::Branched,
            0x041F,
        ),
        (
            "ds_swizzle_b32 bitmode SWAP(1) (guarded, 0x77777777)",
            MaskPattern::ThreeOfFour,
            Divergence::Guarded,
            0x041F,
        ),
        (
            "ds_swizzle_b32 quad_perm [3,2,1,0] (guarded, 0x77777777)",
            MaskPattern::ThreeOfFour,
            Divergence::Guarded,
            0x801B,
        ),
        (
            "ds_swizzle_b32 quad_perm [3,2,1,0] (branched, 0x77777777)",
            MaskPattern::ThreeOfFour,
            Divergence::Branched,
            0x801B,
        ),
        (
            "ds_swizzle_b32 bitmode SWAP(1) (nested branch+guard, 0x07070707)",
            MaskPattern::Nested0707,
            Divergence::NestedBranchAndGuard,
            0x041F,
        ),
    ];

    let items: Vec<_> = specs
        .iter()
        .map(|&(label, pattern, div, offset)| {
            (
                label,
                build_wave32_cs(pattern, div, &ds_swizzle_b32(2, 1, offset)),
                ref_swizzle(offset, pattern.mask()),
            )
        })
        .collect();
    check_cases("ds_swizzle_b32", &items);
}

#[test]
fn pixel_quad_swizzle_and_dpp_validate_and_create_pipeline() {
    // Pixel shader with both unguarded WQM quad_perm (must not depend on
    // waveActiveBallot) and guarded quad ds_swizzle_b32.
    let exp_mrt0 = [
        0xF800_180F, // exp mrt0 v1, v2, v2, v1 done vm (uncompressed)
        0x0102_0201,
    ];
    let mut w = Vec::new();
    // v1 = 1.0 (0x3F80_0000), v2 = 0.5 (0x3F00_0000)
    w.extend_from_slice(&[0x7E02_02FF, 0x3F80_0000]);
    w.extend_from_slice(&[0x7E04_02FF, 0x3F00_0000]);
    // Unguarded WQM quad_perm:[1,0,3,2] (ddx-style quad read).
    w.extend_from_slice(&encode_dpp16(
        1,
        1,
        None,
        Dpp::Row {
            ctrl: 0xB1,
            row_mask: 0xF,
            bank_mask: 0xF,
            bound_ctrl: true,
            fi: false,
        },
    ));
    // Guarded quad swizzle under narrowed EXEC.
    w.push(s_mov_b32(12, EXEC_LO));
    w.push(v_cmpx_eq_u32_imm(0, 0));
    w.extend_from_slice(&ds_swizzle_b32(2, 1, 0x80B1));
    w.push(s_mov_b32(EXEC_LO, 12));
    w.extend_from_slice(&exp_mrt0);
    w.push(S_ENDPGM);

    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 4,
        layout: vec![],
        used: vec![],
        io: StageIo::Ps {
            inputs: vec![],
            input_addr: 0,
            db_shader_control: 0,
            patches: vec![],
        },
        code: w.iter().flat_map(|x| x.to_le_bytes()).collect(),
    };
    let osg1 = container::signature(
        &[SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        }],
        false,
    );
    let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
    let c = Container::parse(&bytes).unwrap();
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let ps = emit(&c, &native, &p, &st, &io, &res).unwrap();

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut ps.dxil.clone())
            .expect("pixel shader DXIL validation failed");
    }

    // Verify graphics pipeline creation on both hardware RTX 5090D and WARP.
    let vs = super::host::stub_vs(&ps).expect("stub_vs");
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let _ = host
            .pipeline(&vs, &ps)
            .unwrap_or_else(|e| panic!("{label}: graphics pipeline creation failed: {e}"));
    }
}

// ---- the SCC of a mask operation read as data ----

const INLINE_0: u8 = 0x80;
const INLINE_1: u8 = 0x81;
const INLINE_2: u8 = 0x82;
const INLINE_5: u8 = 0x85;
const INLINE_64: u8 = 0xC0;
const INLINE_MINUS_1: u8 = 0xC1;

/// SOP2 `op sdst, src0, src1`.
const fn sop2(op: u32, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000 | op << 23 | (sdst as u32) << 16 | (src1 as u32) << 8 | src0 as u32
}

/// SOP1 `op sdst, src0`.
const fn sop1(op: u32, sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | op << 8 | src0 as u32
}

/// SOPC `op src0, src1`.
const fn sopc(op: u32, src0: u8, src1: u8) -> u32 {
    0xBF00_0000 | op << 16 | (src1 as u32) << 8 | src0 as u32
}

/// VOPC `op vcc, src0, vsrc1` (`src0` an SGPR or an inline constant).
const fn vopc(op: u32, src0: u8, vsrc1: u8) -> u32 {
    0x7C00_0000 | op << 17 | (vsrc1 as u32) << 9 | src0 as u32
}

/// `v_mov_b32 vdst, s<src0>`.
const fn v_mov_b32_s(vdst: u8, src0: u8) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | 0x01 << 9 | src0 as u32
}

const S_ADD_U32: u32 = 0x00;
const S_CSELECT_B32: u32 = 0x0A;
const S_AND_B32: u32 = 0x0E;
const S_AND_B64: u32 = 0x0F;
const S_ANDN2_B32: u32 = 0x14;
const S_ORN2_B32: u32 = 0x16;
const S_MOV_B32: u32 = 0x03;
const S_NOT_B32: u32 = 0x07;
const S_CMP_GE_U32: u32 = 0x09;
const V_CMP_LT_U32: u32 = 0xC1;
const V_CMP_GT_U32: u32 = 0xC4;

/// `cmp` into VCC, then the mask operation `op` (into s20), whose SCC
/// `s_cselect_b32 s21, 1, 2` takes; v2 = s21.
fn mask_scc_selected(cmp: u32, op: u32) -> Vec<u32> {
    vec![
        cmp,
        op,
        sop2(S_CSELECT_B32, 21, INLINE_1, INLINE_2),
        v_mov_b32_s(2, 21),
    ]
}

/// v2 of the lanes in `exec`: 1 if `scc`, else 2; the others keep theirs.
fn selected(scc: bool, exec: u32) -> [u32; 32] {
    std::array::from_fn(|l| match (exec >> l & 1 != 0, scc) {
        (false, _) => lane_v2_old(l as u32),
        (true, true) => 1,
        (true, false) => 2,
    })
}

/// The SCC of a bitwise operation that writes a lane mask says whether the
/// whole mask, as a number, is not zero: some lane is set (the lanes
/// outside EXEC too, which a complement sets), the same in every lane of
/// the wave. An `s_cselect` of it (the console's FidelityFX passes poll in
/// such loops) was refused as "a lane mask read as data".
#[test]
fn mask_operation_scc_read_as_data_is_the_waves_any_on_gpu() {
    let (exec, vcc) = (EXEC_LO, VCC_LO);
    // vcc = the lanes below 5: some.
    let some = mask_scc_selected(
        vopc(V_CMP_GT_U32, INLINE_5, 0),
        sop2(S_AND_B32, 20, vcc, exec),
    );
    // vcc = the lanes below 0: none.
    let none = mask_scc_selected(
        vopc(V_CMP_GT_U32, INLINE_0, 0),
        sop2(S_AND_B32, 20, vcc, exec),
    );
    // The even lanes run and all of them set vcc (lanes below 64), so its
    // complement holds only odd lanes, which do not run: still not zero.
    let complement = mask_scc_selected(vopc(V_CMP_GT_U32, INLINE_64, 0), sop1(S_NOT_B32, 20, vcc));
    // EXEC without vcc is zero.
    let none_left = mask_scc_selected(
        vopc(V_CMP_GT_U32, INLINE_64, 0),
        sop2(S_ANDN2_B32, 20, exec, vcc),
    );
    // A wait loop of the console compiler's form: s22 counts up until some
    // lane has lane + 5 < s22, or s22 reaches 64; then v2 = s22. Lane 0
    // ends it at 6 for the whole wave.
    let body = [
        sop2(S_ADD_U32, 22, 22, INLINE_1),
        vopc(V_CMP_GT_U32, 22, 4), // vcc = s22 > v4
        sop2(S_AND_B32, 20, vcc, exec),
        sop2(S_CSELECT_B32, 21, INLINE_0, INLINE_MINUS_1), // s21 = no lane set
        sopc(S_CMP_GE_U32, 22, INLINE_64),
        sop2(S_CSELECT_B32, 23, INLINE_MINUS_1, INLINE_0), // s23 = s22 >= 64
        sop2(S_ORN2_B32, 23, 23, 21),                      // SCC = s23 | ~s21 != 0
    ];
    let back = -(body.len() as i16 + 1);
    let mut poll = vec![
        0x25 << 25 | 4 << 17 | u32::from(INLINE_5), // v_add_nc_u32 v4, 5, v0
        sop1(S_MOV_B32, 22, INLINE_0),
    ];
    poll.extend(body);
    poll.push(0xBF84_0000 | back as u16 as u32); // s_cbranch_scc0 to the add
    poll.push(v_mov_b32_s(2, 22));
    let (all, even) = (MaskPattern::All.mask(), MaskPattern::Even.mask());
    let items = [
        (
            "s_and_b32 of a mask with lanes set",
            build_wave32_cs(MaskPattern::All, Divergence::Full, &some),
            selected(true, all),
        ),
        (
            "s_and_b32 of an empty mask",
            build_wave32_cs(MaskPattern::All, Divergence::Full, &none),
            selected(false, all),
        ),
        (
            "s_not_b32 of a mask set in every running lane (branched, 0x55555555)",
            build_wave32_cs(MaskPattern::Even, Divergence::Branched, &complement),
            selected(true, even),
        ),
        (
            "s_andn2_b32 of EXEC and a mask holding it (branched, 0x55555555)",
            build_wave32_cs(MaskPattern::Even, Divergence::Branched, &none_left),
            selected(false, even),
        ),
        (
            "a loop polling whether some lane compares true",
            build_wave32_cs(MaskPattern::All, Divergence::Full, &poll),
            [6; 32],
        ),
    ];
    check_cases("mask operation SCC", &items);
}

/// The same in a wave64 program: the lanes above 40 compare true, all in
/// the mask's high dword, so its SCC needs both dwords. Run where the host
/// has 64-lane waves, else lowered and validated only.
#[test]
fn wave64_mask_operation_scc_reads_both_dwords() {
    for (above, want) in [(40u8, 1u32), (63, 2)] {
        let mut w = vec![sop1(S_MOV_B32, 1, INLINE_0)];
        w.extend_from_slice(&s_load_dwordx4(4, 0));
        w.extend([
            vopc(V_CMP_LT_U32, INLINE_0 + above, 0), // vcc = above < lane
            sop2(S_AND_B64, 20, VCC_LO, EXEC_LO),    // s_and_b64 s[20:21], vcc, exec
            sop2(S_CSELECT_B32, 22, INLINE_1, INLINE_2),
            v_mov_b32_s(2, 22),
            v_lshlrev_b32_imm(5, 2, 0),
        ]);
        w.extend_from_slice(&buffer_store_dword_offen(2, 5, 4));
        w.push(S_ENDPGM);
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 8,
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
                rsrc2: 1 << 1,
                threads: [64, 1, 1],
            }),
            code: w.iter().flat_map(|x| x.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let st = structure::structure(&p).expect("structure");
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
        let io = io::raise(&p, &native).expect("io::raise");
        let lowered = lower::lower(&lower::Parts {
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
        .expect("lower");
        let shader = HostShader {
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
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            v.validate(&mut shader.dxil.clone())
                .expect("DXIL validation failed");
        }
        let _one = super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let Ok(host) = Host::new(false) else {
            eprintln!("no SM 6.6 hardware device; skipped");
            continue;
        };
        let mut opts1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        // SAFETY: a capability query on a live device, into a structure of
        // the size given.
        let queried = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS1,
                (&raw mut opts1).cast(),
                size_of_val(&opts1) as u32,
            )
        };
        if queried.is_err() || !opts1.WaveOps.as_bool() || opts1.WaveLaneCountMax < 64 {
            eprintln!("hardware device runs no 64-lane waves; lowered and validated only");
            continue;
        }
        let (root, pso) = host.compute_pipeline(&shader).expect("compute_pipeline");
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            assert_eq!(u.root, RootType::Table);
            ud[u.index as usize] = 0;
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(64)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .expect("dispatch");
        assert_eq!(out[0][..64], [want; 64], "lanes above {above}");
    }
}

// ---- one lane of a whole-wave section's value, read after the section ----

/// The OR reduction of `1 << lane` the console compiler writes with DPP
/// steps over each row, then adds row 0's result (lane 0) into every lane,
/// so that only row 1 holds the whole wave's: read at lane 31 after EXEC
/// is restored. Run under a branch on three of four lanes, so the
/// reduction covers only those (0x77777777); with a row dropped it would
/// be half of that, with the inactive lanes counted 0xFFFFFFFF.
#[test]
fn whole_wave_section_read_at_one_lane_after_it_on_gpu() {
    let dpp_or = |ctrl: u16| {
        [
            0x1C << 25 | 7 << 17 | 7 << 9 | 0xFA, // v_or_b32 v7, v7, v7 (DPP16)
            dpp16_word(7, ctrl, 0xF, 0xF, false, false),
        ]
    };
    let mut w = vec![
        0x7E00_0000 | 6 << 17 | 0x01 << 9 | 0x81,  // v_mov_b32 v6, 1
        0x1A << 25 | 6 << 17 | 6 << 9 | 256,       // v_lshlrev_b32 v6, v0, v6
        0xBE80_0000 | 14 << 16 | 0x3D << 8 | 0xC1, // s_or_saveexec_b32 s14, -1
        0xD501_0007,                               // v_cndmask_b32 v7, 0, v6, s14
        0x80 | (256 + 6) << 9 | 14 << 18,
    ];
    // quad_perm [1,0,3,2], [2,3,0,1], row_half_mirror, row_mirror
    for ctrl in [0x0B1, 0x04E, 0x141, 0x140] {
        w.extend(dpp_or(ctrl));
    }
    w.extend([
        0xD760_000F, // v_readlane_b32 s15, v7, 0
        (256 + 7) | 0x80 << 9,
        0x1C << 25 | 7 << 17 | 7 << 9 | 15, // v_or_b32 v7, s15, v7
        s_mov_b32(EXEC_LO, 14),
        0xD760_000F, // v_readlane_b32 s15, v7, 31
        (256 + 7) | (0x80 + 31) << 9,
        0x7E00_0000 | 2 << 17 | 0x01 << 9 | 15, // v_mov_b32 v2, s15
    ]);
    let exec = MaskPattern::ThreeOfFour.mask();
    let want: [u32; 32] = std::array::from_fn(|l| {
        if exec >> l & 1 != 0 {
            exec
        } else {
            lane_v2_old(l as u32)
        }
    });
    let items = [(
        "an OR reduction read at lane 31 (branched, 0x77777777)",
        build_wave32_cs(MaskPattern::ThreeOfFour, Divergence::Branched, &w),
        want,
    )];
    check_cases("section lane read", &items);
}

/// The wave64 form: each half reduced with DPP and `v_permlanex16_b32`,
/// then the low half's result (lane 0) added into every lane, so that
/// only the high half holds the whole wave's, read at lane 63 after the
/// section. The lanes with `lane & 2 == 0` run it on `1 << (lane >> 1)`:
/// 0x55555555. Run where the host has 64-lane waves, else lowered and
/// validated only.
#[test]
fn wave64_section_read_at_lane_63_after_it() {
    check_wave64_section_read_at_lane_63(false);
}

// The first true comparison is in lane 63 at iteration 2. The loop's
// mask SCC has to see the high dword, then the section must read lane 63
// after EXEC is restored. Both results reach the same GPU output.
#[test]
fn wave64_mask_poll_before_section_lane_read() {
    check_wave64_section_read_at_lane_63(true);
}

fn check_wave64_section_read_at_lane_63(poll: bool) {
    let dpp_or = |ctrl: u16| {
        [
            0x1C << 25 | 7 << 17 | 7 << 9 | 0xFA,
            dpp16_word(7, ctrl, 0xF, 0xF, false, false),
        ]
    };
    let mut body = vec![
        0xBE80_0000 | 14 << 16 | 0x25 << 8 | 0xC1, // s_or_saveexec_b64 s[14:15], -1
        0xD501_0007,                               // v_cndmask_b32 v7, 0, v6, s[14:15]
        0x80 | (256 + 6) << 9 | 14 << 18,
    ];
    for ctrl in [0x0B1, 0x04E, 0x141, 0x140] {
        body.extend(dpp_or(ctrl));
    }
    body.extend([
        0xD778_0003, // v_permlanex16_b32 v3, v7, 0, 0
        (256 + 7) | 0x80 << 9 | 0x80 << 18,
        0x1C << 25 | 7 << 17 | 7 << 9 | (256 + 3), // v_or_b32 v7, v3, v7
        0xD760_0010,                               // v_readlane_b32 s16, v7, 0
        (256 + 7) | 0x80 << 9,
        0x1C << 25 | 7 << 17 | 7 << 9 | 16, // v_or_b32 v7, s16, v7
        0xBE80_0000 | u32::from(EXEC_LO) << 16 | 0x04 << 8 | 14, // s_mov_b64 exec, s[14:15]
        0xD760_0010,                        // v_readlane_b32 s16, v7, 63
        (256 + 7) | (0x80 + 63) << 9,
        0x7E00_0000 | 2 << 17 | 0x01 << 9 | 16, // v_mov_b32 v2, s16
    ]);
    if poll {
        body.push(0x25 << 25 | 2 << 17 | 2 << 9 | 22); // v_add_nc_u32 v2, s22, v2
    }
    let mut w = vec![s_mov_b32(1, 0x80)];
    w.extend_from_slice(&s_load_dwordx4(4, 0));
    if poll {
        w.extend([
            0x26 << 25 | 4 << 17 | u32::from(INLINE_64), // v_sub_nc_u32 v4, 64, v0
            sop1(S_MOV_B32, 22, INLINE_0),
        ]);
        let iteration = [
            sop2(S_ADD_U32, 22, 22, INLINE_1),
            vopc(V_CMP_GT_U32, 22, 4),
            sop2(S_AND_B64, 20, VCC_LO, EXEC_LO),
            sop2(S_CSELECT_B32, 24, INLINE_0, INLINE_MINUS_1),
            sopc(S_CMP_GE_U32, 22, INLINE_64),
            sop2(S_CSELECT_B32, 23, INLINE_MINUS_1, INLINE_0),
            sop2(S_ORN2_B32, 23, 23, 24),
        ];
        w.extend(iteration);
        let back = -(iteration.len() as i16 + 1);
        w.push(0xBF84_0000 | u32::from(back as u16)); // s_cbranch_scc0
    }
    w.extend([
        0x7E00_0000 | 2 << 17 | 0x01 << 9 | 0x87,  // v_mov_b32 v2, 7
        0x7E00_0000 | 6 << 17 | 0x01 << 9 | 0x81,  // v_mov_b32 v6, 1
        0x16 << 25 | 5 << 17 | 0x81,               // v_lshrrev_b32 v5, 1, v0
        0x1A << 25 | 6 << 17 | 6 << 9 | (256 + 5), // v_lshlrev_b32 v6, v5, v6
        v_and_b32_imm(4, 2, 0),
        v_cmp_eq_u32_imm(0, 4),
        0xBE80_0000 | 12 << 16 | 0x24 << 8 | u32::from(VCC_LO), // s_and_saveexec_b64 s[12:13], vcc
        s_cbranch_execz(body.len() as i16),
    ]);
    w.extend(body);
    w.extend([
        0x8000_0000 | 0x11 << 23 | u32::from(EXEC_LO) << 16 | 12 << 8 | u32::from(EXEC_LO), // s_or_b64 exec, exec, s[12:13]
        v_lshlrev_b32_imm(5, 2, 0),
    ]);
    w.extend_from_slice(&buffer_store_dword_offen(2, 5, 4));
    w.push(S_ENDPGM);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 8,
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
            rsrc2: 1 << 1,
            threads: [64, 1, 1],
        }),
        code: w.iter().flat_map(|x| x.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
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
    .expect("lower");
    let shader = HostShader {
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
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; skipped");
        return;
    };
    let mut opts1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    // SAFETY: a capability query on a live device, into a structure of the
    // size given.
    let queried = unsafe {
        host.device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS1,
            (&raw mut opts1).cast(),
            size_of_val(&opts1) as u32,
        )
    };
    if queried.is_err() || !opts1.WaveOps.as_bool() || opts1.WaveLaneCountMax < 64 {
        eprintln!("hardware device runs no 64-lane waves; lowered and validated only");
        return;
    }
    let (root, pso) = host.compute_pipeline(&shader).expect("compute_pipeline");
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        assert_eq!(u.root, RootType::Table);
        ud[u.index as usize] = 0;
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(64)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .expect("dispatch");
    let reduction = 0x5555_5555 + if poll { 2 } else { 0 };
    let want: Vec<u32> = (0..64)
        .map(|l| if l & 2 == 0 { reduction } else { 7 })
        .collect();
    assert_eq!(out[0][..64], want[..]);
    if poll {
        println!("Wave64 poll and section readback matched all 64 lanes");
    }
}
