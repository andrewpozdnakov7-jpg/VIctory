//! Pixel programs that interpolate, read I and J as values, or read P0 and
//! P10, drawn on the host adapter and WARP and compared with the weights of
//! the triangle's vertices at each pixel centre. The vertex order matters:
//! the console's P0 is the provoking vertex (the first), I and J weigh
//! vertices 1 and 2, and the host's `SV_Barycentrics` and
//! `attributeAtVertex` must agree with that order.

use super::container::{SigElement, signature};
use super::host::{Draw, Host};
use super::{HostShader, emit};
use crate::native::{HwStage, NativeProgram, PsInput, StageIo, Usage, VsInput, VsOutput};
use crate::{Container, FourCC, io, ir, resource, structure};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_FEATURE_D3D12_OPTIONS3, D3D12_FEATURE_DATA_D3D12_OPTIONS3,
};

const S_ENDPGM: u32 = 0xBF81_0000;
/// `SPI_PS_INPUT_ADDR` PERSP_CENTER: I and J in v0 and v1.
const PERSP_CENTER: u32 = 1 << 1;

/// `exp mrt0, v<a>, v<b>, v<c>, v<d> done vm`.
fn exp_mrt0(r: [u32; 4]) -> [u32; 2] {
    [0xF800_180F, r[0] | r[1] << 8 | r[2] << 16 | r[3] << 24]
}

/// VINTRP: `v_interp_p1_f32` (op 0), `v_interp_p2_f32` (1) or
/// `v_interp_mov_f32` (2, `vsrc` 0 P10, 1 P20, 2 P0) of attr0.
fn vintrp(op: u32, vdst: u32, chan: u32, vsrc: u32) -> u32 {
    0xC800_0000 | vdst << 18 | op << 16 | chan << 8 | vsrc
}

/// `v_mov_b32 v7, 1.0`: the coverage marker in alpha.
const ONE_V7: u32 = 0x7E0E_02F2;

/// TEXCOORD0 of the three vertices: x and y, distinct per vertex.
const ATTR: [[f32; 2]; 3] = [[1.0, 10.0], [2.0, 30.0], [4.0, 70.0]];
/// The vertices in clip space (w 1): a triangle in general position.
const POS: [[f32; 2]; 3] = [[-0.8, -0.9], [-0.6, 0.9], [0.9, -0.5]];

fn sig(name: &str, kind: u8, row: u32, interp: u8) -> SigElement {
    SigElement {
        name: name.into(),
        index: 0,
        kind,
        comp: 9,
        interp,
        row: Some(row),
        col: 0,
        cols: 4,
        usage: 0xF,
        half: false,
    }
}

fn translate(native: &NativeProgram, inputs: &[SigElement], outputs: &[SigElement]) -> HostShader {
    let isg = signature(inputs, true);
    let osg = signature(outputs, false);
    let bytes = crate::dxbc::build(&[(FourCC::ISG1, &isg), (FourCC::OSG1, &osg)]);
    let c = Container::parse(&bytes).expect("container");
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).expect("ir");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource");
    let io = io::raise(&p, native).expect("io");
    let h = emit(&c, native, &p, &st, &io, &res).expect("emit");
    if let Ok(mut val) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        val.validate(&mut h.dxil.clone())
            .expect("DXIL validation failed");
    }
    h
}

/// Weights of vertices 0, 1, 2 at the centre of pixel (x, y) of a 16x16
/// target (linear: every w is 1).
fn weights(x: u32, y: u32) -> [f32; 3] {
    let s = |p: [f32; 2]| [(p[0] + 1.0) * 8.0, (1.0 - p[1]) * 8.0];
    let [a, b, c] = POS.map(s);
    let p = [x as f32 + 0.5, y as f32 + 0.5];
    let area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
    let l1 = ((p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])) / area;
    let l2 = ((b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1])) / area;
    [1.0 - l1 - l2, l1, l2]
}

fn attr(ch: usize, w: [f32; 3]) -> f32 {
    (0..3).map(|k| w[k] * ATTR[k][ch]).sum()
}

type Want = fn([f32; 3]) -> [f32; 2];

/// Each case: a program leaving its two results in v2 and v4, and what
/// they should be from the pixel's weights.
fn cases() -> Vec<(&'static str, Vec<u32>, Want)> {
    let (p1x, p2x) = (vintrp(0, 2, 0, 0), vintrp(1, 2, 0, 1));
    let (p1y, p2y) = (vintrp(0, 4, 1, 0), vintrp(1, 4, 1, 1));
    vec![
        ("x and y interpolated", vec![p1x, p2x, p1y, p2y], |w| {
            [attr(0, w), attr(1, w)]
        }),
        (
            "I and J as values",
            vec![0x7E04_0300, 0x7E08_0301], // v_mov_b32 v2, v0; v_mov_b32 v4, v1
            |w| [w[1], w[2]],
        ),
        (
            "x with a computed I (half of it)",
            vec![0x1006_00F0, vintrp(0, 2, 0, 3), p2x, 0x7E08_0301], // v_mul_f32 v3, 0.5, v0
            |w| {
                let x = ATTR[0][0]
                    + 0.5 * w[1] * (ATTR[1][0] - ATTR[0][0])
                    + w[2] * (ATTR[2][0] - ATTR[0][0]);
                [x, w[2]]
            },
        ),
        (
            "x interpolated, P0 of y (NEW-120)",
            vec![p1x, p2x, vintrp(2, 4, 1, 2)],
            |w| [attr(0, w), ATTR[0][1]],
        ),
        (
            "x interpolated, P10 of x",
            vec![p1x, p2x, vintrp(2, 4, 0, 0)],
            |w| [attr(0, w), ATTR[1][0] - ATTR[0][0]],
        ),
        (
            "P0 of x and of y alone",
            vec![vintrp(2, 2, 0, 2), vintrp(2, 4, 1, 2)],
            |_| [ATTR[0][0], ATTR[0][1]],
        ),
    ]
}

/// Interpolation, barycentrics read as values and P0/P10 beside an
/// interpolated row give, at every covered pixel, what the console's
/// vertex order gives: P0 the provoking (first) vertex, I and J the
/// weights of vertices 1 and 2.
#[test]
fn barycentrics_and_p0_follow_the_vertex_order() {
    // Vertex shader: POSITION in v4-v7, TEXCOORD0 in v8-v11, passed on.
    let vs_code = [
        0xF800_020F, // exp param0, v8, v9, v10, v11
        0x0B0A_0908,
        0xF800_08CF, // exp pos0, v4, v5, v6, v7 done
        0x0706_0504,
        S_ENDPGM,
    ];
    let vs = translate(
        &NativeProgram {
            stage: HwStage::Vs,
            wave64: true,
            vgprs: 12,
            layout: Vec::new(),
            used: Vec::new(),
            io: StageIo::Vs {
                inputs: vec![
                    VsInput {
                        usage: Usage::Generic,
                        row: 0,
                        mask: 0xF,
                        vgpr: 4,
                    },
                    VsInput {
                        usage: Usage::Generic,
                        row: 1,
                        mask: 0xF,
                        vgpr: 8,
                    },
                ],
                outputs: vec![VsOutput {
                    usage: Usage::Generic,
                    row: 0,
                    param: 0,
                    fp16: false,
                    pack2: false,
                }],
            },
            code: vs_code.iter().flat_map(|w| w.to_le_bytes()).collect(),
        },
        &[sig("POSITION", 0, 0, 0), sig("TEXCOORD", 0, 1, 0)],
        &[sig("TEXCOORD", 0, 0, 0), sig("SV_Position", 3, 1, 4)],
    );
    let mut vertices = Vec::new();
    for k in 0..3 {
        vertices.extend([POS[k][0], POS[k][1], 0.5, 1.0]);
        vertices.extend([ATTR[k][0], ATTR[k][1], 0.0, 0.0]);
    }

    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let force_warp = std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1");
    let mut hosts = Vec::new();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        if force_warp && !warp {
            continue;
        }
        let host = match Host::new(warp) {
            Ok(host) => host,
            Err(err) => {
                eprintln!("{label}: skipping barycentric draw: {err}");
                continue;
            }
        };
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
        // SAFETY: OPTIONS3 expects a writable OPTIONS3 structure of this size.
        let support = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS3,
                std::ptr::from_mut(&mut options).cast(),
                std::mem::size_of_val(&options) as u32,
            )
        };
        if support.is_err() || !options.BarycentricsSupported.as_bool() {
            eprintln!("{label}: skipping barycentric draw: SV_Barycentrics unavailable");
            continue;
        }
        hosts.push((label, host));
    }
    if hosts.is_empty() {
        return;
    }
    let mut failures = Vec::new();
    for (name, body, want) in cases() {
        let mut words = body;
        words.push(ONE_V7);
        words.extend(exp_mrt0([2, 4, 7, 7]));
        words.push(S_ENDPGM);
        let ps = translate(
            &NativeProgram {
                stage: HwStage::Ps,
                wave64: false,
                vgprs: 8,
                layout: Vec::new(),
                used: Vec::new(),
                io: StageIo::Ps {
                    inputs: vec![PsInput {
                        usage: Usage::Generic,
                        row: 0,
                        mask: 0b1111,
                        attr: 0,
                        flat: false,
                        fp16: false,
                        pack2: false,
                        row2: 0,
                    }],
                    input_addr: PERSP_CENTER,
                    db_shader_control: 0,
                    patches: Vec::new(),
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            },
            &[sig("TEXCOORD", 0, 0, 0)],
            &[sig("SV_Target", 16, 0, 0)],
        );
        for (label, host) in &hosts {
            let (root, pso) = host
                .pipeline(&vs, &ps)
                .unwrap_or_else(|e| panic!("{name}: {label}: pipeline: {e}"));
            let pixels = host
                .draw(
                    &root,
                    &pso,
                    &Draw {
                        roots: Vec::new(),
                        textures: &[],
                        samplers: &[],
                        vertices: &vertices,
                        stride: 32,
                        size: (16, 16),
                        mesh_groups: 0,
                    },
                )
                .unwrap_or_else(|e| panic!("{name}: {label}: draw: {e}"));
            let (mut covered, mut wrong, mut first) = (0, 0, None);
            for (i, px) in pixels.iter().enumerate() {
                if px[3] != 1.0 {
                    continue;
                }
                covered += 1;
                let (x, y) = (i as u32 % 16, i as u32 / 16);
                let w = weights(x, y);
                let e = want(w);
                let close = |got: f32, want: f32| (got - want).abs() <= 2e-3 * want.abs().max(1.0);
                if !(close(px[0], e[0]) && close(px[1], e[1])) {
                    wrong += 1;
                    if first.is_none() {
                        first = Some(format!(
                            "({x}, {y}) weights {w:.3?}: got [{:.4}, {:.4}], want [{:.4}, {:.4}]",
                            px[0], px[1], e[0], e[1]
                        ));
                    }
                }
            }
            eprintln!(
                "{name}: {label}: {covered} pixels, {wrong} wrong{}",
                first
                    .as_deref()
                    .map(|f| format!(", first {f}"))
                    .unwrap_or_default()
            );
            if covered < 40 || wrong > 0 {
                failures.push(format!("{name}: {label}: {wrong} of {covered}"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
}
