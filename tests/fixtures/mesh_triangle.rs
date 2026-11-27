//! Synthetic fullscreen mesh shader, shared between shader and GPU tests.

use super::shader_api::{
    Container, FourCC, HwStage, NativeProgram, Part, ShaderError, emit, io, ir,
    native::{MeshIo, StageIo},
    resource, structure,
};
use emit::HostShader;

/// Triangle (0, 1, 2), each corner's 9-bit index in its 10-bit field.
pub const TRIANGLE: u32 = 1 << 10 | 2 << 20;

/// A one-wave (32-lane) triangle mesh program whose lanes all export
/// primitive `prim` and, as vertex `lane`, the position
/// (lane & 1 ? 3 : -1, lane & 2 ? 3 : -1, depth, 1): triangle (0, 1, 2)
/// covers the whole target.
pub fn triangle_mesh(prim: u32, depth: f32, whole_wave: bool) -> Result<HostShader, ShaderError> {
    let mut words = vec![
        0xD765_0001u32, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0x0001_00C1,
        0x3604_0281, // v_and_b32 v2, 1, v1
        0x3606_0282, // v_and_b32 v3, 2, v1
        0x7E04_0D02, // v_cvt_f32_u32 v2, v2
        0x7E06_0D03, // v_cvt_f32_u32 v3, v3
        0x1008_04F6, // v_mul_f32 v4, 4.0, v2
        0x0608_08F3, // v_add_f32 v4, -1.0, v4
        0x100A_06F4, // v_mul_f32 v5, 2.0, v3
        0x060A_0AF3, // v_add_f32 v5, -1.0, v5
        0x7E0C_02FF, // v_mov_b32 v6, depth
        depth.to_bits(),
        0x7E0E_02F2, // v_mov_b32 v7, 1.0
        0x7E12_02FF, // v_mov_b32 v9, prim
        prim,
    ];
    if whole_wave {
        words.push(0xBE80_3DC1); // s_or_saveexec_b32 s0, -1
    }
    words.extend([0xF800_0941, 0x0000_0009]); // exp prim v9, off, off, off done
    if whole_wave {
        words.push(0xBEFE_0300); // s_mov_b32 exec_lo, s0
    }
    words.extend([
        0xF800_08CF, // exp pos0 v4, v5, v6, v7 done
        0x0706_0504,
        0xBF81_0000, // s_endpgm
    ]);
    let native = NativeProgram {
        stage: HwStage::Ms,
        wave64: false,
        vgprs: 10,
        layout: vec![],
        used: vec![],
        io: StageIo::Ms(MeshIo {
            outputs: vec![],
            clip_cntl: 0,
            threads: 32,
            max_vertices: 32,
            max_primitives: 32,
            topology: 2,
            lds_bytes: 0,
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    // One OSG1 row for SV_Position (system value 1, f32, four channels).
    let mut osg1 = [1u32, 8, 0, 40, 0, 1, 3, 0, 0xF, 0]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    osg1.extend_from_slice(b"SV_Position\0");
    let c = Container {
        digest: [0; 16],
        version: (1, 0),
        parts: vec![Part {
            fourcc: FourCC::OSG1,
            offset: 36,
            data: &osg1,
        }],
    };
    emit_program(&c, &native)
}

pub fn emit_program(c: &Container, native: &NativeProgram) -> Result<HostShader, ShaderError> {
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, native)?;
    emit::emit(c, native, &p, &st, &io, &res)
}
