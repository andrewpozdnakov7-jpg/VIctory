use super::{Parts, lower};
use crate::native::{HwStage, NativeProgram, StageIo, Usage, VsInput};
use crate::{io, ir, resource, structure};

/// A 16-bit vertex attribute is the host's 32-bit load, packed into the
/// low half the way the fetch shader leaves it (one value per dword).
#[test]
fn a_16_bit_vertex_input_is_unpacked_to_the_low_half() {
    use crate::emit::container::SigElement;

    // exp pos0 v4, v4, v4, v4 done vm; the attribute is in v4.
    let words = [0xF800_18CFu32, 0x0404_0404, 0xBF81_0000];
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Vs,
        wave64: false,
        vgprs: 8,
        layout: vec![],
        used: vec![],
        io: StageIo::Vs {
            inputs: vec![VsInput {
                usage: Usage::Generic,
                row: 0,
                mask: 0b0001,
                vgpr: 4,
            }],
            outputs: vec![],
        },
        code,
    };
    let p = ir::build(&native.code, false, 8).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let el = |name: &str, kind, comp, cols, half| SigElement {
        name: name.into(),
        index: 0,
        kind,
        comp,
        interp: 0,
        row: Some(0),
        col: 0,
        cols,
        usage: 0,
        half,
    };
    let lower_attr = |comp: u8| {
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![el("TEXCOORD", 0, comp, 1, true)],
            outputs: vec![el("SV_Position", 3, 9, 4, false)],
            prims: vec![],
            mesh_stride: 0,
            view_marks: false,
        })
    };
    let ops = |comp: u8| {
        let lowered = lower_attr(comp).unwrap_or_else(|e| panic!("{comp}: {e}"));
        #[cfg(windows)]
        if let Some(path) = crate::emit::validator::Validator::default_path() {
            let mut dxil = lowered.dxil;
            let mut validator = crate::emit::validator::Validator::load(Some(&path)).unwrap();
            validator.validate(&mut dxil).unwrap();
            eprintln!("{comp}: DXIL validated");
        }
        lowered.ops
    };
    let f16 = ops(9);
    assert!(
        f16.contains("call dx.op.legacyF32ToF16"),
        "float16:\n{f16}"
    );
    assert!(!f16.lines().any(|l| l == "and 0xffff"), "float16:\n{f16}");
    for comp in [5u8, 4] {
        let int = ops(comp);
        assert!(
            int.lines().any(|l| l == "and 0xffff"),
            "{comp}:\n{int}"
        );
        assert!(!int.contains("legacyF32ToF16"), "{comp}:\n{int}");
    }
}
