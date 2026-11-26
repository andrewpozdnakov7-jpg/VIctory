//! Whole-wave sections (decision 0003): code between `s_or_saveexec -1`
//! (or `s_nand_saveexec 0`) and the restore of the saved mask runs in every lane of the console's
//! wave, the inactive ones included, which is how the compiler expands
//! HLSL wave reductions and prefix sums (an identity put into the inactive
//! lanes with `v_cndmask`, then DPP, `v_permlanex16` and `v_readlane`
//! steps). A host lane that is elsewhere cannot run that code, so each
//! section is simulated on the console's lanes with symbolic values, and a
//! value it leaves is lowered by what it turned out to be: the reduction or
//! prefix of one operation over the lanes' (masked) inputs, which DXIL's
//! wave operations compute over the host's active lanes. No instruction
//! sequence is matched; any step the simulation does not model leaves the
//! values it writes unknown.

use std::collections::HashMap;

use crate::gfx10::{Dpp, Fields, NULL, Operand};
use crate::ir::{EXEC, Inst, Program, Reg, Value};

/// An associative operation and its identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Op {
    AddU,
    AddF,
    MinU,
    MaxU,
    MinI,
    MaxI,
    MinF,
    MaxF,
    Or,
    And,
    Xor,
    /// `v_add_f16`: on the low halves (see [`Sym::Half`]).
    AddH,
}

impl Op {
    fn of(name: &str) -> Option<Op> {
        Some(match name {
            // The low dwords of a 64-bit sum (the carry is not followed).
            "v_add_nc_u32" | "v_add_co_u32" => Op::AddU,
            "v_add_f32" => Op::AddF,
            "v_min_u32" => Op::MinU,
            "v_max_u32" => Op::MaxU,
            "v_min_i32" => Op::MinI,
            "v_max_i32" => Op::MaxI,
            "v_min_f32" => Op::MinF,
            "v_max_f32" => Op::MaxF,
            "v_or_b32" => Op::Or,
            "v_and_b32" => Op::And,
            "v_xor_b32" => Op::Xor,
            _ => return None,
        })
    }

    /// Whether `k` leaves the other operand unchanged.
    fn identity(self, k: u32) -> bool {
        match self {
            Op::AddU | Op::Or | Op::Xor | Op::MaxU => k == 0,
            Op::AddF => k == 0 || k == 0x8000_0000,
            Op::MinU | Op::And => k == u32::MAX,
            Op::MinI => k == 0x7FFF_FFFF,
            Op::MaxI => k == 0x8000_0000,
            Op::MinF => k == 0x7F80_0000,
            Op::MaxF => k == 0xFF80_0000,
            Op::AddH => k & 0x7FFF == 0,
        }
    }

    /// Counting a lane twice changes the result.
    fn counts(self) -> bool {
        matches!(self, Op::AddU | Op::AddF | Op::Xor | Op::AddH)
    }
}

/// A lane's value inside a section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Sym {
    Unknown,
    K(u32),
    /// The lane's own value of a VGPR value from before the section.
    Own(Value),
    /// A uniform value from before the section.
    Uni(Value),
    /// `op` over `lanes` of `mask ? src : ident` (`op` none: one lane).
    Red {
        op: Option<Op>,
        src: Value,
        mask: Value,
        ident: u32,
        lanes: u64,
    },
    /// A 16-bit operation's result: the low half is `op` over `lanes` of
    /// the low halves of `mask ? src : ident`; the high half is that of
    /// lane `hi`'s `mask ? src : ident` (16-bit results keep the
    /// register's high half).
    Half {
        op: Op,
        src: Value,
        mask: Value,
        ident: u32,
        lanes: u64,
        hi: u32,
    },
}

/// What a value a section leaves holds in every lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Shape {
    K(u32),
    Value(Value),
    /// The lane's own `mask ? src : ident`.
    Masked(Value, Value, u32),
    /// `op` over every lane.
    Active(Op, Value, Value, u32),
    /// `op` over the lanes below (`inclusive`: and including) each lane.
    Prefix(Op, Value, Value, u32, bool),
    /// [`Sym::Half`] over every lane: `op`, src, mask, identity, the lane
    /// giving the high half.
    Half(Op, Value, Value, u32, u32),
}

/// The section starting at instruction `start` of a block: its restore
/// index and the shapes of the values it writes.
pub(crate) struct Section {
    pub end: usize,
    pub shapes: HashMap<Value, Shape>,
}

/// A lane mask by the values of its low and high dword (both the low one in
/// wave32).
type Mask = (Value, Value);

fn mask_pair(value: impl Fn(Reg) -> Option<Value>, code: u8, wave64: bool) -> Option<Mask> {
    let lo = value(Reg::S(code))?;
    let hi = if wave64 {
        value(Reg::S(code.checked_add(1)?))?
    } else {
        lo
    };
    Some((lo, hi))
}

/// Every whole-wave section of `block`, by start index.
pub(crate) fn sections(p: &Program, block: usize) -> HashMap<usize, Section> {
    let insts = &p.blocks[block].insts;
    let wave: u32 = if p.wave64 { 64 } else { 32 };
    let mut out = HashMap::new();
    let mut i = 0;
    while i < insts.len() {
        let inst = &insts[i];
        // EXEC becomes every lane: `s_or_saveexec -1` or `s_nand_saveexec 0`
        // matching the program's wave width (a 32-bit saveexec in wave64
        // leaves EXEC_HI untouched).
        let all = match (p.wave64, inst.name()) {
            (true, "s_or_saveexec_b64") | (false, "s_or_saveexec_b32") => Some(Operand::Const(193)),
            (true, "s_nand_saveexec_b64") | (false, "s_nand_saveexec_b32") => {
                Some(Operand::Const(128))
            }
            _ => None,
        };
        let start = all.is_some_and(
            |k| matches!(&inst.fields, Fields::Sop { src, .. } if src.first() == Some(&k)),
        );
        let saved = match &inst.fields {
            Fields::Sop {
                dst: Some(Operand::Scalar { code, .. }),
                ..
            } if start => mask_pair(|r| inst.write(r), *code, p.wave64).map(|m| (*code, m)),
            _ => None,
        };
        let Some((code, saved)) = saved else {
            i += 1;
            continue;
        };
        let end = (i + 1..insts.len()).find(|&j| {
            mask_pair(|r| insts[j].write(r), EXEC, p.wave64).is_some()
                && mask_pair(|r| insts[j].read(r), code, p.wave64) == Some(saved)
        });
        let Some(end) = end else {
            break;
        };
        let shapes = simulate(&insts[i + 1..end], wave, saved);
        out.insert(i, Section { end, shapes });
        i = end + 1;
    }
    out
}

/// The lane `l` reads with DPP, whether it reads 0 instead, and whether it
/// writes; `None` for patterns not modelled.
fn dpp_lane(d: Dpp, l: u32) -> Option<(u32, bool, bool)> {
    match d {
        Dpp::Lanes { sel, .. } => Some(((l & !7) | (sel >> (3 * (l & 7)) & 7), false, true)),
        Dpp::Row {
            ctrl,
            row_mask,
            bank_mask,
            bound_ctrl,
            ..
        } => {
            let (row, pos, n) = (l & !15, l & 15, u32::from(ctrl & 15));
            let enabled = row_mask >> (l >> 4 & 3) & 1 != 0 && bank_mask >> (l >> 2 & 3) & 1 != 0;
            let (src, inside) = match ctrl {
                0..=0xFF => ((l & !3) | (u32::from(ctrl) >> (2 * (l & 3)) & 3), true),
                0x101..=0x10F => (l + n, pos + n < 16),
                0x111..=0x11F => (l.wrapping_sub(n), pos >= n),
                0x121..=0x12F => (row | (pos.wrapping_sub(n) & 15), true),
                0x140 => (l ^ 15, true),
                0x141 => (l ^ 7, true),
                0x150..=0x15F => (row | n, true),
                0x160..=0x16F => (l ^ n, true),
                _ => return None,
            };
            match (inside, bound_ctrl) {
                (true, _) => Some((src, false, enabled)),
                (false, true) => Some((l, true, enabled)),
                (false, false) => Some((l, false, false)),
            }
        }
    }
}

fn combine(op: Op, a: Sym, b: Sym) -> Sym {
    match (a, b) {
        (Sym::K(k), x) | (x, Sym::K(k)) if op.identity(k) => x,
        (
            Sym::Red {
                op: oa,
                src,
                mask,
                ident,
                lanes: la,
            },
            Sym::Red {
                op: ob,
                src: sb,
                mask: mb,
                ident: ib,
                lanes: lb,
            },
        ) if (src, mask, ident) == (sb, mb, ib)
            && op.identity(ident)
            && oa.is_none_or(|o| o == op)
            && ob.is_none_or(|o| o == op)
            && (!op.counts() || la & lb == 0) =>
        {
            Sym::Red {
                op: Some(op),
                src,
                mask,
                ident,
                lanes: la | lb,
            }
        }
        _ => Sym::Unknown,
    }
}

/// A lane of 16-bit `op` on the low halves of `a` and `b`, the high half
/// kept from the destination's `old` value.
fn half(op: Op, a: Sym, b: Option<Sym>, old: Sym) -> Sym {
    // A low half: the identity, or `op` over lanes of one masked source.
    let low = |s: Sym| match s {
        Sym::K(k) if op.identity(k) => Some(None),
        Sym::Red {
            op: None,
            src,
            mask,
            ident,
            lanes,
        } if op.identity(ident) => Some(Some((src, mask, ident, lanes))),
        Sym::Half {
            op: o,
            src,
            mask,
            ident,
            lanes,
            ..
        } if o == op => Some(Some((src, mask, ident, lanes))),
        _ => None,
    };
    let (src, mask, ident, hi) = match old {
        Sym::Red {
            op: None,
            src,
            mask,
            ident,
            lanes,
        } if lanes.is_power_of_two() => (src, mask, ident, lanes.trailing_zeros()),
        Sym::Half {
            src,
            mask,
            ident,
            hi,
            ..
        } => (src, mask, ident, hi),
        _ => return Sym::Unknown,
    };
    let (Some(x), Some(y)) = (low(a), b.and_then(low)) else {
        return Sym::Unknown;
    };
    let mut lanes = 0;
    for (s, m, i, l) in [x, y].into_iter().flatten() {
        if (s, m, i) != (src, mask, ident) || lanes & l != 0 {
            return Sym::Unknown;
        }
        lanes |= l;
    }
    if lanes == 0 {
        return Sym::Unknown;
    }
    Sym::Half {
        op,
        src,
        mask,
        ident,
        lanes,
        hi,
    }
}

fn simulate(insts: &[Inst], wave: u32, saved: Mask) -> HashMap<Value, Shape> {
    let wave64 = wave == 64;
    let all = if wave64 {
        u64::MAX
    } else {
        u64::from(u32::MAX)
    };
    let mut exec = all;
    let mut vec: HashMap<Value, Vec<Sym>> = HashMap::new();
    let mut uni: HashMap<Value, Sym> = HashMap::new();
    // Masks the section derives from the saved EXEC (`s_and` of wave width with it).
    let mut masked = vec![saved];
    for inst in insts {
        let name = inst.name();
        if inst.write(Reg::S(EXEC)).is_some() {
            // Constant lane sets (prefix sums); anything else ends what
            // the simulation can follow.
            let lo = name
                .starts_with("s_mov_b")
                .then(|| constant(inst))
                .flatten();
            let Some(k) = lo else {
                return shapes(vec, uni, wave);
            };
            exec = (if name == "s_mov_b64" {
                k
            } else {
                (exec & !0xFFFF_FFFF) | (k & 0xFFFF_FFFF)
            }) & all;
            continue;
        }
        if inst.write(Reg::S(EXEC + 1)).is_some() {
            let Some(k) = name
                .starts_with("s_mov_b")
                .then(|| constant(inst))
                .flatten()
            else {
                return shapes(vec, uni, wave);
            };
            exec = ((exec & 0xFFFF_FFFF) | ((k & 0xFFFF_FFFF) << 32)) & all;
            continue;
        }
        if let Fields::Sop { dst, src, .. } = &inst.fields {
            let and_op = if wave64 { "s_and_b64" } else { "s_and_b32" };
            if name == and_op
                && src.iter().any(|o| {
                    matches!(
                        *o,
                        Operand::Scalar { code, .. }
                            if mask_pair(|r| inst.read(r), code, wave64)
                                .is_some_and(|m| masked.contains(&m))
                    )
                })
                && let Some(Operand::Scalar { code, .. }) = *dst
                && let Some(w) = mask_pair(|r| inst.write(r), code, wave64)
            {
                masked.push(w);
            }
            for &(_, w) in &inst.writes {
                uni.entry(w).or_insert(Sym::Unknown);
            }
            continue;
        }
        let Fields::Valu(v) = &inst.fields else {
            for &(_, w) in &inst.writes {
                vec.insert(w, vec![Sym::Unknown; wave as usize]);
            }
            continue;
        };
        let f16_inline = name == "v_add_f16";
        let lanes = |op: &Operand,
                     vec: &HashMap<Value, Vec<Sym>>,
                     uni: &HashMap<Value, Sym>|
         -> Vec<Sym> {
            let one = |s: Sym| vec![s; wave as usize];
            match *op {
                Operand::Vector { reg, .. } => match inst.read(Reg::V(reg)) {
                    Some(x) => vec
                        .get(&x)
                        .cloned()
                        .unwrap_or_else(|| (0..wave).map(|_| Sym::Own(x)).collect()),
                    None => one(Sym::Unknown),
                },
                Operand::Scalar { code, .. } if code != NULL => match inst.read(Reg::S(code)) {
                    Some(x) => one(uni.get(&x).copied().unwrap_or(Sym::Uni(x))),
                    None => one(Sym::Unknown),
                },
                Operand::Const(c @ 240..=248) if f16_inline => {
                    let h = [
                        0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4400, 0xC400, 0x3118,
                    ][usize::from(c - 240)];
                    one(Sym::K(h))
                }
                Operand::Const(c) => one(crate::resource::inline(c).map_or(Sym::Unknown, Sym::K)),
                Operand::Literal(x) => one(Sym::K(x)),
                _ => one(Sym::Unknown),
            }
        };
        let modifiers = v.abs != 0 || v.neg != 0 || v.clamp || v.omod != 0 || v.sdwa.is_some();
        if name == "v_readlane_b32" {
            let (Some(Operand::Scalar { code, .. }), Some(src), Some(lane)) =
                (v.dst, v.src.first(), v.src.get(1))
            else {
                continue;
            };
            let l = match *lane {
                Operand::Const(c) => crate::resource::inline(c),
                Operand::Literal(x) => Some(x),
                _ => None,
            };
            let x = lanes(src, &vec, &uni);
            if let Some(w) = inst.write(Reg::S(code)) {
                let s = l
                    .and_then(|l| x.get(l as usize).copied())
                    .unwrap_or(Sym::Unknown);
                uni.insert(w, s);
            }
            continue;
        }
        let Some(Operand::Vector { reg: dreg, .. }) = v.dst else {
            for &(_, w) in &inst.writes {
                uni.insert(w, Sym::Unknown);
            }
            continue;
        };
        let Some(dst) = inst.write(Reg::V(dreg)) else {
            continue;
        };
        let old = lanes(
            &Operand::Vector {
                reg: dreg,
                dwords: 1,
            },
            &vec,
            &uni,
        );
        // src0 as each lane fetches it (DPP), with the lanes that write.
        let src0 = v.src.first().map(|o| lanes(o, &vec, &uni));
        let mut writes = exec;
        let fetched: Option<Vec<Sym>> = match (v.dpp, &src0) {
            (Some(d), Some(s)) => {
                if exec != all {
                    vec.insert(dst, vec![Sym::Unknown; wave as usize]);
                    continue;
                }
                let mut f = Vec::with_capacity(wave as usize);
                let mut ok = true;
                for l in 0..wave {
                    match dpp_lane(d, l) {
                        Some((src, zero, on)) if src < wave => {
                            f.push(if zero { Sym::K(0) } else { s[src as usize] });
                            if !on {
                                writes &= !(1u64 << l);
                            }
                        }
                        _ => ok = false,
                    }
                }
                ok.then_some(f)
            }
            (None, Some(s)) => Some(s.clone()),
            _ => None,
        };
        let src = |k: usize| v.src.get(k).map(|o| lanes(o, &vec, &uni));
        let result: Vec<Sym> = match (name, fetched) {
            _ if modifiers => vec![Sym::Unknown; wave as usize],
            ("v_mov_b32", Some(a)) => a,
            ("v_bfrev_b32", Some(a)) => a
                .into_iter()
                .map(|s| match s {
                    Sym::K(k) => Sym::K(k.reverse_bits()),
                    _ => Sym::Unknown,
                })
                .collect(),
            ("v_cndmask_b32", Some(a)) => {
                let mask = match v.src.get(2) {
                    Some(&Operand::Scalar { code, .. }) => {
                        mask_pair(|r| inst.read(r), code, wave64)
                    }
                    _ => None,
                };
                let b = src(1).unwrap_or_default();
                (0..wave as usize)
                    .map(|l| match (a[l], b.get(l).copied(), mask) {
                        (Sym::K(ident), Some(Sym::Own(x)), Some(m)) if masked.contains(&m) => {
                            Sym::Red {
                                op: None,
                                src: x,
                                mask: m.0,
                                ident,
                                lanes: 1 << l,
                            }
                        }
                        _ => Sym::Unknown,
                    })
                    .collect()
            }
            ("v_permlanex16_b32", Some(a)) => {
                // Lanes of each row read the other row of their half; lane
                // i's pick is nibble i of {src2, src1}. VOP3 op_sel[0] is FI
                // (fetch inactive) and op_sel[1] is BOUND_CTRL (write 0 vs
                // preserve destination when the source lane is inactive).
                let sel = |k: usize| match v.src.get(k) {
                    Some(&Operand::Const(c)) => crate::resource::inline(c),
                    Some(&Operand::Literal(x)) => Some(x),
                    _ => None,
                };
                let fi = v.opsel & 1 != 0;
                let bound_ctrl = v.opsel & 2 != 0;
                match (sel(1), sel(2)) {
                    (Some(s1), Some(s2)) => {
                        let s = u64::from(s1) | u64::from(s2) << 32;
                        (0..wave)
                            .map(|l| {
                                let other = (l & !31) | ((l & 16) ^ 16);
                                let pick = (s >> (4 * (l & 15)) & 15) as u32;
                                let src = other | pick;
                                if fi || exec >> src & 1 != 0 {
                                    a[src as usize]
                                } else if bound_ctrl {
                                    Sym::K(0)
                                } else {
                                    writes &= !(1u64 << l);
                                    Sym::Unknown
                                }
                            })
                            .collect()
                    }
                    _ => vec![Sym::Unknown; wave as usize],
                }
            }
            ("v_sub_nc_u32", Some(a)) => {
                // A prefix sum less the lane's own term: the exclusive one.
                let b = src(1).unwrap_or_default();
                (0..wave as usize)
                    .map(|l| match (a[l], b.get(l).copied()) {
                        (
                            Sym::Red {
                                op: None | Some(Op::AddU),
                                src,
                                mask,
                                ident: 0,
                                lanes,
                            },
                            Some(Sym::Red {
                                op: None | Some(Op::AddU),
                                src: s2,
                                mask: m2,
                                ident: 0,
                                lanes: l2,
                            }),
                        ) if (src, mask) == (s2, m2) && lanes & l2 == l2 => {
                            let rest = lanes & !l2;
                            if rest == 0 {
                                Sym::K(0)
                            } else {
                                Sym::Red {
                                    op: Some(Op::AddU),
                                    src,
                                    mask,
                                    ident: 0,
                                    lanes: rest,
                                }
                            }
                        }
                        _ => Sym::Unknown,
                    })
                    .collect()
            }
            ("v_add_f16", Some(a)) if v.opsel == 0 => {
                let b = src(1).unwrap_or_default();
                (0..wave as usize)
                    .map(|l| half(Op::AddH, a[l], b.get(l).copied(), old[l]))
                    .collect()
            }
            (n, Some(a)) => match (Op::of(n), src(1)) {
                (Some(op), Some(b)) => (0..wave as usize)
                    .map(|l| combine(op, a[l], b[l]))
                    .collect(),
                _ => vec![Sym::Unknown; wave as usize],
            },
            _ => vec![Sym::Unknown; wave as usize],
        };
        let merged = (0..wave as usize)
            .map(|l| {
                if writes >> l & 1 != 0 {
                    result[l]
                } else {
                    old[l]
                }
            })
            .collect();
        vec.insert(dst, merged);
    }
    shapes(vec, uni, wave)
}

fn constant(inst: &Inst) -> Option<u64> {
    let Fields::Sop { src, .. } = &inst.fields else {
        return None;
    };
    let wide = inst.name() == "s_mov_b64";
    match *src.first()? {
        Operand::Scalar { code: NULL, .. } => Some(0),
        Operand::Const(c @ 128..=208) => {
            let x = crate::resource::inline(c)?;
            Some(if wide {
                i64::from(x as i32) as u64
            } else {
                u64::from(x)
            })
        }
        Operand::Const(c) if !wide => crate::resource::inline(c).map(u64::from),
        Operand::Literal(x) => Some(u64::from(x)),
        _ => None,
    }
}

/// The shape of each value, where every lane agrees on one.
fn shapes(
    vec: HashMap<Value, Vec<Sym>>,
    uni: HashMap<Value, Sym>,
    wave: u32,
) -> HashMap<Value, Shape> {
    let all = if wave == 64 {
        u64::MAX
    } else {
        u64::from(u32::MAX)
    };
    let mut out = HashMap::new();
    let single = |s: Sym| match s {
        Sym::K(k) => Some(Shape::K(k)),
        Sym::Own(x) | Sym::Uni(x) => Some(Shape::Value(x)),
        Sym::Red {
            op: Some(op),
            src,
            mask,
            ident,
            lanes,
        } if lanes == all => Some(Shape::Active(op, src, mask, ident)),
        Sym::Half {
            op,
            src,
            mask,
            ident,
            lanes,
            hi,
        } if lanes == all => Some(Shape::Half(op, src, mask, ident, hi)),
        _ => None,
    };
    for (v, s) in uni {
        if let Some(sh) = single(s) {
            out.insert(v, sh);
        }
    }
    for (v, lanes) in vec {
        if let Some(sh) = per_lane(&lanes, all) {
            out.insert(v, sh);
        }
    }
    out
}

fn per_lane(lanes: &[Sym], all: u64) -> Option<Shape> {
    let first = lanes[0];
    let own = |l: usize, s: Sym| match s {
        Sym::Red {
            op: None,
            src,
            mask,
            ident,
            lanes,
        } if lanes == 1 << l => Some((src, mask, ident)),
        _ => None,
    };
    if let Some(m) = own(0, first)
        && lanes.iter().enumerate().all(|(l, &s)| own(l, s) == Some(m))
    {
        return Some(Shape::Masked(m.0, m.1, m.2));
    }
    if lanes.iter().all(|&s| s == first) {
        return match first {
            Sym::K(k) => Some(Shape::K(k)),
            Sym::Uni(x) | Sym::Own(x) => Some(Shape::Value(x)),
            Sym::Red {
                op: Some(op),
                src,
                mask,
                ident,
                lanes,
            } if lanes == all => Some(Shape::Active(op, src, mask, ident)),
            _ => None,
        };
    }
    // Prefixes: lane l holds lanes 0..l (exclusive) or 0..=l (inclusive).
    let mut meta: Option<(Value, Value, u32, bool)> = None;
    let mut op_found: Option<Op> = None;
    for (l, &s) in lanes.iter().enumerate() {
        let below = (1u64 << l) - 1;
        let (op, src, mask, ident, set) = match s {
            Sym::Red {
                op,
                src,
                mask,
                ident,
                lanes,
            } => (op, src, mask, ident, lanes),
            // Lane 0 of an exclusive prefix holds the identity.
            Sym::K(_) if l == 0 => continue,
            _ => return None,
        };
        let inclusive = set == below | (1u64 << l);
        if !inclusive && set != below {
            return None;
        }
        if l == 0 && !inclusive {
            return None;
        }
        if op.is_none() && set.count_ones() != 1 {
            return None;
        }
        let lane_meta = (src, mask, ident, inclusive);
        match meta {
            None => meta = Some(lane_meta),
            Some(prev) if prev == lane_meta => {}
            _ => return None,
        }
        if let Some(o) = op {
            match op_found {
                None => op_found = Some(o),
                Some(prev) if prev == o => {}
                _ => return None,
            }
        }
    }
    let (src, mask, ident, inclusive) = meta?;
    let op = op_found?;
    if !op.identity(ident) {
        return None;
    }
    match lanes[0] {
        Sym::K(k) if !inclusive && k == ident => {}
        Sym::Red { .. } if inclusive => {}
        _ => return None,
    }
    Some(Shape::Prefix(op, src, mask, ident, inclusive))
}

#[cfg(test)]
mod tests {
    use super::*;

    const S_ENDPGM: u32 = 0xBF81_0000;

    const fn sop1(sdst: u8, op: u8, ssrc0: u8) -> u32 {
        0xBE80_0000 | (sdst as u32) << 16 | (op as u32) << 8 | ssrc0 as u32
    }

    const fn vop1(vdst: u8, op: u8, src0: u16) -> u32 {
        (0x3F << 25) | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
    }

    const fn vop2(vdst: u8, op: u8, src0: u16, vsrc1: u8) -> u32 {
        (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
    }

    const fn dpp16(vsrc0: u8, ctrl: u16, row_mask: u8, bank_mask: u8, bound_ctrl: bool) -> u32 {
        (row_mask as u32) << 28
            | (bank_mask as u32) << 24
            | (bound_ctrl as u32) << 19
            | (ctrl as u32) << 8
            | vsrc0 as u32
    }

    const fn vop3(vdst: u8, op: u16, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
        [
            0xD400_0000 | (op as u32) << 16 | vdst as u32,
            (src0 as u32 & 0x1FF) | ((src1 as u32 & 0x1FF) << 9) | ((src2 as u32 & 0x1FF) << 18),
        ]
    }

    fn build(words: &[u32], wave64: bool) -> Program {
        let code: Vec<u8> = words
            .iter()
            .copied()
            .chain(std::iter::once(S_ENDPGM))
            .flat_map(u32::to_le_bytes)
            .collect();
        crate::ir::build(&code, wave64, 256).unwrap()
    }

    #[test]
    fn dpp_lanes_follow_the_isa() {
        // row_shr:1 with bound_ctrl: lane 16 reads 0, lane 17 lane 16.
        let shr = Dpp::Row {
            ctrl: 0x111,
            row_mask: 0xF,
            bank_mask: 0xF,
            bound_ctrl: true,
            fi: false,
        };
        assert_eq!(dpp_lane(shr, 16), Some((16, true, true)));
        assert_eq!(dpp_lane(shr, 17), Some((16, false, true)));
        // row_mirror: lane 3 reads lane 12.
        let mirror = Dpp::Row {
            ctrl: 0x140,
            row_mask: 0xF,
            bank_mask: 0xF,
            bound_ctrl: true,
            fi: false,
        };
        assert_eq!(dpp_lane(mirror, 3), Some((12, false, true)));
        // dpp8 [1,0,3,2,5,4,7,6] swaps neighbours.
        let swap = Dpp::Lanes {
            sel: 1 | 3 << 6 | 2 << 9 | 5 << 12 | 4 << 15 | 7 << 18 | 6 << 21,
            fi: false,
        };
        assert_eq!(dpp_lane(swap, 8), Some((9, false, true)));
    }

    #[test]
    fn prefixes_are_recognised() {
        let (x, m) = (Value(1), Value(2));
        let red = |lanes: u64| Sym::Red {
            op: Some(Op::AddU),
            src: x,
            mask: m,
            ident: 0,
            lanes,
        };
        let inclusive: Vec<Sym> = (0..64).map(|l| red(u64::MAX >> (63 - l))).collect();
        assert_eq!(
            per_lane(&inclusive, u64::MAX),
            Some(Shape::Prefix(Op::AddU, x, m, 0, true))
        );
        let mut exclusive: Vec<Sym> = (0..64).map(|l| red((1u64 << l) - 1)).collect();
        exclusive[0] = Sym::K(0);
        assert_eq!(
            per_lane(&exclusive, u64::MAX),
            Some(Shape::Prefix(Op::AddU, x, m, 0, false))
        );
        assert_eq!(
            per_lane(&vec![red(u64::MAX); 64], u64::MAX),
            Some(Shape::Active(Op::AddU, x, m, 0))
        );
    }

    #[test]
    fn constant_decodes_width_and_inline_kinds() {
        let b64 = 0x04;
        let b32 = 0x03;
        let check = |words: &[u32], wave64: bool| -> Option<u64> {
            let p = build(words, wave64);
            constant(&p.blocks[0].insts[0])
        };

        assert_eq!(
            check(&[sop1(EXEC, b64, 194)], true),
            Some(0xFFFF_FFFF_FFFF_FFFE)
        );
        assert_eq!(
            check(&[sop1(EXEC, b64, 196)], true),
            Some(0xFFFF_FFFF_FFFF_FFFC)
        );
        assert_eq!(
            check(&[sop1(EXEC, b64, 208)], true),
            Some(0xFFFF_FFFF_FFFF_FFF0)
        );
        assert_eq!(check(&[sop1(EXEC, b64, 193)], true), Some(u64::MAX));
        assert_eq!(check(&[sop1(EXEC, b64, NULL)], true), Some(0));
        assert_eq!(check(&[sop1(EXEC, b64, 243)], true), None);
        assert_eq!(
            check(&[sop1(EXEC, b64, 255), 0x1234_5678], true),
            Some(0x1234_5678)
        );

        assert_eq!(check(&[sop1(EXEC, b32, 194)], false), Some(0xFFFF_FFFE));
        assert_eq!(check(&[sop1(EXEC, b32, 243)], false), Some(0xBF80_0000));
        assert_eq!(check(&[sop1(EXEC, b32, NULL)], false), Some(0));
        assert_eq!(
            check(&[sop1(EXEC, b32, 255), 0x1234_5678], false),
            Some(0x1234_5678)
        );
        assert_eq!(check(&[sop1(EXEC, b32, 0)], false), None);
    }

    #[test]
    fn exec_minus_two_masks_lane_zero_and_enforces_dpp_guard() {
        for wave64 in [false, true] {
            let mut w = Vec::new();
            let save_op = if wave64 { 0x25 } else { 0x3D };
            let mov_exec = if wave64 { 0x04 } else { 0x03 };
            let push_exec = |w: &mut Vec<u32>, k: u8| w.push(sop1(EXEC, mov_exec, k));
            let push_exec_lit = |w: &mut Vec<u32>, lit: u32| {
                w.push(sop1(EXEC, mov_exec, 255));
                w.push(lit);
            };

            w.push(sop1(0, save_op, 193));
            w.extend(vop3(1, 0x101, 128, 256, 0));
            for shift in [1u16, 2, 4, 8] {
                w.push(vop2(1, 0x25, 0xFA, 1));
                w.push(dpp16(1, 0x110 + shift, 0xF, 0xF, false));
            }
            w.extend(vop3(2, 0x360, 257, 128 + 15, 0));
            push_exec_lit(&mut w, 0xFFFF_0000);
            w.push(vop2(1, 0x25, 2, 1));

            if wave64 {
                w.extend(vop3(2, 0x360, 257, 128 + 31, 0));
                w.push(sop1(EXEC, 0x03, 128));
                w.push(sop1(EXEC + 1, 0x03, 255));
                w.push(0x0000_FFFF);
                w.push(vop2(1, 0x25, 2, 1));

                w.extend(vop3(2, 0x360, 257, 128 + 47, 0));
                w.push(sop1(EXEC, 0x03, 128));
                w.push(sop1(EXEC + 1, 0x03, 255));
                w.push(0xFFFF_0000);
                w.push(vop2(1, 0x25, 2, 1));
            }

            push_exec(&mut w, 193);
            w.push(vop1(3, 0x01, 257));
            w.push(vop1(3, 0x01, 0xFA));
            w.push(dpp16(1, 0x111, 0xF, 0xF, false));

            w.extend(vop3(2, 0x360, 257, 128 + 15, 0));
            push_exec_lit(&mut w, 0x0001_0000);
            w.push(vop1(3, 0x01, 2));

            if wave64 {
                w.extend(vop3(2, 0x360, 257, 128 + 31, 0));
                w.push(sop1(EXEC, 0x03, 128));
                w.push(sop1(EXEC + 1, 0x03, 129));
                w.push(vop1(3, 0x01, 2));

                w.extend(vop3(2, 0x360, 257, 128 + 47, 0));
                w.push(sop1(EXEC, 0x03, 128));
                w.push(sop1(EXEC + 1, 0x03, 255));
                w.push(0x0001_0000);
                w.push(vop1(3, 0x01, 2));
            }

            push_exec(&mut w, 193);
            w.push(vop1(2, 0x01, 128));
            w.push(vop1(5, 0x01, 135));
            push_exec(&mut w, 194);
            w.push(vop1(2, 0x01, 259));
            w.push(vop1(4, 0x01, 0xFA));
            w.push(dpp16(5, 0xE4, 0xF, 0xF, false));
            w.push(sop1(EXEC, mov_exec, 0));

            let p = build(&w, wave64);
            let insts = &p.blocks[0].insts;
            let saved = insts[0].write(Reg::S(0)).unwrap();
            let v0_in = p
                .inputs
                .iter()
                .find_map(|&(r, v)| (r == Reg::V(0)).then_some(v))
                .unwrap();
            let v2_val = insts[insts.len() - 4].write(Reg::V(2)).unwrap();
            let v4_val = insts[insts.len() - 3].write(Reg::V(4)).unwrap();
            let sec = &sections(&p, 0)[&0];
            assert_eq!(
                sec.shapes.get(&v2_val),
                Some(&Shape::Prefix(Op::AddU, v0_in, saved, 0, false))
            );
            assert_eq!(sec.shapes.get(&v4_val), None);
        }
    }

    #[test]
    fn wave32_mov_b64_all_lanes_keeps_exec_masked_for_dpp() {
        let w = [
            sop1(0, 0x3D, 193),
            vop1(1, 0x01, 135),
            sop1(EXEC, 0x04, 193),
            vop1(2, 0x01, 0xFA),
            dpp16(1, 0xE4, 0xF, 0xF, false),
            sop1(EXEC + 1, 0x03, 193),
            vop1(3, 0x01, 0xFA),
            dpp16(1, 0xE4, 0xF, 0xF, false),
            sop1(EXEC, 0x03, 0),
        ];
        let p = build(&w, false);
        let insts = &p.blocks[0].insts;
        let v2_val = insts[3].write(Reg::V(2)).unwrap();
        let v3_val = insts[5].write(Reg::V(3)).unwrap();
        let sec = &sections(&p, 0)[&0];
        assert_eq!(sec.shapes.get(&v2_val), Some(&Shape::K(7)));
        assert_eq!(sec.shapes.get(&v3_val), Some(&Shape::K(7)));
    }

    #[test]
    fn per_lane_prefix_soundness() {
        let (v0, v_other, m0, m_other) = (Value(1), Value(2), Value(3), Value(4));
        let red = |op: Option<Op>, src: Value, mask: Value, ident: u32, lanes: u64| Sym::Red {
            op,
            src,
            mask,
            ident,
            lanes,
        };

        for (wave, all) in [(32usize, u64::from(u32::MAX)), (64usize, u64::MAX)] {
            let below_mask = |l: usize| if l == 0 { 0 } else { (1u64 << l) - 1 };

            let mut valid_inc: Vec<Sym> = (0..wave)
                .map(|l| red(Some(Op::AddU), v0, m0, 0, below_mask(l) | (1u64 << l)))
                .collect();
            valid_inc[0] = red(None, v0, m0, 0, 1);
            assert_eq!(
                per_lane(&valid_inc, all),
                Some(Shape::Prefix(Op::AddU, v0, m0, 0, true))
            );

            let mut valid_exc: Vec<Sym> = (0..wave)
                .map(|l| red(Some(Op::AddU), v0, m0, 0, below_mask(l)))
                .collect();
            valid_exc[0] = Sym::K(0);
            valid_exc[1] = red(None, v0, m0, 0, 1);
            assert_eq!(
                per_lane(&valid_exc, all),
                Some(Shape::Prefix(Op::AddU, v0, m0, 0, false))
            );

            let mut mixed_other_src = valid_exc.clone();
            mixed_other_src[0] = red(None, v_other, m0, 0, 1);
            assert_eq!(per_lane(&mixed_other_src, all), None);

            let mut mixed_same_src = valid_exc.clone();
            mixed_same_src[0] = red(None, v0, m0, 0, 1);
            assert_eq!(per_lane(&mixed_same_src, all), None);

            let mut exc_bad_lane1_src = valid_exc.clone();
            exc_bad_lane1_src[1] = red(None, v_other, m0, 0, 1);
            assert_eq!(per_lane(&exc_bad_lane1_src, all), None);

            let mut exc_bad_lane1_set = valid_exc.clone();
            exc_bad_lane1_set[1] = red(None, v0, m0, 0, 0b11);
            assert_eq!(per_lane(&exc_bad_lane1_set, all), None);

            for bad0 in [
                red(None, v0, m0, 1, 1),
                red(None, v_other, m0, 0, 1),
                red(None, v0, m_other, 0, 1),
                red(None, v0, m0, 0, 0),
                Sym::K(0),
            ] {
                let mut bad_inc = valid_inc.clone();
                bad_inc[0] = bad0;
                assert_eq!(per_lane(&bad_inc, all), None);
            }

            let mut bad_ident: Vec<Sym> = (0..wave)
                .map(|l| red(Some(Op::AddU), v0, m0, 1, below_mask(l) | (1u64 << l)))
                .collect();
            bad_ident[0] = red(None, v0, m0, 1, 1);
            assert_eq!(per_lane(&bad_ident, all), None);
        }
    }

    #[test]
    fn v_add_f16_inline_float_constants_are_not_zero_identity() {
        // Wave32 f16 sum reduction with an extra `v_add_f16 v1, <const>, v1`
        // step: 0 (code 128) is a true additive identity, whereas inline float
        // constants (240..=248: 0.5, -0.5, 1.0, ..., 1/(2*pi)) are non-zero
        // f16 values on GFX10 and must not be decoded as f32 bits whose low 15
        // bits happen to be zero.
        let run = |src_const: u16| -> (Option<Shape>, Value, Value) {
            let mut w = vec![sop1(0, 0x3D, 193)]; // s_or_saveexec_b32 s0, -1
            w.extend(vop3(1, 0x101, 128, 256, 0)); // v_cndmask_b32 v1, 0, v0, s0
            w.push(vop2(1, 0x32, src_const, 1)); // v_add_f16 v1, <const>, v1
            for ctrl in [0x00B1u16, 0x004E, 0x0164, 0x0168] {
                w.push(vop2(1, 0x32, 0xFA, 1));
                w.push(dpp16(1, ctrl, 0xF, 0xF, false));
            }
            w.extend(vop3(2, 0x378, 257, 128, 128)); // v_permlanex16_b32 v2, v1, 0, 0
            w.push(vop2(1, 0x32, 258, 1)); // v_add_f16 v1, v2, v1
            w.extend(vop3(1, 0x360, 257, 128 + 31, 0)); // v_readlane_b32 s1, v1, 31
            w.push(sop1(EXEC, 0x03, 0)); // s_mov_b32 exec_lo, s0

            let p = build(&w, false);
            let insts = &p.blocks[0].insts;
            let saved = insts[0].write(Reg::S(0)).unwrap();
            let v0_in = insts[1].read(Reg::V(0)).unwrap();
            let s1 = insts[insts.len() - 3].write(Reg::S(1)).unwrap();
            let sec = &sections(&p, 0)[&0];
            (sec.shapes.get(&s1).copied(), v0_in, saved)
        };

        let (zero_shape, v0_in, saved) = run(128);
        assert_eq!(zero_shape, Some(Shape::Half(Op::AddH, v0_in, saved, 0, 31)));
        for c in 240..=248u16 {
            let (shape, _, _) = run(c);
            assert_eq!(
                shape, None,
                "inline float const {c} was treated as identity"
            );
        }
    }

    const fn sop2(sdst: u8, op: u8, ssrc0: u8, ssrc1: u8) -> u32 {
        0x8000_0000 | (op as u32) << 23 | (sdst as u32) << 16 | (ssrc1 as u32) << 8 | ssrc0 as u32
    }

    const fn vop3_opsel(vdst: u8, op: u16, opsel: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
        [
            0xD400_0000 | (op as u32) << 16 | ((opsel as u32) & 0xF) << 11 | vdst as u32,
            (src0 as u32 & 0x1FF) | ((src1 as u32 & 0x1FF) << 9) | ((src2 as u32 & 0x1FF) << 18),
        ]
    }

    /// Appends a wave32 inclusive row prefix sum (`v_add_nc_u32` DPP `row_shr:1,2,4,8`)
    /// on `v1`, initializes `v2 = old_k`, narrows `EXEC` to `exec_mask`, runs
    /// `v_permlanex16_b32 v2, v1, -1, -1` with `opsel`, adds `v2` into `v1`, and
    /// restores `EXEC` from `s0`.
    fn wave32_permlanex16_prefix_words(exec_mask: u32, old_k: u8, opsel: u8) -> Vec<u32> {
        let mut w = vec![sop1(0, 0x3D, 193)]; // s_or_saveexec_b32 s0, -1
        w.extend(vop3(1, 0x101, 128, 256, 0)); // v_cndmask_b32 v1, 0, v0, s0
        for shift in [1u16, 2, 4, 8] {
            w.push(vop2(1, 0x25, 0xFA, 1));
            w.push(dpp16(1, 0x110 + shift, 0xF, 0xF, false));
        }
        w.push(vop1(2, 0x01, 128 + u16::from(old_k))); // v_mov_b32 v2, old_k
        w.push(sop1(EXEC, 0x03, 255)); // s_mov_b32 exec_lo, exec_mask
        w.push(exec_mask);
        // v_permlanex16_b32 v2, v1, -1, -1 op_sel:[fi, bound_ctrl, 0, 0]
        // (all nibbles = 15, so lanes 16..31 select lane 15).
        w.extend(vop3_opsel(2, 0x378, opsel, 257, 193, 193));
        w.push(vop2(1, 0x25, 258, 1)); // v_add_nc_u32 v1, v2, v1
        w.push(sop1(EXEC, 0x03, 0)); // s_mov_b32 exec_lo, s0
        w
    }

    #[test]
    fn v_permlanex16_honors_exec_fi_and_bound_ctrl() {
        let run = |exec_mask: u32, old_k: u8, opsel: u8| {
            let w = wave32_permlanex16_prefix_words(exec_mask, old_k, opsel);
            let p = build(&w, false);
            let insts = &p.blocks[0].insts;
            let saved = insts[0].write(Reg::S(0)).unwrap();
            let v0_in = insts[1].read(Reg::V(0)).unwrap();
            let v2_val = insts[insts.len() - 4].write(Reg::V(2)).unwrap();
            let v1_val = insts[insts.len() - 3].write(Reg::V(1)).unwrap();
            let sec = &sections(&p, 0)[&0];
            (
                sec.shapes.get(&v2_val).copied(),
                sec.shapes.get(&v1_val).copied(),
                v0_in,
                saved,
            )
        };

        // Under EXEC = 0xFFFF_0000, lane 15 is inactive:
        // FI=0, BOUND_CTRL=0 preserves old v2 (= 7) in every lane and must not
        // produce a wave-wide prefix sum.
        let (v2_fi0_bc0, v1_fi0_bc0, _, _) = run(0xFFFF_0000, 7, 0);
        assert_eq!(v2_fi0_bc0, Some(Shape::K(7)));
        assert_eq!(v1_fi0_bc0, None);

        // FI=0, BOUND_CTRL=1 writes 0 into lanes 16..31 (keeping old=0 in 0..15),
        // leaving v2 = 0 everywhere and never fetching inactive lane 15.
        let (v2_fi0_bc1_zero, v1_fi0_bc1_zero, _, _) = run(0xFFFF_0000, 0, 2);
        assert_eq!(v2_fi0_bc1_zero, Some(Shape::K(0)));
        assert_eq!(v1_fi0_bc1_zero, None);

        // FI=0, BOUND_CTRL=1 with old=7 writes 0 in lanes 16..31 and keeps 7 in
        // lanes 0..15, so lanes disagree and v2 has no uniform Shape::K.
        let (v2_fi0_bc1_seven, v1_fi0_bc1_seven, _, _) = run(0xFFFF_0000, 7, 2);
        assert_eq!(v2_fi0_bc1_seven, None);
        assert_eq!(v1_fi0_bc1_seven, None);

        // FI=1 (op_sel=1 or 3) fetches inactive lane 15, yielding the full
        // inclusive prefix sum.
        for opsel in [1u8, 3] {
            let (_, v1_fi1, v0_in, saved) = run(0xFFFF_0000, 0, opsel);
            assert_eq!(
                v1_fi1,
                Some(Shape::Prefix(Op::AddU, v0_in, saved, 0, true)),
                "opsel={opsel} with FI=1 must fetch inactive lane 15"
            );
        }

        // When EXEC = 0xFFFF_8000, source lane 15 is active, so even FI=0,
        // BOUND_CTRL=0 reads v1[15] for lanes 16..31 (and lane 15 reads active
        // lane 31, so v1 is no prefix sum, while v2[16..31] gets lane 15).
        let (v2_active_src, _, _, _) = run(0xFFFF_8000, 7, 0);
        assert_ne!(
            v2_active_src,
            Some(Shape::K(7)),
            "active source lane 15 must be fetched even with FI=0"
        );
    }

    /// Wave64 whole-wave sum reduction of `v1` into `s4`, after initializing
    /// `v1` via `v_cndmask_b32 v1, 0, v0, s[mask_reg : mask_reg + 1]`.
    fn wave64_sum_section_words(prologue: &[u32], mask_reg: u16, epilogue: &[u32]) -> Vec<u32> {
        let mut w = prologue.to_vec();
        w.extend(vop3(1, 0x101, 128, 256, mask_reg)); // v_cndmask_b32 v1, 0, v0, s[mask:mask+1]
        for ctrl in [0x00B1u16, 0x004E, 0x0164, 0x0168] {
            w.push(vop2(1, 0x25, 0xFA, 1));
            w.push(dpp16(1, ctrl, 0xF, 0xF, false));
        }
        w.extend(vop3(2, 0x378, 257, 128, 128)); // v_permlanex16_b32 v2, v1, 0, 0
        w.push(vop2(1, 0x25, 258, 1)); // v_add_nc_u32 v1, v2, v1
        w.extend(vop3(4, 0x360, 257, 128 + 31, 0)); // v_readlane_b32 s4, v1, 31
        w.extend(vop3(5, 0x360, 257, 128 + 63, 0)); // v_readlane_b32 s5, v1, 63
        w.push(vop1(2, 0x01, 4)); // v_mov_b32 v2, s4
        w.push(vop2(2, 0x25, 5, 2)); // v_add_nc_u32 v2, s5, v2
        w.extend(vop3(4, 0x360, 258, 128, 0)); // v_readlane_b32 s4, v2, 0
        w.extend_from_slice(epilogue);
        w
    }

    #[test]
    fn wave64_whole_wave_section_tracks_full_64bit_masks_and_width() {
        // 1. Positive control: intact 64-bit saved + s_and_b64 mask reduces to Active.
        let valid = wave64_sum_section_words(
            &[
                sop1(0, 0x25, 193),    // s_or_saveexec_b64 s[0:1], -1
                sop2(2, 0x0F, 0, 193), // s_and_b64 s[2:3], s[0:1], -1
            ],
            2,
            &[sop1(EXEC, 0x04, 0)], // s_mov_b64 exec, s[0:1]
        );
        let p = build(&valid, true);
        let insts = &p.blocks[0].insts;
        let s2_val = insts[1].write(Reg::S(2)).unwrap();
        let v0_in = insts[2].read(Reg::V(0)).unwrap();
        let s4_val = insts[insts.len() - 3].write(Reg::S(4)).unwrap();
        let sec = &sections(&p, 0)[&0];
        assert_eq!(
            sec.shapes.get(&s4_val),
            Some(&Shape::Active(Op::AddU, v0_in, s2_val, 0))
        );

        // 2. Clobbering the high half s3 of s[2:3] before v_cndmask_b32 alters
        // lanes 32..63; wave simulation must not treat s[2:3] as the intact mask.
        let clobbered_hi = wave64_sum_section_words(
            &[
                sop1(0, 0x25, 193),    // s_or_saveexec_b64 s[0:1], -1
                sop2(2, 0x0F, 0, 193), // s_and_b64 s[2:3], s[0:1], -1
                sop1(3, 0x03, 128),    // s_mov_b32 s3, 0
            ],
            2,
            &[sop1(EXEC, 0x04, 0)],
        );
        let p = build(&clobbered_hi, true);
        let insts = &p.blocks[0].insts;
        let s4_val = insts[insts.len() - 3].write(Reg::S(4)).unwrap();
        let sec = &sections(&p, 0)[&0];
        assert_eq!(sec.shapes.get(&s4_val), None);

        // 3. A 32-bit s_and_b32 in wave64 writes only s2 and leaves s3 unrelated.
        let narrow_and = wave64_sum_section_words(
            &[
                sop1(0, 0x25, 193),    // s_or_saveexec_b64 s[0:1], -1
                sop2(2, 0x0E, 0, 193), // s_and_b32 s2, s0, -1
            ],
            2,
            &[sop1(EXEC, 0x04, 0)],
        );
        let p = build(&narrow_and, true);
        let insts = &p.blocks[0].insts;
        let s4_val = insts[insts.len() - 3].write(Reg::S(4)).unwrap();
        let sec = &sections(&p, 0)[&0];
        assert_eq!(sec.shapes.get(&s4_val), None);

        // 4. Wrong-width saveexec: s_or_saveexec_b32 in wave64 leaves EXEC_HI
        // untouched and must not open a whole-wave section (and vice versa).
        let w32_in_64 = [sop1(0, 0x3D, 193), sop1(EXEC, 0x03, 0)];
        assert!(sections(&build(&w32_in_64, true), 0).is_empty());
        let w64_in_32 = [sop1(0, 0x25, 193), sop1(EXEC, 0x04, 0)];
        assert!(sections(&build(&w64_in_32, false), 0).is_empty());

        // 5. Partial EXEC_LO restore (s_mov_b32 exec_lo, s0) in wave64 must not
        // close the section before s_mov_b64 exec, s[0:1].
        let partial_restore = [
            sop1(0, 0x25, 193),  // 0: s_or_saveexec_b64 s[0:1], -1
            sop1(EXEC, 0x03, 0), // 1: s_mov_b32 exec_lo, s0 (partial!)
            sop1(EXEC, 0x04, 0), // 2: s_mov_b64 exec, s[0:1] (full restore)
        ];
        let p = build(&partial_restore, true);
        assert_eq!(sections(&p, 0)[&0].end, 2);

        // 6. Clobbering saved high half s1 before s_mov_b64 exec, s[0:1] is not
        // a restore of the saved EXEC mask.
        let clobbered_saved_hi = [
            sop1(0, 0x25, 193),  // s_or_saveexec_b64 s[0:1], -1
            sop1(1, 0x03, 128),  // s_mov_b32 s1, 0
            sop1(EXEC, 0x04, 0), // s_mov_b64 exec, s[0:1]
        ];
        assert!(sections(&build(&clobbered_saved_hi, true), 0).is_empty());
    }

    #[cfg(windows)]
    #[test]
    fn end_to_end_permlanex16_and_wave64_high_mask_do_not_miscompile() {
        use super::super::{HostShader, host, validator};
        use crate::dxbc::Container;
        use crate::native::{
            ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo,
            UsedResource,
        };
        use crate::{io, resource, structure};
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS1,
        };

        let compile = |wave64: bool, body: &[u32], out_vreg: u8| {
            let mut w = vec![
                sop1(8, 0x03, 0),   // s_mov_b32 s8, s0 (table base)
                sop1(9, 0x03, 128), // s_mov_b32 s9, 0
                // s_load_dwordx4 s[12:15], s[8:9], 0
                0x3D << 26 | 0x02 << 18 | (12u32) << 6 | (8u32 >> 1),
                125 << 25,
                // v_lshlrev_b32 v5, 2, v0 (byte offset = lane * 4)
                vop2(5, 0x1A, 128 + 2, 0),
                // v_add_nc_u32 v0, 1, v0 (input value = lane + 1)
                vop2(0, 0x25, 129, 0),
            ];
            w.extend_from_slice(body);
            // buffer_store_dword v[out_vreg], v5, s[12:15], 0 offen
            w.push((0x38 << 26) | (0x1C << 18) | (1 << 12));
            w.push((0x80 << 24) | ((12u32 >> 2) & 0x1F) << 16 | (u32::from(out_vreg) << 8) | 5);
            w.push(S_ENDPGM);
            let code: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            let native = NativeProgram {
                stage: HwStage::Cs,
                wave64,
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
                    threads: [if wave64 { 64 } else { 32 }, 1, 1],
                }),
                code,
            };
            let p = crate::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
            let st = structure::structure(&p).unwrap();
            let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = io::raise(&p, &native).unwrap();
            let container = crate::dxbc::build(&[]);
            let c = Container::parse(&container).unwrap();
            super::super::emit(&c, &native, &p, &st, &io, &res)
        };

        // 1. Wave32 v_permlanex16_b32 under narrowed EXEC = 0xFFFF_0000:
        // FI=0 (opsel=0 and opsel=2) must NOT silently compile into WavePrefixSum.
        for opsel in [0u8, 2] {
            let body = wave32_permlanex16_prefix_words(0xFFFF_0000, 0, opsel);
            assert!(
                compile(false, &body, 1).is_err(),
                "FI=0 v_permlanex16_b32 (opsel={opsel}) under narrowed EXEC must not miscompile into WavePrefixSum"
            );
        }

        let _one = super::super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let host = host::Host::new(false).or_else(|_| host::Host::new(true)).ok();
        let mut validator =
            validator::Validator::load(validator::Validator::default_path().as_deref()).ok();
        let mut readback = |label: &str, shader: &HostShader, expected: &[u32]| {
            if let Some(validator) = &mut validator {
                validator
                    .validate(&mut shader.dxil.clone())
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
            }
            let Some(host) = &host else {
                eprintln!("{label}: no SM 6.6 device; skipped readback");
                return;
            };
            let lanes = expected.len() as u32;
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
            // SAFETY: the query writes one OPTIONS1 structure.
            unsafe {
                host.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS1,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .expect("query wave capabilities");
            }
            if !options.WaveOps.as_bool()
                || options.WaveLaneCountMin > lanes
                || options.WaveLaneCountMax < lanes
            {
                eprintln!("{label}: adapter cannot execute Wave{lanes}; skipped readback");
                return;
            }
            let (root, pso) = host
                .compute_pipeline(shader)
                .unwrap_or_else(|e| panic!("{label}: create pipeline: {e}"));
            let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &host::Dispatch {
                        roots: vec![(0, host::Root::Constants(ud))],
                        views: &[host::View::RawUav(lanes)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch: {e}"));
            assert_eq!(&out[0][..expected.len()], expected, "{label}");
            println!("{label}: read back all {lanes} lanes");
        };

        // FI=1 fetches the inactive lane for either BOUND_CTRL setting.
        let prefixes: Vec<u32> = (1..=32u32).map(|k| k * (k + 1) / 2).collect();
        for opsel in [1, 3] {
            let body = wave32_permlanex16_prefix_words(0xFFFF_0000, 0, opsel);
            let shader = compile(false, &body, 1).expect("FI=1 prefix sum translates");
            readback(
                &format!("Wave32 FI=1 BOUND_CTRL={}", opsel >> 1),
                &shader,
                &prefixes,
            );
        }

        // 2. Wave64 whole-wave reduction where high half s3 of mask s[2:3] is
        // zeroed before v_cndmask_b32: must NOT silently compile into an
        // unmasked 64-lane WaveActiveSum (which computed 2080 instead of 528).
        let mut valid_w64 = wave64_sum_section_words(
            &[sop1(0, 0x25, 193), sop2(2, 0x0F, 0, 193)],
            2,
            &[sop1(EXEC, 0x04, 0)],
        );
        valid_w64.push(vop1(6, 0x01, 4)); // v_mov_b32 v6, s4
        let shader = compile(true, &valid_w64, 6).expect("intact Wave64 reduction translates");
        readback("Wave64 intact reduction", &shader, &[2080; 64]);

        let mut clobbered_w64 = wave64_sum_section_words(
            &[
                sop1(0, 0x25, 193),
                sop2(2, 0x0F, 0, 193),
                sop1(3, 0x03, 128), // s_mov_b32 s3, 0 (clears lanes 32..63!)
            ],
            2,
            &[sop1(EXEC, 0x04, 0)],
        );
        clobbered_w64.push(vop1(6, 0x01, 4)); // v_mov_b32 v6, s4
        assert!(
            compile(true, &clobbered_w64, 6).is_err(),
            "Wave64 reduction with clobbered mask high half must not miscompile into full 64-lane WaveActiveSum"
        );
    }
}
