//! Rounding and output denormal modes for FP16 results.
//!
//! f16 results where the program keeps no 16-bit output denormals: MODE
//! bit 7 (bit 3 of `s_denorm_mode`'s immediate) clear. Measured on gfx1036
//! (RDNA2): the arithmetic gives a zero of its sign for a result below the
//! smallest normal half once rounded to a half's precision; the minimum,
//! maximum and median keep their result.

use super::{Bin, Lower, V, pred};
use crate::gfx10::{Fields, Operand};
use crate::ir::{Inst, Program};

/// The 16/64-bit round mode (MODE bits 3:2: 0 nearest even, 3 toward zero)
/// after `inst`, from `mode` before it (`None`: not known).
pub(super) fn round16_after(inst: &Inst, mode: Option<u8>) -> Option<u8> {
    match (inst.name(), &inst.fields) {
        ("s_round_mode", Fields::Sopp { imm }) => Some((imm >> 2 & 3) as u8),
        (
            "s_setreg_imm32_b32" | "s_setreg_b32",
            Fields::Sop {
                src, imm: Some(h), ..
            },
        ) => {
            let (id, off, size) = (h & 0x3F, h >> 6 & 0x1F, (h >> 11 & 0x1F) + 1);
            if id != 1 || off >= 4 || off + size <= 2 {
                return mode;
            }
            match src.first() {
                Some(&Operand::Literal(x)) if off <= 2 && off + size >= 4 => {
                    Some((x >> (2 - off) & 3) as u8)
                }
                _ => None,
            }
        }
        _ => mode,
    }
}

/// The 16/64-bit round mode each block starts with: nearest even at the
/// program's start, then as the MODE writes on the way set it; `None` where
/// paths that set it differently join (or a write is not a constant).
pub(super) fn round16(p: &Program) -> Vec<Option<u8>> {
    mode_at_blocks(p, Some(0), round16_after)
}

pub(super) fn mode_at_blocks<T: Copy + Eq>(
    p: &Program,
    initial: Option<T>,
    after: fn(&Inst, Option<T>) -> Option<T>,
) -> Vec<Option<T>> {
    // None: not reached yet; Some(m): the mode (m None: not known).
    let mut entry = vec![None; p.blocks.len()];
    entry[0] = Some(initial);
    let order = crate::ir::reverse_postorder(&p.blocks);
    loop {
        let mut changed = false;
        for &b in &order {
            let Some(mut m) = entry[b] else { continue };
            for inst in &p.blocks[b].insts {
                m = after(inst, m);
            }
            for s in p.blocks[b].exit.successors() {
                let new = match entry[s] {
                    None => Some(m),
                    Some(old) if old == m => continue,
                    Some(_) => Some(None),
                };
                if entry[s] != new {
                    entry[s] = new;
                    changed = true;
                }
            }
        }
        if !changed {
            return entry.into_iter().map(Option::flatten).collect();
        }
    }
}


/// Whether 16/64-bit results keep their denormals (MODE bit 7, bit 3 of
/// `s_denorm_mode`'s immediate) after `inst`, from `keep` before it.
pub(super) fn denorm16_after(inst: &Inst, keep: Option<bool>) -> Option<bool> {
    match (inst.name(), &inst.fields) {
        ("s_denorm_mode", Fields::Sopp { imm }) => Some(imm & 8 != 0),
        (
            "s_setreg_imm32_b32" | "s_setreg_b32",
            Fields::Sop {
                src, imm: Some(h), ..
            },
        ) => {
            let (id, off, size) = (h & 0x3F, h >> 6 & 0x1F, (h >> 11 & 0x1F) + 1);
            if id != 1 || off > 7 || off + size <= 7 {
                return keep;
            }
            match src.first() {
                Some(&Operand::Literal(x)) => Some(x >> (7 - off) & 1 != 0),
                _ => None,
            }
        }
        _ => keep,
    }
}

/// Whether 16/64-bit results keep their denormals as each block starts:
/// kept at the program's start (the console's float mode), then as the MODE
/// writes on the way set it, as `round16` follows the round mode.
pub(super) fn denorm16(p: &Program) -> Vec<Option<bool>> {
    mode_at_blocks(p, Some(true), denorm16_after)
}

impl Lower<'_> {
    /// `h`, an f16 result narrowed from `f` in the round mode `mode` (MODE
    /// bits 3:2; `None`: not known, as nearest even), where the program keeps
    /// no 16-bit output denormals (MODE bit 7 clear): then a result that is
    /// below the smallest normal half once rounded to a half's precision (as
    /// if the exponent had no bottom) is a zero of its sign. Measured on
    /// gfx1036 for `v_cvt_f16_f32` in nearest even: 0x387fe000 (2^-14 -
    /// 2^-25) gives 0x0000, 0x387ff000 gives 0x0400. The arithmetic flushes
    /// so; the minimum, maximum and median do not.
    pub(super) fn flush16(&mut self, h: V, f: V, mode: Option<u8>) -> V {
        if self.keep16 != Some(false) {
            return h;
        }
        let c = self.cur;
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let x = self.as_int(f);
        let m = k(self, 0x7FFF_FFFF);
        let a = self.s.m.bin(c, Bin::And, x, m);
        // The smallest magnitude that rounds to 2^-14 or more.
        let down = k(self, 0x3880_0000);
        let even = k(self, 0x387F_F000);
        let up = k(self, 0x387F_E001);
        let limit = match mode {
            Some(3) => down,
            Some(d @ (1 | 2)) => {
                // Toward +inf rounds a positive magnitude up, -inf a negative one.
                let z = k(self, 0);
                let p = if d == 1 { pred::SGE } else { pred::SLT };
                let grows = self.s.m.cmp(c, p, x, z);
                self.s.m.select(c, grows, up, down)
            }
            _ => even,
        };
        let tiny = self.s.m.cmp(c, pred::IULT, a, limit);
        let sign = k(self, 0x8000);
        let signed_zero = self.s.m.bin(c, Bin::And, h, sign);
        self.s.m.select(c, tiny, signed_zero, h)
    }

    /// [`Self::flush16`] for the single-rounding `v_fma_f16`: `rounded` (no
    /// sign), or zero where the exact result, the f32 sum of magnitude bits
    /// `magnitude` with its `error` (positive: the exact value is larger), is
    /// below 2^-14 - 2^-26.
    pub(super) fn flush16_exact(&mut self, rounded: V, magnitude: V, error: V) -> V {
        if self.keep16 != Some(false) {
            return rounded;
        }
        let c = self.cur;
        let m = &mut self.s.m;
        let (limit, zero, fzero) = (m.i32(0x387F_F000), m.i32(0), m.f32(0));
        let below = m.cmp(c, pred::IULT, magnitude, limit);
        let at = m.cmp(c, pred::EQ, magnitude, limit);
        let short = m.cmp(c, pred::OLT, error, fzero);
        let at = m.bin(c, Bin::And, at, short);
        let below = m.bin(c, Bin::Or, below, at);
        m.select(c, below, zero, rounded)
    }
}
