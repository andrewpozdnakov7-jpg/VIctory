//! d16 format loads of UNORM / SNORM channels: the f16 RDNA2 gives, rounded
//! once to nearest even (see [`Lower::norm_f16`]).

use super::{Lower, Numeric, Why, buffer_format};
use crate::emit::bitcode::{Attr, Bin, Cast, V, pred};

impl Lower<'_> {
    /// A UNORM / SNORM channel `x` (its decoded f32 bits) of `w` bits as
    /// f16, rounded once to nearest even as RDNA2 does (`legacyF32ToF16` of
    /// the f32 rounds twice, the second time toward zero, leaving about half
    /// the codes a unit low). `code / (2^w - 1)` repeats the w-bit code in
    /// binary, so its first 32 bits, `(code << 32) / (2^w - 1)`, decide the
    /// rounding and a tie cannot happen. SNORM is `|v| / (2^(w-1) - 1)`.
    pub(super) fn norm_f16(&mut self, x: V, w: u32, snorm: bool) -> V {
        let c = self.cur;
        let bits = if snorm { w - 1 } else { w };
        let d = (1u32 << bits) - 1;
        let scale = self.s.m.f32((d as f32).to_bits());
        let y = self.normalized_code(x, scale);
        // The binary fraction repeats the entire field. Multiply to place
        // each complete copy, then fill any remaining low bits with its prefix.
        // 0 and the all-ones field are selected explicitly below.
        let copies = self.s.m.i32(u32::MAX / d);
        let mut p = self.s.m.bin(c, Bin::Mul, y, copies);
        let tail = 32 % bits;
        if tail != 0 {
            let shift = self.s.m.i32(bits - tail);
            let prefix = self.s.m.bin(c, Bin::LShr, y, shift);
            p = self.s.m.bin(c, Bin::Or, p, prefix);
        }
        let limit = self.s.m.i32(d);
        self.normalized_half(x, y, p, limit, snorm)
    }

    /// A typed view supplies 16-bit UNORM or SNORM at run time.
    fn norm16_from_view(&mut self, x: V, signed: V) -> V {
        let c = self.cur;
        let (u16_max, s16_max) = (65535u32, 32767u32);
        let scale_u = self.s.m.f32((u16_max as f32).to_bits());
        let scale_s = self.s.m.f32((s16_max as f32).to_bits());
        let scale = self.s.m.select(c, signed, scale_s, scale_u);
        let code = self.normalized_code(x, scale);

        let repeats_u = self.s.m.i32(u32::MAX / u16_max);
        let repeats_s = self.s.m.i32(u32::MAX / s16_max);
        let repeats = self.s.m.select(c, signed, repeats_s, repeats_u);
        let fraction = self.s.m.bin(c, Bin::Mul, code, repeats);
        // A 15-bit magnitude leaves two bits after the full copies.
        let k13 = self.s.m.i32(13);
        let prefix = self.s.m.bin(c, Bin::LShr, code, k13);
        let zero = self.s.m.i32(0);
        let prefix = self.s.m.select(c, signed, prefix, zero);
        let fraction = self.s.m.bin(c, Bin::Or, fraction, prefix);
        let limit_u = self.s.m.i32(u16_max);
        let limit_s = self.s.m.i32(s16_max);
        let limit = self.s.m.select(c, signed, limit_s, limit_u);
        // Unsigned normalized input already has its sign bit clear.
        self.normalized_half(x, code, fraction, limit, true)
    }

    fn normalized_code(&mut self, x: V, scale: V) -> V {
        let c = self.cur;
        let mag = self.s.m.i32(0x7fff_ffff);
        let ax = self.s.m.bin(c, Bin::And, x, mag);
        let af = self.as_float(ax);
        let scaled = self.s.m.bin(c, Bin::Mul, af, scale);
        let rounded = self.unary(26, scaled);
        self.s.m.cast(c, Cast::FpToUi, rounded, self.s.t.i32)
    }

    fn normalized_half(&mut self, x: V, y: V, p: V, limit: V, snorm: bool) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let lz = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 33, &[p]);
        // normal: 11 bits after the leading one and the round bit
        let k14 = self.s.m.i32(14);
        let e = self.s.m.bin(c, Bin::Sub, k14, lz);
        let pn = self.s.m.bin(c, Bin::Shl, p, lz);
        let k20 = self.s.m.i32(20);
        let b = self.s.m.bin(c, Bin::LShr, pn, k20);
        let one = self.s.m.i32(1);
        let m = self.s.m.bin(c, Bin::LShr, b, one);
        let r = self.s.m.bin(c, Bin::And, b, one);
        let m = self.s.m.bin(c, Bin::Add, m, r);
        let em1 = self.s.m.bin(c, Bin::Sub, e, one);
        let k10 = self.s.m.i32(10);
        let eh = self.s.m.bin(c, Bin::Shl, em1, k10);
        let normal = self.s.m.bin(c, Bin::Add, eh, m);
        // below 2^-14: a subnormal half
        let k8 = self.s.m.i32(8);
        let k7 = self.s.m.i32(7);
        let s8 = self.s.m.bin(c, Bin::LShr, p, k8);
        let r7 = self.s.m.bin(c, Bin::LShr, p, k7);
        let r7 = self.s.m.bin(c, Bin::And, r7, one);
        let sub = self.s.m.bin(c, Bin::Add, s8, r7);
        let is_normal = self.s.m.cmp(c, pred::SGE, e, one);
        let h = self.s.m.select(c, is_normal, normal, sub);
        // 0 and 1.0 (the latter's expansion has no leading one in 32 bits' reach)
        let zero = self.s.m.i32(0);
        let is0 = self.s.m.cmp(c, pred::EQ, y, zero);
        let h = self.s.m.select(c, is0, zero, h);
        let full = self.s.m.cmp(c, pred::IUGE, y, limit);
        let h1 = self.s.m.i32(0x3C00);
        let h = self.s.m.select(c, full, h1, h);
        if !snorm {
            return h;
        }
        let k31 = self.s.m.i32(31);
        let sign = self.s.m.bin(c, Bin::LShr, x, k31);
        let k15 = self.s.m.i32(15);
        let sign = self.s.m.bin(c, Bin::Shl, sign, k15);
        self.s.m.bin(c, Bin::Or, h, sign)
    }

    /// The instruction supplies the channel widths and number format for MTBUF.
    pub(super) fn d16_channels(&mut self, format: u8, channels: Vec<V>) -> Result<Vec<V>, Why> {
        let (widths, numeric) = buffer_format(format).ok_or("a buffer format")?;
        let c = self.cur;
        let mut halves = Vec::with_capacity(channels.len());
        for (k, x) in channels.into_iter().enumerate() {
            let width = widths.get(k).copied().unwrap_or(0);
            let half = match numeric {
                Numeric::Uint => {
                    let max = self.s.m.i32(0xffff);
                    let below = self.s.m.cmp(c, pred::IULT, x, max);
                    self.s.m.select(c, below, x, max)
                }
                Numeric::Sint => {
                    let lo = self.s.m.i32((-32768i32) as u32);
                    let hi = self.s.m.i32(32767);
                    let below = self.s.m.cmp(c, pred::SLT, x, lo);
                    let x = self.s.m.select(c, below, lo, x);
                    let above = self.s.m.cmp(c, pred::SGT, x, hi);
                    let x = self.s.m.select(c, above, hi, x);
                    let mask = self.s.m.i32(0xffff);
                    self.s.m.bin(c, Bin::And, x, mask)
                }
                Numeric::Unorm | Numeric::Snorm if (2..=16).contains(&width) => {
                    self.norm_f16(x, width, matches!(numeric, Numeric::Snorm))
                }
                Numeric::Uscaled | Numeric::Sscaled => {
                    let f = self.as_float(x);
                    self.narrow16_bits(f, false)
                }
                _ => {
                    let f = self.as_float(x);
                    self.s.op(c, "legacyF32ToF16", Attr::ReadNone, self.s.t.i32, 130, &[f])
                }
            };
            halves.push(half);
        }
        Ok(self.pack_d16(&halves))
    }

    /// A buffer view supplies the number format for MUBUF.
    pub(super) fn d16_view_channels(&mut self, channels: Vec<V>, kinds: Option<[V; 3]>) -> Vec<V> {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let mut halves = Vec::with_capacity(channels.len());
        for &f in &channels {
            let truncate = self.s.op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
            let half = if let Some([integer, normalized, _]) = kinds {
                let rounded = self.narrow16_bits(f, false);
                let half = self.s.m.select(c, normalized, rounded, truncate);
                let bits = self.as_int(f);
                let mask = self.s.m.i32(0xffff);
                let low = self.s.m.bin(c, Bin::And, bits, mask);
                self.s.m.select(c, integer, low, half)
            } else {
                truncate
            };
            halves.push(half);
        }
        let Some([integer, signed_norm, normal16]) = kinds else {
            return halves;
        };
        // The bit selecting exact normalized widths also denotes SINT when
        // the view is integer. Enter this branch only for a normalized view.
        let t = self.s.m.bool(true);
        let float = self.s.m.bin(c, Bin::Xor, integer, t);
        let exact = self.s.m.bin(c, Bin::And, normal16, float);
        let at = self.begin_guarded(Some(exact));
        let pairs = channels
            .into_iter()
            .zip(halves)
            .map(|(f, fallback)| {
                let bits = self.as_int(f);
                (self.norm16_from_view(bits, signed_norm), fallback)
            })
            .collect::<Vec<_>>();
        self.end_guarded(at, &pairs)
    }

    /// Pack two halfwords per dword with zero in an odd last high half.
    pub(super) fn pack_d16(&mut self, halves: &[V]) -> Vec<V> {
        let c = self.cur;
        halves
            .chunks(2)
            .map(|pair| {
                let lo = pair[0];
                if let Some(&hi) = pair.get(1) {
                    let k16 = self.s.m.i32(16);
                    let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                    self.s.m.bin(c, Bin::Or, lo, hi)
                } else {
                    lo
                }
            })
            .collect()
    }
}
