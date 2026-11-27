//! FP32 median lowering.

use super::{Attr, Bin, Lower, V, pred};

impl Lower<'_> {
    /// f32 median using integer-ordered keys, with the same denormal,
    /// signed-zero and NaN rules as [`Self::med3`].
    pub(super) fn med3_f32(&mut self, a: V, b: V, third: V) -> V {
        let c = self.cur;
        let k = |l: &mut Self, x| l.s.m.i32(x);
        let (k31, mag, inf) = (k(self, 31), k(self, 0x7FFF_FFFF), k(self, 0x7F80_0000));
        // Reversing the negative half of the f32 bit patterns gives signed
        // integer keys: -inf < ... < -0 < +0 < ... < +inf. The transform is
        // its own inverse; no floating-point arithmetic rounds the result.
        let key = |l: &mut Self, x| {
            let sign = l.s.m.bin(c, Bin::AShr, x, k31);
            let flip = l.s.m.bin(c, Bin::And, sign, mag);
            l.s.m.bin(c, Bin::Xor, x, flip)
        };
        let input = |l: &mut Self, f| {
            let x = l.as_int(f);
            let abs = l.s.m.bin(c, Bin::And, x, mag);
            let nan = l.s.m.cmp(c, pred::IUGT, abs, inf);
            let sign = l.s.m.bin(c, Bin::AShr, x, k31);
            let flip = l.s.m.bin(c, Bin::And, sign, mag);
            let ordered = l.s.m.bin(c, Bin::Xor, x, flip);
            // A flushed zero's key is exactly the arithmetic sign mask.
            // NaNs sort above every number, independent of sign or payload.
            let min_normal = l.s.m.i32(0x0080_0000);
            let den = l.s.m.cmp(c, pred::IULT, abs, min_normal);
            let ordered = l.s.m.select(c, den, sign, ordered);
            (l.s.m.select(c, nan, mag, ordered), nan)
        };
        let ((ka, na), (kb, nb), (kc, nc)) = (input(self, a), input(self, b), input(self, third));
        let minmax = |l: &mut Self, max: bool, x, y| {
            l.s.op(
                c,
                "binary.i32",
                Attr::ReadNone,
                l.s.t.i32,
                if max { 37 } else { 38 },
                &[x, y],
            )
        };
        let low = minmax(self, false, ka, kb);
        let high = minmax(self, true, ka, kb);
        let t = minmax(self, false, high, kc);
        let median = minmax(self, true, low, t);
        // Any NaN selects the minimum of the remaining numbers instead.
        let least = minmax(self, false, low, kc);
        let ab = self.s.m.bin(c, Bin::Or, na, nb);
        let any = self.s.m.bin(c, Bin::Or, ab, nc);
        let r = self.s.m.select(c, any, least, median);
        let r = key(self, r);
        let r = self.as_float(r);
        // Only three NaNs can leave the minimum at the sentinel. Preserve
        // the original minNum chain's all-NaN result in that case.
        let all = self.s.m.cmp(c, pred::EQ, least, mag);
        let lo = self.binary(36, a, b);
        let nan = self.binary(36, lo, third);
        self.s.m.select(c, all, nan, r)
    }
}
