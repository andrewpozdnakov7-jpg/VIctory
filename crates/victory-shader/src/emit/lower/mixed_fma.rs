//! Single rounding of mixed-precision FMA results to binary16.

use super::super::bitcode::{Attr, Bin, Cast, V, pred};
use super::Lower;

impl<'a> Lower<'a> {
    /// Shift a pair of dwords right and fold any discarded bits into bit 0.
    fn wide_shr_sticky(&mut self, [lo, hi]: [V; 2], shift: V) -> [V; 2] {
        let c = self.cur;
        let m = &mut self.s.m;
        let (zero, one, k31, k32, k64) = (m.i32(0), m.i32(1), m.i32(31), m.i32(32), m.i32(64));
        let within_word = m.cmp(c, pred::IULT, shift, k32);
        let within_pair = m.cmp(c, pred::IULT, shift, k64);
        let right = m.bin(c, Bin::And, shift, k31);
        let left = m.bin(c, Bin::Sub, zero, right);
        let left = m.bin(c, Bin::And, left, k31);
        let has_crossing = m.cmp(c, pred::NE, right, zero);
        let crossing = m.bin(c, Bin::Shl, hi, left);
        let crossing = m.select(c, has_crossing, crossing, zero);
        let lower = m.bin(c, Bin::LShr, lo, right);
        let lower = m.bin(c, Bin::Or, lower, crossing);
        let upper = m.bin(c, Bin::LShr, hi, right);
        let shifted_lo = m.select(c, within_word, lower, upper);
        let shifted_lo = m.select(c, within_pair, shifted_lo, zero);
        let shifted_hi = m.select(c, within_word, upper, zero);

        let mask = m.bin(c, Bin::Shl, one, right);
        let mask = m.bin(c, Bin::Sub, mask, one);
        let partial = m.select(c, within_word, lo, hi);
        let discarded_bits = m.bin(c, Bin::And, partial, mask);
        let discarded_lo = m.select(c, within_word, zero, lo);
        let discarded = m.bin(c, Bin::Or, discarded_bits, discarded_lo);
        let all = m.bin(c, Bin::Or, hi, lo);
        let discarded = m.select(c, within_pair, discarded, all);
        let sticky = m.cmp(c, pred::NE, discarded, zero);
        let sticky = m.cast(c, Cast::ZExt, sticky, self.s.t.i32);
        [m.bin(c, Bin::Or, shifted_lo, sticky), shifted_hi]
    }

    fn wide_negate_if(&mut self, [lo, hi]: [V; 2], negate: V) -> [V; 2] {
        let c = self.cur;
        let m = &mut self.s.m;
        let zero = m.i32(0);
        let borrow = m.cmp(c, pred::NE, lo, zero);
        let borrow = m.cast(c, Cast::ZExt, borrow, self.s.t.i32);
        let neg_lo = m.bin(c, Bin::Sub, zero, lo);
        let neg_hi = m.bin(c, Bin::Sub, zero, hi);
        let neg_hi = m.bin(c, Bin::Sub, neg_hi, borrow);
        [
            m.select(c, negate, neg_lo, lo),
            m.select(c, negate, neg_hi, hi),
        ]
    }

    /// Round a nonnegative value in units of `2^(exponent - 61)` to f16.
    /// Also report whether both input dwords were zero.
    fn round_wide_to_f16(&mut self, [lo, hi]: [V; 2], exponent: V) -> (V, V) {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let m = &mut self.s.m;
        let (zero, one, k31, k32) = (m.i32(0), m.i32(1), m.i32(31), m.i32(32));
        let hi_nonzero = m.cmp(c, pred::NE, hi, zero);
        let head = m.select(c, hi_nonzero, hi, lo);
        let tail = m.select(c, hi_nonzero, lo, zero);
        let leading = self
            .s
            .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 33, &[head]); // FirstbitHi
        let m = &mut self.s.m;
        let leading = m.bin(c, Bin::And, leading, k31); // FirstbitHi(0) is -1.
        let head_left = m.bin(c, Bin::Shl, head, leading);
        let right = m.bin(c, Bin::Sub, zero, leading);
        let right = m.bin(c, Bin::And, right, k31);
        let crossing = m.bin(c, Bin::LShr, tail, right);
        let shifted = m.cmp(c, pred::NE, leading, zero);
        let crossing = m.select(c, shifted, crossing, zero);
        let top = m.bin(c, Bin::Or, head_left, crossing);
        let tail = m.bin(c, Bin::Shl, tail, leading);
        let (high_offset, low_offset) = (m.i32(2), m.i32(-30i32 as u32));
        let offset = m.select(c, hi_nonzero, high_offset, low_offset);
        let exp = m.bin(c, Bin::Add, exponent, offset);
        let exp = m.bin(c, Bin::Sub, exp, leading);

        // Keep the guard bit below the 11 normal bits. For subnormals move
        // the guard up to bit 31 at exponent -25, the midpoint above zero.
        let (k20, k6, min_normal, min_tie) = (
            m.i32(20),
            m.i32(6),
            m.i32(-14i32 as u32),
            m.i32(-25i32 as u32),
        );
        let normal = m.cmp(c, pred::SGE, exp, min_normal);
        let tiny = m.cmp(c, pred::SLT, exp, min_tie);
        let denorm_shift = m.bin(c, Bin::Sub, k6, exp);
        let shift = m.select(c, normal, k20, denorm_shift);
        let shift = m.select(c, tiny, k31, shift);
        let with_guard = m.bin(c, Bin::LShr, top, shift);
        let kept = m.bin(c, Bin::LShr, with_guard, one);
        let right = m.bin(c, Bin::Sub, k32, shift);
        let below_guard = m.bin(c, Bin::Shl, top, right);
        let below_guard = m.bin(c, Bin::Or, below_guard, tail);
        let sticky = m.cmp(c, pred::NE, below_guard, zero);
        let sticky = m.cast(c, Cast::ZExt, sticky, i32t);
        let odd_or_sticky = m.bin(c, Bin::Or, kept, sticky);
        let guard = m.bin(c, Bin::And, with_guard, one);
        let increment = m.bin(c, Bin::And, guard, odd_or_sticky);
        let rounded = m.bin(c, Bin::Add, kept, increment);

        // Adding the rounded significand also carries into the exponent at
        // the subnormal/normal boundary and at overflow.
        let (k10, bias_minus_one, max_exp, inf) = (m.i32(10), m.i32(14), m.i32(15), m.i32(0x7C00));
        let exp_field = m.bin(c, Bin::Add, exp, bias_minus_one);
        let exp_field = m.select(c, normal, exp_field, zero);
        let exp_field = m.bin(c, Bin::Shl, exp_field, k10);
        let half = m.bin(c, Bin::Add, exp_field, rounded);
        let overflow = m.cmp(c, pred::SGT, exp, max_exp);
        let half = m.select(c, overflow, inf, half);
        let is_zero = m.cmp(c, pred::EQ, head, zero);
        let zero_result = m.bin(c, Bin::Or, is_zero, tiny);
        (m.select(c, zero_result, zero, half), is_zero)
    }

    /// Fused multiply-add of three binary32 bit patterns, rounded directly
    /// to binary16, with binary32 denormals read as signed zeros.
    /// DXIL FMad may round the product separately.
    pub(super) fn fma_mix_f16(&mut self, [a, b, addend]: [V; 3], clamp: bool) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let m = &mut self.s.m;
        let (zero, one, k23, bias) = (m.i32(0), m.i32(1), m.i32(23), m.i32(127));
        let (abs_mask, frac_mask, hidden) =
            (m.i32(0x7FFF_FFFF), m.i32(0x7F_FFFF), m.i32(0x80_0000));
        let magnitudes = [a, b, addend].map(|x| m.bin(c, Bin::And, x, abs_mask));
        let [(sig_a, exp_a), (sig_b, exp_b), (sig_c, exp_c)] = magnitudes.map(|mag| {
            let raw_exp = m.bin(c, Bin::LShr, mag, k23);
            let has_exponent = m.cmp(c, pred::NE, raw_exp, zero);
            let exp = m.select(c, has_exponent, raw_exp, one);
            let exp = m.bin(c, Bin::Sub, exp, bias);
            let frac = m.bin(c, Bin::And, mag, frac_mask);
            let sig = m.bin(c, Bin::Or, frac, hidden);
            (m.select(c, has_exponent, sig, zero), exp)
        });

        // Each significand has at most 24 bits, so the cross sum fits in i32.
        // Split the exact 48-bit product across two i32s to avoid Int64.
        let (k16, mask16) = (m.i32(16), m.i32(0xFFFF));
        let a0 = m.bin(c, Bin::And, sig_a, mask16);
        let a1 = m.bin(c, Bin::LShr, sig_a, k16);
        let b0 = m.bin(c, Bin::And, sig_b, mask16);
        let b1 = m.bin(c, Bin::LShr, sig_b, k16);
        let p00 = m.bin(c, Bin::Mul, a0, b0);
        let p01 = m.bin(c, Bin::Mul, a0, b1);
        let p10 = m.bin(c, Bin::Mul, a1, b0);
        let cross = m.bin(c, Bin::Add, p01, p10);
        let p11 = m.bin(c, Bin::Mul, a1, b1);
        let carry_in = m.bin(c, Bin::Shl, cross, k16);
        let product_lo = m.bin(c, Bin::Add, p00, carry_in);
        let wrapped = m.cmp(c, pred::IULT, product_lo, p00);
        let wrapped = m.cast(c, Cast::ZExt, wrapped, i32t);
        let cross_hi = m.bin(c, Bin::LShr, cross, k16);
        let product_hi = m.bin(c, Bin::Add, p11, cross_hi);
        let product_hi = m.bin(c, Bin::Add, product_hi, wrapped);

        // Put the highest possible bit of each significand at bit 61 so the
        // signed sum fits in two dwords. The implied unit is 2^(exponent-61).
        let (k6, k14, k18) = (m.i32(6), m.i32(14), m.i32(18));
        let product_hi = m.bin(c, Bin::Shl, product_hi, k14);
        let crossing = m.bin(c, Bin::LShr, product_lo, k18);
        let product_hi = m.bin(c, Bin::Or, product_hi, crossing);
        let product_lo = m.bin(c, Bin::Shl, product_lo, k14);
        let addend_hi = m.bin(c, Bin::Shl, sig_c, k6);
        let product_exp = m.bin(c, Bin::Add, exp_a, exp_b);
        let product_exp = m.bin(c, Bin::Add, product_exp, one);
        let product_first = m.cmp(c, pred::SGE, product_exp, exp_c);
        let exponent = m.select(c, product_first, product_exp, exp_c);
        let other_exp = m.select(c, product_first, exp_c, product_exp);
        let shift = m.bin(c, Bin::Sub, exponent, other_exp);
        let base_lo = m.select(c, product_first, product_lo, zero);
        let base_hi = m.select(c, product_first, product_hi, addend_hi);
        let to_shift = [
            m.select(c, product_first, zero, product_lo),
            m.select(c, product_first, addend_hi, product_hi),
        ];

        // The unshifted term is even. An inexact alignment makes the sum odd
        // and leaves it within one unit of the exact sum. At exponent <= 35,
        // all half midpoints are even in these units (at most 2^-26).
        // Above that exponent the unshifted term starts at bit 60 or 61, so a
        // shift that loses bits (>= 15) leaves the result beyond the half range.
        let aligned = self.wide_shr_sticky(to_shift, shift);
        let m = &mut self.s.m;
        let sign_mask = m.i32(0x8000);
        let product_sign = m.bin(c, Bin::Xor, a, b);
        let product_sign = m.bin(c, Bin::LShr, product_sign, k16);
        let product_sign = m.bin(c, Bin::And, product_sign, sign_mask);
        let addend_sign = m.bin(c, Bin::LShr, addend, k16);
        let addend_sign = m.bin(c, Bin::And, addend_sign, sign_mask);
        let subtract = m.cmp(c, pred::NE, product_sign, addend_sign);
        let [aligned_lo, aligned_hi] = self.wide_negate_if(aligned, subtract);
        let m = &mut self.s.m;
        let lo = m.bin(c, Bin::Add, base_lo, aligned_lo);
        let carry = m.cmp(c, pred::IULT, lo, base_lo);
        let carry = m.cast(c, Cast::ZExt, carry, i32t);
        let hi = m.bin(c, Bin::Add, base_hi, aligned_hi);
        let hi = m.bin(c, Bin::Add, hi, carry);
        let reversed = m.cmp(c, pred::SLT, hi, zero);
        let absolute = self.wide_negate_if([lo, hi], reversed);
        let (rounded, is_zero) = self.round_wide_to_f16(absolute, exponent);
        let m = &mut self.s.m;
        let from_product = m.bin(c, Bin::Xor, product_first, reversed);
        let sign = m.select(c, from_product, product_sign, addend_sign);
        let both_negative = m.bin(c, Bin::And, product_sign, addend_sign);
        let sign = m.select(c, is_zero, both_negative, sign);
        let rounded = m.bin(c, Bin::Or, rounded, sign);

        let (f32_inf, f16_inf, quiet_nan) = (m.i32(0x7F80_0000), m.i32(0x7C00), m.i32(0x7E00));
        let [a_inf, b_inf, c_inf] = magnitudes.map(|x| m.cmp(c, pred::EQ, x, f32_inf));
        let product_inf = m.bin(c, Bin::Or, a_inf, b_inf);
        let zero_a = m.cmp(c, pred::EQ, sig_a, zero);
        let zero_b = m.cmp(c, pred::EQ, sig_b, zero);
        let zero_factor = m.bin(c, Bin::Or, zero_a, zero_b);
        let cancel_inf = m.bin(c, Bin::And, c_inf, subtract);
        let invalid = m.bin(c, Bin::Or, zero_factor, cancel_inf);
        let invalid = m.bin(c, Bin::And, product_inf, invalid);
        let nan = magnitudes.into_iter().fold(invalid, |nan, x| {
            let input_nan = m.cmp(c, pred::IUGT, x, f32_inf);
            m.bin(c, Bin::Or, nan, input_nan)
        });
        let inf_sign = m.select(c, product_inf, product_sign, addend_sign);
        let inf = m.bin(c, Bin::Or, inf_sign, f16_inf);
        let any_inf = m.bin(c, Bin::Or, product_inf, c_inf);
        let rounded = m.select(c, any_inf, inf, rounded);
        let rounded = m.select(c, nan, quiet_nan, rounded);
        if clamp {
            // Saturate the rounded half. Negative results (including -0)
            // and NaNs become +0. Positive values above one become one.
            let half_one = m.i32(0x3C00);
            let above = m.cmp(c, pred::IUGT, rounded, half_one);
            let negative = m.cmp(c, pred::IUGE, rounded, sign_mask);
            let to_zero = m.bin(c, Bin::Or, negative, nan);
            let bounded = m.select(c, above, half_one, rounded);
            m.select(c, to_zero, zero, bounded)
        } else {
            rounded
        }
    }
}
