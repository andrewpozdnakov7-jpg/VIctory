//! Single rounding of mixed-precision FMA results to binary16.

use super::super::bitcode::{Attr, Bin, Cast, V, pred};
use super::Lower;

impl<'a> Lower<'a> {
    /// Clamps a binary16 value (low 16 bits of an `i32`) to `[0.0, 1.0]`.
    pub(super) fn clamp_f16(&mut self, h: V) -> V {
        let c = self.cur;
        let f = self
            .s
            .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h]);
        let f = self.unary(7, f);
        self.s
            .op(c, "legacyF32ToF16", Attr::ReadNone, self.s.t.i32, 130, &[f])
    }

    /// Unpacks an `f32` into `(sign, mant24, exp24, is_nan, is_inf, is_zero)`
    /// where `|x| = mant24 * 2^exp24` with bit 23 of `mant24` set for non-zero
    /// finite inputs; f32 denormals flush to signed zero per RDNA2 console
    /// float mode (widened f16 denormals are normal f32 and stay intact).
    fn unpack_f32_parts(&mut self, x: V) -> (V, V, V, V, V, V) {
        let c = self.cur;
        let ix = self.as_int(x);
        let k0 = self.s.m.i32(0);
        let k23 = self.s.m.i32(23);
        let k31 = self.s.m.i32(31);
        let mask_abs = self.s.m.i32(0x7FFF_FFFF);
        let mask_frac = self.s.m.i32(0x007F_FFFF);
        let inf_bits = self.s.m.i32(0x7F80_0000);
        let implicit = self.s.m.i32(0x0080_0000);
        let k150 = self.s.m.i32(150);

        let sign = self.s.m.bin(c, Bin::LShr, ix, k31);
        let mag = self.s.m.bin(c, Bin::And, ix, mask_abs);
        let exp_raw = self.s.m.bin(c, Bin::LShr, mag, k23);
        let frac = self.s.m.bin(c, Bin::And, mag, mask_frac);
        let is_nan = self.s.m.cmp(c, pred::IUGT, mag, inf_bits);
        let is_inf = self.s.m.cmp(c, pred::EQ, mag, inf_bits);
        let is_zero = self.s.m.cmp(c, pred::EQ, exp_raw, k0);
        let m_norm = self.s.m.bin(c, Bin::Or, frac, implicit);
        let e_norm = self.s.m.bin(c, Bin::Sub, exp_raw, k150);
        (sign, m_norm, e_norm, is_nan, is_inf, is_zero)
    }

    /// Exact 24x24 -> 48-bit unsigned product, returned as `(hi24, lo24)`.
    fn mul24_exact(&mut self, ma: V, mb: V) -> (V, V) {
        let c = self.cur;
        let k12 = self.s.m.i32(12);
        let k24 = self.s.m.i32(24);
        let m12 = self.s.m.i32(0xFFF);
        let m24 = self.s.m.i32(0xFF_FFFF);
        let a_hi = self.s.m.bin(c, Bin::LShr, ma, k12);
        let a_lo = self.s.m.bin(c, Bin::And, ma, m12);
        let b_hi = self.s.m.bin(c, Bin::LShr, mb, k12);
        let b_lo = self.s.m.bin(c, Bin::And, mb, m12);
        let ll = self.s.m.bin(c, Bin::Mul, a_lo, b_lo);
        let m1 = self.s.m.bin(c, Bin::Mul, a_hi, b_lo);
        let m2 = self.s.m.bin(c, Bin::Mul, a_lo, b_hi);
        let hh = self.s.m.bin(c, Bin::Mul, a_hi, b_hi);
        let mid = self.s.m.bin(c, Bin::Add, m1, m2);
        let mid_lo = self.s.m.bin(c, Bin::And, mid, m12);
        let mid_lo_sh = self.s.m.bin(c, Bin::Shl, mid_lo, k12);
        let lo_sum = self.s.m.bin(c, Bin::Add, ll, mid_lo_sh);
        let p_lo = self.s.m.bin(c, Bin::And, lo_sum, m24);
        let mid_hi = self.s.m.bin(c, Bin::LShr, mid, k12);
        let carry = self.s.m.bin(c, Bin::LShr, lo_sum, k24);
        let t = self.s.m.bin(c, Bin::Add, hh, mid_hi);
        let p_hi = self.s.m.bin(c, Bin::Add, t, carry);
        (p_hi, p_lo)
    }

    /// Applies one round-to-nearest-even step to `sig` given round bit `rbit`
    /// and sticky sources `(rem_hi, rem_lo, st)`.
    fn rne_step(&mut self, sig: V, rbit: V, rem_hi: V, rem_lo: V, st: V) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let k0 = self.s.m.i32(0);
        let k1 = self.s.m.i32(1);
        let stk_or = self.s.m.bin(c, Bin::Or, rem_hi, rem_lo);
        let stk_or = self.s.m.bin(c, Bin::Or, stk_or, st);
        let stk_nz = self.s.m.cmp(c, pred::NE, stk_or, k0);
        let stk = self.s.m.cast(c, Cast::ZExt, stk_nz, i32t);
        let lsb = self.s.m.bin(c, Bin::And, sig, k1);
        let tos = self.s.m.bin(c, Bin::Or, stk, lsb);
        let rup = self.s.m.bin(c, Bin::And, rbit, tos);
        self.s.m.bin(c, Bin::Add, sig, rup)
    }

    /// Exact single-rounding `fma(a, b, third)` from `f32` inputs to IEEE-754
    /// binary16 round-to-nearest-even (low 16 bits of an `i32`).
    pub(super) fn fma_f32_to_f16(&mut self, a: V, b: V, third: V) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let (sa, ma, ea, nan_a, inf_a, zero_a) = self.unpack_f32_parts(a);
        let (sb, mb, eb, nan_b, inf_b, zero_b) = self.unpack_f32_parts(b);
        let (sc, mc, ec, nan_c, inf_c, zero_c) = self.unpack_f32_parts(third);

        let k0 = self.s.m.i32(0);

        let k1 = self.s.m.i32(1);
        let k7 = self.s.m.i32(7);
        let k8 = self.s.m.i32(8);
        let k10 = self.s.m.i32(10);
        let k15 = self.s.m.i32(15);
        let k16 = self.s.m.i32(16);
        let k20 = self.s.m.i32(20);
        let k21 = self.s.m.i32(21);
        let k23 = self.s.m.i32(23);
        let k24 = self.s.m.i32(24);
        let k31 = self.s.m.i32(31);
        let k32 = self.s.m.i32(32);
        let k47 = self.s.m.i32(47);
        let k64 = self.s.m.i32(64);
        let k2048 = self.s.m.i32(2048);
        let km14 = self.s.m.i32((-14i32) as u32);
        let km25 = self.s.m.i32((-25i32) as u32);
        let qnan = self.s.m.i32(0x7E00);
        let f16_inf = self.s.m.i32(0x7C00);
        let m16 = self.s.m.i32(0xFFFF);
        let m10 = self.s.m.i32(0x3FF);
        let m20 = self.s.m.i32(0xF_FFFF);
        let m24 = self.s.m.i32(0xFF_FFFF);
        let bit23_mask = self.s.m.i32(0x0080_0000);
        let sign_bit_mask = self.s.m.i32(0x8000_0000);
        let max_s32 = self.s.m.i32(0x7FFF_FFFF);
        let b_true = self.s.m.bool(true);
        let b_false = self.s.m.bool(false);

        let s_prod = self.s.m.bin(c, Bin::Xor, sa, sb);
        let any_nan = self.s.m.bin(c, Bin::Or, nan_a, nan_b);
        let any_nan = self.s.m.bin(c, Bin::Or, any_nan, nan_c);
        let ab_inf = self.s.m.bin(c, Bin::Or, inf_a, inf_b);
        let ab_zero = self.s.m.bin(c, Bin::Or, zero_a, zero_b);
        let iz1 = self.s.m.bin(c, Bin::And, inf_a, zero_b);
        let iz2 = self.s.m.bin(c, Bin::And, inf_b, zero_a);
        let inf_zero = self.s.m.bin(c, Bin::Or, iz1, iz2);
        let s_diff = self.s.m.cmp(c, pred::NE, s_prod, sc);
        let inf_cancel = self.s.m.bin(c, Bin::And, ab_inf, inf_c);
        let inf_cancel = self.s.m.bin(c, Bin::And, inf_cancel, s_diff);
        let out_nan = self.s.m.bin(c, Bin::Or, any_nan, inf_zero);
        let out_nan = self.s.m.bin(c, Bin::Or, out_nan, inf_cancel);

        let any_inf = self.s.m.bin(c, Bin::Or, ab_inf, inf_c);
        let not_nan = self.s.m.bin(c, Bin::Xor, out_nan, b_true);
        let out_inf = self.s.m.bin(c, Bin::And, not_nan, any_inf);
        let inf_sign = self.s.m.select(c, ab_inf, s_prod, sc);
        let inf_sign_15 = self.s.m.bin(c, Bin::Shl, inf_sign, k15);
        let inf_val = self.s.m.bin(c, Bin::Or, inf_sign_15, f16_inf);

        let both_neg_zero = self.s.m.bin(c, Bin::And, s_prod, sc);
        let zero_res_if_c0 = self.s.m.bin(c, Bin::Shl, both_neg_zero, k15);
        let ab_zero_and_c0 = self.s.m.bin(c, Bin::And, ab_zero, zero_c);

        // Exact 48-bit product normalized to [2^47, 2^48 - 1], placed at bits 16..63.
        let (p_hi, p_lo) = self.mul24_exact(ma, mb);
        let bit23 = self.s.m.bin(c, Bin::And, p_hi, bit23_mask);
        let p_lt = self.s.m.cmp(c, pred::EQ, bit23, k0);
        let p_hi_sh = self.s.m.bin(c, Bin::Shl, p_hi, k1);
        let p_lo_top = self.s.m.bin(c, Bin::LShr, p_lo, k23);
        let p_hi_n = self.s.m.bin(c, Bin::Or, p_hi_sh, p_lo_top);
        let p_lo_sh = self.s.m.bin(c, Bin::Shl, p_lo, k1);
        let p_lo_n = self.s.m.bin(c, Bin::And, p_lo_sh, m24);
        let mp_hi = self.s.m.select(c, p_lt, p_hi_n, p_hi);
        let mp_lo = self.s.m.select(c, p_lt, p_lo_n, p_lo);
        let e_ab = self.s.m.bin(c, Bin::Add, ea, eb);
        let p_lt_i = self.s.m.cast(c, Cast::ZExt, p_lt, i32t);
        let e_p = self.s.m.bin(c, Bin::Sub, e_ab, p_lt_i);

        let mp_hi_8 = self.s.m.bin(c, Bin::Shl, mp_hi, k8);
        let mp_lo_16 = self.s.m.bin(c, Bin::LShr, mp_lo, k16);
        let p64_hi = self.s.m.bin(c, Bin::Or, mp_hi_8, mp_lo_16);
        let mp_lo_low = self.s.m.bin(c, Bin::And, mp_lo, m16);
        let p64_lo = self.s.m.bin(c, Bin::Shl, mp_lo_low, k16);

        let c64_hi = self.s.m.bin(c, Bin::Shl, mc, k8);
        let c64_lo = k0;
        let e_c64 = self.s.m.bin(c, Bin::Sub, ec, k24);

        let e_gt = self.s.m.cmp(c, pred::SGT, e_p, e_c64);
        let e_eq = self.s.m.cmp(c, pred::EQ, e_p, e_c64);
        let hi_gt = self.s.m.cmp(c, pred::IUGT, p64_hi, c64_hi);
        let hi_eq = self.s.m.cmp(c, pred::EQ, p64_hi, c64_hi);
        let lo_gt = self.s.m.cmp(c, pred::IUGT, p64_lo, c64_lo);
        let hi_lo_gt = self.s.m.bin(c, Bin::And, hi_eq, lo_gt);
        let mag_gt = self.s.m.bin(c, Bin::Or, hi_gt, hi_lo_gt);
        let eq_gt = self.s.m.bin(c, Bin::And, e_eq, mag_gt);
        let p_ge_c = self.s.m.bin(c, Bin::Or, zero_c, e_gt);
        let p_ge_c = self.s.m.bin(c, Bin::Or, p_ge_c, eq_gt);

        let a_hi = self.s.m.select(c, p_ge_c, p64_hi, c64_hi);
        let a_lo = self.s.m.select(c, p_ge_c, p64_lo, c64_lo);
        let e_big = self.s.m.select(c, p_ge_c, e_p, e_c64);
        let s_big = self.s.m.select(c, p_ge_c, s_prod, sc);
        let b_hi = self.s.m.select(c, p_ge_c, c64_hi, p64_hi);
        let b_lo = self.s.m.select(c, p_ge_c, c64_lo, p64_lo);
        let e_small = self.s.m.select(c, p_ge_c, e_c64, e_p);
        let s_small = self.s.m.select(c, p_ge_c, sc, s_prod);

        // Shift smaller operand right by diff = e_big - e_small, tracking sticky.
        let diff = self.s.m.bin(c, Bin::Sub, e_big, e_small);
        let k = self.s.m.bin(c, Bin::And, diff, k31);
        let k_is_0 = self.s.m.cmp(c, pred::EQ, k, k0);
        let inv_k = self.s.m.bin(c, Bin::Sub, k32, k);
        let inv_k = self.s.m.bin(c, Bin::And, inv_k, k31);
        let is_lt32 = self.s.m.cmp(c, pred::IULT, diff, k32);
        let is_lt64 = self.s.m.cmp(c, pred::IULT, diff, k64);

        let b_hi_shl = self.s.m.bin(c, Bin::Shl, b_hi, inv_k);
        let b_lo_shr = self.s.m.bin(c, Bin::LShr, b_lo, k);
        let sh_lo_nz = self.s.m.bin(c, Bin::Or, b_hi_shl, b_lo_shr);
        let sh_lo = self.s.m.select(c, k_is_0, b_lo, sh_lo_nz);
        let sh_hi = self.s.m.bin(c, Bin::LShr, b_hi, k);
        let b_lo_lost = self.s.m.bin(c, Bin::Shl, b_lo, inv_k);
        let b_lo_lost_nz = self.s.m.cmp(c, pred::NE, b_lo_lost, k0);
        let sh_st_b = self.s.m.select(c, k_is_0, b_false, b_lo_lost_nz);
        let sh_st = self.s.m.cast(c, Cast::ZExt, sh_st_b, i32t);

        let ge32_lo = self.s.m.select(c, k_is_0, b_hi, sh_hi);
        let b_lo_nz = self.s.m.cmp(c, pred::NE, b_lo, k0);
        let b_hi_lost_nz = self.s.m.cmp(c, pred::NE, b_hi_shl, k0);
        let k_ne_0 = self.s.m.cmp(c, pred::NE, k, k0);
        let ge32_hi_lost = self.s.m.bin(c, Bin::And, k_ne_0, b_hi_lost_nz);
        let ge32_st_b = self.s.m.bin(c, Bin::Or, b_lo_nz, ge32_hi_lost);
        let ge32_st = self.s.m.cast(c, Cast::ZExt, ge32_st_b, i32t);

        let not_lt32 = self.s.m.bin(c, Bin::Xor, is_lt32, b_true);
        let not_lt64 = self.s.m.bin(c, Bin::Xor, is_lt64, b_true);
        let zero_hi = self.s.m.bin(c, Bin::Or, zero_c, not_lt32);
        let zero_lo = self.s.m.bin(c, Bin::Or, zero_c, not_lt64);
        let b_s_hi = self.s.m.select(c, zero_hi, k0, sh_hi);
        let b_s_lo_nz = self.s.m.select(c, is_lt32, sh_lo, ge32_lo);
        let b_s_lo = self.s.m.select(c, zero_lo, k0, b_s_lo_nz);
        let st_nz = self.s.m.select(c, is_lt64, ge32_st, k1);
        let st_nz = self.s.m.select(c, is_lt32, sh_st, st_nz);
        let st = self.s.m.select(c, zero_c, k0, st_nz);

        // Add or subtract aligned 64-bit magnitudes.
        let s_eq = self.s.m.cmp(c, pred::EQ, s_big, s_small);
        let same_sign = self.s.m.bin(c, Bin::Or, s_eq, zero_c);

        let add_lo0 = self.s.m.bin(c, Bin::Add, a_lo, b_s_lo);
        let c_lo = self.s.m.cmp(c, pred::IULT, add_lo0, a_lo);
        let c_lo = self.s.m.cast(c, Cast::ZExt, c_lo, i32t);
        let t_hi = self.s.m.bin(c, Bin::Add, a_hi, b_s_hi);
        let c1 = self.s.m.cmp(c, pred::IULT, t_hi, a_hi);
        let add_hi0 = self.s.m.bin(c, Bin::Add, t_hi, c_lo);
        let c2 = self.s.m.cmp(c, pred::IULT, add_hi0, t_hi);
        let ovf = self.s.m.bin(c, Bin::Or, c1, c2);
        let add_lo_lsb = self.s.m.bin(c, Bin::And, add_lo0, k1);
        let add_st_ovf = self.s.m.bin(c, Bin::Or, st, add_lo_lsb);
        let add_hi_bot = self.s.m.bin(c, Bin::Shl, add_hi0, k31);
        let add_lo_shr = self.s.m.bin(c, Bin::LShr, add_lo0, k1);
        let add_lo_ovf = self.s.m.bin(c, Bin::Or, add_hi_bot, add_lo_shr);
        let add_hi_shr = self.s.m.bin(c, Bin::LShr, add_hi0, k1);
        let add_hi_ovf = self.s.m.bin(c, Bin::Or, sign_bit_mask, add_hi_shr);
        let add_hi = self.s.m.select(c, ovf, add_hi_ovf, add_hi0);
        let add_lo = self.s.m.select(c, ovf, add_lo_ovf, add_lo0);
        let add_st = self.s.m.select(c, ovf, add_st_ovf, st);
        let ovf_i = self.s.m.cast(c, Cast::ZExt, ovf, i32t);
        let e_add = self.s.m.bin(c, Bin::Add, e_big, ovf_i);

        let b1 = self.s.m.cmp(c, pred::IULT, a_lo, b_s_lo);
        let t_lo = self.s.m.bin(c, Bin::Sub, a_lo, b_s_lo);
        let b2 = self.s.m.cmp(c, pred::IULT, t_lo, st);
        let sub_lo = self.s.m.bin(c, Bin::Sub, t_lo, st);
        let borrow = self.s.m.bin(c, Bin::Or, b1, b2);
        let borrow = self.s.m.cast(c, Cast::ZExt, borrow, i32t);
        let sub_hi0 = self.s.m.bin(c, Bin::Sub, a_hi, b_s_hi);
        let sub_hi = self.s.m.bin(c, Bin::Sub, sub_hi0, borrow);

        let s_hi = self.s.m.select(c, same_sign, add_hi, sub_hi);
        let s_lo = self.s.m.select(c, same_sign, add_lo, sub_lo);
        let s_st = self.s.m.select(c, same_sign, add_st, st);
        let e_s = self.s.m.select(c, same_sign, e_add, e_big);

        let s_hi = self.s.m.select(c, ab_zero, c64_hi, s_hi);
        let s_lo = self.s.m.select(c, ab_zero, k0, s_lo);
        let s_st = self.s.m.select(c, ab_zero, k0, s_st);
        let e_s = self.s.m.select(c, ab_zero, e_c64, e_s);
        let s_big = self.s.m.select(c, ab_zero, sc, s_big);

        // Normalize (s_hi, s_lo) to bit 63 = 1.
        let hi_z = self.s.m.cmp(c, pred::EQ, s_hi, k0);
        let lo_z = self.s.m.cmp(c, pred::EQ, s_lo, k0);
        let exact_zero = self.s.m.bin(c, Bin::And, hi_z, lo_z);
        let hi_nz = self.s.m.cmp(c, pred::NE, s_hi, k0);
        let lz_hi = self
            .s
            .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 33, &[s_hi]);
        let lz_lo = self
            .s
            .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 33, &[s_lo]);
        let lz = self.s.m.select(c, hi_nz, lz_hi, lz_lo);
        let lz = self.s.m.bin(c, Bin::And, lz, k31);
        let lz_0 = self.s.m.cmp(c, pred::EQ, lz, k0);
        let inv_lz = self.s.m.bin(c, Bin::Sub, k32, lz);
        let inv_lz = self.s.m.bin(c, Bin::And, inv_lz, k31);
        let hi_shl = self.s.m.bin(c, Bin::Shl, s_hi, lz);
        let lo_shr = self.s.m.bin(c, Bin::LShr, s_lo, inv_lz);
        let norm_hi_s = self.s.m.bin(c, Bin::Or, hi_shl, lo_shr);
        let norm_hi_h = self.s.m.select(c, lz_0, s_hi, norm_hi_s);
        let lo_shl = self.s.m.bin(c, Bin::Shl, s_lo, lz);
        let norm_hi = self.s.m.select(c, hi_nz, norm_hi_h, lo_shl);
        let norm_lo = self.s.m.select(c, hi_nz, lo_shl, k0);
        let lz_plus_32 = self.s.m.bin(c, Bin::Add, k32, lz);
        let shift_total = self.s.m.select(c, hi_nz, lz, lz_plus_32);
        let e_s47 = self.s.m.bin(c, Bin::Add, e_s, k47);
        let e_top = self.s.m.bin(c, Bin::Sub, e_s47, shift_total);

        // Single round-to-nearest-even to binary16.
        let s_res_15 = self.s.m.bin(c, Bin::Shl, s_big, k15);
        let inf_signed = self.s.m.bin(c, Bin::Or, s_res_15, f16_inf);

        // Normal range (e_top >= -14).
        let sig11 = self.s.m.bin(c, Bin::LShr, norm_hi, k21);
        let rbit = self.s.m.bin(c, Bin::LShr, norm_hi, k20);
        let rbit = self.s.m.bin(c, Bin::And, rbit, k1);
        let hi_rem = self.s.m.bin(c, Bin::And, norm_hi, m20);
        let sigr = self.rne_step(sig11, rbit, hi_rem, norm_lo, s_st);
        let carry = self.s.m.cmp(c, pred::EQ, sigr, k2048);
        let carry_i = self.s.m.cast(c, Cast::ZExt, carry, i32t);
        let expf = self.s.m.bin(c, Bin::Add, e_top, k15);
        let expf = self.s.m.bin(c, Bin::Add, expf, carry_i);
        let mantf_nc = self.s.m.bin(c, Bin::And, sigr, m10);
        let mantf = self.s.m.select(c, carry, k0, mantf_nc);
        let expf_sh = self.s.m.bin(c, Bin::Shl, expf, k10);
        let norm_pk = self.s.m.bin(c, Bin::Or, s_res_15, expf_sh);
        let norm_pk = self.s.m.bin(c, Bin::Or, norm_pk, mantf);
        let norm_ovf = self.s.m.cmp(c, pred::SGE, expf, k31);
        let norm_res = self.s.m.select(c, norm_ovf, inf_signed, norm_pk);

        // Subnormal range (-25 <= e_top <= -15).
        let is_e25 = self.s.m.cmp(c, pred::EQ, e_top, km25);
        let sub_sh0 = self.s.m.bin(c, Bin::Sub, k7, e_top);
        let sub_sh0 = self.s.m.bin(c, Bin::And, sub_sh0, k31);
        let sub_sh = self.s.m.select(c, is_e25, k31, sub_sh0);
        let sub_sh_m1 = self.s.m.bin(c, Bin::Sub, sub_sh, k1);
        let sub_sh_m1 = self.s.m.bin(c, Bin::And, sub_sh_m1, k31);
        let sub_sig_n = self.s.m.bin(c, Bin::LShr, norm_hi, sub_sh);
        let sub_sig = self.s.m.select(c, is_e25, k0, sub_sig_n);
        let sub_r_e25 = self.s.m.bin(c, Bin::LShr, norm_hi, k31);
        let sub_r_n = self.s.m.bin(c, Bin::LShr, norm_hi, sub_sh_m1);
        let sub_r_n = self.s.m.bin(c, Bin::And, sub_r_n, k1);
        let sub_rbit = self.s.m.select(c, is_e25, sub_r_e25, sub_r_n);
        let sub_mask_n = self.s.m.bin(c, Bin::Shl, k1, sub_sh_m1);
        let sub_mask_n = self.s.m.bin(c, Bin::Sub, sub_mask_n, k1);
        let sub_mask = self.s.m.select(c, is_e25, max_s32, sub_mask_n);
        let sub_rem = self.s.m.bin(c, Bin::And, norm_hi, sub_mask);
        let sub_sig_r = self.rne_step(sub_sig, sub_rbit, sub_rem, norm_lo, s_st);
        let sub_res = self.s.m.bin(c, Bin::Or, s_res_15, sub_sig_r);

        let ge_16 = self.s.m.cmp(c, pred::SGE, e_top, k16);
        let ge_m14 = self.s.m.cmp(c, pred::SGE, e_top, km14);
        let ge_m25 = self.s.m.cmp(c, pred::SGE, e_top, km25);
        let fin = self.s.m.select(c, ge_m25, sub_res, s_res_15);
        let fin = self.s.m.select(c, ge_m14, norm_res, fin);
        let fin = self.s.m.select(c, ge_16, inf_signed, fin);
        let fin = self.s.m.select(c, exact_zero, k0, fin);

        let r = self.s.m.select(c, ab_zero_and_c0, zero_res_if_c0, fin);
        let r = self.s.m.select(c, out_inf, inf_val, r);
        self.s.m.select(c, out_nan, qnan, r)
    }
}
