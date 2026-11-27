//! Binary64 scaling in two dwords. Adjusting the exponent directly avoids
//! rounding an intermediate power of two or requiring native DXIL doubles.

use super::{Attr, Bin, Cast, Fields, Inst, Lower, NULL, Operand, Reg, V, Why, pred};
use crate::emit::bitcode::Module;

impl Lower<'_> {
    pub(super) fn ldexp_f64(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.clamp || v.omod != 0 || v.sdwa.is_some() || v.dpp.is_some() || (v.abs | v.neg) & !1 != 0 {
            return Err("binary64 scaling modifiers");
        }
        let (Some(round), Some(preserve_input), Some(preserve_output)) = (
            self.mode16.round,
            self.mode16.denorm_input,
            self.mode16.denorm_output,
        ) else {
            return Err("binary64 scaling with unknown floating-point modes");
        };
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a binary64 destination pair");
        };
        let dst_high = reg.checked_add(1).ok_or("a missing destination high word")?;
        let source = *v.src.first().ok_or("a missing binary64 source")?;
        let (low, mut high) = self.scale_source_words(inst, source)?;
        let power = self.operand(inst, v.src.get(1).ok_or("a missing scale exponent")?)?;

        let c = self.cur;
        let i32t = self.s.t.i32;
        let m = &mut self.s.m;
        let zero = m.i32(0);
        let one = m.i32(1);
        let sign_mask = m.i32(0x8000_0000);
        let magnitude_mask = m.i32(0x7fff_ffff);
        let mantissa_mask = m.i32(0x000f_ffff);
        let implicit = m.i32(0x0010_0000);
        let magnitude_high = m.bin(c, Bin::And, high, magnitude_mask);
        if v.abs & 1 != 0 {
            high = magnitude_high;
        }
        if v.neg & 1 != 0 {
            high = m.bin(c, Bin::Xor, high, sign_mask);
        }
        let sign = m.bin(c, Bin::And, high, sign_mask);
        let negative = m.cmp(c, pred::NE, sign, zero);
        let away = match round {
            1 => m.cmp(c, pred::EQ, sign, zero),
            2 => negative,
            _ => m.bool(false),
        };
        let shift20 = m.i32(20);
        let exponent = m.bin(c, Bin::LShr, magnitude_high, shift20);
        let frac_high = m.bin(c, Bin::And, magnitude_high, mantissa_mask);
        let input_fraction = m.bin(c, Bin::Or, low, frac_high);
        let fraction_nonzero = m.cmp(c, pred::NE, input_fraction, zero);
        let exponent_zero = m.cmp(c, pred::EQ, exponent, zero);

        // Subnormal input: shift its first set bit into the normal implicit
        // bit position and adjust the exponent by the same distance.
        let leading_low = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 33, &[low]);
        let leading_high = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 33, &[frac_high]);
        let m = &mut self.s.m;
        let high_nonzero = m.cmp(c, pred::NE, frac_high, zero);
        let low_shift_base = m.i32(21); // 52 - 31
        let high_shift_base = m.i32((-11i32) as u32); // 52 - 63
        let low_shift = m.bin(c, Bin::Add, low_shift_base, leading_low);
        let high_shift = m.bin(c, Bin::Add, high_shift_base, leading_high);
        let normalize = m.select(c, high_nonzero, high_shift, low_shift);
        let (norm_low, norm_high) = shift_words(m, c, (low, frac_high), normalize, true);
        let normal_high = m.bin(c, Bin::Or, frac_high, implicit);
        let significand_low = m.select(c, exponent_zero, norm_low, low);
        let significand_high = m.select(c, exponent_zero, norm_high, normal_high);
        let subnormal_exp = m.bin(c, Bin::Sub, one, normalize);
        let exponent = m.select(c, exponent_zero, subnormal_exp, exponent);

        // Powers beyond these bounds cannot change the overflow/underflow
        // decision. Bound the add so an arbitrary i32 power cannot wrap it.
        let min_power = m.i32((-4096i32) as u32);
        let max_power = m.i32(4096);
        let too_small = m.cmp(c, pred::SLT, power, min_power);
        let too_large = m.cmp(c, pred::SGT, power, max_power);
        let power = m.select(c, too_small, min_power, power);
        let power = m.select(c, too_large, max_power, power);
        let scaled_exp = m.bin(c, Bin::Add, exponent, power);
        let subnormal = m.cmp(c, pred::SLE, scaled_exp, zero);
        let infinity_exp = m.i32(0x7ff);
        let overflow = m.cmp(c, pred::SGE, scaled_exp, infinity_exp);
        let guard_shift = m.bin(c, Bin::Sub, zero, scaled_exp);
        let denorm_shift_limit = m.i32(53);
        let shift_in_range = m.cmp(c, pred::IULT, guard_shift, denorm_shift_limit);
        let guard_shift = m.select(c, shift_in_range, guard_shift, denorm_shift_limit);

        let (mut sub_low, mut sub_high) = self.round_scaled_subnormal(
            (significand_low, significand_high),
            guard_shift,
            (round != 0).then_some(away),
        );
        let m = &mut self.s.m;
        if !preserve_output {
            // A carry can round the largest subnormal up to the minimum
            // normal. Flush only a result that is still subnormal.
            let rounds_to_normal = m.cmp(c, pred::IUGE, sub_high, implicit);
            sub_low = m.select(c, rounds_to_normal, sub_low, zero);
            sub_high = m.select(c, rounds_to_normal, sub_high, zero);
        }
        let scaled_high = m.bin(c, Bin::And, significand_high, mantissa_mask);
        let scaled_exp = m.bin(c, Bin::Shl, scaled_exp, shift20);
        let scaled_high = m.bin(c, Bin::Or, scaled_high, scaled_exp);
        let result_low = m.select(c, subnormal, sub_low, significand_low);
        let result_high = m.select(c, subnormal, sub_high, scaled_high);

        let overflow_to_inf = if round == 0 { m.bool(true) } else { away };
        let max_low = m.i32(u32::MAX);
        let max_high = m.i32(0x7fef_ffff);
        let inf_high = m.i32(0x7ff0_0000);
        let overflow_low = m.select(c, overflow_to_inf, zero, max_low);
        let overflow_high = m.select(c, overflow_to_inf, inf_high, max_high);
        let result_low = m.select(c, overflow, overflow_low, result_low);
        let result_high = m.select(c, overflow, overflow_high, result_high);

        let zero_operand = if !preserve_input {
            exponent_zero
        } else {
            let no_fraction = m.cmp(c, pred::EQ, input_fraction, zero);
            m.bin(c, Bin::And, exponent_zero, no_fraction)
        };
        let result_low = m.select(c, zero_operand, zero, result_low);
        let result_high = m.select(c, zero_operand, zero, result_high);
        let result_high = m.bin(c, Bin::Or, result_high, sign);

        let original_exp = m.bin(c, Bin::LShr, magnitude_high, shift20);
        let nonfinite = m.cmp(c, pred::EQ, original_exp, infinity_exp);
        let quiet_bit = m.i32(0x0008_0000);
        let quiet_nan = m.bin(c, Bin::Or, high, quiet_bit);
        let special_high = m.select(c, fraction_nonzero, quiet_nan, high);
        let result_low = m.select(c, nonfinite, low, result_low);
        let result_high = m.select(c, nonfinite, special_high, result_high);
        self.write(inst, Reg::V(reg), result_low, guard);
        self.write(inst, Reg::V(dst_high), result_high, guard);
        Ok(())
    }

    fn scale_source_words(&mut self, inst: &Inst, source: Operand) -> Result<(V, V), Why> {
        let second = |reg: u8| reg.checked_add(1).ok_or("a missing binary64 high word");
        let pair = |l: &mut Self, high| {
            let high = inst.read(high).ok_or("an unread binary64 high word")?;
            Ok((l.operand(inst, &source)?, l.i32_of(high)?))
        };
        match source {
            Operand::Vector { reg, .. } => pair(self, Reg::V(second(reg)?)),
            Operand::Scalar { code, .. } if code != NULL => pair(self, Reg::S(second(code)?)),
            Operand::Literal(high) => Ok((self.s.m.i32(0), self.s.m.i32(high))),
            Operand::Const(code @ 240..=248) => {
                let bits = if code == 248 {
                    0x3fc4_5f30_6dc9_c882 // Binary64 1/(2 pi).
                } else {
                    let value = crate::resource::inline(code).ok_or("an unknown inline float")?;
                    f64::from(f32::from_bits(value)).to_bits()
                };
                Ok((self.s.m.i32(bits as u32), self.s.m.i32((bits >> 32) as u32)))
            }
            _ => {
                let high = if matches!(source, Operand::Const(193..=208)) { u32::MAX } else { 0 };
                Ok((self.operand(inst, &source)?, self.s.m.i32(high)))
            }
        }
    }

    /// Shift the significand down to subnormal units. All bits below the
    /// guard contribute to sticky, including at the 32-bit word boundary.
    fn round_scaled_subnormal(&mut self, words: (V, V), guard_shift: V, directed: Option<V>) -> (V, V) {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let (low, high) = words;
        let first_low = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[low]);
        let first_high = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[high]);
        let m = &mut self.s.m;
        let zero = m.i32(0);
        let one = m.i32(1);
        let bits31 = m.i32(31);
        let bits32 = m.i32(32);
        let first_high = m.bin(c, Bin::Add, first_high, bits32);
        let low_nonzero = m.cmp(c, pred::NE, low, zero);
        let first = m.select(c, low_nonzero, first_low, first_high);
        let sticky = m.cmp(c, pred::IULT, first, guard_shift);
        let (lo_guard, hi_guard) = shift_words(m, c, words, guard_shift, false);
        let guard = m.bin(c, Bin::And, lo_guard, one);
        let guard = m.cmp(c, pred::NE, guard, zero);
        let low = m.bin(c, Bin::LShr, lo_guard, one);
        let carry = m.bin(c, Bin::Shl, hi_guard, bits31);
        let low = m.bin(c, Bin::Or, low, carry);
        let high = m.bin(c, Bin::LShr, hi_guard, one);
        let up = if let Some(away) = directed {
            let discarded = m.bin(c, Bin::Or, guard, sticky);
            m.bin(c, Bin::And, away, discarded)
        } else {
            let odd = m.bin(c, Bin::And, low, one);
            let odd = m.cmp(c, pred::NE, odd, zero);
            let sticky_or_odd = m.bin(c, Bin::Or, sticky, odd);
            m.bin(c, Bin::And, guard, sticky_or_odd)
        };
        let up = m.cast(c, Cast::ZExt, up, i32t);
        let rounded = m.bin(c, Bin::Add, low, up);
        let carry = m.cmp(c, pred::IULT, rounded, low);
        let carry = m.cast(c, Cast::ZExt, carry, i32t);
        (rounded, m.bin(c, Bin::Add, high, carry))
    }
}

/// Logical shift up to 53 bits across two dwords. All individual IR shifts
/// are masked to 0..31, including the carry when the offset is zero.
fn shift_words(m: &mut Module, c: usize, (low, high): (V, V), count: V, left: bool) -> (V, V) {
    let (near, far, shift, cross) = if left {
        (low, high, Bin::Shl, Bin::LShr)
    } else {
        (high, low, Bin::LShr, Bin::Shl)
    };
    let zero = m.i32(0);
    let bits31 = m.i32(31);
    let bits32 = m.i32(32);
    let offset = m.bin(c, Bin::And, count, bits31);
    let shifted = m.bin(c, shift, near, offset);
    let far = m.bin(c, shift, far, offset);
    let carry_shift = m.bin(c, Bin::Sub, zero, offset);
    let carry_shift = m.bin(c, Bin::And, carry_shift, bits31);
    let carry = m.bin(c, cross, near, carry_shift);
    let nonzero_offset = m.cmp(c, pred::NE, offset, zero);
    let carry = m.select(c, nonzero_offset, carry, zero);
    let far = m.bin(c, Bin::Or, far, carry);
    let large = m.cmp(c, pred::IUGE, count, bits32);
    let near = m.select(c, large, zero, shifted);
    let far = m.select(c, large, shifted, far);
    if left { (near, far) } else { (far, near) }
}
