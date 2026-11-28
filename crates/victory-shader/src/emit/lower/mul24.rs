use super::{Attr, Bin, Cast, Fields, Inst, Lower, Operand, Reg, V, Why, pred};

impl Lower<'_> {
    pub(super) fn valu_mul24(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.omod != 0 {
            return Err("integer output modifiers");
        }
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a result without a register");
        };
        let signed = inst.name().ends_with("i24");
        let (a, b) = (self.src(inst, 0)?, self.src(inst, 1)?);
        let addend = inst
            .name()
            .starts_with("v_mad_")
            .then(|| self.src(inst, 2))
            .transpose()?;
        let a = self.extend24(a, signed);
        let b = self.extend24(b, signed);
        let c = self.cur;
        let result = match (addend, v.clamp) {
            (Some(addend), false) => {
                let mad = if signed { 48 } else { 49 };
                self.s.op(
                    c,
                    "tertiary.i32",
                    Attr::ReadNone,
                    self.s.t.i32,
                    mad,
                    &[a, b, addend],
                )
            }
            (None, false) => self.s.m.bin(c, Bin::Mul, a, b),
            (addend, true) => {
                let m = &mut self.s.m;
                let low = m.bin(c, Bin::Mul, a, b);
                let sum = addend.map_or(low, |x| m.bin(c, Bin::Add, low, x));
                self.clamp_mul24(a, b, low, addend, sum, signed)
            }
        };
        self.write(inst, Reg::V(reg), result, guard);
        Ok(())
    }

    fn clamp_mul24(&mut self, a: V, b: V, low: V, addend: Option<V>, sum: V, signed: bool) -> V {
        let c = self.cur;
        let m = &mut self.s.m;
        let shr = if signed { Bin::AShr } else { Bin::LShr };
        let sixteen = m.i32(16);
        let mask = m.i32(0xFFFF);
        let a_lo = m.bin(c, Bin::And, a, mask);
        let b_lo = m.bin(c, Bin::And, b, mask);
        let a_hi = m.bin(c, shr, a, sixteen);
        let b_hi = m.bin(c, shr, b, sixteen);
        // Split each 24-bit operand into 16 low bits and 8 high bits.
        // Each partial product and the middle sum fit in 32 bits.
        let low_product = m.bin(c, Bin::Mul, a_lo, b_lo);
        let carry = m.bin(c, Bin::LShr, low_product, sixteen);
        let left = m.bin(c, Bin::Mul, a_hi, b_lo);
        let right = m.bin(c, Bin::Mul, a_lo, b_hi);
        let middle = m.bin(c, Bin::Add, left, right);
        let middle = m.bin(c, Bin::Add, middle, carry);
        let middle_hi = m.bin(c, shr, middle, sixteen);
        let high_product = m.bin(c, Bin::Mul, a_hi, b_hi);
        let mut high = m.bin(c, Bin::Add, high_product, middle_hi);
        let thirty_one = m.i32(31);
        if let Some(x) = addend {
            let carry = m.cmp(c, pred::IULT, sum, low);
            let carry = m.cast(c, Cast::ZExt, carry, self.s.t.i32);
            high = m.bin(c, Bin::Add, high, carry);
            if signed {
                let extension = m.bin(c, Bin::AShr, x, thirty_one);
                high = m.bin(c, Bin::Add, high, extension);
            }
        }
        let zero = m.i32(0);
        let (extension, saturated) = if signed {
            let extension = m.bin(c, Bin::AShr, sum, thirty_one);
            let negative = m.cmp(c, pred::SLT, high, zero);
            let min = m.i32(i32::MIN as u32);
            let max = m.i32(i32::MAX as u32);
            (extension, m.select(c, negative, min, max))
        } else {
            (zero, m.i32(u32::MAX))
        };
        let fits = m.cmp(c, pred::EQ, high, extension);
        m.select(c, fits, sum, saturated)
    }
}
