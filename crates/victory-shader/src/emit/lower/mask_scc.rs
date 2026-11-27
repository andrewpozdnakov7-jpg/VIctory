use super::{Bin, LANE_MASK, Lower, V, Why, pred};
use crate::ir::{EXEC, Program, Reg, Value};
use crate::structure::Structured;
use std::collections::HashSet;

/// The SCC values of mask operations (bitwise scalar operations writing a
/// lane mask, not EXEC: SCC is whether the result is not zero) that an
/// instruction other than a branch reads, as an `s_cselect` of "some lane
/// is set" (the console's FidelityFX passes wait in such loops). A branch
/// on such an SCC is the lane's own bit (`structure`); these read the
/// console's number tested for zero.
pub(super) fn mask_sccs_read_as_data(p: &Program, st: &Structured) -> HashSet<Value> {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let bitwise = |name: &str| {
        let op = name
            .strip_prefix("s_")
            .and_then(|n| n.strip_suffix("_b32").or_else(|| n.strip_suffix("_b64")));
        matches!(
            op,
            Some("and" | "or" | "xor" | "andn2" | "orn2" | "nand" | "nor" | "xnor" | "not")
        )
    };
    let insts = || p.blocks.iter().flat_map(|b| &b.insts);
    let sccs: HashSet<Value> = insts()
        .filter(|i| bitwise(i.name()) && i.write(Reg::S(EXEC)).is_none())
        .filter(|i| {
            i.writes
                .iter()
                .any(|&(r, w)| matches!(r, Reg::S(_)) && is_mask(w))
        })
        .filter_map(|i| i.write(Reg::Scc))
        .collect();
    insts()
        .filter(|i| !i.name().starts_with("s_cbranch"))
        .filter_map(|i| i.read(Reg::Scc))
        .filter(|v| sccs.contains(v))
        .collect()
}

impl Lower<'_> {
    /// Whether lane mask `lo` (its low dword's value), as a number, is not
    /// zero: both dwords of a wave64 mask.
    pub(super) fn mask_nonzero(&mut self, lo: Value) -> Result<V, Why> {
        let mut n = self.mask_dword(lo)?;
        if self.x.native.wave64 {
            // `mask_dword` keeps both dwords of the mask it made.
            let (_, high) = *self.ballots.get(&lo).ok_or(LANE_MASK)?;
            n = self.s.m.bin(self.cur, Bin::Or, n, high);
        }
        let zero = self.s.m.i32(0);
        Ok(self.s.m.cmp(self.cur, pred::NE, n, zero))
    }
}
