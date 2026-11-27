use super::{Attr, Bin, Lower, V, pred};

impl Lower<'_> {
    /// On WARP 1.0.22, divergent WaveReadLaneAt can select the wrong
    /// four-lane group. Hold the group constant in each read, then select
    /// the requested group's result. See #2949.
    pub(super) fn read_lane_on_warp(&mut self, value: V, lane: V, active: [V; 2]) -> V {
        let c = self.cur;
        let me = self.lane_index();
        let i32t = self.s.t.i32;
        let first = self.s.op(c, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[me]);
        let inner_mask = self.s.m.i32(3);
        let outer_mask = self.s.m.i32(!3);
        let inner = self.s.m.bin(c, Bin::And, lane, inner_mask);
        let outer = self.s.m.bin(c, Bin::And, lane, outer_mask);
        let zero = self.s.m.i32(0);
        let one = self.s.m.i32(1);
        let bank_mask = self.s.m.i32(15);
        let bit = self.s.m.bin(c, Bin::Shl, one, inner);
        let mut result = zero;
        for start in (0..self.wave()).step_by(4) {
            let base = self.s.m.i32(start);
            let shift = self.s.m.i32(start % 32);
            let mask = self.s.m.bin(c, Bin::LShr, active[(start / 32) as usize], shift);
            let mask = self.s.m.bin(c, Bin::And, mask, bank_mask);
            let nonempty = self.s.m.cmp(c, pred::NE, mask, zero);
            let possible = self.s.m.bin(c, Bin::And, mask, bit);
            let live = self.s.m.cmp(c, pred::NE, possible, zero);
            let fallback = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[mask]);
            let selected = self.s.m.select(c, live, inner, fallback);
            let bank_lane = self.s.m.bin(c, Bin::Or, selected, base);
            // An unselected read also needs a live host source. Use the
            // wave's first lane for a group with no active source.
            let source = self.s.m.select(c, nonempty, bank_lane, first);
            let read = self.read_lane(value, source);
            let requested = self.s.m.cmp(c, pred::EQ, outer, base);
            result = self.s.m.select(c, requested, read, result);
        }
        result
    }
}
