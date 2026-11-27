use super::Lower;
use crate::emit::bitcode::{Attr, Rmw, V, pred};

impl Lower<'_> {
    /// A scalar mask branch in one logical wave can span several host
    /// waves. All group threads reach this point, including inactive guest
    /// lanes. Only the eligible lanes contribute to the vote.
    pub(super) fn group_vote(&mut self, bit: V) -> V {
        let i32t = self.s.t.i32;
        let slot = *self
            .group_vote_slot
            .get_or_insert_with(|| self.s.m.global("branch_vote", i32t, 3));
        let flat = self.s.op(
            self.s.entry,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            i32t,
            96,
            &[],
        );
        let zero = self.s.m.i32(0);
        let one = self.s.m.i32(1);
        let first = self.s.m.cmp(self.cur, pred::EQ, flat, zero);
        let guard = self.begin_guarded(Some(first));
        self.s.m.store(self.cur, slot, zero);
        self.end_guarded(guard, &[]);

        let mode = self.s.m.i32(1 | 8);
        let void = self.s.t.void;
        self.s
            .op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        let guard = self.begin_guarded(Some(bit));
        self.s.m.atomic(self.cur, Rmw::Or, slot, one);
        self.end_guarded(guard, &[]);
        self.s
            .op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        let vote = self.s.m.load(self.cur, slot);
        // Finish every read before the next branch can reuse the slot.
        self.s
            .op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        self.s.m.cmp(self.cur, pred::NE, vote, zero)
    }
}
