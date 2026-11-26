use super::{Lower, V};
use crate::emit::bitcode::{Bin, pred};
use crate::emit::{VIEW_FLOAT_TEXTURE, VIEW_INTEGER, VIEW_SIGNED};

impl Lower<'_> {
    pub(super) fn scrub_nan(&mut self, value: V) -> V {
        if super::super::nan_scrub() {
            self.scrub_nonfinite(value)
        } else {
            value
        }
    }

    pub(super) fn scrub_texture_store(&mut self, handle: V, values: &mut [V; 4]) {
        if !super::super::nan_scrub() {
            return;
        }
        let Some(&index) = self.s.texture_uav_index.get(&handle) else {
            return;
        };
        let word = self.view_word(index);
        let mask = self.s.m.i32(VIEW_INTEGER | VIEW_SIGNED);
        let kind = self.s.m.bin(self.cur, Bin::And, word, mask);
        let float = self.s.m.i32(VIEW_FLOAT_TEXTURE);
        let scrub = self.s.m.cmp(self.cur, pred::EQ, kind, float);
        for value in values {
            let cleaned = self.scrub_nonfinite(*value);
            *value = self.s.m.select(self.cur, scrub, cleaned, *value);
        }
    }

    // An unordered comparison of |value| with infinity catches NaNs too.
    fn scrub_nonfinite(&mut self, value: V) -> V {
        let c = self.cur;
        let magnitude = self.unary(6, value);
        let infinity = self.s.m.f32(0x7f80_0000);
        let bad = self.s.m.cmp(c, pred::UGE, magnitude, infinity);
        let zero = self.s.m.f32(0);
        self.s.m.select(c, bad, zero, value)
    }
}
