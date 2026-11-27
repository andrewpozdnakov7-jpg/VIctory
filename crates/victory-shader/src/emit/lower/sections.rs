use std::collections::HashMap;

use super::{Fields, Inst, Lower, Operand, Reg, V, Val, Value, Why};

impl Lower<'_> {
    /// Finds the whole-wave sections and the values they leave.
    pub(super) fn find_sections(&mut self) {
        let p = self.x.p;
        let mut inside: HashMap<Value, (usize, usize, usize)> = HashMap::new();
        for b in 0..p.blocks.len() {
            let found = super::super::wave::sections(p, b);
            for (&start, sec) in &found {
                for i in start + 1..sec.end {
                    for &(_, w) in &p.blocks[b].insts[i].writes {
                        inside.insert(w, (b, start, sec.end));
                    }
                }
            }
            if !found.is_empty() {
                self.sections.insert(b, found);
            }
        }
        if inside.is_empty() {
            return;
        }
        for (b, block) in p.blocks.iter().enumerate() {
            for phi in &block.phis {
                self.section_out
                    .extend(phi.args.iter().filter(|a| inside.contains_key(a)));
            }
            for (i, inst) in block.insts.iter().enumerate() {
                for &(_, r) in &inst.reads {
                    if let Some(&(sb, start, end)) = inside.get(&r)
                        && !(sb == b && i > start && i < end)
                    {
                        self.section_out.insert(r);
                        if inst.name() == "v_readlane_b32"
                            && let Some((src, lane)) = Self::section_lane_operand(inst)
                            && src == r
                        {
                            self.section_lane_reads.entry(r).or_default().insert(lane);
                        }
                    }
                }
            }
        }
    }

    /// An instruction inside a whole-wave section: the values that leave
    /// the section are made from what the simulation found them to be.
    pub(super) fn section_inst(&mut self, b: usize, i: usize, start: usize) {
        const UNKNOWN: Why = "a whole-wave section step the simulation does not follow";
        let inst = &self.x.p.blocks[b].insts[i];
        for &(_, w) in &inst.writes {
            let val = if self.section_out.contains(&w) {
                let sec = &self.sections[&b][&start];
                let shape = sec.shapes.get(&w).copied();
                if shape.is_none()
                    && let Some(lanes) = sec.lanes.get(&w)
                {
                    // Evaluate reductions where the whole section runs. A
                    // later v_readlane can be inside a divergent branch.
                    let reads: Vec<_> = self
                        .section_lane_reads
                        .get(&w)
                        .into_iter()
                        .flatten()
                        .filter_map(|&lane| Some((lane, (*lanes.get(lane as usize)?)?)))
                        .collect();
                    for (lane, sh) in reads {
                        let value = self.shape(sh);
                        self.section_lanes.insert((w, lane), value);
                    }
                }
                match shape.ok_or(UNKNOWN).and_then(|sh| self.shape(sh)) {
                    Ok(x) => Val::I32(x),
                    Err(why) => Val::Missing(why),
                }
            } else {
                Val::Missing(UNKNOWN)
            };
            self.vals.insert(w, val);
        }
    }

    /// The value and constant lane a v_readlane reads.
    fn section_lane_operand(inst: &Inst) -> Option<(Value, u32)> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        let Some(&Operand::Vector { reg, .. }) = v.src.first() else {
            return None;
        };
        let src = inst.read(Reg::V(reg))?;
        let lane = match *v.src.get(1)? {
            Operand::Const(c) => crate::resource::inline(c)?,
            Operand::Literal(x) => x,
            _ => return None,
        };
        Some((src, lane))
    }

    pub(super) fn section_lane(&self, inst: &Inst) -> Option<Result<V, Why>> {
        self.section_lanes
            .get(&Self::section_lane_operand(inst)?)
            .copied()
    }
}
