//! Pixel input, interpolation and barycentric lowering.
//!
//! Console pixel shaders receive interpolation weights in VGPRs. Most only
//! feed them to `v_interp_*`, but programs can also read or compute them;
//! those forms are host `SV_Barycentrics` values and per-vertex attribute
//! reads. The lane-edge proofs here show whether a register the console
//! compiler reused still holds the weight in the lanes that interpolate.

use super::{Lower, Val, Why, packed_half};
use crate::emit::bitcode::{Attr, Bin, Cast, V};
use crate::emit::container::SigElement;
use crate::emit::shader::flags;
use crate::gfx10::{Fields, Operand};
use crate::io::{Attribute, Barycentric, Elsewhere, Input, Interp};
use crate::ir::{Inst, Reg, Value};
use crate::native::StageIo;
use crate::resource::Site;
use crate::structure::{Atom, Bool, LaneExit};

/// How an interpolation evaluates an input channel.
#[derive(Debug, Clone, Copy)]
enum Eval {
    /// Loaded with this DXIL interpolation mode (1: constant, P0).
    Mode(u8),
    /// From the vertex values of a constant element: P0 when `p0`, plus
    /// i (P1 - P0), plus j (P2 - P0), weights as floats.
    Weights {
        p0: bool,
        i: Option<V>,
        j: Option<V>,
    },
}

impl Lower<'_> {
    pub(super) fn input(&mut self, input: Input) -> Val {
        let (kind, col) = match input {
            Input::VertexId => (1, 0),
            Input::InstanceId => (2, 0),
            Input::Position(c) => (3, c),
            Input::Attribute { row, channel } => {
                return match self.find_input(row, channel) {
                    Some((id, col)) => self.load_input(id, col),
                    None => Val::Missing("a vertex input the signature lacks"),
                };
            }
            Input::Barycentric { kind, index } => return self.barycentric(kind, index),
            Input::PrimitiveId => return Val::Missing("the primitive id"),
            // The console code tests it against 0 (`v_cmp_ne_u32`), as the
            // host's value can be.
            Input::FrontFace => (13, 0),
            Input::ThreadIdInGroup(c) => return self.thread_id(c),
            Input::GroupId(c) => return self.compute_id("groupId.i32", 94, c),
            Input::GroupInfo => return Val::Missing("the thread-group info SGPR"),
            Input::MeshGroup => {
                let Val::I32(g) = self.compute_id("groupId.i32", 94, 0) else {
                    unreachable!()
                };
                let n = self.s.m.i32(self.x.mesh_stride);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Mul, g, n));
            }
            Input::MergedWaveInfo => {
                // shortcut: only the wave index (bits 24-27) is set; the
                // console's other bits (lane counts) are read by no program.
                let f = self.mesh_thread();
                let shift = self.s.m.i32(self.wave().trailing_zeros());
                let w = self.s.m.bin(self.s.entry, Bin::LShr, f, shift);
                let k24 = self.s.m.i32(24);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Shl, w, k24));
            }
            _ => return Val::Missing("a pixel system input"),
        };
        let free = self
            .s
            .inputs
            .iter()
            .filter_map(|e| e.row)
            .map(|r| r + 1)
            .max()
            .unwrap_or(0);
        let id = match self.s.inputs.iter().position(|e| e.kind == kind) {
            Some(id) => {
                if kind != 3 {
                    self.s.inputs[id].row.get_or_insert(free);
                }
                id
            }
            None if kind >= 3 => return Val::Missing("a pixel system input the signature lacks"),
            None => {
                // The console signature leaves out vertex and instance ids; PC
                // gives them a row after the others.
                let row = self
                    .s
                    .inputs
                    .iter()
                    .filter_map(|e| e.row)
                    .map(|r| r + 1)
                    .max()
                    .unwrap_or(0);
                self.s.inputs.push(SigElement {
                    name: ["", "SV_VertexID", "SV_InstanceID"][kind as usize].into(),
                    index: 0,
                    kind,
                    comp: 5,
                    interp: 0,
                    row: Some(row),
                    col: 0,
                    cols: 1,
                    usage: 0,
                    half: false,
                });
                self.s.inputs.len() - 1
            }
        };
        self.load_input(id, col)
    }

    fn find_input(&self, row: u8, channel: u8) -> Option<(usize, u8)> {
        self.s.inputs.iter().enumerate().find_map(|(id, e)| {
            let r = e.row?;
            // `then`, not `then_some`: the channel is only past `col` when
            // the condition holds.
            (r == u32::from(row) && channel >= e.col && channel < e.col + e.cols)
                .then(|| (id, channel - e.col))
        })
    }

    fn load_input(&mut self, id: usize, col: u8) -> Val {
        if self.s.inputs[id].half {
            // shortcut: 16-bit vertex inputs (the fetch routine's) are not
            // unpacked; pixel inputs are (`interp`).
            return Val::Missing("a 16-bit input");
        }
        self.load_any_input(id, col)
    }

    /// An input element's channel as declared (16-bit ones as 32-bit floats).
    fn load_any_input(&mut self, id: usize, col: u8) -> Val {
        let e = &mut self.s.inputs[id];
        e.usage |= 1 << col;
        let (int, bit) = (e.comp != 9, e.comp == 1);
        let ty = if bit {
            self.s.t.i1
        } else if int {
            self.s.t.i32
        } else {
            self.s.t.f32
        };
        let suffix = if bit {
            ".i1"
        } else if int {
            ".i32"
        } else {
            ".f32"
        };
        let entry = self.s.entry;
        let args = [
            self.s.m.i32(id as u32),
            self.s.m.i32(0),
            self.s.m.i8(col),
            self.s.m.undef(self.s.t.i32),
        ];
        let v = self.s.op(
            entry,
            &format!("loadInput{suffix}"),
            Attr::ReadNone,
            ty,
            4,
            &args,
        );
        if bit {
            let z = self.s.m.cast(entry, Cast::ZExt, v, self.s.t.i32);
            Val::I32(z)
        } else if int {
            Val::I32(v)
        } else {
            Val::I32(self.s.m.cast(entry, Cast::Bitcast, v, self.s.t.i32))
        }
    }

    /// Loads `v` if it is a barycentric input VGPR not loaded yet (see
    /// `inputs`): `None` if it is not one.
    pub(super) fn lazy_barycentric(&mut self, v: Value) -> Option<Result<(), Why>> {
        let (kind, index) = self.x.io.inputs.iter().find_map(|&(w, i)| match i {
            Input::Barycentric { kind, index } if w == v => Some((kind, index)),
            _ => None,
        })?;
        let val = self.barycentric(kind, index);
        self.vals.insert(v, val);
        Some(match val {
            Val::Missing(w) => Err(w),
            _ => Ok(()),
        })
    }

    /// Barycentric `index` (0 i, 1 j) of `kind` as a value: a channel of
    /// `SV_Barycentrics` declared with the same interpolation mode.
    ///
    /// GFX10 interpolates P0 + i (P1 - P0) + j (P2 - P0), so i and j are
    /// the weights of vertices 1 and 2. The host's weights (x, y, z) are
    /// those of vertices 0, 1 and 2, vertex 0 being the provoking vertex,
    /// which is the console's P0 (`v_interp_mov` P0 is lowered to a flat
    /// input): i is y and j is z.
    /// shortcut: triangles only; on lines and points the console's i and j
    /// are not checked against the host's weights.
    fn barycentric(&mut self, kind: Barycentric, index: u8) -> Val {
        let Some(mode) = interp_mode(kind) else {
            return Val::Missing("pull-model barycentrics");
        };
        if index > 1 {
            return Val::Missing("pull-model barycentrics");
        }
        // One element per interpolation mode (the console's own, if its
        // signature declares one). DXIL allows two, one perspective and
        // one linear: a second perspective (or linear) mode is refused,
        // since its weights differ from the first's where the modes'
        // sample points do (centroid weights stay inside the triangle,
        // center ones need not), so reusing the declared element would
        // be a silent wrong value.
        let bary = |e: &SigElement| e.kind == 28;
        let clash = self.s.inputs.iter().any(|e| {
            bary(e) && e.usage != 0 && e.interp != mode && linear(e.interp) == linear(mode)
        });
        if clash {
            return Val::Missing(if linear(mode) {
                "barycentrics of two linear modes (SV_Barycentrics allows one)"
            } else {
                "barycentrics of two perspective modes (SV_Barycentrics allows one)"
            });
        }
        let id = match self
            .s
            .inputs
            .iter()
            .position(|e| bary(e) && (e.interp == mode || e.usage == 0))
        {
            Some(id) => id,
            None => {
                let index = self.s.inputs.iter().filter(|e| bary(e)).count() as u32;
                if index >= 2 {
                    return Val::Missing("barycentrics of a third interpolation mode");
                }
                // A system value the validator wants unpacked (packing
                // location -1), like other non-packed system values.
                self.s.inputs.push(SigElement {
                    name: "SV_Barycentrics".into(),
                    index,
                    kind: 28,
                    comp: 9,
                    interp: mode,
                    row: None,
                    col: 0,
                    cols: 3,
                    usage: 0,
                    half: false,
                });
                self.s.inputs.len() - 1
            }
        };
        let e = &mut self.s.inputs[id];
        if e.col != 0 || e.cols < 3 {
            return Val::Missing("a console barycentrics element without i and j");
        }
        e.row = None;
        e.interp = mode;
        self.s.flags |= flags::BARYCENTRICS;
        self.load_any_input(id, index + 1)
    }

    pub(super) fn interp(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        let interp = self.interps.get(&site).copied();
        // A first half whose pair's registers may be overwritten where it
        // runs is a computed interpolation too (see `computed_first`).
        let computed_first = if interp.is_none() {
            self.computed_first(site)
        } else {
            None
        };
        if interp.is_none() && computed_first.is_none() {
            return Err("the first half of an interpolation");
        }
        let name = inst.name();
        let (dst, src) = match &inst.fields {
            Fields::Vintrp { dst, src, .. } => (*dst, Some(*src)),
            Fields::Valu(v) => {
                if v.abs != 0 || v.neg != 0 {
                    return Err("interpolation operand modifiers");
                }
                (v.dst.ok_or("an interpolation")?, v.src.get(1).copied())
            }
            _ => unreachable!(),
        };
        let Operand::Vector { reg, .. } = dst else {
            return Err("an interpolation");
        };
        // A float operand of the instruction.
        let float = |l: &mut Self, op: Option<Operand>| -> Result<V, Why> {
            let x = l.operand(inst, &op.ok_or("an interpolation operand")?)?;
            Ok(l.as_float(x))
        };
        // What the result adds to the evaluated channel (a second half's
        // first half, when the halves evaluate on their own).
        let mut add = None;
        let weighted_first = |l: &mut Self| -> Result<Eval, Why> {
            let i = Some(float(l, src)?);
            Ok(Eval::Weights {
                p0: true,
                i,
                j: None,
            })
        };
        let weighted_second = |l: &mut Self, add: &mut Option<V>| -> Result<Eval, Why> {
            let old = match &inst.fields {
                Fields::Valu(v) => v.src.get(2).copied(),
                _ => Some(dst),
            };
            *add = Some(float(l, old)?);
            let j = Some(float(l, src)?);
            Ok(Eval::Weights {
                p0: false,
                i: None,
                j,
            })
        };
        let (attribute, eval) = if let Some(attribute) = computed_first {
            (attribute, weighted_first(self)?)
        } else {
            match interp.expect("an interpolation") {
                Interp::Interpolated {
                    attribute,
                    barycentric,
                    unless,
                    ..
                } => {
                    let (attribute, barycentric) = (*attribute, *barycentric);
                    if !self.avoids(site, unless) {
                        // The register the instruction reads holds the
                        // overwritten value in the lanes that run it: use it
                        // as the weight, as the hardware does.
                        (attribute, weighted_second(self, &mut add)?)
                    } else {
                        let mode = interp_mode(barycentric).ok_or("pull-model interpolation")?;
                        let eval = if self.per_vertex(attribute.row) {
                            // The element is read per vertex elsewhere, so it is
                            // constant: weigh its vertices with SV_Barycentrics.
                            let mut ij = [None; 2];
                            for (index, w) in ij.iter_mut().enumerate() {
                                let x = match self.barycentric(barycentric, index as u8) {
                                    Val::I32(x) => x,
                                    Val::Missing(w) => return Err(w),
                                    Val::Bit(_) => unreachable!(),
                                };
                                *w = Some(self.as_float(x));
                            }
                            Eval::Weights {
                                p0: true,
                                i: ij[0],
                                j: ij[1],
                            }
                        } else {
                            Eval::Mode(mode)
                        };
                        (attribute, eval)
                    }
                }
                // P0, the provoking vertex: a flat (nointerpolation) input.
                Interp::Vertex { attribute, vertex } if *vertex == 2 => {
                    (*attribute, Eval::Mode(1))
                }
                // P10 and P20: what the first and second halves weigh, P1 - P0
                // and P2 - P0, which is what the hardware holds for an
                // attribute the program also interpolates.
                Interp::Vertex { attribute, vertex } if self.interpolated(attribute.row) => {
                    let one = Some(self.s.m.f32(1f32.to_bits()));
                    let (i, j) = if *vertex == 0 { (one, None) } else { (None, one) };
                    (*attribute, Eval::Weights { p0: false, i, j })
                }
                Interp::Vertex { .. } => return Err("a non-provoking vertex value"),
                // A half of an interpolation with computed i or j: P0 + i P10,
                // or the old dst (f16: the third operand) + j P20.
                Interp::Weighted { attribute } if name.starts_with("v_interp_p1") => {
                    (*attribute, weighted_first(self)?)
                }
                Interp::Weighted { attribute } => {
                    (*attribute, weighted_second(self, &mut add)?)
                }
            }
        };
        let packing = match &self.x.native.io {
            StageIo::Ps { inputs, .. } => inputs
                .iter()
                .find(|i| i.row == attribute.row && i.fp16)
                .map(|i| i.pack2),
            _ => None,
        };
        let f16 = name.ends_with("_f16");
        // The evaluated channel plus `add`, as a float.
        let total = |l: &mut Self, row: u8, ch: u8| -> Result<V, Why> {
            let x = l.eval_channel(row, ch, eval)?;
            let x = l.as_float(x);
            Ok(match add {
                Some(a) => l.s.m.bin(l.cur, Bin::Add, a, x),
                None => x,
            })
        };
        let x = match packing {
            None if f16 || attribute.high => return Err("a 16-bit read of a 32-bit input"),
            None => {
                let x = total(self, attribute.row, attribute.channel)?;
                self.as_int(x)
            }
            Some(pack2) => {
                // 16-bit values: one in the low half of each dword, or two
                // per dword (`packed_half`); the host's are 32-bit floats.
                let half = |l: &mut Self, h: u8| -> Result<Option<V>, Why> {
                    let src = if pack2 {
                        packed_half(&l.s.inputs, attribute.row, h)
                    } else {
                        h.is_multiple_of(2).then_some((attribute.row, h / 2))
                    };
                    match src {
                        Some((row, ch)) => total(l, row, ch).map(Some),
                        None => Ok(None),
                    }
                };
                let h = 2 * attribute.channel + u8::from(attribute.high);
                let c = self.cur;
                if name == "v_interp_p1ll_f16" {
                    // An f32 result from 16-bit vertex values.
                    let f = half(self, h)?.ok_or("a packed input half no element holds")?;
                    self.as_int(f)
                } else if f16 {
                    let f = half(self, h)?.ok_or("a packed input half no element holds")?;
                    let lo = self.narrow16(f);
                    // omod and clamp are part of the f16 instruction; op_sel[3]
                    // picks the destination half.
                    let lo = self.half_modifiers(inst, lo);
                    return self.write_half(inst, lo, None);
                } else if matches!(eval, Eval::Mode(1)) {
                    // The whole dword of a flat input: both halves.
                    let h = 2 * attribute.channel;
                    let z = self.s.m.i32(0);
                    let lo = match half(self, h)? {
                        Some(f) => self.narrow16(f),
                        None => z,
                    };
                    let hi = match half(self, h + 1)? {
                        Some(f) => self.narrow16(f),
                        None => z,
                    };
                    let k16 = self.s.m.i32(16);
                    let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                    self.s.m.bin(c, Bin::Or, lo, hi)
                } else {
                    return Err("32-bit interpolation of a 16-bit input");
                }
            }
        };
        // A pixel interpolates in every lane; guards only hide inactive ones.
        self.write(inst, Reg::V(reg), x, None);
        Ok(())
    }

    /// The attribute of a first interpolation half at `site` whose pair's
    /// barycentric register is overwritten in lanes that reach the second
    /// half (so both halves evaluate with the values they read).
    fn computed_first(&self, site: Site) -> Option<Attribute> {
        self.x.io.interps.iter().find_map(|(second, interp)| match interp {
            Interp::Interpolated {
                attribute,
                first,
                unless,
                ..
            } if first.contains(&site) && !self.avoids(*second, unless) => Some(*attribute),
            _ => None,
        })
    }

    /// Whether no lane that runs the instruction at `site` has been where
    /// `unless` says: a lane that came along such an edge or ran (skipped)
    /// such an instruction meets its condition, which the instruction's must
    /// contradict. Those lane bits must not change in between, so none of
    /// those blocks is on a cycle. A lane mask joined where such an edge
    /// enters holds, for the lanes that came along it, the edge's mask.
    fn avoids(&self, (b, i): Site, unless: &[Elsewhere]) -> bool {
        let st = self.x.st;
        let runs = [
            (st.reach[b], true),
            (st.guards[b][i].unwrap_or(Bool::TRUE), true),
        ];
        let mut atoms = st.bools.support(runs[0].0);
        atoms.extend(st.bools.support(runs[1].0));
        unless.iter().all(|&u| {
            let (mut all, at, mut links) = match u {
                Elsewhere::Edge(q, to) => {
                    let all = self.lane_edge(q, to);
                    let block = &self.x.p.blocks[to];
                    let n = block.preds.iter().position(|&p| p == q).expect("an edge");
                    let links: Vec<(Bool, Bool)> = block
                        .phis
                        .iter()
                        .filter(|ph| atoms.contains(&Atom::Phi(ph.value)))
                        .filter_map(|ph| {
                            Some((*st.masks.get(&ph.value)?, *st.masks.get(&ph.args[n])?))
                        })
                        .collect();
                    (all, to, links)
                }
                Elsewhere::Write((w, k)) | Elsewhere::Skip((w, k)) => {
                    let guard = st.guards[w][k].unwrap_or(Bool::TRUE);
                    let ran = matches!(u, Elsewhere::Write(_));
                    (vec![(st.reach[w], true), (guard, ran)], w, vec![])
                }
            };
            all.extend(runs);
            // Each joined mask equals the incoming one: try both values.
            links.truncate(6);
            let some_lane = (0..1u32 << links.len()).any(|bits| {
                let mut all = all.clone();
                for (k, &(joined, incoming)) in links.iter().enumerate() {
                    let v = bits >> k & 1 != 0;
                    all.extend([(joined, v), (incoming, v)]);
                }
                st.bools.meet(&all)
            });
            !self.on_cycle(at) && !some_lane
        })
    }

    /// Whether the program reads attribute row `row` per vertex (an
    /// interpolation with computed barycentrics, or P10 or P20): then its
    /// element is declared constant and read with `attributeAtVertex`.
    fn per_vertex(&self, row: u8) -> bool {
        self.x.io.interps.iter().any(|(site, i)| match i {
            Interp::Weighted { attribute } => attribute.row == row,
            Interp::Vertex { attribute, vertex } => {
                attribute.row == row && *vertex != 2 && self.interpolated(row)
            }
            Interp::Interpolated {
                attribute, unless, ..
            } => attribute.row == row && !self.avoids(*site, unless),
        })
    }

    /// Whether the program interpolates attribute row `row` (so the
    /// hardware holds P1 - P0 and P2 - P0 for it).
    fn interpolated(&self, row: u8) -> bool {
        self.x.io.interps.iter().any(|(_, i)| match i {
            Interp::Interpolated { attribute, .. } | Interp::Weighted { attribute } => {
                attribute.row == row
            }
            Interp::Vertex { .. } => false,
        })
    }

    /// Channel `ch` of input row `row` evaluated as `eval` says, as raw bits.
    fn eval_channel(&mut self, row: u8, ch: u8, eval: Eval) -> Result<V, Why> {
        let (id, col) = self
            .find_input(row, ch)
            .ok_or("an input the signature lacks")?;
        let mode = match eval {
            Eval::Mode(mode) => mode,
            Eval::Weights { .. } => 1,
        };
        let e = &mut self.s.inputs[id];
        if e.interp != mode && e.usage != 0 {
            return Err("one input interpolated two ways");
        }
        e.interp = mode;
        let Eval::Weights { p0, i, j } = eval else {
            return match self.load_any_input(id, col) {
                Val::I32(x) => Ok(x),
                _ => Err("an input"),
            };
        };
        let v0 = self.vertex_input(id, col, 0);
        let c = self.cur;
        let mut acc = p0.then_some(v0);
        for (k, w) in [(1, i), (2, j)] {
            let Some(w) = w else { continue };
            let vk = self.vertex_input(id, col, k);
            let d = self.s.m.bin(c, Bin::Sub, vk, v0);
            acc = Some(match acc {
                Some(a) => self.fmad(w, d, a),
                None => self.s.m.bin(c, Bin::Mul, w, d),
            });
        }
        let acc = acc.ok_or("an interpolation")?;
        Ok(self.as_int(acc))
    }

    /// Vertex `k` (0 the provoking vertex, as `barycentric` orders them)
    /// of a constant input element's channel, as a float.
    fn vertex_input(&mut self, id: usize, col: u8, k: u8) -> V {
        let e = &mut self.s.inputs[id];
        e.usage |= 1 << col;
        let int = e.comp != 9;
        let (ty, suffix) = if int {
            (self.s.t.i32, ".i32")
        } else {
            (self.s.t.f32, ".f32")
        };
        self.s.flags |= flags::BARYCENTRICS;
        let entry = self.s.entry;
        let args = [
            self.s.m.i32(id as u32),
            self.s.m.i32(0),
            self.s.m.i8(col),
            self.s.m.i8(k),
        ];
        let v = self.s.op(
            entry,
            &format!("attributeAtVertex{suffix}"),
            Attr::ReadNone,
            ty,
            137,
            &args,
        );
        if int {
            self.s.m.cast(entry, Cast::Bitcast, v, self.s.t.f32)
        } else {
            v
        }
    }

    /// What holds in a lane that goes from IR block `q` to `b`.
    fn lane_edge(&self, q: usize, b: usize) -> Vec<(Bool, bool)> {
        let st = self.x.st;
        let mut all = vec![(st.reach[q], true)];
        if let LaneExit::Branch {
            when,
            taken,
            not_taken,
        } = st.exits[q]
            && taken != not_taken
        {
            all.push((when, taken == b));
        }
        all
    }

    /// Whether every lane enters IR block `b` along the edge the console's
    /// wave takes, so that an SGPR joined there holds what that edge
    /// brought: every branch into it is uniform (SCC values). The wave runs
    /// one path; a per-lane branch into the join (an EXEC skip, a loop's
    /// per-lane exit) is where lanes would take edges the wave did not, and
    /// the console compiler's if/else always skips into its join that way.
    /// Lanes that left EXEC inside a uniform loop (the BVH walks) take no
    /// edge into its header; the others take the wave's.
    pub(super) fn uniform_join(&self, b: usize) -> bool {
        let st = self.x.st;
        let preds = &self.x.p.blocks[b].preds;
        preds.iter().filter(|&&q| self.block[q].is_some()).all(|&q| match st.exits[q] {
            LaneExit::Branch { when, taken, not_taken } if taken != not_taken => {
                st.bools.support(when).iter().all(|a| matches!(a, Atom::Uniform(_)))
            }
            _ => true,
        })
    }

    /// Whether no lane goes from IR block `q` to `b`.
    pub(super) fn never_taken(&self, q: usize, b: usize) -> bool {
        !self.x.st.bools.meet(&self.lane_edge(q, b))
    }

    /// Whether, at a per-lane join `b` of a lane mask whose register holds
    /// data along edge `n`, some edge brings a mask and no lane taking edge
    /// `n` is in any of them (a compare inside a region those lanes skip).
    /// The console wave runs one path: on a mask edge, the lanes of edge `n`
    /// read their bit of that mask, 0. The compiler merges no value for them,
    /// so it holds none for them either where the whole wave skips (the data
    /// there is a reused register): 0 is theirs.
    pub(super) fn no_mask_lane_on_edge(&self, b: usize, args: &[Value], n: usize) -> bool {
        let st = self.x.st;
        let preds = &self.x.p.blocks[b].preds;
        let on_n = self.lane_edge(preds[n], b);
        let mut some = false;
        for (k, arg) in args.iter().enumerate() {
            let Some(&m) = st.masks.get(arg).filter(|_| k != n) else {
                continue;
            };
            some = true;
            let mut lanes = on_n.clone();
            lanes.push((m, true));
            if st.bools.meet(&lanes) {
                return false;
            }
        }
        some
    }

    /// Whether IR block `b` can reach itself.
    fn on_cycle(&self, b: usize) -> bool {
        let blocks = &self.x.p.blocks;
        let mut seen = vec![false; blocks.len()];
        let mut stack = blocks[b].exit.successors();
        while let Some(x) = stack.pop() {
            if x == b {
                return true;
            }
            if !std::mem::replace(&mut seen[x], true) {
                stack.extend(blocks[x].exit.successors());
            }
        }
        false
    }

    /// VOP3 f16 output modifiers: omod, then clamp, applied in f32 so the
    /// half conversion rounds once (the order `write_f` uses).
    fn half_modifiers(&mut self, inst: &Inst, h: V) -> V {
        let Fields::Valu(v) = &inst.fields else {
            return h;
        };
        if v.omod == 0 && !v.clamp {
            return h;
        }
        let c = self.cur;
        let f32t = self.s.t.f32;
        let mut f = self
            .s
            .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
        if v.omod != 0 {
            let factor = [1.0f32, 2.0, 4.0, 0.5][usize::from(v.omod)];
            let k = self.s.m.f32(factor.to_bits());
            f = self.s.m.bin(c, Bin::Mul, f, k);
        }
        if v.clamp {
            f = self.unary(7, f);
        }
        self.s
            .op(c, "legacyF32ToF16", Attr::ReadNone, self.s.t.i32, 130, &[f])
    }
}

/// Whether DXIL interpolation mode `mode` is a linear (noperspective) one.
fn linear(mode: u8) -> bool {
    matches!(mode, 4 | 5 | 7)
}

/// The DXIL interpolation mode of a kind of barycentrics (none for the
/// pull model).
fn interp_mode(kind: Barycentric) -> Option<u8> {
    Some(match kind {
        Barycentric::PerspectiveCenter => 2,
        Barycentric::PerspectiveCentroid => 3,
        Barycentric::PerspectiveSample => 6,
        Barycentric::LinearCenter => 4,
        Barycentric::LinearCentroid => 5,
        Barycentric::LinearSample => 7,
        Barycentric::PerspectivePullModel => return None,
    })
}
