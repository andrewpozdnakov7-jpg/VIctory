//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler; a 5.1 shader with its root signature inside).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint arg_offset; };
RWByteAddressBuffer args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
// Clamps the record at `at` in place; its cost.
uint clamped(uint at) {
    uint cost;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint count = min(args.Load(at), 1u << 24);
        uint instances = min(args.Load(at + 4), (1u << 26) / max(count, 1u));
        cost = count * instances;
        args.Store2(at, uint2(count, instances));
    } else if (kind == 2) {
        uint3 g = min(args.Load3(at), uint3(65535, 65535, 65535));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        args.Store3(at, g);
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = args.Load3(at);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128;
        args.Store3(at, g);
    }
    return cost;
}
// A record past the budget (no cost reaches it).
static const uint PAST = 0xffffffffu;
groupshared uint costs[64];
// One group walks the records in tiles of 64: every lane clamps its own
// record, then the first lane charges the tile's records in order. No
// other group shares the budget: a shared add wraps once enough records
// are rejected, and a compare-exchange retries more the more records fit.
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    uint left = 0;
    if (lane == 0) {
        left = BUDGET - min(budget.Load(0), BUDGET);
    }
    for (uint tile = 0; tile < records / 64 + min(records % 64, 1u); tile++) {
        uint record = tile * 64 + lane;
        uint at = record * stride + arg_offset;
        uint cost = 0;
        if (record < records) {
            cost = clamped(at);
        }
        costs[lane] = cost;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0) {
            for (uint i = 0; i < 64; i++) {
                if (costs[i] > left) {
                    // The first record that does not fit takes what is
                    // left, so those after it run empty too.
                    costs[i] = PAST;
                    left = 0;
                } else {
                    left -= costs[i];
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (costs[lane] == PAST) {
            // Draws keep their layout with no vertices; groups become zero.
            args.Store(at, 0);
        }
    }
    if (lane == 0) {
        budget.Store(0, BUDGET - left);
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at `arg_offset` bytes into each
/// record, at host address `gpu` in `buffer`, and leaves the buffer ready to
/// be read as arguments. Replaces the list's compute root signature and
/// pipeline (callers bind theirs after). False if the clamp is not available.
pub(crate) fn bound(
    list: &HostList,
    buffer: &ID3D12Resource,
    gpu: u64,
    (kind, stride, records, arg_offset): (u32, u32, u32, u32),
) -> bool {
    let (Some(c), Some(budget)) = (clamp(), list.budget()) else {
        return false;
    };
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) || !arg_offset.is_multiple_of(4) {
        return false;
    }
    let l = &list.list;
    let constants = [kind, stride, records, arg_offset];
    // SAFETY: an open list; our root signature, pipeline and a live buffer
    // the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        // One group walks every record.
        l.Dispatch(records.min(1), 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(
                buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
        ]);
    }
    true
}

/// One argument of a host command signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HostIndirectArg {
    Constant { param: u32, offset: u32, count: u32 },
    Execute(u32),
}

/// One stage's user-data root-constant parameter, for mapping title indirect
/// root constants onto our host root signature.
pub(crate) struct StageUserData<'a> {
    pub(crate) param: u32,
    pub(crate) dwords: u32,
    pub(crate) entries: &'a [victory_shader::emit::UserData],
}

/// Host command-signature layout and trailing draw/dispatch offset for an
/// indirect execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IndirectPlan {
    /// Final execution kind (PC numbering: 0 draw, 1 indexed, 2 dispatch, 10 mesh).
    pub(crate) kind: u32,
    /// Byte offset of the final draw/dispatch argument inside each record.
    pub(crate) arg_offset: u32,
    /// Host arguments in record order (0 or more `Constant`s, then `Execute(kind)`).
    pub(crate) args: Vec<HostIndirectArg>,
}

/// Maps a title command signature (`stride`, `title_args`) onto `stages`'s
/// host user-data parameters. Supports single-argument draw/dispatch/mesh
/// signatures and multi-argument signatures whose prefix arguments are root
/// constants (`D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT`).
pub(crate) fn plan_indirect(
    stride: u32,
    title_args: &[[u32; 4]],
    stages: &[StageUserData<'_>],
) -> Option<IndirectPlan> {
    const DRAW: u32 = 0;
    const INDEXED: u32 = 1;
    const DISPATCH: u32 = 2;
    const CONSTANT: u32 = 5;
    const MESH: u32 = 10;
    const DISPATCH_SINGLE: u32 = 105;

    let (last, prefix) = title_args.split_last()?;
    let kind = if last[0] == DISPATCH_SINGLE {
        DISPATCH
    } else {
        last[0]
    };
    let exec_bytes = match kind {
        DRAW => 16,
        INDEXED => 20,
        DISPATCH | MESH => 12,
        _ => return None,
    };
    if prefix.is_empty() {
        return Some(IndirectPlan {
            kind,
            arg_offset: 0,
            args: vec![HostIndirectArg::Execute(kind)],
        });
    }

    let mut host_args = Vec::with_capacity(title_args.len());
    let mut arg_offset = 0u32;
    for arg in prefix {
        if arg[0] != CONSTANT {
            return None;
        }
        let title_slot = u8::try_from(arg[1]).ok()?;
        let dest_off = arg[2];
        let num = arg[3];
        if num == 0 {
            return None;
        }
        let end_dword = dest_off.checked_add(num)?;

        // Find every stage/index reading any dword in [dest_off, dest_off + num).
        let mut target_stage: Option<usize> = None;
        let mut base_index: Option<u32> = None;
        for (si, stage) in stages.iter().enumerate() {
            for u in stage.entries {
                if u.root == victory_shader::RootType::Constants
                    && u.slot == title_slot
                    && u.dword >= dest_off
                    && u.dword < end_dword
                {
                    let k = u.dword - dest_off;
                    let candidate = u.index.checked_sub(k)?;
                    match (target_stage, base_index) {
                        (None, None) => {
                            target_stage = Some(si);
                            base_index = Some(candidate);
                        }
                        (Some(s), Some(b)) if s == si && b == candidate => {}
                        _ => return None,
                    }
                }
            }
        }

        let mapped = if let (Some(si), Some(base)) = (target_stage, base_index) {
            let stage = &stages[si];
            let end_idx = base.checked_add(num)?;
            if end_idx > stage.dwords {
                return None;
            }
            // No other user-data entry may live in [base, base + num).
            let collides = stage.entries.iter().any(|u| {
                u.index >= base
                    && u.index < end_idx
                    && !(u.root == victory_shader::RootType::Constants
                        && u.slot == title_slot
                        && u.dword == dest_off + (u.index - base))
            });
            if collides {
                return None;
            }
            HostIndirectArg::Constant {
                param: stage.param,
                offset: base,
                count: num,
            }
        } else {
            // No stage reads these constants; keep record alignment by landing
            // them in an unused contiguous range of some stage's user data.
            let mut free = None;
            'find: for stage in stages {
                if num > stage.dwords {
                    continue;
                }
                for base in 0..=stage.dwords - num {
                    let end_idx = base + num;
                    let used_by_shader = stage
                        .entries
                        .iter()
                        .any(|u| u.index >= base && u.index < end_idx);
                    let used_by_prev = host_args.iter().any(|a| match *a {
                        HostIndirectArg::Constant {
                            param,
                            offset,
                            count,
                        } => param == stage.param && offset < end_idx && offset + count > base,
                        HostIndirectArg::Execute(_) => false,
                    });
                    if !used_by_shader && !used_by_prev {
                        free = Some(HostIndirectArg::Constant {
                            param: stage.param,
                            offset: base,
                            count: num,
                        });
                        break 'find;
                    }
                }
            }
            free?
        };

        host_args.push(mapped);
        arg_offset = arg_offset.checked_add(num.checked_mul(4)?)?;
    }
    if arg_offset.checked_add(exec_bytes)? > stride {
        return None;
    }
    host_args.push(HostIndirectArg::Execute(kind));
    Some(IndirectPlan {
        kind,
        arg_offset,
        args: host_args,
    })
}

/// Converts [`HostIndirectArg`]s to D3D12 argument descriptions.
pub(crate) fn to_d3d12_descs(args: &[HostIndirectArg]) -> Vec<D3D12_INDIRECT_ARGUMENT_DESC> {
    args.iter()
        .map(|&arg| match arg {
            HostIndirectArg::Constant {
                param,
                offset,
                count,
            } => D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT,
                Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0 {
                    Constant: D3D12_INDIRECT_ARGUMENT_DESC_0_1 {
                        RootParameterIndex: param,
                        DestOffsetIn32BitValues: offset,
                        Num32BitValuesToSet: count,
                    },
                },
            },
            HostIndirectArg::Execute(kind) => D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
                ..Default::default()
            },
        })
        .collect()
}

/// After the execution: the buffer back to unordered access, where the
/// title's passes keep using it.
pub(crate) fn done(list: &HostList, buffer: &ID3D12Resource) {
    // SAFETY: an open list and a live buffer.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            buffer,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )])
    };
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_shader::RootType;
    use victory_shader::emit::UserData;

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    /// Clamps `records` records of `kind`, filled by repeating `words`,
    /// `calls` times in one list: each record's first word after (zero when
    /// it runs empty). One record more follows them and must stay as
    /// written. `None` without a host GPU.
    fn clamped(kind: u32, words: &[u32], records: u32, calls: u32) -> Option<Vec<u32>> {
        let host = super::host()?;
        // Words in a record of `kind`.
        let per = match kind {
            0 => 4,
            1 => 5,
            _ => 3,
        };
        let spec = (kind, per as u32 * 4, records, 0);
        let all = per * (records as usize + 1);
        let (buffer, data) = host.shared_buffer(all as u64 * 4).unwrap();
        let data = data.cast::<u32>();
        let list = host.new_list().unwrap();
        // SAFETY: our mapped buffer, written before the list runs and read
        // after it has; a live buffer.
        unsafe {
            for at in 0..all {
                data.add(at).write_volatile(words[at % words.len()]);
            }
            let gpu = buffer.GetGPUVirtualAddress();
            for _ in 0..calls {
                assert!(super::bound(&list, &buffer, gpu, spec));
                super::done(&list, &buffer);
            }
            list.close();
            host.execute(&[&list]);
            for at in all - per..all {
                assert_eq!(data.add(at).read_volatile(), words[at % words.len()]);
            }
            let first = |i: usize| data.add(i * per).read_volatile();
            Some((0..records as usize).map(first).collect())
        }
    }

    /// Rejected records do not wrap the budget: of 128 garbage draws (2^26
    /// vertex instances each once clamped) two run, where a wrapping add
    /// let four; of 600 garbage mesh dispatches 16, not 31.
    #[test]
    fn rejected_records_do_not_wrap_the_budget() {
        let Some(draws) = clamped(0, &[u32::MAX], 128, 1) else {
            return; // no host GPU here
        };
        assert_eq!(draws.iter().filter(|&&count| count != 0).count(), 2);
        let meshes = clamped(10, &[u32::MAX], 600, 1).unwrap();
        assert_eq!(meshes.iter().filter(|&&groups| groups != 0).count(), 16);
    }

    /// The budget goes to the first records, in order: of 200 draws of
    /// 3 * 2^18 vertices the first 170 run (it ends inside the third tile);
    /// small records that all fit all run.
    #[test]
    fn the_budget_goes_to_the_first_records() {
        let Some(draws) = clamped(0, &[3 << 18, 1, 0, 0], 200, 1) else {
            return; // no host GPU here
        };
        assert!(draws[..170].iter().all(|&count| count != 0));
        assert!(draws[170..].iter().all(|&count| count == 0));
        let small = clamped(0, &[1], 1 << 16, 1).unwrap();
        assert!(small.iter().all(|&count| count == 1));
    }

    /// A record that does not fit ends the budget: after two draws of
    /// 3 * 2^24 vertex instances a third takes what is left, and the small
    /// ones after it run empty.
    #[test]
    fn a_record_that_does_not_fit_ends_the_budget() {
        let mut words = [1 << 24, 3, 0, 0].repeat(3);
        words.extend([3, 1, 0, 0].repeat(6));
        let Some(draws) = clamped(0, &words, 8, 1) else {
            return; // no host GPU here
        };
        let ran: Vec<bool> = draws.iter().map(|&count| count != 0).collect();
        assert_eq!(ran, [true, true, false, false, false, false, false, false]);
    }

    /// The budget is the list's: a second execution of the same records
    /// finds it spent, and they all run empty.
    #[test]
    fn a_spent_budget_stays_spent() {
        let Some(twice) = clamped(1, &[u32::MAX], 64, 2) else {
            return; // no host GPU here
        };
        assert!(twice.iter().all(|&count| count == 0));
    }

    #[test]
    fn plan_indirect_maps_root_constants_and_rejects_unsupported() {
        let entries = [
            UserData {
                index: 2,
                slot: 1,
                root: RootType::Constants,
                dword: 0,
            },
            UserData {
                index: 3,
                slot: 1,
                root: RootType::Constants,
                dword: 1,
            },
            UserData {
                index: 4,
                slot: 0,
                root: RootType::Table,
                dword: 0,
            },
        ];
        let stages = [StageUserData {
            param: 0,
            dwords: 8,
            entries: &entries,
        }];

        // Single-argument dispatch stays at offset 0.
        let single = plan_indirect(12, &[[2, 0, 0, 0]], &stages).unwrap();
        assert_eq!(single.kind, 2);
        assert_eq!(single.arg_offset, 0);
        assert_eq!(single.args, vec![HostIndirectArg::Execute(2)]);

        // Console DISPATCH_SINGLE (105) normalizes to DISPATCH (2).
        let single_console = plan_indirect(12, &[[105, 0, 0, 0]], &stages).unwrap();
        assert_eq!(single_console.kind, 2);

        // Two root constants followed by Dispatch (stride 20 = 8 + 12).
        let multi = plan_indirect(20, &[[5, 1, 0, 2], [2, 0, 0, 0]], &stages).unwrap();
        assert_eq!(multi.kind, 2);
        assert_eq!(multi.arg_offset, 8);
        assert_eq!(
            multi.args,
            vec![
                HostIndirectArg::Constant {
                    param: 0,
                    offset: 2,
                    count: 2,
                },
                HostIndirectArg::Execute(2),
            ]
        );

        // Unused constant lands in an unoccupied slot (0..1) without colliding with entries.
        let unused = plan_indirect(16, &[[5, 7, 0, 1], [2, 0, 0, 0]], &stages).unwrap();
        assert_eq!(unused.arg_offset, 4);
        assert_eq!(
            unused.args[0],
            HostIndirectArg::Constant {
                param: 0,
                offset: 0,
                count: 1,
            }
        );

        // Collision with an occupied table slot at index 4 is rejected.
        assert!(plan_indirect(24, &[[5, 1, 0, 3], [2, 0, 0, 0]], &stages).is_none());

        // Stride smaller than arg_offset + exec_bytes is rejected.
        assert!(plan_indirect(16, &[[5, 1, 0, 2], [2, 0, 0, 0]], &stages).is_none());

        // Non-constant prefix argument (e.g. CBV = 6) is rejected.
        assert!(plan_indirect(20, &[[6, 0, 0, 0], [2, 0, 0, 0]], &stages).is_none());

        // Same constant read by two different stages cannot map to one host parameter.
        let stage2 = [
            StageUserData {
                param: 0,
                dwords: 4,
                entries: &entries[..1],
            },
            StageUserData {
                param: 1,
                dwords: 4,
                entries: &entries[..1],
            },
        ];
        assert!(plan_indirect(16, &[[5, 1, 0, 1], [2, 0, 0, 0]], &stage2).is_none());
    }

    /// Real D3D12 execution: `bound` clamps a multi-argument indirect dispatch
    /// at `arg_offset` without disturbing the preceding root constants, and
    /// `ExecuteIndirect` passes each record's root constants to the shader.
    #[test]
    fn multi_argument_indirect_dispatch_sets_root_constants_and_clamps() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        const CS: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0)"
cbuffer Params : register(b0) { uint index; uint value; uint pad0; uint pad1; };
RWByteAddressBuffer out_buf : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    out_buf.Store(index * 4, value);
}
"#;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: static HLSL source and valid out pointer.
        unsafe {
            D3DCompile(
                CS.as_ptr().cast(),
                CS.len(),
                s!("indirect_test"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )
            .unwrap();
        }
        let code = code.unwrap();
        // SAFETY: compiled DXBC bytes live through creation.
        let (root, pso) = unsafe {
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        };

        // Plan: 2 root constants (index, value) at offset 0, then Dispatch at byte offset 8.
        let ud = [
            UserData {
                index: 0,
                slot: 0,
                root: RootType::Constants,
                dword: 0,
            },
            UserData {
                index: 1,
                slot: 0,
                root: RootType::Constants,
                dword: 1,
            },
        ];
        let plan = plan_indirect(
            20,
            &[[5, 0, 0, 2], [2, 0, 0, 0]],
            &[StageUserData {
                param: 0,
                dwords: 4,
                entries: &ud,
            }],
        )
        .unwrap();
        assert_eq!(plan.arg_offset, 8);
        let descs = to_d3d12_descs(&plan.args);
        let sig_desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: 20,
            NumArgumentDescs: descs.len() as u32,
            pArgumentDescs: descs.as_ptr(),
            NodeMask: 0,
        };
        let mut sig: Option<ID3D12CommandSignature> = None;
        // SAFETY: `descs` and `root` are live.
        unsafe {
            host.device
                .CreateCommandSignature(&sig_desc, Some(&root), &mut sig)
                .unwrap();
        }
        let sig = sig.unwrap();

        let make_buf = |heap: D3D12_HEAP_TYPE, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES, size: u64| {
            let mut r: Option<ID3D12Resource> = None;
            let props = D3D12_HEAP_PROPERTIES {
                Type: heap,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            };
            // SAFETY: valid heap and resource descriptions.
            unsafe {
                host.device
                    .CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r)
                    .unwrap();
            }
            r.unwrap()
        };

        // Two records:
        // Record 0: index=0, value=0x11223344, groups=(1, 1, 1) -> writes 0x11223344
        // Record 1: index=1, value=0x55667788, groups=(100000, 100000, 1) -> clamped to (0,0,0) by GROUPS limit!
        // Record 2: index=2, value=0xCAFEBABE, groups=(1, 1, 1) -> writes 0xCAFEBABE
        let records: [u32; 15] = [
            0, 0x1122_3344, 1, 1, 1,
            1, 0x5566_7788, 100_000, 100_000, 1,
            2, 0xCAFE_BABE, 1, 1, 1,
        ];
        let upload = make_buf(
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            64,
        );
        let arg_buf = make_buf(
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_DEST,
            64,
        );
        let out_buf = make_buf(
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            64,
        );
        let readback = make_buf(
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
            64,
        );

        // SAFETY: mapping upload buffer of 64 bytes and writing 60 bytes.
        unsafe {
            let mut ptr = std::ptr::null_mut();
            upload.Map(0, None, Some(&mut ptr)).unwrap();
            std::ptr::copy_nonoverlapping(records.as_ptr().cast::<u8>(), ptr.cast::<u8>(), 60);
            upload.Unmap(0, None);
        }

        let list = host.new_list().unwrap();
        // SAFETY: recording valid D3D12 commands on an open list.
        unsafe {
            list.list.CopyBufferRegion(&arg_buf, 0, &upload, 0, 60);
            list.list.ResourceBarrier(&[transition(
                &arg_buf,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            assert!(bound(
                &list,
                &arg_buf,
                arg_buf.GetGPUVirtualAddress(),
                (plan.kind, 20, 3, plan.arg_offset),
            ));
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pso);
            list.list
                .SetComputeRootUnorderedAccessView(1, out_buf.GetGPUVirtualAddress());
            list.list
                .ExecuteIndirect(&sig, 3, &arg_buf, 0, None::<&ID3D12Resource>, 0);
            done(&list, &arg_buf);
            list.list.ResourceBarrier(&[
                uav_barrier(),
                transition(
                    &out_buf,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                ),
            ]);
            list.list.CopyBufferRegion(&readback, 0, &out_buf, 0, 16);
        }
        list.close();
        host.execute(&[&list]);

        let mut got = [0u32; 4];
        // SAFETY: reading 16 bytes from mapped readback buffer after GPU completion.
        unsafe {
            let mut ptr = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut ptr)).unwrap();
            std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), got.as_mut_ptr().cast::<u8>(), 16);
            readback.Unmap(0, None);
        }
        assert_eq!(got[0], 0x1122_3344, "record 0 root constants reached shader");
        assert_eq!(got[1], 0, "record 1 over-limit dispatch was clamped to 0");
        assert_eq!(got[2], 0xCAFE_BABE, "record 2 root constants reached shader");
    }
}
