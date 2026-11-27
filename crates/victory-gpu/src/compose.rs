//! A present's second plane laid over its first (`VICTORY_COMPOSE=1`).
//!
//! A Series S title presents two planes (plane 0 the game, plane 1 the
//! interface): the console's display lays the second over the first and scales
//! both to the screen. The host shows one texture, so the interface is laid
//! over the game here, at the game plane's size, into a texture of ours that
//! is shown instead (filtered: the interface is scaled down). shortcut: the
//! planes' placement and source rectangles are not read; both cover the
//! screen. The interface's alpha is taken as straight
//! (`VICTORY_PLANE_PREMULTIPLIED=1`: premultiplied).

use std::sync::Mutex;
use std::sync::OnceLock;

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::s;

/// Frames of descriptors in the ring (three each): presents in flight
/// must not find theirs rewritten.
const SLOTS: u32 = 16;

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=3, b0), DescriptorTable(SRV(t0, numDescriptors=2), UAV(u0)), StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP)"
cbuffer Size : register(b0) { uint w; uint h; uint premultiplied; };
Texture2D<float4> game : register(t0);
Texture2D<float4> ui : register(t1);
RWTexture2D<float4> outp : register(u0);
SamplerState linear_clamp : register(s0);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= w || id.y >= h) return;
    float2 uv = (float2(id.xy) + 0.5) / float2(w, h);
    float4 g = game.Load(int3(id.xy, 0));
    float4 u = ui.SampleLevel(linear_clamp, uv, 0);
    float3 c = premultiplied != 0 ? g.rgb * (1.0 - u.a) + u.rgb : lerp(g.rgb, u.rgb, u.a);
    outp[id.xy] = float4(c, 1.0);
}
"#;

struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    views: ID3D12DescriptorHeap,
    cpu: usize,
    gpu: u64,
    step: u32,
    next: u32,
    /// The texture shown instead, by size.
    out: Option<(u64, u32, ID3D12Resource)>,
}
// SAFETY: D3D12 objects are free-threaded; the pass is used under a lock.
unsafe impl Send for Pass {}

/// The pass: not tried yet, could not be made, or made.
static PASS: Mutex<Option<Option<Pass>>> = Mutex::new(None);

fn make(device: &ID3D12Device) -> windows::core::Result<Pass> {
    let mut code: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; the out pointer
    // is ours.
    unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("present planes"),
            None,
            None,
            windows::core::s!("main"),
            windows::core::s!("cs_5_1"),
            0,
            0,
            &mut code,
            None,
        )?;
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes (the root signature is inside the
    // shader); plain object creation and queries.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso: ID3D12PipelineState =
            device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        let views: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: kind,
                NumDescriptors: SLOTS * 3,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })?;
        Ok(Pass {
            cpu: views.GetCPUDescriptorHandleForHeapStart().ptr,
            gpu: views.GetGPUDescriptorHandleForHeapStart().ptr,
            step: device.GetDescriptorHandleIncrementSize(kind),
            root,
            pso,
            views,
            next: 0,
            out: None,
        })
    }
}

/// Whether a host texture's format is one this pass reads and writes
/// (8-bit RGBA, typeless or not).
fn rgba8(format: DXGI_FORMAT) -> bool {
    matches!(format.0, 27 | 28)
}

fn flag(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_some_and(|v| v == "1")
}

fn env_flag(name: &str) -> bool {
    flag(std::env::var_os(name).as_deref())
}

/// Off by default: the multi-plane compositor already in the host shows
/// ordinary frames; this pass is the opt-in Series S path.
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| env_flag("VICTORY_COMPOSE"))
}

fn premultiplied() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| env_flag("VICTORY_PLANE_PREMULTIPLIED"))
}

fn constants(width: u32, height: u32, premultiplied: bool) -> [u32; 3] {
    [width, height, u32::from(premultiplied)]
}

fn dispatch(width: u32, height: u32) -> (u32, u32, u32) {
    (width.div_ceil(8), height.div_ceil(8), 1)
}

/// Records, into the present's `list`, `ui` laid over `game` into a
/// texture of ours, and gives that texture, left in the copy-source
/// state. Both input planes return to their incoming states. `None` if
/// a format is not 8-bit RGBA or the pass could not be made.
pub(crate) fn record(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    game: &ID3D12Resource,
    game_state: u32,
    ui: &ID3D12Resource,
    ui_state: u32,
) -> Option<ID3D12Resource> {
    // SAFETY: live resources; plain queries.
    let (gd, ud) = unsafe { (game.GetDesc(), ui.GetDesc()) };
    if !rgba8(gd.Format) || !rgba8(ud.Format) {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                game = gd.Format.0,
                ui = ud.Format.0,
                "present planes not laid over each other: a format this pass does not read"
            );
        });
        return None;
    }
    let mut kept = PASS.lock().unwrap_or_else(|e| e.into_inner());
    if kept.is_none() {
        *kept = Some(match make(device) {
            Ok(p) => {
                tracing::info!(
                    premultiplied = premultiplied(),
                    "present planes: the interface is laid over the game"
                );
                Some(p)
            }
            Err(e) => {
                tracing::warn!(error = %e, "present planes: the pass that lays them over each other could not be made");
                None
            }
        });
    }
    let pass = kept.as_mut()?.as_mut()?;
    let (w, h) = (gd.Width, gd.Height);
    if pass.out.as_ref().is_none_or(|o| (o.0, o.1) != (w, h)) {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: w,
            Height: h,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a device-local texture.
        unsafe {
            device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    None,
                    &mut out,
                )
                .ok()?;
        }
        pass.out = Some((w, h, out?));
    }
    let out = pass.out.as_ref()?.2.clone();
    let first = (pass.next % SLOTS) * 3;
    pass.next = pass.next.wrapping_add(1);
    let cpu = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
        ptr: pass.cpu + ((first + i) * pass.step) as usize,
    };
    let gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
        ptr: pass.gpu + u64::from(first * pass.step),
    };
    let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_UAV {
                MipSlice: 0,
                PlaneSlice: 0,
            },
        },
    };
    let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    let t = crate::host::transition;
    let mut before = vec![t(
        &out,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    )];
    let mut after = vec![t(
        &out,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    )];
    let game_state = D3D12_RESOURCE_STATES(game_state as i32);
    if game_state != read {
        before.push(t(game, game_state, read));
        after.push(t(game, read, game_state));
    }
    let ui_state = D3D12_RESOURCE_STATES(ui_state as i32);
    if ui != game && ui_state != read {
        before.push(t(ui, ui_state, read));
        after.push(t(ui, read, ui_state));
    }
    let values = constants(w as u32, h, premultiplied());
    // SAFETY: an open list (the present's); slots of our heap; live
    // resources and objects (the pass's own live as long as the process,
    // the texture is kept by the present's list).
    unsafe {
        device.CreateShaderResourceView(game, Some(&srv), cpu(0));
        device.CreateShaderResourceView(ui, Some(&srv), cpu(1));
        device.CreateUnorderedAccessView(&out, None::<&ID3D12Resource>, Some(&uav), cpu(2));
        list.ResourceBarrier(&before);
        list.SetDescriptorHeaps(&[Some(pass.views.clone())]);
        list.SetComputeRootSignature(&pass.root);
        list.SetPipelineState(&pass.pso);
        list.SetComputeRoot32BitConstants(0, values.len() as u32, values.as_ptr().cast(), 0);
        list.SetComputeRootDescriptorTable(1, gpu);
        let (x, y, z) = dispatch(w as u32, h);
        list.Dispatch(x, y, z);
        list.ResourceBarrier(&after);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;

    use super::{constants, dispatch, flag};

    #[test]
    fn plane_composition_setup_is_straight_or_premultiplied() {
        assert_eq!(constants(1920, 1080, false), [1920, 1080, 0]);
        assert_eq!(constants(1920, 1080, true), [1920, 1080, 1]);
        assert_eq!(dispatch(1920, 1080), (240, 135, 1));
        assert_eq!(dispatch(3840, 2160), (480, 270, 1));
    }

    #[test]
    fn plane_gates_are_off_unless_one() {
        assert!(!flag(None));
        assert!(!flag(Some(OsStr::new("0"))));
        assert!(!flag(Some(OsStr::new("true"))));
        assert!(flag(Some(OsStr::new("1"))));
    }

    #[cfg(feature = "diagnostics")]
    #[test]
    fn compute_planes_restore_render_targets_and_scale_the_overlay() {
        use super::*;
        use crate::host::{
            host,
            tests::{issued, replay},
        };
        use windows::Win32::Foundation::RECT;
        use windows::core::Interface;

        let Some(host) = host() else {
            eprintln!("no D3D12 device; skipped plane readback");
            return;
        };
        let recorded = host.new_list().unwrap();
        let list = &recorded.list;
        let state = D3D12_RESOURCE_STATE_RENDER_TARGET;
        let make_plane = |width, height| {
            let mut texture: Option<ID3D12Resource> = None;
            // SAFETY: a live device and storage for one output texture.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES {
                            Type: D3D12_HEAP_TYPE_DEFAULT,
                            ..Default::default()
                        },
                        D3D12_HEAP_FLAG_NONE,
                        &D3D12_RESOURCE_DESC {
                            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                            Width: width,
                            Height: height,
                            DepthOrArraySize: 1,
                            MipLevels: 1,
                            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                            SampleDesc: DXGI_SAMPLE_DESC {
                                Count: 1,
                                Quality: 0,
                            },
                            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                            ..Default::default()
                        },
                        state,
                        None,
                        &mut texture,
                    )
                    .unwrap();
            }
            texture.unwrap()
        };
        let (width, height) = (9usize, 7usize);
        let game = make_plane(width as u64, height as u32);
        let ui = make_plane(2, 1);
        let a = 128.0 / 255.0;
        let rtv = host.rtv(&ui, None);
        // SAFETY: the lists are open and both textures are render targets.
        unsafe {
            list.ClearRenderTargetView(host.rtv(&game, None), &[1.0, 0.0, 0.0, 1.0], None);
            list.ClearRenderTargetView(
                rtv,
                &[0.0, 0.0, 64.0 / 255.0, a],
                Some(&[RECT {
                    left: 0,
                    top: 0,
                    right: 1,
                    bottom: 1,
                }]),
            );
            list.ClearRenderTargetView(
                rtv,
                &[0.0, 128.0 / 255.0, 0.0, a],
                Some(&[RECT {
                    left: 1,
                    top: 0,
                    right: 2,
                    bottom: 1,
                }]),
            );
        }
        let replay_inputs = |second: &ID3D12Resource| {
            let game_id = game.as_raw() as usize;
            let mut states = std::collections::HashMap::from([
                (game_id, state.0 as u32),
                (second.as_raw() as usize, state.0 as u32),
            ]);
            let barriers = issued();
            let first = barriers.iter().find(|&&(id, _, _)| id == game_id).unwrap();
            println!(
                "plane transitions primary_initial={} declared_before={}",
                state.0, first.1
            );
            assert!(
                replay(&barriers, &mut states).is_ok(),
                "input transitions must start in their real states"
            );
            assert!(states.values().all(|&current| current == state.0 as u32));
        };
        issued();
        let out = record(
            &host.device,
            list,
            &game,
            state.0 as u32,
            &ui,
            state.0 as u32,
        )
        .expect("compute compositor");
        replay_inputs(&ui);
        let (pixels, footprint) = crate::capture::read_back(&host.device, list, &out).unwrap();

        // The same resource can be shown as both planes. Its transition is
        // recorded once, and the two input bindings still sample it.
        let out = record(
            &host.device,
            list,
            &game,
            state.0 as u32,
            &game,
            state.0 as u32,
        )
        .expect("aliased compositor inputs");
        replay_inputs(&game);
        let (alias_pixels, alias_footprint) =
            crate::capture::read_back(&host.device, list, &out).unwrap();
        recorded.close();
        host.execute(&[&recorded]);

        let read = |buffer: &ID3D12Resource, footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| {
            assert_eq!(
                (footprint.Footprint.Width, footprint.Footprint.Height),
                (9, 7)
            );
            let mut data = std::ptr::null_mut();
            // SAFETY: the copy completed and the footprint contains all
            // 63 RGBA8 texels; the buffer remains mapped through the reads.
            unsafe {
                buffer.Map(0, None, Some(&mut data)).unwrap();
                let mut rows = Vec::new();
                for y in 0..height {
                    let offset =
                        footprint.Offset as usize + y * footprint.Footprint.RowPitch as usize;
                    let row = data.cast::<u8>().add(offset);
                    rows.extend((0..width).map(|x| row.add(x * 4).cast::<[u8; 4]>().read()));
                }
                buffer.Unmap(0, None);
                rows
            }
        };
        let factor = if premultiplied() { 1.0 } else { a };
        let mut worst = 0u8;
        for (i, actual) in read(&pixels, footprint).iter().enumerate() {
            let x = (i % width) as f32 + 0.5;
            let k = (2.0 * x / width as f32 - 0.5).clamp(0.0, 1.0);
            let expected = [
                127,
                (128.0 * k * factor).round() as u8,
                (64.0 * (1.0 - k) * factor).round() as u8,
                255,
            ];
            let delta = actual
                .iter()
                .zip(expected)
                .map(|(&a, b)| a.abs_diff(b))
                .max()
                .unwrap();
            worst = worst.max(delta);
            assert!(delta <= 1, "texel {i}: {actual:?}, expected {expected:?}");
        }
        assert!(
            read(&alias_pixels, alias_footprint)
                .iter()
                .all(|p| *p == [255, 0, 0, 255])
        );
        println!(
            "plane readback premultiplied={} texels=63 alias_texels=63 worst={worst}",
            premultiplied()
        );
    }
}
