//! Synthetic shaders, color target and readback verifying `RSSetShadingRate`
//! through the title's command list entry points.

use super::*;
use crate::object::{S_OK, slot};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::{Fxc::D3DCompile, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
};
use windows::core::{GUID, Interface, PCSTR, s};

fn compile_hlsl(source: &[u8], profile: PCSTR, stage: &'static str, hw: victory_shader::HwStage) -> Arc<Shader> {
    let mut code: Option<ID3DBlob> = None;
    // SAFETY: `source` and `profile` are valid and `code` is the output pointer.
    unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            PCSTR::null(),
            None,
            None,
            s!("main"),
            profile,
            0,
            0,
            &mut code,
            None,
        )
        .unwrap();
    }
    let code = code.unwrap();
    // SAFETY: the blob owns these bytes.
    let bytes = unsafe {
        std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
    };
    let shader = shader(stage, bytes);
    shader.translated.get_or_init(|| {
        Some(Translated {
            host: victory_shader::emit::HostShader {
                stage: hw,
                dxil: bytes.to_vec(),
                user_data: Vec::new(),
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                memory: false,
                memory_writes: false,
                view_addresses: false,
                heap: false,
                heap_writes: false,
                view_marks: false,
            },
            used: Vec::new(),
        })
    });
    shader
}

fn made(pso: This) -> &'static host::HostPipeline {
    let started = std::time::Instant::now();
    loop {
        if let Some(p) = host_pipeline(pso) {
            return p;
        }
        assert!(
            started.elapsed() < std::time::Duration::from_secs(60),
            "the host refused the shading-rate test pipeline, or did not make it in a minute"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

#[test]
fn shading_rate_controls_coarse_pixel_rasterization() {
    let Some(host) = crate::host::host() else {
        eprintln!("skipped shading rate render: no host D3D12 device");
        return;
    };
    let device = &host.device;
    // SAFETY: plain adapter query on a live device.
    let adapter = unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let adapter: IDXGIAdapter1 = factory.EnumAdapterByLuid(device.GetAdapterLuid()).unwrap();
        adapter.GetDesc1().unwrap()
    };
    let name = String::from_utf16_lossy(&adapter.Description);
    let tier = host.shading_rate_tier.0 as u32;
    eprintln!(
        "shading rate on {}: tier={tier}, additional={}, sum={}",
        name.trim_end_matches('\0'),
        host.additional_shading_rates,
        host.shading_rate_sum_combiner
    );

    let vs = compile_hlsl(
        b"float4 main(uint id : SV_VertexID) : SV_Position {
            return float4(id == 1 ? 3 : -1, id == 2 ? -3 : 1, 0.5, 1);
        }",
        s!("vs_5_1"),
        "vertex",
        victory_shader::HwStage::Vs,
    );
    let ps = compile_hlsl(
        b"float4 main(float4 pos : SV_Position) : SV_Target {
            return float4(pos.x, pos.y, 0.0, 1.0);
        }",
        s!("ps_5_1"),
        "pixel",
        victory_shader::HwStage::Ps,
    );

    #[repr(C)]
    struct RtvFormats {
        kind: u32,
        formats: [u32; 8],
        count: u32,
    }
    #[repr(C)]
    struct Stream {
        vs_kind: u64,
        vs: Bytecode,
        ps_kind: u64,
        ps: Bytecode,
        topology: [u32; 2],
        rtv: RtvFormats,
    }
    let stream = Stream {
        vs_kind: 1,
        vs: Bytecode {
            code: vs.container.as_ptr(),
            size: vs.container.len(),
        },
        ps_kind: 2,
        ps: Bytecode {
            code: ps.container.as_ptr(),
            size: ps.container.len(),
        },
        topology: [14, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE.0 as u32],
        rtv: RtvFormats {
            kind: 15,
            formats: [
                DXGI_FORMAT_R32G32B32A32_FLOAT.0 as u32,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            count: 1,
        },
    };
    let desc = StreamDesc {
        size: size_of_val(&stream),
        stream: (&raw const stream).cast(),
    };
    let null = std::ptr::null();
    let mut pso = null;
    assert_eq!(create_stream(null, &desc, &PIPELINE_IIDS[0], &mut pso), S_OK);
    let _ = made(pso);

    let mut allocator = null;
    let mut list = null;
    let iid = |g: GUID| guid(g.data1, g.data2, g.data3, g.data4);
    assert_eq!(
        crate::list::create_allocator(null, 0, &iid(ID3D12CommandAllocator::IID), &mut allocator),
        S_OK
    );
    assert_eq!(
        crate::list::create_list(
            null,
            0,
            0,
            allocator,
            null,
            &iid(ID3D12GraphicsCommandList::IID),
            &mut list
        ),
        S_OK
    );
    let commands = crate::list::host_list(list).unwrap();
    let l = &commands.list;
    let newer = crate::list::has_newer_table(list);
    let set_rate =
        slot::<extern "system" fn(This, u32, *const u32)>(list, if newer { 0x3f0 } else { 0x3e0 });
    let scissors = slot::<extern "system" fn(This, u32, *const RECT)>(list, 0xb0);
    let draw = slot::<extern "system" fn(This, u32, u32, u32, u32)>(list, 0x60);

    let rt_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 4,
        Height: 4,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        ..Default::default()
    };
    let buffer_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: 256 * 4,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: rt_desc.SampleDesc,
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let resource = |desc: &D3D12_RESOURCE_DESC, kind, state| {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: kind,
            ..Default::default()
        };
        let mut result: Option<ID3D12Resource> = None;
        // SAFETY: valid resource descriptions and output pointer.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, desc, state, None, &mut result)
                .unwrap();
        }
        result.unwrap()
    };
    let target = resource(
        &rt_desc,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
    );
    let readback = resource(
        &buffer_desc,
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    let rtv = host.rtv(&target, None);

    let bind = || {
        // SAFETY: an open list and a live render-target descriptor.
        unsafe { l.OMSetRenderTargets(1, Some(&rtv), false, None) };
        slot::<extern "system" fn(This, This)>(list, 0xc8)(list, pso);
        slot::<extern "system" fn(This, u32)>(list, 0xa0)(list, 4); // triangle list
        slot::<extern "system" fn(This, u32, *const D3D12_VIEWPORT)>(list, 0xa8)(
            list,
            1,
            &D3D12_VIEWPORT {
                Width: 4.0,
                Height: 4.0,
                MaxDepth: 1.0,
                ..Default::default()
            },
        );
    };
    let draw_quad = |x: i32, y: i32| {
        scissors(
            list,
            1,
            &RECT {
                left: x,
                top: y,
                right: x + 2,
                bottom: y + 2,
            },
        );
        draw(list, 3, 1, 0, 0);
    };
    let read_target = || -> [[[f32; 4]; 4]; 4] {
        let mut from = D3D12_TEXTURE_COPY_LOCATION {
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            ..Default::default()
        };
        let mut to = D3D12_TEXTURE_COPY_LOCATION {
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: rt_desc.Format,
                        Width: 4,
                        Height: 4,
                        Depth: 1,
                        RowPitch: 256,
                    },
                },
            },
            ..Default::default()
        };
        // SAFETY: live resources and exact full-subresource copy.
        unsafe {
            from.pResource = std::mem::transmute_copy(&target);
            to.pResource = std::mem::transmute_copy(&readback);
            l.ResourceBarrier(&[crate::host::transition(
                &target,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            l.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            l.ResourceBarrier(&[crate::host::transition(
                &target,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            )]);
        }
        assert_eq!(
            slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list),
            S_OK
        );
        host.execute(&[commands]);
        let mut mapped = std::ptr::null_mut();
        let mut rows = [[[0.0f32; 4]; 4]; 4];
        // SAFETY: GPU copy completed; buffer has 4 rows at 256-byte stride.
        unsafe {
            readback
                .Map(
                    0,
                    Some(&D3D12_RANGE {
                        Begin: 0,
                        End: 256 * 4,
                    }),
                    Some(&mut mapped),
                )
                .unwrap();
            for (y, row) in rows.iter_mut().enumerate() {
                *row = mapped.cast::<u8>().add(y * 256).cast::<[[f32; 4]; 4]>().read();
            }
            readback.Unmap(0, Some(&D3D12_RANGE::default()));
        }
        rows
    };

    bind();
    // Top-left (0..2, 0..2): fresh list defaults to 1x1.
    draw_quad(0, 0);
    // Top-right (2..4, 0..2): 2x2 with null combiners.
    set_rate(list, 0x5, std::ptr::null());
    draw_quad(2, 0);
    // Bottom-left (0..2, 2..4): 2x1 with null combiners.
    set_rate(list, 0x4, std::ptr::null());
    draw_quad(0, 2);
    // Bottom-right (2..4, 2..4): 1x2 with explicit passthrough combiners.
    let passthrough = [0u32, 0u32];
    set_rate(list, 0x1, passthrough.as_ptr());
    draw_quad(2, 2);
    // Leave a coarse rate active before reset to verify Reset restores 1x1.
    set_rate(list, 0x5, std::ptr::null());
    let pass1 = read_target();

    // Fresh list 1x1 quadrant: every pixel has a distinct SV_Position.
    assert_ne!(pass1[0][0][0], pass1[0][1][0], "fresh 1x1 must separate X");
    assert_ne!(pass1[0][0][1], pass1[1][0][1], "fresh 1x1 must separate Y");

    if tier >= 1 {
        // 2x2: all four pixels share one coarse PS invocation.
        assert_eq!(pass1[0][2][0], pass1[0][3][0], "2x2 must broadcast X");
        assert_eq!(pass1[0][2][1], pass1[1][2][1], "2x2 must broadcast Y");
        assert_eq!(pass1[0][2], pass1[1][3], "2x2 must broadcast across block");
        // 2x1: horizontal pairs match, rows remain distinct.
        assert_eq!(pass1[2][0][0], pass1[2][1][0], "2x1 must broadcast X");
        assert_eq!(pass1[3][0][0], pass1[3][1][0], "2x1 must broadcast X on row 3");
        assert_ne!(pass1[2][0][1], pass1[3][0][1], "2x1 must keep Y distinct");
        // 1x2: vertical pairs match, columns remain distinct.
        assert_eq!(pass1[2][2][1], pass1[3][2][1], "1x2 must broadcast Y");
        assert_eq!(pass1[2][3][1], pass1[3][3][1], "1x2 must broadcast Y on col 3");
        assert_ne!(pass1[2][2][0], pass1[2][3][0], "1x2 must keep X distinct");
    } else {
        // Tier 0 ignores RSSetShadingRate cleanly and rasterizes at 1x1.
        assert_ne!(pass1[0][2][0], pass1[0][3][0]);
        assert_ne!(pass1[0][2][1], pass1[1][2][1]);
    }

    // Reset must restore default 1x1 even though 2x2 was set prior to submission.
    assert_eq!(
        slot::<extern "system" fn(This, This, This) -> HRESULT>(list, 0x50)(list, allocator, null),
        S_OK
    );
    bind();
    // Top-left (0..2, 0..2): right after Reset, must be 1x1.
    draw_quad(0, 0);
    // Top-right (2..4, 0..2): invalid base rate (0x7) must fall back to 1x1
    // cleanly without tripping the D3D12 debug layer.
    set_rate(list, 0x7, std::ptr::null());
    draw_quad(2, 0);
    // Bottom-left (0..2, 2..4): 2x2 with non-passthrough combiner [PASSTHROUGH, MAX].
    // Tier 1 strips combiners (avoiding debug-layer error); Tier 2 computes MAX(2x2, 1x1) = 2x2.
    let non_passthrough = [0u32, 3u32];
    set_rate(list, 0x5, non_passthrough.as_ptr());
    draw_quad(0, 2);
    // The shading-rate image defaults to 1x1, so SUM in the second
    // combiner or its passthrough fallback preserves the 2x2 rate.
    let sum = [0u32, 4];
    set_rate(list, 0x5, sum.as_ptr());
    draw_quad(2, 2);
    let pass2 = read_target();

    assert_ne!(pass2[0][0][0], pass2[0][1][0], "reset must restore 1x1 X");
    assert_ne!(pass2[0][0][1], pass2[1][0][1], "reset must restore 1x1 Y");
    assert_ne!(
        pass2[0][2][0], pass2[0][3][0],
        "invalid rate must fall back to 1x1 X"
    );
    assert_ne!(
        pass2[0][2][1], pass2[1][2][1],
        "invalid rate must fall back to 1x1 Y"
    );
    if tier >= 1 {
        assert_eq!(
            pass2[2][0], pass2[3][1],
            "2x2 with [PASSTHROUGH, MAX] must produce 2x2 on Tier 1 and Tier 2"
        );
        assert_eq!(
            pass2[2][2], pass2[3][3],
            "2x2 with SUM or its fallback must broadcast across the block"
        );
    } else {
        assert_ne!(pass2[2][0][0], pass2[2][1][0]);
        assert_ne!(pass2[2][2][0], pass2[2][3][0]);
    }

    // When the adapter advertises AdditionalShadingRatesSupported, 4x4 (0xA)
    // must broadcast one coarse PS invocation across the full 4x4 target.
    if tier >= 1 && host.additional_shading_rates {
        assert_eq!(
            slot::<extern "system" fn(This, This, This) -> HRESULT>(list, 0x50)(
                list, allocator, null
            ),
            S_OK
        );
        bind();
        set_rate(list, 0xA, std::ptr::null());
        scissors(
            list,
            1,
            &RECT {
                left: 0,
                top: 0,
                right: 4,
                bottom: 4,
            },
        );
        draw(list, 3, 1, 0, 0);
        let pass3 = read_target();
        for row in &pass3 {
            for px in row {
                assert_eq!(*px, pass3[0][0], "4x4 must broadcast across the entire 4x4 block");
            }
        }
    }

    for object in [list, allocator, pso] {
        assert_eq!(slot::<extern "system" fn(This) -> u32>(object, 0x10)(object), 0);
    }
}
