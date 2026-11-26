use super::*;
use std::ptr::null;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{IUnknown, Interface};

// Each guest call returns one COM reference. Keep it through the readback,
// including when a failed assertion unwinds the test.
fn create(call: impl FnOnce(*mut This) -> HRESULT) -> IUnknown {
    let mut out = null();
    assert_eq!(call(&mut out), S_OK);
    // SAFETY: the successful call returned a new reference to a guest object.
    unsafe { IUnknown::from_raw(out.cast_mut().cast()) }
}

fn depth_texture(size: u32, samples: u32) -> IUnknown {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: size.into(),
        Height: size,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT(118),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: samples,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        ..Default::default()
    };
    // The guest heap and resource descriptions have the D3D12 ABI.
    create(|out| {
        crate::resource::create_committed(
            null(),
            (&raw const heap).cast(),
            0,
            (&raw const desc).cast(),
            0,
            null(),
            &PAGEABLE_IIDS[0],
            out,
        )
    })
}

fn check_depth_resolve(size: u32) {
    let Some(host) = crate::host::host() else {
        eprintln!("skipping format-118 resolve (no host D3D12 device)");
        return;
    };
    let allocator = create(|out| create_allocator(null(), 0, &ALLOCATOR_IIDS[0], out));
    let title = create(|out| {
        create_list(
            null(),
            0,
            0,
            allocator.as_raw().cast(),
            null(),
            &LIST_IIDS[1],
            out,
        )
    });
    let src = depth_texture(4, 4);
    let dst = depth_texture(size, 1);
    let (src, dst, title) = (
        src.as_raw().cast(),
        dst.as_raw().cast(),
        title.as_raw().cast(),
    );
    let list = host_list(title).expect("host command list");

    // Empty and full depth occupy two columns each. The single-sample
    // destination starts at 1/3, distinct from either half.
    for (resource, dimension, initial) in [
        (src, D3D12_DSV_DIMENSION_TEXTURE2DMS, 1.0),
        (dst, D3D12_DSV_DIMENSION_TEXTURE2D, 1.0 / 3.0),
    ] {
        ensure_state(list, resource, D3D12_RESOURCE_STATE_DEPTH_WRITE.0 as u32);
        let texture = crate::resource::host_resource(resource).expect("host depth texture");
        let view = host.dsv(
            &texture,
            Some(&D3D12_DEPTH_STENCIL_VIEW_DESC {
                Format: DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
                ViewDimension: dimension,
                ..Default::default()
            }),
        );
        // SAFETY: live textures in DEPTH_WRITE, with a matching DSV and an in-bounds rectangle.
        unsafe {
            list.list
                .ClearDepthStencilView(view, D3D12_CLEAR_FLAG_DEPTH, initial, 0, None);
            if resource == src {
                let left = RECT {
                    left: 0,
                    top: 0,
                    right: 2,
                    bottom: 4,
                };
                list.list.ClearDepthStencilView(
                    view,
                    D3D12_CLEAR_FLAG_DEPTH,
                    0.0,
                    0,
                    Some(&[left]),
                );
            }
        }
    }

    resolve(title, dst, 0, src, 0, 118);
    ensure_state(list, dst, COPY_SOURCE);
    let result = crate::resource::host_resource(dst).unwrap();
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut bytes = 0;
    // SAFETY: one output footprint for the live destination's depth plane.
    unsafe {
        host.device.GetCopyableFootprints(
            &result.GetDesc(),
            0,
            1,
            0,
            Some(&mut footprint),
            None,
            None,
            Some(&mut bytes),
        );
    }
    let (buffer, mapped) = host.shared_buffer(bytes).unwrap();
    // SAFETY: both resources outlive submission. The copy locations borrow
    // their pointers and the footprint describes the complete depth plane.
    unsafe {
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&result),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&buffer),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        list.list
            .Close()
            .expect("record a format-118 depth resolve");
    }
    host.execute(&[list]);
    for y in 0..size {
        for x in 0..size {
            let expected: f32 = if x >= 4 || y >= 4 {
                1.0 / 3.0
            } else if x < 2 {
                0.0
            } else {
                1.0
            };
            let offset = footprint.Offset as usize
                + y as usize * footprint.Footprint.RowPitch as usize
                + x as usize * 4;
            // SAFETY: the completed copy fills size × size f32 depth values at the device's row pitch.
            let depth = unsafe { mapped.add(offset).cast::<f32>().read_unaligned() };
            assert!(
                (depth - expected).abs() < 1e-6,
                "depth at ({x}, {y}) in {size}x{size} destination: got {depth}, expected {expected}"
            );
        }
    }
    eprintln!(
        "format-118 readback: 16 depth pixels and {} border pixels matched",
        size * size - 16
    );
}

#[test]
fn format_118_resolves_the_whole_depth_plane() {
    check_depth_resolve(4);
}

#[test]
fn format_118_resolves_the_overlapping_depth_region() {
    check_depth_resolve(6);
}

/// A triangle writes stencil 0x80 into the lower left half of an 8x8 4x
/// MSAA texture cleared to zero. For the two regions where all samples
/// agree, resolving the stencil plane must preserve both stencil values.
fn check_stencil_resolve(guest_format: u32) {
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    use windows::core::s;
    const N: u32 = 8;
    const REFERENCE: u8 = 0x80;
    let Some(host) = crate::host::host() else {
        eprintln!("skipping stencil resolve (no host D3D12 device)");
        return;
    };
    let allocator = create(|out| create_allocator(null(), 0, &ALLOCATOR_IIDS[0], out));
    let title = create(|out| {
        create_list(null(), 0, 0, allocator.as_raw().cast(), null(), &LIST_IIDS[1], out)
    });
    let src = depth_texture(N, 4);
    let dst = depth_texture(N, 1);
    let (src, dst, title) = (src.as_raw().cast(), dst.as_raw().cast(), title.as_raw().cast());
    let list = host_list(title).expect("host command list");

    // A vertex shader only: stencil replace writes where the triangle covers
    // a sample.
    let source = "[RootSignature(\"\")] float4 main(uint i : SV_VertexID) : SV_Position {\n\
        float2 p[3] = { float2(-1, 1), float2(-1, -1), float2(1, -1) };\n\
        return float4(p[i], 0, 1); }";
    let mut blob = None;
    // SAFETY: the source outlives the call; the blob is read below.
    unsafe {
        D3DCompile(source.as_ptr().cast(), source.len(), None, None, None, s!("main"), s!("vs_5_1"), 0, 0, &mut blob, None)
            .expect("compile the vertex shader");
    }
    let blob = blob.unwrap();
    // SAFETY: the blob's own bytes.
    let code = unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()) };
    // SAFETY: a compiled shader with its root signature inside.
    let root: ID3D12RootSignature = unsafe { host.device.CreateRootSignature(0, code) }.expect("root signature");
    let face = D3D12_DEPTH_STENCILOP_DESC {
        StencilFailOp: D3D12_STENCIL_OP_KEEP,
        StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
        StencilPassOp: D3D12_STENCIL_OP_REPLACE,
        StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
    };
    let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        // SAFETY: a borrowed pointer the description does not release.
        pRootSignature: unsafe { std::mem::transmute_copy(&root) },
        VS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr().cast(), BytecodeLength: code.len() },
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            DepthClipEnable: true.into(),
            ..Default::default()
        },
        DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
            StencilEnable: true.into(),
            StencilReadMask: 0xff,
            StencilWriteMask: 0xff,
            FrontFace: face,
            BackFace: face,
            ..Default::default()
        },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        DSVFormat: DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 4, Quality: 0 },
        ..Default::default()
    };
    desc.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xf;
    // SAFETY: a complete depth/stencil-only description.
    let pso: ID3D12PipelineState = unsafe { host.device.CreateGraphicsPipelineState(&desc) }.expect("pipeline state");

    ensure_state(list, src, D3D12_RESOURCE_STATE_DEPTH_WRITE.0 as u32);
    let texture = crate::resource::host_resource(src).expect("host depth texture");
    let view = host.dsv(
        &texture,
        Some(&D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DMS,
            ..Default::default()
        }),
    );
    // SAFETY: an open list, a live texture in DEPTH_WRITE with its view, a
    // pipeline that needs no other state.
    unsafe {
        list.list.ClearDepthStencilView(view, D3D12_CLEAR_FLAG_STENCIL, 0.0, 0, None);
        list.list.SetGraphicsRootSignature(&root);
        list.list.SetPipelineState(&pso);
        list.list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        let viewport = D3D12_VIEWPORT { Width: N as f32, Height: N as f32, MaxDepth: 1.0, ..Default::default() };
        list.list.RSSetViewports(&[viewport]);
        list.list.RSSetScissorRects(&[RECT { left: 0, top: 0, right: N as i32, bottom: N as i32 }]);
        list.list.OMSetRenderTargets(0, None, false, Some(&view));
        list.list.OMSetStencilRef(u32::from(REFERENCE));
        list.list.DrawInstanced(3, 1, 0, 0);
    }

    // The stencil plane is subresource 1 of both textures.
    resolve(title, dst, 1, src, 1, guest_format);
    ensure_state(list, dst, COPY_SOURCE);
    let result = crate::resource::host_resource(dst).unwrap();
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut bytes = 0;
    // SAFETY: one output footprint for the live destination's stencil plane.
    unsafe {
        host.device.GetCopyableFootprints(&result.GetDesc(), 1, 1, 0, Some(&mut footprint), None, None, Some(&mut bytes));
    }
    let (buffer, mapped) = host.shared_buffer(bytes).unwrap();
    // SAFETY: both resources outlive submission; the footprint describes the
    // complete stencil plane.
    unsafe {
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&result),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 1 },
        };
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&buffer),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
        };
        list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        list.list.Close().expect("record a stencil resolve: the list closes");
    }
    host.execute(&[list]);
    for y in 0..N {
        for x in 0..N {
            // A diagonal pixel has both values. Only check pixels wholly
            // covered by one of the two stencil values.
            if y == x {
                continue;
            }
            let expected = if y > x { REFERENCE } else { 0 };
            let offset = footprint.Offset as usize + y as usize * footprint.Footprint.RowPitch as usize + x as usize;
            // SAFETY: the completed copy fills N x N stencil bytes at the device's row pitch.
            let stencil = unsafe { mapped.add(offset).read() };
            assert_eq!(stencil, expected, "stencil at ({x}, {y}), format {guest_format}");
        }
    }
}

#[test]
fn format_120_resolves_a_stencil_plane_away_from_the_edge() {
    check_stencil_resolve(120);
}

#[test]
fn format_118_resolves_its_stencil_plane_away_from_the_edge() {
    check_stencil_resolve(118);
}
