use crate::object::{S_OK, This, guid, slot};
use std::sync::Arc;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

struct Guest(This);

impl Drop for Guest {
    fn drop(&mut self) {
        slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
    }
}

const TEXTURE_ALIGNMENT: usize = 64 << 10;

/// A compute dispatch reading a raw buffer and storing it at texel (0, 0).
fn program() -> victory_shader::NativeProgram {
    use victory_shader::native::{ComputeIo, StageIo};
    use victory_shader::{HwStage, NativeProgram, ResourceType, RootParam, RootType, UsedResource};

    let words: [u32; 16] = [
        0xBE82_0300,
        0xBE83_0380, // s_mov_b32 s2, s0; s3, 0
        0xF408_0201,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[2:3], 0
        0xBE84_0301,
        0xBE85_0380, // s_mov_b32 s4, s1; s5, 0
        0xF40C_0402,
        0xFA00_0000, // s_load_dwordx8 s[16:23], s[4:5], 0
        0x7E00_0280, // v_mov_b32 v0, 0
        0xE030_1000,
        0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 offen
        0x7E04_0280,
        0x7E06_0280,
        0x7E08_0280, // v_mov_b32 v[2:4], 0
        0xF020_1F08,
        0x0004_0102, // image_store v[1:4], v[2:3], s[16:23], 2D dmask:0xf
    ];

    let mut code: Vec<u8> = words.into_iter().flat_map(u32::to_le_bytes).collect();
    code.extend_from_slice(&0xBF81_0000u32.to_le_bytes());
    let table = |slot| RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data: slot,
        slot,
    };
    let access = |slot, kind, dwords| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data: slot,
        slot,
    };
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![table(0), table(1)],
        used: vec![
            access(0, ResourceType::Srv, 4),
            access(1, ResourceType::Uav, 8),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [1, 1, 1],
        }),
        code,
    }
}

#[test]
fn registered_buffer_replaces_loose_shadow_on_next_dispatch() {
    let Some(host) = crate::host::host() else {
        return;
    };
    let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL {
        HighestShaderModel: D3D_SHADER_MODEL_6_6,
    };
    // SAFETY: this holds the full output structure for the feature query.
    let supported = unsafe {
        host.device.CheckFeatureSupport(
            D3D12_FEATURE_SHADER_MODEL,
            (&raw mut sm).cast(),
            size_of_val(&sm) as u32,
        )
    };
    if supported.is_err() || sm.HighestShaderModel.0 < D3D_SHADER_MODEL_6_6.0 {
        eprintln!("placed buffer dispatch needs SM 6.6");
        return;
    }
    let null = std::ptr::null();
    let mut pipe = null;
    assert_eq!(
        crate::pipeline::test_compute_pipeline(program(), &mut pipe),
        S_OK
    );
    let pipe = Guest(pipe);
    assert!(crate::pipeline::host_pipeline(pipe.0).is_some(), "synthetic host pipeline");

    // Loose copies cover 4 MiB windows and survive the test. Reserve a
    // complete aligned window so later submissions can still read it.
    let window = 4 << 20;
    let start = crate::watch::tests::pool(2 * window / crate::watch::PAGE);
    let source = start.next_multiple_of(window) as u64;
    let source_words = [
        D3D12_RESOURCE_DIMENSION_BUFFER.0 as u32,
        0,
        0,
        0,
        TEXTURE_ALIGNMENT as u32,
        0, // width
        1,
        1 | (1 << 16), // height, depth, mip count
        0,
        1,
        0, // format, samples, quality
        1,
        0,
        0, // row-major layout, flags, padding
    ];
    let source_view = [
        DXGI_FORMAT_R32_TYPELESS.0 as u32,
        D3D12_SRV_DIMENSION_BUFFER.0 as u32,
        D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        0,
        0,
        0, // padding and first element
        4,
        0,
        D3D12_BUFFER_SRV_FLAG_RAW.0 as u32, // length, stride, flags
        0,
        source as u32,
        (source >> 32) as u32,
        0,
        0,
        0,
        0,
    ];
    // The placed texture cache keeps the address for the process.
    let dest_address = crate::watch::tests::pool(1) as u64;
    let format = DXGI_FORMAT_R32G32B32A32_FLOAT.0 as u32;
    let dest_words = [
        D3D12_RESOURCE_DIMENSION_TEXTURE2D.0 as u32,
        0,
        0,
        0,
        16,
        0,
        1,
        1 | (1 << 16), // width, height, slice and mip counts
        format,
        1,
        0,
        1,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 as u32,
        0,
    ];
    let output =
        crate::resource::texture_at(dest_address, &dest_words).expect("placed output texture");
    let texture = crate::resource::host_resource(output).expect("host output texture");
    let output_view = [
        format,
        D3D12_UAV_DIMENSION_TEXTURE2D.0 as u32,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ];

    let new_heap = || {
        #[repr(C)]
        struct HeapDescWithPlacement(D3D12_DESCRIPTOR_HEAP_DESC, [u64; 2]);
        let desc = HeapDescWithPlacement(
            D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 2,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            },
            [0; 2],
        );
        let iid = ID3D12DescriptorHeap::IID;
        let iid = guid(iid.data1, iid.data2, iid.data3, iid.data4);
        let mut views = null;
        assert_eq!(
            crate::descriptor::create_heap(
                null,
                std::ptr::from_ref(&desc).cast(),
                &iid,
                &mut views
            ),
            S_OK
        );
        let handle = crate::descriptor::heap(views).start;
        crate::descriptor::create_placed_srv(null, &source_words, &source_view, handle);
        crate::descriptor::create_uav(null, output, null, &output_view, handle + 32);
        Guest(views)
    };
    struct Recorded {
        state: crate::draw::State,
        list: crate::host::HostList,
        readback: ID3D12Resource,
        footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    }

    let record = |views: &Guest| {
        let handle = crate::descriptor::heap(views.0).start;
        let list = host.new_list().unwrap();
        crate::texture::upload(&list, output, texture);
        let mut state = crate::draw::State::default();
        state.set_pipeline(pipe.0);
        state.set_heaps(&[views.0]);
        state.set_table(true, 0, handle);
        state.set_table(true, 1, handle + 32);
        let before = crate::draw::dispatched();
        state.dispatch(&list, 1, 1, 1);
        assert!(
            crate::draw::dispatched() > before,
            "dispatch did not record"
        );
        crate::list::ensure_state(&list, output, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
        let (readback, footprint) =
            crate::capture::read_back(&host.device, &list.list, texture).unwrap();
        list.close();
        Recorded { state, list, readback, footprint }
    };
    let submit = |recorded: &Recorded| {
        host.execute(&[&recorded.list]);
        // SAFETY: the readback contains the RGBA32_FLOAT texel after completion.
        unsafe {
            let mut data = std::ptr::null_mut();
            recorded.readback.Map(0, None, Some(&mut data)).unwrap();
            let texel = data
                .cast::<u8>()
                .add(recorded.footprint.Offset as usize)
                .cast::<[f32; 4]>()
                .read_unaligned();
            recorded.readback.Unmap(0, None);
            assert!(host.device.GetDeviceRemovedReason().is_ok());
            (texel, !recorded.state.fresh_tables.is_empty())
        }
    };

    let render = |views: &Guest| submit(&record(views));
    let heap = new_heap();
    let loose = crate::shadow::find(source, TEXTURE_ALIGNMENT as u64).expect("loose shadow");
    let old = record(&heap);
    assert_eq!(submit(&old), ([0.0; 4], false), "before buffer registration");
    // Keep a second old list recorded but unsubmitted during the change.
    let unsubmitted = record(&heap);

    let mut buffer = null;
    let iid = ID3D12Resource::IID;
    let iid = guid(iid.data1, iid.data2, iid.data3, iid.data4);
    assert_eq!(
        crate::resource::create_at_address(
            null,
            source,
            source_words.as_ptr().cast(),
            0,
            std::ptr::null(),
            &iid,
            &mut buffer,
        ),
        S_OK
    );
    let _buffer = Guest(buffer);
    let registered =
        crate::shadow::find(source, TEXTURE_ALIGNMENT as u64).expect("registered shadow");
    assert!(!Arc::ptr_eq(&loose, &registered));
    // SAFETY: the first dword of source_mem, still live throughout the test.
    unsafe { (source as *mut u32).write(1.25f32.to_bits()) };
    let want = [1.25, 0.0, 0.0, 0.0];
    assert_eq!(
        render(&heap),
        (want, true),
        "a new dispatch uses a fresh table while old lists can still execute"
    );
    drop(old);
    assert_eq!(
        render(&heap),
        (want, true),
        "the recorded but unsubmitted list still needs the original slot"
    );
    drop(unsubmitted);
    assert_eq!(
        render(&heap),
        (want, false),
        "the permanent slot follows the backing once old lists retire"
    );
    assert_eq!(render(&new_heap()), (want, false), "fresh descriptor control");
}
