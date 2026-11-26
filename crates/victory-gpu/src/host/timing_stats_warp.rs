use super::*;

const SHADER: &str = r#"
RWByteAddressBuffer output : register(u0);
[RootSignature("UAV(u0)")]
[numthreads(8, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    output.Store(id.x * 4, id.x);
}
"#;

#[test]
#[ignore = "requires an isolated WARP process with VICTORY_GPU_TIMING set"]
fn timing_stats_follow_commands_through_close_and_reset() {
    assert!(timing());
    assert!(std::env::var_os("VICTORY_D3D12_WARP").is_some());
    println!("DRED requested: {}", dred_on());
    let host = host().expect("WARP device");
    let (root, pipeline) = crate::draw::tests::compile_cs(SHADER);
    let list = host.new_list().expect("command list");
    let mut output: Option<ID3D12Resource> = None;
    // SAFETY: the descriptors describe a buffer on the default heap.
    unsafe {
        host.device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 4096,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            None,
            &mut output,
        )
    }
    .expect("compute UAV");
    let output = output.unwrap();
    // SAFETY: query the live queue.
    let frequency = unsafe { host.queue.GetTimestampFrequency() }.unwrap();
    assert_ne!(frequency, 0);

    let first = [("first", 2), ("middle", 7), ("last", 3)];
    let second = [("new-first", 5), ("new-last", 1)];
    let mut actual = Vec::new();
    for (run, commands) in [first.as_slice(), second.as_slice()]
        .into_iter()
        .enumerate()
    {
        if run != 0 {
            list.reset();
        }
        // SAFETY: the list is open and the pipeline and root signature are live.
        unsafe {
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pipeline);
        }
        for (i, &(label, groups)) in commands.iter().enumerate() {
            list.note_op(|| label.to_owned());
            // SAFETY: each dispatch writes to its own region of the live buffer.
            unsafe {
                let address = output.GetGPUVirtualAddress() + 1024 * i as u64;
                list.list.SetComputeRootUnorderedAccessView(0, address);
                list.list.Dispatch(groups, 1, 1);
            }
        }
        list.close();
        host.execute(&[&list]);

        let mut rows = list.slowest(frequency, usize::MAX);
        println!("run {}: {} command rows", run + 1, rows.len());
        for (ms, text) in &rows {
            println!("slow list command, {ms:.4} ms: {text}");
        }
        let row_count = rows.len();
        if row_count == commands.len() {
            let (_, timestamps) = list.stamps().unwrap();
            let range = D3D12_RANGE {
                Begin: 0,
                End: (commands.len() + 1) * 8,
            };
            let mut data = std::ptr::null_mut();
            // SAFETY: this submission resolved all its timestamps and execute waited for it.
            let ticks = unsafe {
                timestamps.Map(0, Some(&range), Some(&mut data)).unwrap();
                let ticks =
                    std::slice::from_raw_parts(data.cast::<u64>(), commands.len() + 1).to_vec();
                timestamps.Unmap(0, None);
                ticks
            };
            for (i, &(label, groups)) in commands.iter().enumerate() {
                let expected = format!(
                    "[primitives to the rasterizer 0, rasterized 0, pixel shader runs 0, compute runs {}] {label}",
                    groups * 8,
                );
                let index = rows
                    .iter()
                    .position(|(_, text)| *text == expected)
                    .expect(&expected);
                let (ms, _) = rows.remove(index);
                assert!(
                    ticks[i + 1] >= ticks[i],
                    "timestamps went backwards at {label}"
                );
                let expected_ms = (ticks[i + 1] - ticks[i]) as f64 * 1e3 / frequency as f64;
                assert_eq!(ms, expected_ms, "timestamp interval for {label}");
            }
            assert!(rows.is_empty());
        }
        actual.push(row_count);
    }
    assert_eq!(actual, [3, 2]);
}
