use super::*;

#[test]
#[ignore = "requires an isolated WARP process with VICTORY_D3D12_WARP set"]
fn pooled_staging_keeps_live_lists_independent_through_reset_and_drop() {
    assert!(crate::switch::on("VICTORY_D3D12_WARP"));
    let host = host().expect("WARP device");
    const WORDS: usize = 1024;

    let record = |list: &HostList, tag: u32| {
        let size = (WORDS * size_of::<u32>()) as u64;
        let (upload, src) = list.upload(host, size).unwrap();
        let (readback, dst) = list.readback(host, size).unwrap();
        // SAFETY: two live mapped buffers of the requested size. The upload
        // belongs to this open list and the copy runs before reset or drop.
        unsafe {
            for i in 0..WORDS {
                src.cast::<u32>()
                    .add(i)
                    .write_unaligned(tag.wrapping_add(i as u32).to_le());
            }
            list.list.CopyBufferRegion(&readback, 0, &upload, 0, size);
        }
        (src, dst)
    };
    let verify = |dst: *mut u8, tag: u32| {
        for i in 0..WORDS {
            // SAFETY: the copy completed and the list still owns the readback.
            let value = unsafe { dst.cast::<u32>().add(i).read_unaligned() };
            assert_eq!(u32::from_le(value), tag.wrapping_add(i as u32));
        }
    };

    let first = host.new_list().unwrap();
    let second = host.new_list().unwrap();
    let first_buffers = record(&first, 0x1020_3000);
    let second_buffers = record(&second, 0x4050_6000);
    assert_ne!(first_buffers.0, second_buffers.0);
    assert_ne!(first_buffers.1, second_buffers.1);
    first.close();
    second.close();
    host.execute(&[&second, &first]);
    verify(first_buffers.1, 0x1020_3000);
    verify(second_buffers.1, 0x4050_6000);

    first.reset();
    let reset_buffers = record(&first, 0x7080_9000);
    assert_eq!(reset_buffers, first_buffers);
    first.close();
    host.execute(&[&first]);
    verify(reset_buffers.1, 0x7080_9000);
    verify(second_buffers.1, 0x4050_6000);
    drop(first);

    let third = host.new_list().unwrap();
    let drop_buffers = record(&third, 0xa0b0_c000);
    assert_eq!(drop_buffers, reset_buffers);
    third.close();
    host.execute(&[&third]);
    verify(drop_buffers.1, 0xa0b0_c000);
    verify(second_buffers.1, 0x4050_6000);
    // SAFETY: the queue completed all submitted work on the host device.
    unsafe { host.device.GetDeviceRemovedReason() }.expect("host device still available");
}
