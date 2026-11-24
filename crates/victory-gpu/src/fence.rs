//! Fences: a 64-bit value and the events waiting for it. Nothing runs on a
//! GPU yet, so only signals (the title's, later our queues') move one.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::object::{self, E_NOTIMPL, FENCE, HRESULT, Iid, Object, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

pub(crate) const FENCE_IIDS: [Iid; 4] = [
    guid(
        0x0a753dcf,
        0xc4d8,
        0x4b91,
        [0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Fence {
    object: Object,
    /// Titles also read it straight from memory, at the fence's GPU
    /// address.
    value: AtomicU64,
    flags: u32,
    /// Events to set once the value reaches theirs (handles as integers).
    waiters: Mutex<Vec<(u64, usize)>>,
}

fn fence<'a>(this: This) -> &'a Fence {
    // SAFETY: called from fence methods only.
    unsafe { &*this.cast::<Fence>() }
}

pub(crate) extern "system" fn create(
    _: This,
    initial: u64,
    flags: u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::debug!(initial, flags, "fence");
    let fence = Box::new(Fence {
        object: Object::new(&FUNCTIONS),
        value: AtomicU64::new(initial),
        flags,
        waiters: Mutex::new(Vec::new()),
    });
    object::hand_out(FENCE, fence, iid, &FENCE_IIDS, out)
}

fn set_event(event: usize) {
    #[cfg(windows)]
    // SAFETY: the title passed an event handle.
    unsafe {
        windows_sys::Win32::System::Threading::SetEvent(event as *mut c_void)
    };
    #[cfg(not(windows))]
    let _ = event;
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        FENCE,
        iid,
        object::is(iid, &FENCE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Fence>(this)
}
extern "system" fn completed(this: This) -> u64 {
    fence(this).value.load(Ordering::Acquire)
}
extern "system" fn set_event_on_completion(this: This, value: u64, event: *mut c_void) -> HRESULT {
    let fence = fence(this);
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    if fence.value.load(Ordering::Acquire) >= value {
        if !event.is_null() {
            set_event(event as usize);
        }
        return S_OK;
    }
    if event.is_null() {
        // shortcut: a null event means "block until reached"; nothing can
        // reach it while no GPU work runs, so fail instead of hanging.
        tracing::warn!(value, "blocking fence wait not implemented");
        return E_NOTIMPL;
    }
    waiters.push((value, event as usize));
    S_OK
}
pub(crate) extern "system" fn signal(this: This, value: u64) -> HRESULT {
    let fence = fence(this);
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    // New waits check the latest signal value, including after a rewind.
    fence.value.store(value, Ordering::Release);
    waiters.retain(|&(wanted, event)| {
        let reached = wanted <= value;
        if reached {
            set_event(event);
        }
        !reached
    });
    S_OK
}

extern "system" fn creation_flags(this: This) -> u32 {
    fence(this).flags
}
/// Where the value lives (GPU addresses are CPU addresses).
extern "system" fn gpu_address(this: This) -> u64 {
    (&raw const fence(this).value) as u64
}

/// IUnknown 3, object 4, device child 1, fence 3, fence 1 (2, a console
/// one among them).
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<FENCE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = completed as *const (); // 0x40
    f[9] = set_event_on_completion as *const (); // 0x48
    f[10] = signal as *const (); // 0x50
    f[11] = creation_flags as *const (); // 0x58
    f[12] = gpu_address as *const (); // 0x60
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{E_NOINTERFACE, E_POINTER, S_FALSE, slot};

    #[cfg(windows)]
    #[test]
    fn signals_set_waiting_events() {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 1, 0, &FENCE_IIDS[0], &mut f), S_OK);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(f, 0x40)(f), 1);
        // SAFETY: plain auto-reset event.
        let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
        assert!(!event.is_null());
        let wait = slot::<extern "system" fn(This, u64, *mut c_void) -> HRESULT>(f, 0x48);
        let signal = slot::<extern "system" fn(This, u64) -> HRESULT>(f, 0x50);
        assert_eq!(wait(f, 5, event), S_OK);
        // SAFETY: a valid event.
        assert_ne!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(signal(f, 5), S_OK);
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(wait(f, 5, event), S_OK, "already reached");
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(wait(f, 9, std::ptr::null_mut()), E_NOTIMPL);
        // Grand Theft Auto VI reads the value at the fence's address.
        let at = slot::<extern "system" fn(This) -> u64>(f, 0x60)(f);
        // SAFETY: the fence's value.
        assert_eq!(unsafe { *(at as *const u64) }, 5);
        // SAFETY: ours.
        unsafe { CloseHandle(event) };
        assert_eq!(slot::<extern "system" fn(This) -> u32>(f, 0x10)(f), 0);
    }

    #[test]
    fn signal_can_rewind_the_fence_value() {
        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 3, 0, &FENCE_IIDS[0], &mut f), S_OK);
        assert_eq!(signal(f, 5), S_OK);
        assert_eq!(completed(f), 5);
        assert_eq!(signal(f, 2), S_OK);
        assert_eq!(completed(f), 2);
        assert_eq!(release(f), 0);
    }

    /// Creation parameters, and the value where the title reads it.
    #[test]
    fn creation_parameters_and_gpu_address() {
        let null = std::ptr::null();
        let (mut a, mut b) = (null, null);
        assert_eq!(create(null, 7, 4, &FENCE_IIDS[0], &mut a), S_OK);
        assert_eq!(create(null, 0, 0, &FENCE_IIDS[1], &mut b), S_OK);
        assert_eq!(creation_flags(a), 4);
        assert_eq!(creation_flags(b), 0);
        assert_eq!(completed(a), 7);
        assert_eq!(completed(b), 0);
        let at = gpu_address(a);
        // SAFETY: the fence's value, read only.
        assert_eq!(unsafe { (at as *const u64).read() }, 7);
        assert_eq!(gpu_address(a), at, "the value does not move");
        assert_ne!(gpu_address(b), at, "each fence has its own value");
        assert_eq!(signal(a, 9), S_OK);
        // SAFETY: as above.
        assert_eq!(unsafe { (at as *const u64).read() }, 9);
        assert_eq!(completed(b), 0);
        assert_eq!(release(a), 0);
        assert_eq!(release(b), 0);
    }

    /// Interfaces, the references they hand out and the last release.
    #[test]
    fn interfaces_and_references() {
        let unknown = guid(0, 0, 0, [0; 8]);
        // Asked only whether creation would succeed.
        assert_eq!(
            create(std::ptr::null(), 0, 0, &FENCE_IIDS[0], std::ptr::null_mut()),
            S_FALSE
        );
        let mut out = std::ptr::null();
        assert_eq!(
            create(std::ptr::null(), 0, 0, &unknown, &mut out),
            E_NOINTERFACE
        );
        assert!(out.is_null(), "the title's pointer is cleared");
        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 0, 0, &FENCE_IIDS[0], &mut f), S_OK);
        // Every interface is the same object, with a new reference.
        let mut q = std::ptr::null();
        assert_eq!(query(f, &PAGEABLE_IIDS[1], &mut q), S_OK);
        assert_eq!(q, f);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(f, 0x8)(f), 3);
        assert_eq!(release(f), 2);
        // An unknown interface answers none, and clears the pointer.
        let mut none = f;
        assert_eq!(query(f, &unknown, &mut none), E_NOINTERFACE);
        assert!(none.is_null());
        assert_eq!(query(f, &FENCE_IIDS[0], std::ptr::null_mut()), E_POINTER);
        assert_eq!(release(f), 1);
        assert_eq!(release(f), 0, "the last reference drops the fence");
    }

    /// The table's slots are where the interface puts them.
    #[test]
    fn methods_sit_at_their_offsets() {
        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 0, 8, &FENCE_IIDS[0], &mut f), S_OK);
        let completed_at = slot::<extern "system" fn(This) -> u64>(f, 0x40);
        let wait_at = slot::<extern "system" fn(This, u64, *mut c_void) -> HRESULT>(f, 0x48);
        let signal_at = slot::<extern "system" fn(This, u64) -> HRESULT>(f, 0x50);
        let flags_at = slot::<extern "system" fn(This) -> u32>(f, 0x58);
        let address_at = slot::<extern "system" fn(This) -> u64>(f, 0x60);
        assert_eq!(completed_at(f), 0);
        assert_eq!(flags_at(f), 8);
        assert_eq!(signal_at(f, 4), S_OK);
        assert_eq!(completed_at(f), 4);
        // SAFETY: the fence's value.
        assert_eq!(unsafe { *(address_at(f) as *const u64) }, 4);
        assert_eq!(wait_at(f, 4, std::ptr::null_mut()), S_OK, "reached");
        assert_eq!(
            wait_at(f, 5, std::ptr::null_mut()),
            E_NOTIMPL,
            "never reached"
        );
        assert_eq!(slot::<extern "system" fn(This) -> u32>(f, 0x10)(f), 0);
    }

    /// Waiters fire when their value is reached, and only then.
    #[cfg(windows)]
    #[test]
    fn events_fire_when_their_value_is_reached() {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 1, 0, &FENCE_IIDS[0], &mut f), S_OK);
        // SAFETY: plain auto-reset events, all ours.
        let (low, high) = unsafe {
            (
                CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()),
                CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()),
            )
        };
        assert!(!low.is_null() && !high.is_null());
        let signalled = |event: *mut c_void| {
            // SAFETY: a live event of ours.
            unsafe { WaitForSingleObject(event, 0) == WAIT_OBJECT_0 }
        };
        assert_eq!(set_event_on_completion(f, 2, low), S_OK);
        assert_eq!(set_event_on_completion(f, 4, high), S_OK);
        assert!(!signalled(low) && !signalled(high));
        assert_eq!(signal(f, 2), S_OK);
        assert!(signalled(low), "the waiter for 2");
        assert!(!signalled(high), "4 is not reached yet");
        assert_eq!(signal(f, 4), S_OK);
        assert!(signalled(high), "the waiter for 4");
        // A value that was reached before the wait fires at once.
        assert_eq!(set_event_on_completion(f, 3, low), S_OK);
        assert!(signalled(low));
        assert_eq!(signal(f, 1), S_OK);
        assert_eq!(set_event_on_completion(f, 3, low), S_OK);
        assert!(!signalled(low), "a new wait checks the rewound value");
        assert_eq!(signal(f, 3), S_OK);
        assert!(signalled(low));
        // Nothing is left waiting: a later signal sets nothing.
        assert_eq!(signal(f, 9), S_OK);
        assert!(!signalled(low) && !signalled(high));
        // SAFETY: ours.
        unsafe {
            CloseHandle(low);
            CloseHandle(high);
        }
        assert_eq!(release(f), 0);
    }
}
