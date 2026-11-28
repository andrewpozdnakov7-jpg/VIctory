//! Hidden diagnostics for the D3D12 Agility SDK opt-in.
//!
//! The root-signature serialization probe deliberately goes through
//! `victory_gpu::root::serialize_versioned`, the same title-facing path the
//! game uses, rather than a separate copy of the host call.
//! Set VICTORY_WARP=1 to select the software adapter for the device check.

use anyhow::Result;

#[cfg(windows)]
use std::ffi::c_void;

#[cfg(windows)]
#[repr(C)]
struct ProbeDesc {
    parameter_count: u32,
    parameters: *const c_void,
    sampler_count: u32,
    samplers: *const c_void,
    flags: u32,
}

#[cfg(windows)]
#[repr(C)]
struct ProbeVersionedDesc {
    version: u32,
    desc: ProbeDesc,
}

#[cfg(windows)]
const _: () = {
    assert!(size_of::<ProbeDesc>() == size_of::<victory_gpu::root::Desc>());
    assert!(align_of::<ProbeDesc>() == align_of::<victory_gpu::root::Desc>());
    assert!(size_of::<ProbeVersionedDesc>() == size_of::<victory_gpu::root::VersionedDesc>());
    assert!(align_of::<ProbeVersionedDesc>() == align_of::<victory_gpu::root::VersionedDesc>());
};

/// `root::serialize_versioned` looks at `desc + 0x38` when recovering the
/// console's compact input. Keep a zero there so the probe does not interpret
/// unrelated stack bytes as an input end pointer.
#[cfg(windows)]
#[repr(C)]
struct ProbeFrame {
    desc: ProbeVersionedDesc,
    tail: [usize; 4],
}

#[cfg(windows)]
pub fn run() -> Result<()> {
    match option_env!("VICTORY_AGILITY_SDK_VERSION") {
        Some(version) => println!("agility probe: executable configured for Agility SDK {version}"),
        None => println!("agility probe: executable is not configured for the Agility SDK"),
    }

    let mut failures = 0;
    for flags in [0xc00, 0xc01] {
        let frame = ProbeFrame {
            desc: ProbeVersionedDesc {
                version: 2,
                desc: ProbeDesc {
                    parameter_count: 0,
                    parameters: std::ptr::null(),
                    sampler_count: 0,
                    samplers: std::ptr::null(),
                    flags,
                },
            },
            tail: [0; 4],
        };
        let mut blob = std::ptr::null_mut();
        let mut error = std::ptr::null_mut();
        let result = victory_gpu::root::serialize_versioned(
            (&raw const frame.desc).cast(),
            &raw mut blob,
            &raw mut error,
        );
        let text = blob_text(error)
            .map(|t| format!("; host error: {t}"))
            .unwrap_or_default();
        println!(
            "root serialize flags {flags:#x}: hresult {result:#x} {}{text}",
            if result >= 0 { "succeeded" } else { "failed" }
        );
        release(blob);
        release(error);
        if result < 0 {
            failures += 1;
        }
    }

    match victory_shader::emit::host::Host::new(false) {
        Ok(host) => {
            let adapter =
                adapter_name(&host.device).unwrap_or_else(|| "adapter name unknown".into());
            println!("D3D12 SM 6.6 device: created ({adapter})");
            match host.root_signature_for(&[]) {
                Ok(_) => println!("D3D12 root signature 0xc01: created"),
                Err(e) => {
                    println!("D3D12 root signature 0xc01: failed: {e}");
                    failures += 1;
                }
            }
        }
        Err(e) => {
            println!("D3D12 SM 6.6 device: unavailable: {e}");
            failures += 1;
        }
    }

    anyhow::ensure!(failures == 0, "Agility probe failed at {failures} step(s)");
    Ok(())
}

#[cfg(windows)]
fn adapter_name(device: &windows::Win32::Graphics::Direct3D12::ID3D12Device) -> Option<String> {
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1,
        IDXGIFactory4,
    };
    // SAFETY: plain DXGI/D3D12 queries with live COM objects.
    unsafe {
        let luid = device.GetAdapterLuid();
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
        let adapter: IDXGIAdapter1 = factory.EnumAdapterByLuid(luid).ok()?;
        let desc = adapter.GetDesc1().ok()?;
        let len = desc
            .Description
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(desc.Description.len());
        let name = String::from_utf16_lossy(&desc.Description[..len]);
        let software = if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
            ", software"
        } else {
            ""
        };
        Some(format!(
            "{name}, {} MiB dedicated{software}",
            desc.DedicatedVideoMemory as u64 / (1024 * 1024)
        ))
    }
}

#[cfg(windows)]
fn blob_text(object: *mut c_void) -> Option<String> {
    if object.is_null() {
        return None;
    }
    type GetBufferPointer = unsafe extern "system" fn(*mut c_void) -> *mut c_void;
    type GetBufferSize = unsafe extern "system" fn(*mut c_void) -> usize;
    // SAFETY: the serializer returned a live ID3DBlob. ID3DBlob vtable slots
    // 3 and 4 are GetBufferPointer/GetBufferSize.
    let (pointer, size) = unsafe {
        let vtable = (object as *const usize).read() as *const usize;
        let get_pointer: GetBufferPointer = std::mem::transmute(vtable.add(3).read());
        let get_size: GetBufferSize = std::mem::transmute(vtable.add(4).read());
        (get_pointer(object), get_size(object))
    };
    if pointer.is_null() || size == 0 {
        return None;
    }
    // SAFETY: an ID3DBlob's buffer is readable for its reported size.
    let bytes = unsafe { std::slice::from_raw_parts(pointer.cast::<u8>(), size) };
    Some(
        String::from_utf8_lossy(bytes)
            .trim_end_matches(['\0', '\n', '\r'])
            .to_owned(),
    )
}

#[cfg(windows)]
fn release(object: *mut c_void) {
    if object.is_null() {
        return;
    }
    type Release = unsafe extern "system" fn(*mut c_void) -> u32;
    // SAFETY: the serializer returned a live COM object. IUnknown slot 2 is
    // Release; calling it once balances the reference the call returned.
    unsafe {
        let vtable = (object as *const usize).read() as *const usize;
        let release: Release = std::mem::transmute(vtable.add(2).read());
        release(object);
    }
}

#[cfg(not(windows))]
pub fn run() -> Result<()> {
    anyhow::bail!("the Agility probe is Windows-only")
}
