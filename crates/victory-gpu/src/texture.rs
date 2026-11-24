//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory changed since the last detiling
//! ([`crate::watch`], which follows writes through every mapping of the
//! same physical pages) in the runtime's memory; elsewhere, a content hash
//! of the whole memory at every submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Textures read through descriptor tables are brought up to date, and for
//! shaders that index the heap themselves every texture with a resource
//! object the heap views (`crate::bindless`); textures the title placed in
//! memory itself are not made for the heap (see `descriptor::host`). Plane
//! 0 is uploaded (every mip and array slice), depth and stencil planes are
//! not.

use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;
use windows::core::Interface;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    buffer: ID3D12Resource,
    /// Its bytes.
    size: u64,
    data: *mut u8,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    memory: *const u8,
    /// The memory's pages changed since detiled (none: not the runtime's
    /// memory, hashed instead).
    tracked: Option<crate::watch::Tracked>,
    last: Mutex<Last>,
    #[cfg(test)]
    gpu_uploads: std::sync::atomic::AtomicU64,
}

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not tracked), with room to read tracked memory into.
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
    read: Vec<u8>,
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

/// A host copy or resolve writes `resource`: its host texture's contents
/// are the host's, not title memory's, which stays as it was.
pub(crate) fn host_written(resource: This) {
    if let Some(copy) = crate::resource::host_copy(resource) {
        copy.host_written.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

fn is_host_written(resource: This) -> bool {
    crate::resource::host_copy(resource)
        .is_some_and(|c| c.host_written.load(std::sync::atomic::Ordering::Relaxed))
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Copies [`alias_copy`] recorded since the last frame-rate line.
pub(crate) static ALIAS_COPIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Records, once per list, the copy of `from`'s host texture into `to`'s:
/// two title textures at one memory that cannot share a host texture
/// (`resource::alias_source`: block-compressed against plain formats of
/// the same bytes, or a cast the host refused), `to` (a placed view's)
/// reading what the GPU wrote through `from`. Through a buffer, each
/// subresource with its own format, as many as have the same bytes per
/// row and rows in both; `to` no longer takes its contents from title
/// memory.
pub(crate) fn alias_copy(list: &HostList, from: This, to: This) {
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    let (Some(a), Some(b), Some(host)) = (
        crate::resource::host_resource(from),
        crate::resource::host_resource(to),
        host(),
    ) else {
        return;
    };
    // An odd key: textures' own first uses are keyed by the host texture.
    if !list.first_use(to as usize | 1) {
        return;
    }
    // SAFETY: live resources.
    let (da, db) = unsafe { (a.GetDesc(), b.GetDesc()) };
    let count = |d: &D3D12_RESOURCE_DESC| {
        if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
            u32::from(d.MipLevels)
        } else {
            u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize).max(1)
        }
    };
    let footprints = |d: &D3D12_RESOURCE_DESC, n: u32| {
        let mut f = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n as usize];
        let mut rows = vec![0u32; n as usize];
        let mut total = 0;
        // SAFETY: outputs sized for `n` subresources.
        unsafe {
            host.device
                .GetCopyableFootprints(d, 0, n, 0, Some(f.as_mut_ptr()), Some(rows.as_mut_ptr()), None, Some(&mut total));
        }
        (f, rows, total)
    };
    let (fa, ra, total) = footprints(&da, count(&da));
    let (fb, rb, _) = footprints(&db, count(&db));
    // Subresources in order while both have the same bytes there (mips of
    // a block-compressed texture past its last whole block do not).
    let n = (0..fa.len().min(fb.len()))
        .take_while(|&i| {
            let (x, y) = (fa[i], fb[i]);
            x.Offset == y.Offset && x.Footprint.RowPitch == y.Footprint.RowPitch && ra[i] == rb[i] && x.Footprint.Depth == y.Footprint.Depth
        })
        .count() as u32;
    if n == 0 {
        once!(?da, ?db, "an aliased texture not copied: its layouts differ on the host");
        return;
    }
    let Some(buffer) = host.default_buffer(total) else { return };
    list.keep(&buffer);
    crate::texture::host_written(to);
    crate::list::ensure_state(list, from, COPY_SOURCE);
    crate::list::ensure_state(list, to, COPY_DEST);
    let place = |r: &ID3D12Resource, f: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the copy does not release.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: f },
    };
    let sub = |r: &ID3D12Resource, i: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i },
    };
    let l = &list.list;
    // SAFETY: an open list; live resources in the states named.
    unsafe {
        for i in 0..n {
            l.CopyTextureRegion(&place(&buffer, fa[i as usize]), 0, 0, 0, &sub(a, i), None);
        }
        l.ResourceBarrier(&[crate::host::transition(
            &buffer,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        for i in 0..n {
            l.CopyTextureRegion(&sub(b, i), 0, 0, 0, &place(&buffer, fb[i as usize]), None);
        }
    }
    ALIAS_COPIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static LOGGED: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
    let mut logged = LOGGED.lock().unwrap_or_else(|e| e.into_inner());
    if logged.len() < 64 && !logged.contains(&(from as usize, to as usize)) {
        logged.push((from as usize, to as usize));
        tracing::info!(from = ?from, to = ?to, a = da.Format.0, b = db.Format.0, w = da.Width, h = da.Height, subresources = n, "aliased texture copied");
    }
}

/// Queues the texture's upload once per list. Its submission prelude
/// refreshes the staging buffer and copies it only when the memory changed.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    // Once per list for each host texture, however many title textures of
    // its memory the list samples.
    if is_host_written(resource) || !list.first_use(texture.as_raw() as usize) {
        return;
    }
    let Some(copy) = crate::resource::host_copy(resource) else {
        return;
    };
    let Some(staging) = staging(resource, texture) else {
        return;
    };
    #[cfg(feature = "diagnostics")]
    for i in 0..staging.footprints.len() {
        staging.note_copy(list, resource, texture, i);
    }
    list.defer_texture(staging, copy);
}

/// For textures shaders reach through the heap itself, at submission
/// (`crate::bindless::uploads`): brings the staging buffer up to date and
/// records its copy into `list` only if the memory changed since (or was
/// never uploaded). True if it recorded; the texture is then in the copy
/// destination state.
pub(crate) fn upload_changed(list: &HostList, resource: This, texture: &ID3D12Resource) -> bool {
    if is_host_written(resource) {
        return false;
    }
    let Some(staging) = staging(resource, texture) else {
        return false;
    };
    if !staging.refresh() {
        return false;
    }
    record(list, resource, texture, &staging);
    list.keep(&staging.buffer);
    true
}

/// Records the copy of every subresource of `staging` into `texture`,
/// moving it to the copy destination state first.
fn record(list: &HostList, resource: This, texture: &ID3D12Resource, staging: &Staging) {
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, resource, COPY_DEST);
    for i in 0..staging.footprints.len() {
        #[cfg(feature = "diagnostics")]
        staging.note_copy(list, resource, texture, i);
        staging.record_copy(&list.list, texture, i);
    }
}

impl Staging {
    #[cfg(feature = "diagnostics")]
    fn note_copy(&self, list: &HostList, resource: This, texture: &ID3D12Resource, i: usize) {
        let footprint = &self.footprints[i];
        let f = footprint.Footprint;
        // SAFETY: live resources.
        let (gpu, d) = unsafe { (self.buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
        let len = u64::from(f.RowPitch) * u64::from(self.rows[i]) * u64::from(f.Depth);
        list.note_read(
            || {
                format!(
                    "texture staging for {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                    d.Width,
                    d.Height,
                    d.DepthOrArraySize,
                    d.Format.0,
                    d.MipLevels,
                    self.footprints.len(),
                    self.size,
                    footprint.Offset,
                    f.Width,
                    f.Height,
                    f.Depth,
                    f.RowPitch
                )
            },
            (gpu + footprint.Offset, len),
        );
    }

    /// Copies the staged subresources during the list's submission prelude.
    pub(crate) fn record_copies(&self, list: &ID3D12GraphicsCommandList, texture: &ID3D12Resource) {
        for i in 0..self.footprints.len() {
            self.record_copy(list, texture, i);
        }
    }

    fn record_copy(&self, list: &ID3D12GraphicsCommandList, texture: &ID3D12Resource, i: usize) {
        let footprint = &self.footprints[i];
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&self.buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        #[cfg(test)]
        self.gpu_uploads.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    #[cfg(test)]
    fn gpu_uploads(&self) -> u64 {
        self.gpu_uploads.load(std::sync::atomic::Ordering::Relaxed)
    }
    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all): whether it did.
    pub(crate) fn refresh(&self) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let (read, changed) = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
        changed
    }

    /// Returns the bytes of title memory read, and whether it detiled.
    fn refresh_now(&self) -> (usize, bool) {
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        if let Some(t) = &self.tracked {
            if t.take(start, start + size).is_empty() {
                return (0, false);
            }
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            last.read.resize(size, 0);
            crate::host::copy_mapped(last.read.as_mut_ptr(), self.memory, size, false);
            self.detile(&last.read);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            return (size, true);
        }
        if !committed(self.memory, size) {
            #[cfg(feature = "diagnostics")]
            tracing::debug!(
                memory = ?self.memory,
                size,
                desc = ?self.layout.desc,
                "a sampled texture's memory is not committed"
            );
            return (0, false);
        }
        // SAFETY: committed memory of the texture's layout size.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        let hash = blake3::hash(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some(hash) {
            return (size, false);
        }
        last.hash = Some(hash);
        self.detile(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        (size, true)
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer.
    fn detile(&self, tiled: &[u8]) {
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = subresource_len(footprint.Offset, pitch, rows, f.Depth, self.size);
            // SAFETY: `subresource_len` clamps `len` to `self.size - footprint.Offset`,
            // keeping the slice strictly inside the mapped staging buffer even when
            // GetCopyableFootprints does not pad the final row of the last subresource.
            let linear = unsafe {
                std::slice::from_raw_parts_mut(self.data.add(footprint.Offset as usize), len)
            };
            if let Err(e) =
                self.layout
                    .detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
            {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return;
            }
        }
    }
}

/// Bytes available in the staging buffer for a subresource at `offset`.
/// D3D12's `GetCopyableFootprints` omits trailing row-pitch padding on the
/// final row of the final subresource, so `pitch * rows * depth` can exceed
/// `total - offset`.
fn subresource_len(offset: u64, pitch: usize, rows: u32, depth: u32, total: u64) -> usize {
    let padded = pitch
        .saturating_mul(rows as usize)
        .saturating_mul(depth as usize);
    let remaining = usize::try_from(total.saturating_sub(offset)).unwrap_or(usize::MAX);
    padded.min(remaining)
}

/// Whether `[at, at + len)` is committed, readable memory (streamed textures
/// may be unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    let mut p = at as usize;
    let Some(end) = p.checked_add(len) else {
        return false;
    };
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        let unreadable = info.Protect == 0
            || info.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0;
        if n == 0 || info.State != MEM_COMMIT || unreadable {
            return false;
        }
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= p {
            return false;
        }
        p = next;
    }
    true
}

/// The staging state of a title texture's host texture, made on first use
/// (one for all title textures of its memory).
fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let copy = crate::resource::host_copy(resource)?;
    let mut held = copy.staging.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = held.as_ref() {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let s = create_staging(host()?, resource, texture, layout, memory)?;
    *held = Some(s.clone());
    Some(s)
}

fn create_staging(
    host: &crate::host::Host,
    resource: This,
    texture: &ID3D12Resource,
    layout: victory_formats::texture::Layout,
    memory: *const u8,
) -> Option<Arc<Staging>> {
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    let (buffer, data) = host.upload_buffer(total)?;
    crate::host::name(&buffer, || {
        format!(
            "staging of {resource:?} {}x{}x{} f{}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0
        )
    });
    tracing::debug!(desc = ?layout.desc, total, "texture contents uploaded from title memory");
    let layout_size = layout.size as usize;
    Some(Arc::new(Staging {
        buffer,
        size: total,
        data,
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        memory,
        tracked: crate::watch::track(memory as usize, layout_size, true),
        last: Mutex::default(),
        #[cfg(test)]
        gpu_uploads: std::sync::atomic::AtomicU64::new(0),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subresource_len_stays_inside_staging_total() {
        // A 4x4 RGBA8 texture has RowPitch=256, rows=4, last_row=16 -> total=784.
        // Padded size (1024) must be clamped to 784 so from_raw_parts_mut stays in bounds.
        assert_eq!(subresource_len(0, 256, 4, 1, 784), 784);
        assert_eq!(subresource_len(512, 256, 4, 1, 784), 272);
        assert_eq!(subresource_len(0, 256, 4, 1, 2048), 1024);
    }

    #[test]
    fn committed_rejects_overflow_and_noaccess_pages() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        assert!(!committed(usize::MAX as *const u8, 16));
        // SAFETY: reserving and committing one private page for the test, freed below.
        let noaccess =
            unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS) };
        assert!(!noaccess.is_null());
        assert!(!committed(noaccess.cast::<u8>(), 4096));
        // SAFETY: freeing the allocation above.
        unsafe { VirtualFree(noaccess, 0, MEM_RELEASE) };

        // Negative control: a readable committed page still passes.
        // SAFETY: reserving and committing one private page for the test, freed below.
        let rw =
            unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        assert!(!rw.is_null());
        assert!(committed(rw.cast::<u8>(), 4096));
        // SAFETY: freeing the allocation above.
        unsafe { VirtualFree(rw, 0, MEM_RELEASE) };
    }
}

#[cfg(test)]
mod prelude_tests {
    use super::*;
    use crate::host::Host;
    use crate::resource::HostTexture;
    use std::sync::atomic::Ordering;
    use victory_formats::texture::{Dimension, Swizzle, Texture, TextureDesc};
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};

    const SHADER_RESOURCE: u32 = (D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0
        | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0) as u32;

    fn make_rgba_texture(host: &Host, width: u32, height: u32) -> (Arc<HostTexture>, Texture) {
        let layout_tex = Texture::new(&TextureDesc {
            dimension: Dimension::Tex2d,
            format: DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32,
            width,
            height,
            depth_or_array: 1,
            mips: 1,
            samples: 1,
            swizzle: Swizzle::LINEAR,
        })
        .expect("linear RGBA layout");
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(width),
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            ..Default::default()
        };
        let mut resource: Option<ID3D12Resource> = None;
        // SAFETY: plain default-heap 2D texture creation on our test device.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut resource,
                )
                .expect("create host texture");
        }
        let copy = Arc::new(HostTexture::new(
            resource.unwrap(),
            D3D12_RESOURCE_STATE_COMMON.0 as u32,
            Vec::new(),
        ));
        (copy, layout_tex)
    }

    /// Reads a 2D RGBA8 texture back from `host` into a tightly packed row buffer.
    fn read_back_rgba(host: &Host, copy: &Arc<HostTexture>, width: u32, height: u32) -> Vec<u8> {
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut rows = 0u32;
        let mut total = 0u64;
        // SAFETY: querying footprint of a live texture.
        let desc = unsafe { copy.texture.GetDesc() };
        // SAFETY: single-subresource footprint query.
        unsafe {
            host.device.GetCopyableFootprints(
                &desc,
                0,
                1,
                0,
                Some(&mut footprint),
                Some(&mut rows),
                None,
                Some(&mut total),
            );
        }
        let readback_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let readback_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: total,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain readback buffer creation.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &readback_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &readback_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .expect("create readback buffer");
        }
        let readback = readback.unwrap();
        let list = host.new_list().expect("readback list");
        let before = copy.state.load(Ordering::Relaxed);
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed COM pointer for the copy call.
            pResource: unsafe { std::mem::transmute_copy(&readback) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed COM pointer for the copy call.
            pResource: unsafe { std::mem::transmute_copy(&copy.texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        // SAFETY: recording a valid readback copy on an open list.
        unsafe {
            if before != D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32 {
                list.list.ResourceBarrier(&[crate::host::transition(
                    &copy.texture,
                    D3D12_RESOURCE_STATES(before as i32),
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
            }
            list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            if before != D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32 {
                list.list.ResourceBarrier(&[crate::host::transition(
                    &copy.texture,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATES(before as i32),
                )]);
            }
        }
        list.close();
        host.execute(&[&list]);

        let mut ptr = std::ptr::null_mut();
        // SAFETY: GPU finished writing `readback`; map for CPU read.
        unsafe {
            readback.Map(0, None, Some(&mut ptr)).expect("map readback");
            let pitch = footprint.Footprint.RowPitch as usize;
            let row_bytes = width as usize * 4;
            let src = std::slice::from_raw_parts(ptr.cast::<u8>(), total as usize);
            let mut out = Vec::with_capacity(row_bytes * height as usize);
            for y in 0..height as usize {
                let start = footprint.Offset as usize + y * pitch;
                out.extend_from_slice(&src[start..start + row_bytes]);
            }
            readback.Unmap(0, None);
            out
        }
    }

    fn fill_linear_rgba(layout: &victory_formats::texture::Layout, mem: &mut [u8], pixel: [u8; 4]) {
        let mut linear = vec![0u8; layout.desc.width as usize * layout.desc.height as usize * 4];
        for chunk in linear.as_chunks_mut::<4>().0 {
            *chunk = pixel;
        }
        let pitch = layout.desc.width as usize * 4;
        layout
            .tile(mem, 0, 0, &linear, pitch, linear.len())
            .expect("tile pixels");
    }

    fn all_pixels_eq(bytes: &[u8], expected: [u8; 4]) -> bool {
        let (chunks, rem) = bytes.as_chunks::<4>();
        rem.is_empty() && !chunks.is_empty() && chunks.iter().all(|&px| px == expected)
    }

    /// Exercises the real D3D12 WARP prelude upload path end to end:
    /// (a) initial use uploads title memory to GPU,
    /// (b) subsequent frames with unchanged title memory record zero GPU copies,
    /// (c) multi-list submissions after a modification upload only once,
    /// (d) modifying title memory triggers a fresh GPU upload with new pixels.
    #[test]
    fn warp_skips_clean_texture_uploads_and_reuploads_on_change() {
        let Some(host) = Host::new_warp() else {
            return; // WARP software adapter unavailable here
        };
        let (width, height) = (4, 4);
        let (copy, tex) = make_rgba_texture(&host, width, height);
        let layout = tex.planes[0].layout.clone();
        let mut title_mem = vec![0u8; layout.size as usize];
        let v1 = [0x11, 0x22, 0x33, 0xff];
        fill_linear_rgba(&layout, &mut title_mem, v1);

        let staging = create_staging(
            &host,
            std::ptr::null(),
            &copy.texture,
            layout.clone(),
            title_mem.as_ptr(),
        )
        .expect("create staging");
        assert_eq!(staging.gpu_uploads(), 0);

        let make_sampling_list = || {
            let list = host.new_list().expect("new list");
            list.defer_texture(staging.clone(), copy.clone());
            list.expect_texture_state(&copy, SHADER_RESOURCE);
            list.close();
            list
        };

        // (a) Submission 1: first use must upload onto the prelude and land on GPU.
        let list1 = make_sampling_list();
        host.execute(&[&list1]);
        assert_eq!(staging.gpu_uploads(), 1, "initial use must record 1 copy");
        assert_eq!(copy.state.load(Ordering::Relaxed), SHADER_RESOURCE);
        assert!(all_pixels_eq(
            &read_back_rgba(&host, &copy, width, height),
            v1
        ));

        // (b) Submission 2: unchanged title memory across two lists records ZERO copies.
        let list2a = make_sampling_list();
        let list2b = make_sampling_list();
        host.execute(&[&list2a, &list2b]);
        assert_eq!(
            staging.gpu_uploads(),
            1,
            "unchanged texture must record zero CopyTextureRegion uploads"
        );
        // Re-submitting the same closed list with unchanged memory also records zero copies.
        host.execute(&[&list2a]);
        assert_eq!(staging.gpu_uploads(), 1);
        assert!(all_pixels_eq(
            &read_back_rgba(&host, &copy, width, height),
            v1
        ));

        // (c) Submission 3: CPU modifies title memory; two lists sample it -> exactly 1 upload.
        let v2 = [0xaa, 0xbb, 0xcc, 0xff];
        fill_linear_rgba(&layout, &mut title_mem, v2);
        let list3a = make_sampling_list();
        let list3b = make_sampling_list();
        host.execute(&[&list3a, &list3b]);
        assert_eq!(
            staging.gpu_uploads(),
            2,
            "modified texture across two lists must upload exactly once"
        );
        assert_eq!(copy.state.load(Ordering::Relaxed), SHADER_RESOURCE);
        assert!(all_pixels_eq(
            &read_back_rgba(&host, &copy, width, height),
            v2
        ));

        // (d) Submission 4: unchanged again -> zero additional uploads.
        let list4 = make_sampling_list();
        host.execute(&[&list4]);
        assert_eq!(staging.gpu_uploads(), 2);
    }

    /// Verifies the write-watched (`MEM_WRITE_WATCH`) dirty-tracking path on
    /// real D3D12 WARP: first use uploads, clean frames skip, and a CPU page
    /// write triggers a single fresh upload.
    #[test]
    fn warp_write_watched_texture_uploads_only_when_written() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;

        let Some(host) = Host::new_warp() else {
            return;
        };
        let (width, height) = (4, 4);
        let (copy, tex) = make_rgba_texture(&host, width, height);
        let layout = tex.planes[0].layout.clone();
        let alloc_size = (layout.size as usize).max(crate::watch::PAGE);
        // SAFETY: allocating a private write-watched page buffer freed before returning.
        let raw = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                alloc_size,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as *mut u8;
        assert!(!raw.is_null());

        let v1 = [0x33, 0x44, 0x55, 0xff];
        // SAFETY: `raw` is committed read-write memory of `alloc_size` bytes.
        unsafe {
            let mem = std::slice::from_raw_parts_mut(raw, layout.size as usize);
            fill_linear_rgba(&layout, mem, v1);
        }

        let staging = create_staging(&host, std::ptr::null(), &copy.texture, layout.clone(), raw)
            .expect("create staging");

        let run_once = || {
            let list = host.new_list().expect("new list");
            list.defer_texture(staging.clone(), copy.clone());
            list.expect_texture_state(&copy, SHADER_RESOURCE);
            list.close();
            host.execute(&[&list]);
        };

        // Initial submission uploads.
        run_once();
        assert_eq!(staging.gpu_uploads(), 1);
        assert!(all_pixels_eq(
            &read_back_rgba(&host, &copy, width, height),
            v1
        ));

        // Unchanged write-watched memory skips GPU upload.
        run_once();
        assert_eq!(staging.gpu_uploads(), 1);

        // CPU write dirties the watched page -> next submission uploads.
        let v2 = [0xde, 0xad, 0xbe, 0xff];
        // SAFETY: `raw` is still committed read-write memory.
        unsafe {
            let mem = std::slice::from_raw_parts_mut(raw, layout.size as usize);
            fill_linear_rgba(&layout, mem, v2);
        }
        run_once();
        assert_eq!(staging.gpu_uploads(), 2);
        assert!(all_pixels_eq(
            &read_back_rgba(&host, &copy, width, height),
            v2
        ));

        // Clean again -> zero upload.
        run_once();
        assert_eq!(staging.gpu_uploads(), 2);

        // SAFETY: freeing our own allocation.
        unsafe { VirtualFree(raw.cast(), 0, MEM_RELEASE) };
    }
}
