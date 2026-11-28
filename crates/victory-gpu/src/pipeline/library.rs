//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early (decision 0007, "Pipeline cache").
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The library is saved under the cache
//!    folder; the host refuses a library of another adapter or driver, and
//!    an entry whose description differs, so neither is ever used. A
//!    pipeline with static samplers is named by their places in the
//!    driver's table too (`sampler_order`), or, where those cannot be
//!    vouched for, made every run on NVIDIA ([`keeps_static_samplers`]).
//! 2. The title pipelines a run drew or dispatched with (made in time or
//!    not) are listed by a key known when the title creates them (their
//!    shader containers and graphics state); the next run hands those to
//!    the workers at creation, during loading. A full list keeps the
//!    latest run's first.
//!
//! 3. The pre-build (`victory translate --cache --pipelines`) lists the
//!    title pipelines it has put in the library, by the same key, so a
//!    second pre-build skips them without translating or loading anything.
//!    The list holds only for the library bytes it was saved with and for
//!    what names the pipelines (the translator, this module and `host`).
//!
//! The library is free-threaded but for loads of one pipeline on several
//! threads (`ID3D12Device1::CreatePipelineLibrary`, "Thread Safety"), so
//! pipelines of different names load at the same time. A load is not always
//! quick: the library of an AMD driver holds about 140 bytes per pipeline,
//! and what its own cache lacks is compiled inside the load, for seconds.
//! Loaded one at a time, 650 pipelines kept the eight workers to one core
//! for minutes, while each title frame could wait out its 500 ms for
//! them (`make`). Stores and serialising keep the lock: the size asked for
//! must still be right when the bytes are written. Loads take no lock at
//! all, since a load adds nothing to the library: a save holds the lock
//! while it serialises the whole library (hundreds of megabytes in the
//! world), and loads that waited on it stalled every worker for as long.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Keeps the pipeline library and the early list in `dir` from now on.
pub fn use_pipeline_cache(dir: PathBuf) {
    let _ = DIR.set(dir);
}

/// The cache folder, once one is given.
pub(super) fn dir() -> Option<&'static PathBuf> {
    DIR.get()
}

/// Host pipelines created, loaded from the library, and title pipelines
/// started at creation, since the start.
pub(crate) static MADE: AtomicU64 = AtomicU64::new(0);
pub(crate) static LOADED: AtomicU64 = AtomicU64::new(0);
pub(crate) static EARLY: AtomicU64 = AtomicU64::new(0);

struct Library {
    library: ID3D12PipelineLibrary1,
    /// The blob the library was made from: it must outlive the library.
    _blob: Vec<u8>,
    /// Pipelines stored since the last save.
    added: u64,
    /// Title pipelines the pre-build put in the library or found there, by
    /// key, and whether that list changed since the last save.
    prebuilt: HashSet<String>,
    prebuilt_changed: bool,
}

// SAFETY: D3D12 objects are free-threaded (a pipeline library with the
// exception `NAMES` takes care of).
unsafe impl Send for Library {}

/// The library for loads, beside the one stores and saves lock.
struct Shared {
    /// The same library as `state`'s, loaded from without its lock.
    loads: ID3D12PipelineLibrary1,
    state: Mutex<Library>,
}

// SAFETY: as for `Library`.
unsafe impl Send for Shared {}
// SAFETY: the library is free-threaded but for loads of one name, which
// `NAMES` keeps apart; the rest is behind `state`'s lock.
unsafe impl Sync for Shared {}

/// Names one thread at a time holds.
struct Names {
    held: Mutex<Option<HashSet<String>>>,
    released: Condvar,
}

/// A name held until this is dropped.
struct Held<'a> {
    names: &'a Names,
    name: &'a str,
}

impl Names {
    const fn new() -> Names {
        Names {
            held: Mutex::new(None),
            released: Condvar::new(),
        }
    }

    /// Holds `name`, once no other thread does.
    fn hold<'a>(&'a self, name: &'a str) -> Held<'a> {
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        while held.as_ref().is_some_and(|h| h.contains(name)) {
            held = self.released.wait(held).unwrap_or_else(|e| e.into_inner());
        }
        held.get_or_insert_default().insert(name.to_owned());
        Held { names: self, name }
    }
}

impl Drop for Held<'_> {
    fn drop(&mut self) {
        let mut held = self.names.held.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = held.as_mut() {
            h.remove(self.name);
        }
        drop(held);
        self.names.released.notify_all();
    }
}

/// Pipelines being loaded, or made and stored.
static NAMES: Names = Names::new();

const LIBRARY_FILE: &str = "host-pipelines.bin";
const EARLY_FILE: &str = "early-pipelines.txt";
const PREBUILT_FILE: &str = "prebuilt-pipelines.txt";
/// Title pipelines listed for an early start, at most.
const EARLY_MAX: usize = 8192;

fn library() -> Option<&'static Shared> {
    static LIBRARY: OnceLock<Option<Shared>> = OnceLock::new();
    LIBRARY
        .get_or_init(|| {
            let device = crate::host::host()?.device.cast::<ID3D12Device1>().ok()?;
            let read = |file| DIR.get().and_then(|d| std::fs::read(d.join(file)).ok());
            let blob = read(LIBRARY_FILE).unwrap_or_default();
            let mut library = open(&device, blob);
            // Fingerprint the accepted bytes only when there is a list to verify.
            if let Some(l) = library.as_mut()
                && !l._blob.is_empty()
                && let Some(list) = read(PREBUILT_FILE)
            {
                l.prebuilt = listed(&list, &tag(&l._blob));
            }
            if library.is_none() {
                tracing::warn!("no host pipeline library: pipelines are not kept");
            }
            library.map(|l| Shared { loads: l.library.clone(), state: Mutex::new(l) })
        })
        .as_ref()
}

/// The library in `blob`, or an empty one if the host rejects the blob (an
/// adapter or driver change, damage).
fn open(device: &ID3D12Device1, blob: Vec<u8>) -> Option<Library> {
    let make = |bytes: &[u8]| -> windows::core::Result<ID3D12PipelineLibrary1> {
        // SAFETY: the bytes outlive the library (kept beside it).
        unsafe {
            device.CreatePipelineLibrary::<ID3D12PipelineLibrary>(bytes)?.cast()
        }
    };
    if !blob.is_empty() {
        let started = std::time::Instant::now();
        match make(&blob) {
            Ok(library) => {
                let ms = started.elapsed().as_millis() as u64;
                tracing::info!(bytes = blob.len(), ms, "host pipeline library loaded");
                return Some(Library {
                    library,
                    _blob: blob,
                    added: 0,
                    prebuilt: HashSet::new(),
                    prebuilt_changed: false,
                });
            }
            Err(e) => tracing::info!(error = %e, "host pipeline library rejected; starting an empty one"),
        }
    } else {
        tracing::info!("no host pipeline library yet; starting an empty one");
    }
    let library = make(&[]).ok()?;
    Some(Library {
        library,
        _blob: Vec::new(),
        added: 0,
        prebuilt: HashSet::new(),
        prebuilt_changed: false,
    })
}

/// What the pre-build's list holds for: the library's bytes, and what
/// names its pipelines (the translation, as the shader cache keys it; the
/// host descriptions; the static samplers' list).
fn tag(blob: &[u8]) -> String {
    let limit = victory_shader::emit::loop_limit().to_le_bytes();
    let translator = victory_shader::SOURCE_HASH.as_bytes();
    let samplers = super::sampler_order::listed();
    let mut h = blake3::Hasher::new();
    for part in [
        blob,
        translator,
        include_bytes!("../shader_cache.rs"),
        &limit,
        include_bytes!("host.rs"),
        include_bytes!("library.rs"),
        include_bytes!("sampler_order.rs"),
        samplers.as_bytes(),
    ] {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex().to_string()
}

/// The pre-build's list as saved: the tag it holds for, then its keys.
fn list(tag: &str, keys: &HashSet<String>) -> String {
    let keys = keys.iter().map(String::as_str);
    std::iter::once(tag).chain(keys).collect::<Vec<_>>().join("\n")
}

/// The keys of a saved list, if it holds for `tag` (else none).
fn listed(list: &[u8], tag: &str) -> HashSet<String> {
    let list = String::from_utf8_lossy(list);
    let mut lines = list.lines();
    match lines.next() == Some(tag) {
        true => lines.map(str::to_owned).collect(),
        false => HashSet::new(),
    }
}

/// Whether the pre-build has the title pipeline `key` in the library.
pub(crate) fn prebuilt(key: &str) -> bool {
    library().is_some_and(|l| l.state.lock().unwrap_or_else(|e| e.into_inner()).prebuilt.contains(key))
}

/// The pre-build has the title pipeline `key` in the library now.
pub(crate) fn note_prebuilt(key: String) {
    if let Some(l) = library() {
        let mut l = l.state.lock().unwrap_or_else(|e| e.into_inner());
        l.prebuilt_changed |= l.prebuilt.insert(key);
    }
}

/// A pipeline's name in the library: the hash of what its host
/// description holds.
pub(crate) fn name(kind: &str, root: &[u8], code: &[&[u8]], state: &str) -> String {
    let mut h = blake3::Hasher::new();
    for part in [kind.as_bytes(), root].into_iter().chain(code.iter().copied()).chain([state.as_bytes()]) {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex()[..32].to_owned()
}

const NVIDIA: u32 = 0x10DE;

/// Whether a pipeline with static samplers whose places in the driver's
/// table are not in its name (`sampler_order`: the order is off, or not
/// vouched for) can be kept in the library: not on NVIDIA, nor where the
/// adapter cannot be told.
///
/// NVIDIA's driver puts each distinct static sampler in a table of its
/// own, in the order a process first creates them, and a compiled
/// pipeline names its samplers by their places in that table. The
/// library keeps those places, so a pipeline loaded in a run that made
/// its static samplers in another order (the workers translate in no
/// fixed order) samples through other samplers, and the load still
/// succeeds: a linear sampler read as a point one, every result a
/// texel (RTX 3070, driver 616.56; the colour-grading tables of the
/// title's composite pass, 4070 SUPER). Made again in the run, such a
/// pipeline names this run's places: the cost is a compile each run,
/// which the driver's own shader cache on disk may serve.
fn keeps_static_samplers() -> bool {
    static KEEPS: OnceLock<bool> = OnceLock::new();
    *KEEPS.get_or_init(|| {
        let vendor = crate::host::host().and_then(|h| adapter_vendor(&h.device));
        let keeps = vendor.is_some_and(|v| v != NVIDIA);
        if !keeps {
            tracing::info!(
                vendor = ?vendor.map(|v| format!("{v:#x}")),
                "host pipelines with static samplers are made every run, not kept"
            );
        }
        keeps
    })
}

/// The PCI vendor of the adapter `device` runs on.
fn adapter_vendor(device: &ID3D12Device) -> Option<u32> {
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
    };
    // SAFETY: plain device and DXGI queries.
    unsafe {
        let luid = device.GetAdapterLuid();
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
        let adapter: IDXGIAdapter1 = factory.EnumAdapterByLuid(luid).ok()?;
        Some(adapter.GetDesc1().ok()?.VendorId)
    }
}

/// The pipeline named `name`: loaded from the library by `load` if it is
/// there with the same description, else made by `create` and stored. A
/// name asked for by several threads is loaded or made by one at a time
/// (the others then find it in the library); different names overlap.
/// `unguarded`: its root signature has static samplers whose places are not
/// in `name`, and such a pipeline is only made, neither loaded nor stored,
/// where the library cannot keep them ([`keeps_static_samplers`]).
pub(crate) fn pipeline(
    name: &str,
    unguarded: bool,
    load: impl FnOnce(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    let wide = HSTRING::from(name);
    #[cfg(feature = "diagnostics")]
    LAST.set(Some((name.to_owned(), true)));
    let library = library().filter(|_| !unguarded || keeps_static_samplers());
    let Some(library) = library else {
        #[cfg(feature = "diagnostics")]
        LAST.set(Some((name.to_owned(), false)));
        MADE.fetch_add(1, Ordering::Relaxed);
        return create();
    };
    let _held = NAMES.hold(name);
    // Outside the lock: loads take up to seconds, and a save holds the lock
    // for as long as it serialises. The library's blob lives as long as it
    // does, which is never dropped.
    if let Ok(pso) = load(&library.loads, PCWSTR(wide.as_ptr())) {
        LOADED.fetch_add(1, Ordering::Relaxed);
        return Ok(pso);
    }
    // Compiled outside the lock: compiles take up to seconds.
    #[cfg(feature = "diagnostics")]
    LAST.set(Some((name.to_owned(), false)));
    let pso = create()?;
    MADE.fetch_add(1, Ordering::Relaxed);
    let mut l = library.state.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: a live pipeline; a name already stored (a description that
    // differs) is refused, which leaves the library as it was.
    if unsafe { l.library.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }.is_ok() {
        l.added += 1;
    }
    Ok(pso)
}

#[cfg(feature = "diagnostics")]
thread_local! {
    /// This thread's last pipeline: its name, and whether it was loaded
    /// (else made), for the pipeline log (diagnostics).
    pub(crate) static LAST: std::cell::Cell<Option<(String, bool)>> =
        const { std::cell::Cell::new(None) };
}

/// Writes the library and the early list if pipelines were added since the
/// last save and `SAVE_EVERY` has passed (or `now`).
pub(crate) fn save(now: bool) {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    const SAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(20);
    let (Some(dir), Some(library)) = (DIR.get(), library()) else { return };
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if !now && last.is_some_and(|t| t.elapsed() < SAVE_EVERY) {
            return;
        }
        *last = Some(std::time::Instant::now());
    }
    let (blob, prebuilt) = {
        let mut l = library.state.lock().unwrap_or_else(|e| e.into_inner());
        if l.added == 0 && !l.prebuilt_changed && !USED_CHANGED.swap(false, Ordering::Relaxed) {
            return;
        }
        // SAFETY: a plain size query on a live library.
        let size = unsafe { l.library.GetSerializedSize() };
        let Some(mut bytes) = zeroed(size) else {
            // Left for a later save: the pipelines stay counted as added.
            tracing::warn!(bytes = size, "no memory to serialise the host pipeline library; saved later");
            return;
        };
        // SAFETY: a buffer of the size the library asks for.
        let bytes = unsafe { l.library.Serialize(&mut bytes) }.map(|()| bytes);
        match bytes {
            Ok(bytes) => {
                l.added = 0;
                l.prebuilt_changed = false;
                // Kept beside the bytes it holds for (also when a run adds
                // pipelines).
                let prebuilt = (!l.prebuilt.is_empty()).then(|| list(&tag(&bytes), &l.prebuilt));
                (bytes, prebuilt)
            }
            Err(e) => {
                tracing::warn!(error = %e, "host pipeline library not serialised");
                return;
            }
        }
    };
    let used = used().as_ref().map(Used::list).unwrap_or_default();
    let written = std::fs::create_dir_all(dir)
        .and_then(|()| write_atomically(&dir.join(LIBRARY_FILE), &blob))
        .and_then(|()| write_atomically(&dir.join(EARLY_FILE), used.join("\n").as_bytes()))
        .and_then(|()| match &prebuilt {
            Some(list) => write_atomically(&dir.join(PREBUILT_FILE), list.as_bytes()),
            None => Ok(()),
        });
    let (created, distinct) = created();
    let (shaders_cached, shaders_translated) = crate::shader_cache::counts();
    match written {
        Ok(()) => tracing::info!(
            bytes = blob.len(),
            early = used.len(),
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            created,
            distinct,
            shaders_cached,
            shaders_translated,
            "host pipeline library saved"
        ),
        Err(e) => tracing::warn!(error = %e, "host pipeline library not saved"),
    }
}

/// `n` zero bytes, or `None` when the host has no memory for them. The
/// library is about 1.7 GB on NVIDIA (each pipeline's machine code), and an
/// infallible allocation of that size aborted the whole process mid-run
/// ("memory allocation of 2650101862 bytes failed", RTX 4050 Laptop).
fn zeroed(n: usize) -> Option<Vec<u8>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).ok()?;
    v.resize(n, 0);
    Some(v)
}

/// Writes `bytes` to a temporary file beside `path`, then renames it over
/// `path`: a reader never sees a part. Serialised so concurrent worker and
/// shutdown saves do not race on the same staging file or target rename.
pub(super) fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::sync::atomic::AtomicU64;
    static WRITING: Mutex<()> = Mutex::new(());
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let _guard = WRITING.lock().unwrap_or_else(|e| e.into_inner());
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".tmp-{}-{n}", std::process::id()));
    let tmp = path.with_file_name(name);
    let res = std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, path));
    if res.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    res
}

/// Title pipelines created, and their distinct keys.
static CREATED: Mutex<(u64, Option<HashSet<String>>)> = Mutex::new((0, None));

/// The title created a pipeline with `key`.
pub(crate) fn note_created(key: &str) {
    let mut created = CREATED.lock().unwrap_or_else(|e| e.into_inner());
    created.0 += 1;
    created.1.get_or_insert_with(HashSet::new).insert(key.to_owned());
}

/// Title pipelines created so far, and how many distinct ones.
fn created() -> (u64, usize) {
    let created = CREATED.lock().unwrap_or_else(|e| e.into_inner());
    (created.0, created.1.as_ref().map_or(0, HashSet::len))
}

/// Title pipeline keys drawn or dispatched with: in earlier runs (the
/// list as loaded), and in this one.
#[derive(Default)]
struct Used {
    /// Each earlier key with its place in the list as loaded: the list is
    /// kept most recent run first, so a lower place is a more recent run.
    before: HashMap<String, usize>,
    now: HashSet<String>,
}

impl Used {
    /// The list as saved: one key a line, most recent run first (a list
    /// in no order, as earlier builds saved it, reads as well).
    fn read(listed: &str) -> Used {
        let mut before = HashMap::new();
        for (place, key) in listed.lines().filter(|l| !l.is_empty()).enumerate() {
            before.entry(key.to_owned()).or_insert(place);
        }
        Used { before, now: HashSet::new() }
    }

    /// The list to keep: this run's keys, then earlier runs' others, most
    /// recent first, at most `EARLY_MAX` (the oldest go once it is full).
    fn list(&self) -> Vec<String> {
        let mut earlier: Vec<_> = self.before.iter().filter(|(k, _)| !self.now.contains(*k)).collect();
        earlier.sort_unstable_by_key(|&(_, &place)| place);
        let earlier = earlier.into_iter().map(|(k, _)| k);
        self.now.iter().chain(earlier).take(EARLY_MAX).cloned().collect()
    }
}

static USED: Mutex<Option<Used>> = Mutex::new(None);
static USED_CHANGED: AtomicBool = AtomicBool::new(false);

fn used() -> std::sync::MutexGuard<'static, Option<Used>> {
    let mut used = USED.lock().unwrap_or_else(|e| e.into_inner());
    if used.is_none() {
        let listed = DIR
            .get()
            .and_then(|d| std::fs::read_to_string(d.join(EARLY_FILE)).ok())
            .unwrap_or_default();
        let read = Used::read(&listed);
        tracing::info!(pipelines = read.before.len(), "title pipelines to start early");
        *used = Some(read);
    }
    used
}

/// Reads the early list now if not yet (a pre-build asks for no title
/// pipeline).
pub(crate) fn keep_early_list() {
    drop(used());
}

/// Whether the title pipeline with `key` was drawn with in an earlier run.
pub(crate) fn wanted_early(key: &str) -> bool {
    used().as_ref().is_some_and(|u| u.before.contains_key(key))
}

/// The title pipeline with `key` was drawn or dispatched with.
pub(crate) fn note_used(key: &str) {
    if used().get_or_insert_with(Used::default).now.insert(key.to_owned()) {
        USED_CHANGED.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A size the host cannot give is refused, not fatal; others are zeros.
    #[test]
    fn a_save_buffer_the_host_cannot_give_is_refused() {
        assert!(zeroed(usize::MAX / 2).is_none());
        assert_eq!(zeroed(100).as_deref(), Some(&[0u8; 100][..]));
    }

    #[test]
    fn names_change_with_what_goes_into_the_pipeline() {
        let a = name("graphics", b"root", &[b"vs", b"ps"], "state");
        assert_eq!(a, name("graphics", b"root", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps2"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps"], "state2"));
        assert_ne!(a, name("graphics", b"root2", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("mesh", b"root", &[b"vs", b"ps"], "state"));
        // Parts do not run into each other.
        assert_ne!(a, name("graphics", b"root", &[b"vsp", b"s"], "state"));
    }

    /// This run's keys stay in the list; earlier runs' fill it up to the cap.
    #[test]
    fn the_early_list_keeps_this_run_first() {
        let keys = |r: std::ops::Range<usize>| r.map(|i| format!("{i:032x}")).collect::<HashSet<_>>();
        let before = keys(0..EARLY_MAX).into_iter().enumerate().map(|(i, k)| (k, i)).collect();
        let used = Used { before, now: keys(EARLY_MAX - 10..EARLY_MAX + 10) };
        let list = used.list();
        assert_eq!(list.len(), EARLY_MAX);
        assert!(used.now.iter().all(|k| list.contains(k)));
        assert_eq!(list.iter().collect::<HashSet<_>>().len(), EARLY_MAX, "no key twice");
    }

    /// A full list makes room by the oldest runs' keys, and keeps that
    /// order across saves.
    #[test]
    fn a_full_early_list_drops_the_oldest_runs_keys() {
        let key = |i: usize| format!("{i:032x}");
        // As saved: the last run's keys, then the run before's.
        let saved: Vec<String> = (0..EARLY_MAX).map(key).collect();
        let mut used = Used::read(&saved.join("\n"));
        used.now = (EARLY_MAX..EARLY_MAX + 10).map(key).collect();
        used.now.insert(key(EARLY_MAX - 1)); // the oldest key, used again
        let list = used.list();
        assert_eq!(list.len(), EARLY_MAX);
        assert!(used.now.iter().all(|k| list.contains(k)));
        // The ten that go are the oldest ones not used again.
        let dropped: HashSet<String> = (EARLY_MAX - 11..EARLY_MAX - 1).map(key).collect();
        assert!(list.iter().all(|k| !dropped.contains(k)));
        // Read back, this run's keys come first, then the earlier ones in order.
        let again = Used::read(&list.join("\n"));
        assert!(used.now.iter().all(|k| again.before[k] < used.now.len()));
        assert!(again.before[&key(0)] < again.before[&key(1)]);
    }

    /// The pre-build's list holds only for the library bytes it was
    /// saved with.
    #[test]
    fn the_prebuilt_list_holds_for_its_library_only() {
        let keys: HashSet<String> = ["a".repeat(32), "b".repeat(32)].into();
        let saved = list(&tag(b"library"), &keys);
        assert_eq!(listed(saved.as_bytes(), &tag(b"library")), keys);
        assert!(listed(saved.as_bytes(), &tag(b"library2")).is_empty());
        assert!(listed(b"", &tag(b"library")).is_empty());
    }

    #[test]
    fn different_names_are_held_at_the_same_time() {
        static NAMES: Names = Names::new();
        // Each thread holds its name until every thread holds its own.
        let all = std::sync::Barrier::new(8);
        std::thread::scope(|s| {
            for i in 0..8 {
                let all = &all;
                s.spawn(move || {
                    let name = format!("pipeline {i}");
                    let _held = NAMES.hold(&name);
                    all.wait();
                });
            }
        });
        assert!(NAMES.held.lock().unwrap().as_ref().unwrap().is_empty());
    }

    #[test]
    fn one_name_is_held_by_one_thread_at_a_time() {
        static NAMES: Names = Names::new();
        static INSIDE: AtomicU64 = AtomicU64::new(0);
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    for _ in 0..500 {
                        let _held = NAMES.hold("pipeline");
                        assert_eq!(INSIDE.fetch_add(1, Ordering::SeqCst), 0);
                        std::hint::spin_loop();
                        INSIDE.fetch_sub(1, Ordering::SeqCst);
                    }
                });
            }
        });
        // A holder that panics lets go of it.
        let panicked = std::thread::spawn(|| {
            let _held = NAMES.hold("pipeline");
            panic!("on purpose");
        });
        assert!(panicked.join().is_err());
        drop(NAMES.hold("pipeline"));
    }

    /// A library's pipelines load on many threads at once, each with its
    /// own description (a load with another's is refused), while others are
    /// stored and the library is serialised.
    #[test]
    fn pipelines_load_on_many_threads_at_once() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::core::s;
        const N: usize = 48;
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok())
        else {
            return; // no host GPU here
        };
        let shader = |i: usize| {
            let source = format!(
                "#define RS \"UAV(u0)\"\nRWByteAddressBuffer b : register(u0);\n\
                 [RootSignature(RS)] [numthreads(1, 1, 1)] void main() {{ b.Store(0, b.Load(4) * {i} + {i}); }}"
            );
            let mut blob = None;
            // SAFETY: the source outlives the call; the blob is read within it.
            unsafe {
                D3DCompile(
                    source.as_ptr().cast(),
                    source.len(),
                    None,
                    None,
                    None,
                    s!("main"),
                    s!("cs_5_1"),
                    0,
                    0,
                    &mut blob,
                    None,
                )
                .ok()?;
                let blob = blob?;
                Some(
                    std::slice::from_raw_parts(
                        blob.GetBufferPointer().cast::<u8>(),
                        blob.GetBufferSize(),
                    )
                    .to_vec(),
                )
            }
        };
        let Some(codes) = (0..2 * N).map(shader).collect::<Option<Vec<_>>>() else {
            return; // no shader compiler here
        };
        // SAFETY: the code holds its root signature.
        let root: ID3D12RootSignature =
            unsafe { device.CreateRootSignature(0, &codes[0]) }.unwrap();
        struct Shared(ID3D12Device1, ID3D12RootSignature, ID3D12PipelineLibrary1);
        // SAFETY: D3D12 objects are free-threaded.
        unsafe impl Sync for Shared {}
        impl Shared {
            fn desc(&self, code: &[u8]) -> D3D12_COMPUTE_PIPELINE_STATE_DESC {
                D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    // SAFETY: a borrowed pointer the description never releases.
                    pRootSignature: unsafe { std::mem::transmute_copy(&self.1) },
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                }
            }
            fn store(&self, i: usize, code: &[u8]) {
                let name = HSTRING::from(format!("p{i}"));
                // SAFETY: the description points at live data.
                unsafe {
                    let pso: ID3D12PipelineState =
                        self.0.CreateComputePipelineState(&self.desc(code)).unwrap();
                    self.2.StorePipeline(&name, &pso).unwrap();
                }
            }
            fn load(&self, i: usize, code: &[u8]) -> windows::core::Result<ID3D12PipelineState> {
                let name = HSTRING::from(format!("p{i}"));
                // SAFETY: as above.
                unsafe { self.2.LoadComputePipeline(&name, &self.desc(code)) }
            }
            fn serialized(&self) -> Vec<u8> {
                // SAFETY: a buffer of the size asked for; nothing is stored
                // meanwhile.
                unsafe {
                    let mut blob = vec![0; self.2.GetSerializedSize()];
                    self.2.Serialize(&mut blob).unwrap();
                    blob
                }
            }
        }
        let first = open(&device, Vec::new()).expect("an empty library");
        let made = Shared(device.clone(), root.clone(), first.library.clone());
        (0..N).for_each(|i| made.store(i, &codes[i]));
        let again = open(&device, made.serialized()).expect("the library again");
        let shared = Shared(device, root, again.library.clone());
        let stores = Mutex::new(());
        std::thread::scope(|s| {
            for t in 0..8 {
                let (shared, codes, stores) = (&shared, &codes, &stores);
                s.spawn(move || {
                    for i in (t..N).step_by(8) {
                        shared.load(i, &codes[i]).expect("a stored pipeline");
                        assert!(
                            shared.load(i, &codes[(i + 1) % N]).is_err(),
                            "another description"
                        );
                        assert!(shared.load(N + i, &codes[N + i]).is_err(), "not stored yet");
                        let _one = stores.lock().unwrap();
                        shared.store(N + i, &codes[N + i]);
                        shared.serialized();
                    }
                });
            }
        });
        for (i, code) in codes.iter().enumerate() {
            shared.load(i, code).expect("every pipeline");
        }
    }

    /// A stored pipeline loads while a save holds the library's lock (one
    /// serialises the whole library, for seconds in the world).
    #[test]
    fn a_pipeline_loads_while_a_save_holds_the_library() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::core::s;
        let (Some(device), Some(library)) = (
            crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()),
            library(),
        ) else {
            return; // no host GPU here
        };
        // No resources, and the root signature made apart: a compiler need
        // not keep one the source names (Wine's does not).
        let source = "[numthreads(1, 1, 1)] void main() {}";
        let mut blob = None;
        // SAFETY: the source outlives the call; the blob is read within it.
        let code = unsafe {
            if D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                None,
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut blob,
                None,
            )
            .is_err()
            {
                return; // no shader compiler here
            }
            let blob = blob.unwrap();
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
                .to_vec()
        };
        // SAFETY: an empty description; the blob is read while it lives.
        let root: ID3D12RootSignature = unsafe {
            let mut blob = None;
            D3D12SerializeRootSignature(
                &D3D12_ROOT_SIGNATURE_DESC::default(),
                D3D_ROOT_SIGNATURE_VERSION_1,
                &mut blob,
                None,
            )
            .unwrap();
            let blob = blob.unwrap();
            let bytes = std::slice::from_raw_parts(
                blob.GetBufferPointer().cast::<u8>(),
                blob.GetBufferSize(),
            );
            device.CreateRootSignature(0, bytes).unwrap()
        };
        struct Made(ID3D12Device1, ID3D12RootSignature, Vec<u8>);
        // SAFETY: D3D12 objects are free-threaded.
        unsafe impl Sync for Made {}
        impl Made {
            fn get(&self, made: bool) -> windows::core::Result<ID3D12PipelineState> {
                let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    // SAFETY: a borrowed pointer the description never releases.
                    pRootSignature: unsafe { std::mem::transmute_copy(&self.1) },
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: self.2.as_ptr().cast(),
                        BytecodeLength: self.2.len(),
                    },
                    ..Default::default()
                };
                pipeline(
                    "test: a pipeline loads while a save holds the library",
                    false,
                    // SAFETY: the description points at live data.
                    |library, name| unsafe { library.LoadComputePipeline(name, &desc) },
                    || match made {
                        // SAFETY: as above.
                        true => unsafe { self.0.CreateComputePipelineState(&desc) },
                        false => panic!("made again, not loaded"),
                    },
                )
            }
        }
        let made = Made(device, root, code);
        made.get(true).expect("made and stored");
        let saving = library.state.lock().unwrap_or_else(|e| e.into_inner());
        let (done, loaded) = std::sync::mpsc::channel();
        std::thread::scope(|s| {
            let made = &made;
            s.spawn(move || done.send(made.get(false).is_ok()));
            let loaded = loaded.recv_timeout(std::time::Duration::from_secs(30));
            drop(saving);
            assert_eq!(loaded, Ok(true), "the load waited for the save");
        });
    }

    /// A pipeline with static samplers kept in the library by one run
    /// samples through its own samplers in a later run that made other
    /// static samplers first: the later run created the listed ones first,
    /// in the list's order, so it loads the pipeline. (Without that, the
    /// RTX 3070 loaded it reading its linear samplers as point ones, 0 for
    /// 0.25.) The runs are child processes: the driver's sampler table is a
    /// process's own.
    #[test]
    fn a_kept_pipeline_samples_through_its_own_static_samplers() {
        static_samplers::runs(
            "a_kept_pipeline_samples_through_its_own_static_samplers",
            &["store", "load"],
        );
    }

    #[test]
    fn prebuilt_keys_survive_static_sampler_replay() {
        static_samplers::runs(
            "prebuilt_keys_survive_static_sampler_replay",
            &["store prebuilt", "load prebuilt", "load changed prebuilt"],
        );
    }

    /// A sampler first met in a later run is appended to the list, and a
    /// pipeline with it, stored then, loads in the run after that, which
    /// met yet other samplers first.
    #[test]
    fn a_static_sampler_met_in_a_later_run_is_appended() {
        static_samplers::runs(
            "a_static_sampler_met_in_a_later_run_is_appended",
            &["store", "append", "load appended"],
        );
    }

    /// With the list lost, the pipeline's name misses (its samplers have
    /// other places now): it is made again and samples right.
    #[test]
    fn a_lost_static_sampler_list_gives_a_miss() {
        static_samplers::runs("a_lost_static_sampler_list_gives_a_miss", &["store", "load lost"]);
    }

    /// The control (NVIDIA only, so not run by default): the list read but
    /// its samplers not created first, the name still found, the loaded
    /// pipeline samples through the other samplers.
    #[test]
    #[ignore = "a control for NVIDIA's sampler table: it asserts wrong samples"]
    fn unprimed_static_samplers_load_wrong() {
        static_samplers::runs("unprimed_static_samplers_load_wrong", &["store", "load unprimed"]);
    }

    /// The runs of the tests above.
    mod static_samplers {
        use super::super::super::host::root_signature;
        use super::super::super::sampler_order;
        use super::*;
        use windows::Win32::Foundation::CloseHandle;
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_FLOAT, DXGI_SAMPLE_DESC};
        use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
        use windows::core::PCSTR;

        pub(super) const ROLE: &str = "VICTORY_TEST_STATIC_SAMPLERS";
        pub(super) const DIR_VAR: &str = "VICTORY_TEST_STATIC_SAMPLERS_DIR";

        /// A 2x1 texture of 0 and 1, then the probe's three samples of it
        /// at u = 0.375 through its point, linear and trilinear sampler.
        const SOURCE: &str = "
            RWTexture2D<float> lut : register(u0);
            Texture2D<float> tex : register(t0);
            RWByteAddressBuffer result : register(u1);
            SamplerState point_ : register(s0);
            SamplerState linear_ : register(s1);
            SamplerState trilinear : register(s2);
            [numthreads(2, 1, 1)] void fill(uint3 id : SV_DispatchThreadID) { lut[id.xy] = id.x; }
            [numthreads(1, 1, 1)] void probe() {
                float2 uv = float2(0.375, 0.5);
                result.Store3(0, asuint(float3(tex.SampleLevel(point_, uv, 0),
                    tex.SampleLevel(linear_, uv, 0), tex.SampleLevel(trilinear, uv, 0))));
            }";
        /// What the three samples are: the texel left of 0.375, and a
        /// quarter of the way to the right one.
        const EXPECTED: [f32; 3] = [0.0, 0.25, 0.25];
        /// Other static samplers the later run makes first.
        const OTHERS: u32 = 10;
        /// The probe samplers' greatest LOD: another one makes other
        /// samplers that sample the same (the texture has one level).
        const FIRST: f32 = 12.0;
        const PREBUILT_KEY: &str = "0123456789abcdef0123456789abcdef";
        const APPENDED: f32 = 13.0;

        /// Runs `steps` of `test` in child processes, in order, on one
        /// cache folder; in a child, its step.
        pub(super) fn runs(test: &str, steps: &[&str]) {
            if let Ok(step) = std::env::var(ROLE) {
                return run(&step);
            }
            if crate::host::host().is_none() {
                return; // no host GPU here
            }
            let dir = tempfile::tempdir().unwrap();
            let path = module_path!().split_once("::").map_or(module_path!(), |(_, p)| p);
            let test = format!("{}::{test}", path.trim_end_matches("::static_samplers"));
            for step in steps {
                let run = std::process::Command::new(std::env::current_exe().unwrap())
                    .args([test.as_str(), "--exact", "--nocapture", "--test-threads=1", "--include-ignored"])
                    .env(ROLE, step)
                    .env(DIR_VAR, dir.path())
                    .output()
                    .unwrap();
                let out = String::from_utf8_lossy(&run.stdout);
                let err = String::from_utf8_lossy(&run.stderr);
                println!("{step}: {}", out.lines().filter(|l| l.contains("sampled")).collect::<Vec<_>>().join("; "));
                assert!(run.status.success(), "the {step} run failed:\n{out}\n{err}");
                assert!(out.contains("1 passed"), "the {step} run ran no test:\n{out}\n{err}");
            }
        }

        fn run(step: &str) {
            let (step, with_prebuilt) = step
                .strip_suffix(" prebuilt")
                .map_or((step, false), |plain| (plain, true));
            let dir = PathBuf::from(std::env::var_os(DIR_VAR).expect("the library's folder"));
            let list = dir.join(sampler_order::FILE);
            if step == "load lost" {
                let lost = std::fs::remove_file(&list);
                assert!(lost.is_ok() || !sampler_order::on(), "no list to lose");
            }
            if step == "load unprimed" {
                sampler_order::UNPRIMED.store(true, Ordering::Relaxed);
            }
            // Before the device: it creates the listed samplers.
            use_pipeline_cache(dir.clone());
            let device = device();
            let ordered = sampler_order::on();
            match step {
                "store" => {
                    let (made, values) = probe(&device, FIRST, &[]);
                    check(values);
                    assert!(made, "a first run loaded the probe");
                    if with_prebuilt {
                        note_prebuilt(PREBUILT_KEY.to_owned());
                    }
                    store_and_save(&dir);
                    assert_eq!(dir.join(PREBUILT_FILE).is_file(), with_prebuilt);
                    if ordered {
                        assert_eq!(lines(&list), 3, "the probe's samplers listed");
                    }
                }
                "load" | "load changed" | "load lost" | "load unprimed" => {
                    if ordered && matches!(step, "load" | "load changed") {
                        // The prebuilt tag covers the complete sampler list.
                        assert_eq!(prebuilt(PREBUILT_KEY), with_prebuilt && step == "load");
                    }
                    let (made, values) = probe(&device, FIRST, &others(0..OTHERS));
                    match (step, ordered) {
                        ("load" | "load changed", true) => {
                            check(values);
                            assert!(!made, "the probe was made again, not loaded");
                            assert_eq!(lines(&list), 3 + OTHERS as usize, "the others appended");
                        }
                        ("load lost", true) => {
                            check(values);
                            assert!(made, "the probe loaded with its samplers in other places");
                        }
                        ("load unprimed", true) => {
                            assert!(!made, "the control did not load the probe");
                            println!("sampled (point, linear, trilinear), control: {values:?}");
                            assert_ne!(values, EXPECTED, "the control sampled right without priming");
                        }
                        // Without the order: such a pipeline is made on NVIDIA.
                        (_, false) => {
                            check(values);
                            assert!(keeps_static_samplers() || made, "a pipeline with static samplers was loaded");
                        }
                        _ => unreachable!(),
                    }
                }
                "append" => {
                    let (made, values) = probe(&device, APPENDED, &others(0..OTHERS));
                    check(values);
                    assert!(made, "a probe with new samplers was loaded");
                    store_and_save(&dir);
                    if ordered {
                        assert_eq!(lines(&list), 3 + OTHERS as usize + 3, "the new samplers appended");
                    }
                }
                "load appended" => {
                    // Yet others first, new to the list too.
                    let (made, values) = probe(&device, APPENDED, &others(OTHERS..OTHERS + 5));
                    check(values);
                    assert!(!ordered || !made, "the probe with appended samplers was made again, not loaded");
                }
                _ => panic!("no step {step}"),
            }
        }

        /// Lines in the list of static samplers.
        fn lines(list: &std::path::Path) -> usize {
            std::fs::read_to_string(list).map_or(0, |l| l.lines().count())
        }

        /// Root signatures with one other static sampler each, created
        /// through the host's path and kept while they are returned.
        fn others(range: std::ops::Range<u32>) -> Vec<ID3D12RootSignature> {
            let device = device();
            range
                .map(|i| {
                    let other = sampler(0, D3D12_FILTER_MIN_MAG_MIP_POINT, (i + 1) as f32 / 64.0, FIRST);
                    root_signature(&device, &[], &[other], true).expect("another root signature").0
                })
                .collect()
        }

        /// The probe with its samplers' greatest LOD `max_lod`, after
        /// `_others`: whether it was made (else loaded), and its samples.
        fn probe(device: &ID3D12Device, max_lod: f32, _others: &[ID3D12RootSignature]) -> (bool, [f32; 3]) {
            let (root, blob, unguarded) = root(device, max_lod);
            let code = fxc(windows::core::s!("probe"));
            let loaded = LOADED.load(Ordering::Relaxed);
            let (name, pso) = pipeline_of(device, &root, &blob, &code, unguarded);
            let made = LOADED.load(Ordering::Relaxed) == loaded;
            println!("sampled with the probe {}", if made { "made" } else { "loaded" });
            if unguarded && !keeps_static_samplers() {
                // As earlier builds did: kept regardless, for a later run's
                // check that it is not loaded.
                let library = library().expect("a pipeline library");
                let mut l = library.state.lock().unwrap();
                assert_eq!(l.added, 0, "a pipeline with static samplers was stored");
                let wide = HSTRING::from(name.as_str());
                // SAFETY: a live pipeline; one already stored is refused.
                if unsafe { l.library.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }.is_ok() {
                    l.added += 1;
                }
            }
            (made, sample(device, &root, &pso))
        }

        fn store_and_save(dir: &std::path::Path) {
            save(true);
            assert!(dir.join(LIBRARY_FILE).exists());
        }

        fn fxc(entry: PCSTR) -> Vec<u8> {
            let (mut blob, mut errors) = (None, None);
            // SAFETY: the source outlives the call; the blobs are read within it.
            unsafe {
                let compiled = D3DCompile(
                    SOURCE.as_ptr().cast(),
                    SOURCE.len(),
                    None,
                    None,
                    None,
                    entry,
                    windows::core::s!("cs_5_1"),
                    0,
                    0,
                    &mut blob,
                    Some(&mut errors),
                );
                if let (Err(e), Some(errors)) = (&compiled, &errors) {
                    let errors: &windows::Win32::Graphics::Direct3D::ID3DBlob = errors;
                    let text = std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize());
                    panic!("the probe does not compile: {e}: {}", String::from_utf8_lossy(text));
                }
                compiled.expect("the probe compiles");
                let blob: windows::Win32::Graphics::Direct3D::ID3DBlob = blob.unwrap();
                std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
                    .to_vec()
            }
        }

        fn sampler(register: u32, filter: D3D12_FILTER, bias: f32, max_lod: f32) -> D3D12_STATIC_SAMPLER_DESC {
            D3D12_STATIC_SAMPLER_DESC {
                Filter: filter,
                AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                MipLODBias: bias,
                MaxAnisotropy: 1,
                ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
                MaxLOD: max_lod,
                ShaderRegister: register,
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                ..Default::default()
            }
        }

        /// The probe's root signature as the host makes one (`host`): a
        /// table of the texture's two views, the output, three samplers.
        fn root(device: &ID3D12Device, max_lod: f32) -> (ID3D12RootSignature, Vec<u8>, bool) {
            let range = |kind, offset| D3D12_DESCRIPTOR_RANGE1 {
                RangeType: kind,
                NumDescriptors: 1,
                OffsetInDescriptorsFromTableStart: offset,
                ..Default::default()
            };
            let ranges = [
                range(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0),
                range(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1),
            ];
            let params = [
                D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 {
                            NumDescriptorRanges: ranges.len() as u32,
                            pDescriptorRanges: ranges.as_ptr(),
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
                D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR1 { ShaderRegister: 1, ..Default::default() },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
            ];
            let samplers = [
                sampler(0, D3D12_FILTER_MIN_MAG_MIP_POINT, 0.0, max_lod),
                sampler(1, D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT, 0.0, max_lod),
                sampler(2, D3D12_FILTER_MIN_MAG_MIP_LINEAR, 0.0, max_lod),
            ];
            let (root, mut blob, order) =
                root_signature(device, &params, &samplers, true).expect("the probe's root signature");
            // Named as the host names its pipelines (`host`).
            let unguarded = order.is_none();
            blob.extend_from_slice(order.unwrap_or_default().as_bytes());
            (root, blob, unguarded)
        }

        /// The probe pipeline through the library, as the host gets one.
        fn pipeline_of(device: &ID3D12Device, root: &ID3D12RootSignature, blob: &[u8], code: &[u8], unguarded: bool) -> (String, ID3D12PipelineState) {
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                // SAFETY: a borrowed pointer the description never releases.
                pRootSignature: unsafe { std::mem::transmute_copy(root) },
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr().cast(), BytecodeLength: code.len() },
                ..Default::default()
            };
            let name = name("compute", blob, &[code], "test: static samplers");
            let pso = pipeline(
                &name,
                unguarded,
                // SAFETY: the description points at live data.
                |library, name| unsafe { library.LoadComputePipeline(name, &desc) },
                // SAFETY: as above.
                || unsafe { device.CreateComputePipelineState(&desc) },
            )
            .expect("the probe pipeline");
            (name, pso)
        }

        /// The three samples `pso` takes.
        fn sample(device: &ID3D12Device, root: &ID3D12RootSignature, pso: &ID3D12PipelineState) -> [f32; 3] {
            let fill_code = fxc(windows::core::s!("fill"));
            // SAFETY: plain D3D12 object creation and one list's commands on
            // live objects, read back once the fence says the GPU is done.
            unsafe {
                let fill: ID3D12PipelineState = device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: fill_code.as_ptr().cast(),
                            BytecodeLength: fill_code.len(),
                        },
                        ..Default::default()
                    })
                    .unwrap();
                let resource = |desc: &D3D12_RESOURCE_DESC, kind, state| {
                    let heap = D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() };
                    let mut made: Option<ID3D12Resource> = None;
                    device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, desc, state, None, &mut made).unwrap();
                    made.unwrap()
                };
                let samples = DXGI_SAMPLE_DESC { Count: 1, Quality: 0 };
                let texture = resource(
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                        Width: 2,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        Format: DXGI_FORMAT_R32_FLOAT,
                        SampleDesc: samples,
                        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                        ..Default::default()
                    },
                    D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
                let buffer = |flags| D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: 256,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: samples,
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: flags,
                    ..Default::default()
                };
                let out = resource(
                    &buffer(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                    D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_RESOURCE_STATE_COMMON,
                );
                let readback = resource(
                    &buffer(D3D12_RESOURCE_FLAG_NONE),
                    D3D12_HEAP_TYPE_READBACK,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                );
                let heap: ID3D12DescriptorHeap = device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: 2,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        ..Default::default()
                    })
                    .unwrap();
                let step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
                let cpu = heap.GetCPUDescriptorHandleForHeapStart();
                device.CreateUnorderedAccessView(
                    &texture,
                    None,
                    Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                        Format: DXGI_FORMAT_R32_FLOAT,
                        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                        ..Default::default()
                    }),
                    cpu,
                );
                device.CreateShaderResourceView(
                    &texture,
                    Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: DXGI_FORMAT_R32_FLOAT,
                        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
                        },
                    }),
                    D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu.ptr + step },
                );
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
                        ..Default::default()
                    })
                    .unwrap();
                let allocator: ID3D12CommandAllocator =
                    device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE).unwrap();
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &allocator, &fill)
                    .unwrap();
                list.SetDescriptorHeaps(&[Some(heap.clone())]);
                list.SetComputeRootSignature(root);
                list.SetComputeRootDescriptorTable(0, heap.GetGPUDescriptorHandleForHeapStart());
                list.SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
                list.Dispatch(1, 1, 1);
                list.ResourceBarrier(&[crate::host::transition(
                    &texture,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                )]);
                list.SetPipelineState(pso);
                list.Dispatch(1, 1, 1);
                list.ResourceBarrier(&[crate::host::transition(
                    &out,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                list.CopyBufferRegion(&readback, 0, &out, 0, 16);
                list.Close().unwrap();
                queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
                queue.Signal(&fence, 1).unwrap();
                let event = CreateEventW(None, false, false, None).unwrap();
                fence.SetEventOnCompletion(1, event).unwrap();
                WaitForSingleObject(event, INFINITE);
                CloseHandle(event).unwrap();
                device.GetDeviceRemovedReason().expect("the device is not removed");
                let mut mapped = std::ptr::null_mut();
                readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 12 }), Some(&mut mapped)).unwrap();
                let values = mapped.cast::<[f32; 3]>().read();
                readback.Unmap(0, Some(&D3D12_RANGE::default()));
                values
            }
        }

        fn check(values: [f32; 3]) {
            println!("sampled (point, linear, trilinear): {values:?}, expected {EXPECTED:?}");
            for (got, want) in values.into_iter().zip(EXPECTED) {
                assert!((got - want).abs() < 1e-3, "sampled {values:?}, not {EXPECTED:?}");
            }
        }

        fn device() -> ID3D12Device {
            crate::host::host().expect("a host GPU, as the parent had").device.clone()
        }
    }

    #[test]
    fn a_rejected_blob_gives_an_empty_library() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()) else {
            return; // no host GPU here
        };
        let library = open(&device, b"not a pipeline library".repeat(16)).expect("an empty library");
        // SAFETY: a plain query on a live library.
        assert!(unsafe { library.library.GetSerializedSize() } > 0);
        assert!(library._blob.is_empty());
    }

    #[test]
    fn concurrent_write_atomically_never_fails_or_truncates() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("host-pipelines.bin");
        let threads: Vec<_> = (0..4)
            .map(|t| {
                let target = target.clone();
                std::thread::spawn(move || {
                    let payload = vec![b'a' + t as u8; 4096];
                    for _ in 0..30 {
                        write_atomically(&target, &payload).unwrap();
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        let final_bytes = std::fs::read(&target).unwrap();
        assert_eq!(final_bytes.len(), 4096);
        assert!(final_bytes.iter().all(|&b| b == final_bytes[0]));
    }
}
