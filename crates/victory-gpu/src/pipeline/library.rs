//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early (decision 0007, "Pipeline cache").
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The library is saved under the cache
//!    folder in [`SHARDS`] files, chosen by that name; the host refuses a
//!    library of another adapter or driver, and an entry whose description
//!    differs, so neither is ever used. A save serialises and writes only
//!    the shards that gained pipelines, because the host can only serialise
//!    a library whole and one blob of the world's pipelines is over 600 MB.
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
//! All library calls are serialised: the interface does not promise that
//! stores, loads and serialising may overlap.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Keeps the pipeline library and the early list in `dir` from now on.
pub fn use_pipeline_cache(dir: PathBuf) {
    let _ = DIR.set(dir);
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

// SAFETY: D3D12 objects are free-threaded; every call is made under the
// mutex that holds the library.
unsafe impl Send for Library {}

/// How many files the library is split into. A save serialises and writes
/// only the shards that gained pipelines; see the module comment.
const SHARDS: usize = 16;
const EARLY_FILE: &str = "early-pipelines.txt";
/// Title pipelines listed for an early start, at most.
const EARLY_MAX: usize = 8192;

/// The file shard `shard` is kept in.
fn shard_file(shard: usize) -> String {
    format!("host-pipelines.{shard:02}.bin")
}

/// The pre-build's list for shard `shard`.
fn prebuilt_file(shard: usize) -> String {
    format!("prebuilt-pipelines.{shard:02}.txt")
}

/// The shard a pipeline named `name` belongs to. The name is a hash, so this
/// is spread and stable across runs: a pipeline always lands in the same
/// file, which is what lets a save write only the files that changed.
fn shard_of(name: &str) -> usize {
    let mut h = 0u32;
    for b in name.bytes() {
        h = h.wrapping_mul(31).wrapping_add(u32::from(b));
    }
    h as usize % SHARDS
}

/// One save at a time: serialising and publishing must not interleave, or an
/// older complete snapshot could be published over a newer one.
static SAVING: Mutex<()> = Mutex::new(());

fn libraries() -> Option<&'static [Mutex<Library>; SHARDS]> {
    static LIBRARIES: OnceLock<Option<[Mutex<Library>; SHARDS]>> = OnceLock::new();
    LIBRARIES
        .get_or_init(|| {
            let device = crate::host::host()?.device.cast::<ID3D12Device1>().ok()?;
            let read = |file| DIR.get().and_then(|d| std::fs::read(d.join(file)).ok());
            let mut shards: Vec<Mutex<Library>> = Vec::with_capacity(SHARDS);
            for shard in 0..SHARDS {
                let blob = read(shard_file(shard)).unwrap_or_default();
                let tag = tag(&blob);
                let mut library = open(&device, blob)?;
                // The pre-build's list, if it was saved with this shard's bytes.
                if let Some(list) = read(prebuilt_file(shard))
                    && !library._blob.is_empty()
                {
                    library.prebuilt = listed(&list, &tag);
                }
                shards.push(Mutex::new(library));
            }
            shards.try_into().ok()
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
/// host descriptions).
fn tag(blob: &[u8]) -> String {
    let limit = victory_shader::emit::loop_limit().to_le_bytes();
    let translator = victory_shader::SOURCE_HASH.as_bytes();
    let mut h = blake3::Hasher::new();
    for part in [
        blob,
        translator,
        include_bytes!("../shader_cache.rs"),
        &limit,
        include_bytes!("host.rs"),
        include_bytes!("library.rs"),
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
    libraries().is_some_and(|l| {
        l[shard_of(key)]
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .prebuilt
            .contains(key)
    })
}

/// The pre-build has the title pipeline `key` in the library now.
pub(crate) fn note_prebuilt(key: String) {
    if let Some(libraries) = libraries() {
        let mut shard = libraries[shard_of(&key)].lock().unwrap_or_else(|e| e.into_inner());
        shard.prebuilt_changed |= shard.prebuilt.insert(key);
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

/// The pipeline named `name`: loaded from the library by `load` if it is
/// there with the same description, else made by `create` and stored.
pub(crate) fn pipeline(
    name: &str,
    load: impl FnOnce(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    let wide = HSTRING::from(name);
    #[cfg(feature = "diagnostics")]
    LAST.set(Some((name.to_owned(), true)));
    let Some(libraries) = libraries() else {
        MADE.fetch_add(1, Ordering::Relaxed);
        return create();
    };
    let shard = &libraries[shard_of(name)];
    if let Ok(pso) = load(&shard.lock().unwrap_or_else(|e| e.into_inner()).library, PCWSTR(wide.as_ptr())) {
        LOADED.fetch_add(1, Ordering::Relaxed);
        return Ok(pso);
    }
    // Compiled outside the lock: compiles take up to seconds.
    #[cfg(feature = "diagnostics")]
    LAST.set(Some((name.to_owned(), false)));
    let pso = create()?;
    MADE.fetch_add(1, Ordering::Relaxed);
    let mut l = shard.lock().unwrap_or_else(|e| e.into_inner());
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

/// Writes the shards that gained pipelines since the last save, and the early
/// list if it changed, when `SAVE_EVERY` has passed (or `now`).
pub(crate) fn save(now: bool) {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    const SAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(20);
    let (Some(dir), Some(libraries)) = (DIR.get(), libraries()) else { return };
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if !now && last.is_some_and(|t| t.elapsed() < SAVE_EVERY) {
            return;
        }
        *last = Some(std::time::Instant::now());
    }
    // Serialising and publishing are both inside this lock, so two saves
    // cannot interleave and publish an older snapshot over a newer one.
    let _saving = SAVING.lock().unwrap_or_else(|e| e.into_inner());
    if let Err(e) = std::fs::create_dir_all(dir) {
        tracing::warn!(error = %e, "host pipeline library folder not made");
        return;
    }
    let used_changed = USED_CHANGED.load(Ordering::Relaxed);
    let used = USED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(Used::list)
        .unwrap_or_default();
    let mut saved = 0u32;
    let mut total = 0usize;
    for (shard, library) in libraries.iter().enumerate() {
        if let Some(bytes) = save_shard(dir, shard, library, write_atomically) {
            saved += 1;
            total += bytes;
        }
    }
    if used_changed || saved > 0 {
        match write_atomically(&dir.join(EARLY_FILE), used.join("\n").as_bytes()) {
            Ok(()) => USED_CHANGED.store(false, Ordering::Relaxed),
            Err(e) => tracing::warn!(error = %e, "early pipeline list not saved"),
        }
    }
    if saved > 0 {
        let (created, distinct) = created();
        let (shaders_cached, shaders_translated) = crate::shader_cache::counts();
        tracing::info!(
            shards = saved,
            bytes = total,
            early = used.len(),
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            created,
            distinct,
            shaders_cached,
            shaders_translated,
            "host pipeline library saved"
        );
    }
}

fn save_shard(
    dir: &std::path::Path,
    shard: usize,
    library: &Mutex<Library>,
    write: impl Fn(&std::path::Path, &[u8]) -> std::io::Result<()>,
) -> Option<usize> {
    let (blob, prebuilt, added, prebuilt_count) = {
        let l = library.lock().unwrap_or_else(|e| e.into_inner());
        if l.added == 0 && !l.prebuilt_changed {
            return None;
        }
        // SAFETY: a buffer of the size the library asks for.
        let bytes = unsafe {
            let mut bytes = vec![0u8; l.library.GetSerializedSize()];
            l.library.Serialize(&mut bytes).map(|()| bytes)
        };
        match bytes {
            Ok(bytes) => {
                // Kept beside the bytes it holds for.
                let prebuilt = (!l.prebuilt.is_empty()).then(|| list(&tag(&bytes), &l.prebuilt));
                (bytes, prebuilt, l.added, l.prebuilt.len())
            }
            Err(e) => {
                tracing::warn!(error = %e, shard, "host pipeline library shard not serialised");
                return None;
            }
        }
    };

    let written = write(&dir.join(shard_file(shard)), &blob).and_then(|()| match &prebuilt {
        Some(list) => write(&dir.join(prebuilt_file(shard)), list.as_bytes()),
        None => Ok(()),
    });
    if let Err(e) = written {
        tracing::warn!(error = %e, shard, "host pipeline library shard not saved");
        return None;
    }

    // Acknowledge only successfully published changes. Updates that arrived
    // while the write ran must stay dirty for the next save.
    let mut l = library.lock().unwrap_or_else(|e| e.into_inner());
    l.added -= added;
    l.prebuilt_changed = l.prebuilt.len() != prebuilt_count;
    Some(blob.len())
}

/// Writes `bytes` to a temporary file beside `path`, then renames it over
/// `path`: a reader never sees a part.
fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
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
    before: HashSet<String>,
    now: HashSet<String>,
}

impl Used {
    /// The list to keep: this run's keys, then earlier runs' others, at
    /// most `EARLY_MAX` (the oldest go once it is full).
    fn list(&self) -> Vec<String> {
        let earlier = self.before.iter().filter(|k| !self.now.contains(*k));
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
        let before: HashSet<String> = listed.lines().filter(|l| !l.is_empty()).map(str::to_owned).collect();
        tracing::info!(pipelines = before.len(), "title pipelines to start early");
        *used = Some(Used { before, now: HashSet::new() });
    }
    used
}

/// Reads the early list now if not yet, so a save keeps it (a pre-build
/// asks for no title pipeline).
pub(crate) fn keep_early_list() {
    drop(used());
}

/// Whether the title pipeline with `key` was drawn with in an earlier run.
pub(crate) fn wanted_early(key: &str) -> bool {
    used().as_ref().is_some_and(|u| u.before.contains(key))
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
        let used = Used { before: keys(0..EARLY_MAX), now: keys(EARLY_MAX - 10..EARLY_MAX + 10) };
        let list = used.list();
        assert_eq!(list.len(), EARLY_MAX);
        assert!(used.now.iter().all(|k| list.contains(k)));
        assert_eq!(list.iter().collect::<HashSet<_>>().len(), EARLY_MAX, "no key twice");
    }

    /// A pipeline always lands in the same shard, and a corpus of names
    /// spreads over all of them: that is what makes a save a fraction of the
    /// library rather than all of it.
    #[test]
    fn a_pipeline_lands_in_one_shard_and_the_shards_spread() {
        let first = name("graphics", b"root", &[b"vs", b"ps"], "state");
        assert_eq!(shard_of(&first), shard_of(&first), "stable across runs");
        let mut seen = [0usize; SHARDS];
        for i in 0..2048u32 {
            seen[shard_of(&format!("{i:032x}"))] += 1;
        }
        assert!(seen.iter().all(|&n| n > 0), "every shard is used: {seen:?}");
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
    fn a_prebuilt_key_recorded_during_publication_is_saved_next_time() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()) else {
            return; // no host GPU here
        };
        let dir = tempfile::tempdir().unwrap();
        let mut library = open(&device, Vec::new()).expect("an empty library");
        library.prebuilt_changed |= library.prebuilt.insert("before".to_owned());
        let library = Mutex::new(library);
        let shard = 0;
        let published_keys = || {
            let blob = std::fs::read(dir.path().join(shard_file(shard))).unwrap();
            let records = std::fs::read(dir.path().join(prebuilt_file(shard))).unwrap();
            listed(&records, &tag(&blob))
        };

        let first = save_shard(dir.path(), shard, &library, |path, bytes| {
            // The saver has already captured the keys and released the mutex.
            {
                let mut l = library.lock().unwrap();
                l.prebuilt_changed |= l.prebuilt.insert("during".to_owned());
            }
            write_atomically(path, bytes)
        });
        assert!(first.is_some());
        assert_eq!(published_keys(), HashSet::from(["before".to_owned()]));

        let second = save_shard(dir.path(), shard, &library, write_atomically);
        assert!(second.is_some(), "a key recorded during publication still needs saving");
        assert_eq!(published_keys(), HashSet::from(["before".to_owned(), "during".to_owned()]));
        assert!(save_shard(dir.path(), shard, &library, write_atomically).is_none());
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
}
