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
//! The library is free-threaded but for loads of one pipeline on several
//! threads (`ID3D12Device1::CreatePipelineLibrary`, "Thread Safety"), so
//! pipelines of different names load at the same time. A load is not always
//! quick: the library of an AMD driver holds about 140 bytes per pipeline,
//! and what its own cache lacks is compiled inside the load, for seconds.
//! Loaded one at a time, 650 pipelines kept the eight workers to one core
//! for minutes, while each title frame could wait out its 500 ms for
//! them (`make`). Stores and serialising keep the lock: the size asked for
//! must still be right when the bytes are written.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Keeps the pipeline library and the early list in `dir` from now on.
///
/// The library's pipelines are named after the shaders they were made from,
/// so a change to the translator (`victory_shader::SOURCE_HASH`) makes every
/// name in it useless. The first run after one used to write the few
/// pipelines it made over the old library: measured on the title, a 926 MB
/// library (27 160 pipelines, a 7.5 minute warm-up) became 612 KB, and
/// bisecting between two builds paid that warm-up for each of them, in
/// turn, every time. So each translation keeps its library beside the
/// others, in `dir/<SOURCE_HASH>`, and a run only ever writes its own;
/// [`KEPT`] of the newest are kept.
pub fn use_pipeline_cache(dir: PathBuf) {
    let _ = DIR.set(keyed(&dir));
}

/// Whether `dir` is a library directory this wrote: a name that is 16
/// lowercase hex digits (a [`victory_shader::SOURCE_HASH`]) holding a
/// library.
fn is_translation(dir: &Path) -> bool {
    let ours = dir
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            name.len() == 16
                && name
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        });
    ours && dir.is_dir() && library_in(dir)
}

/// A library may have the old single file or only the shards that gained
/// pipelines. Recognize every shard filename the writer uses.
fn library_in(dir: &Path) -> bool {
    dir.join("host-pipelines.bin").is_file()
        || (0..SHARDS).any(|shard| dir.join(shard_file(shard)).is_file())
}

/// How many translations keep a library: the one in use and the previous
/// build's, which is what a bisection between two builds needs.
const KEPT: usize = 2;

/// `dir` under this build's translation, with older libraries pruned to
/// [`KEPT`] (by modification time).
fn keyed(dir: &Path) -> PathBuf {
    let path = dir.join(victory_shader::SOURCE_HASH);
    let Ok(entries) = std::fs::read_dir(dir) else {
        return path;
    };
    let mut others: Vec<(std::time::SystemTime, PathBuf)> = entries
        .flatten()
        .map(|entry| entry.path())
        // Only a directory this wrote: named like a translation and holding
        // a library. Anything else in the cache directory (`backup/`, a
        // copy of one) is not ours to delete.
        .filter(|other| *other != path && is_translation(other))
        .filter_map(|other| {
            let at = other.metadata().ok()?.modified().ok()?;
            Some((at, other))
        })
        .collect();
    others.sort_by_key(|(at, _)| std::cmp::Reverse(*at));
    for (_, older) in others.into_iter().skip(KEPT - 1) {
        // A library and a list or two: gone with the translation that made
        // them. Nothing this run reads is ever one of them.
        let _ = std::fs::remove_dir_all(&older);
    }
    path
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

/// A shard of the library: the interface loads use, and the state stores and
/// saves lock.
struct Shared {
    /// The same pipeline library, for loads only. A load holds nothing here,
    /// so it never waits out a save: a save holds the shard's lock while it
    /// serialises (230-600 MB in the world, #1683), some of them on the
    /// pipeline workers themselves, and every load of the shard waited for
    /// it (#1615's remainder, #2179).
    loads: ID3D12PipelineLibrary1,
    /// The library stores, saves and pre-build notes lock: the size a save
    /// asks for must still be right when it serialises.
    state: Mutex<Library>,
}

fn libraries() -> Option<&'static [Shared; SHARDS]> {
    static LIBRARIES: OnceLock<Option<[Shared; SHARDS]>> = OnceLock::new();
    LIBRARIES
        .get_or_init(|| {
            let device = crate::host::host()?.device.cast::<ID3D12Device1>().ok()?;
            let read = |file| DIR.get().and_then(|d| std::fs::read(d.join(file)).ok());
            let mut shards: Vec<Shared> = Vec::with_capacity(SHARDS);
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
                // Loads take the interface alone: the library is
                // free-threaded but for loads of one name, which `NAMES`
                // keeps apart from each other.
                let loads = library.library.clone();
                shards.push(Shared { loads, state: Mutex::new(library) });
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
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .prebuilt
            .contains(key)
    })
}

/// The pre-build has the title pipeline `key` in the library now.
pub(crate) fn note_prebuilt(key: String) {
    if let Some(libraries) = libraries() {
        let mut shard = libraries[shard_of(&key)]
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner());
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
/// there with the same description, else made by `create` and stored. A
/// name asked for by several threads is loaded or made by one at a time
/// (the others then find it in the library); different names overlap.
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
    let _held = NAMES.hold(name);
    // No lock at all: the shard's load interface lives as long as the
    // library, which is never dropped, and loads add nothing to it.
    let loads = &libraries[shard_of(name)].loads;
    if let Ok(pso) = load(loads, PCWSTR(wide.as_ptr())) {
        LOADED.fetch_add(1, Ordering::Relaxed);
        return Ok(pso);
    }
    // Compiled outside the lock: compiles take up to seconds.
    #[cfg(feature = "diagnostics")]
    LAST.set(Some((name.to_owned(), false)));
    let pso = create()?;
    MADE.fetch_add(1, Ordering::Relaxed);
    let mut l = libraries[shard_of(name)]
        .state
        .lock()
        .unwrap_or_else(|e| e.into_inner());
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
    let used_generation = USED_GENERATION.load(Ordering::Relaxed);
    let used = USED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(Used::list)
        .unwrap_or_default();
    let mut saved = 0u32;
    let mut total = 0usize;
    for (shard, library) in libraries.iter().enumerate() {
        if let Some(bytes) = save_shard(dir, shard, &library.state, write_atomically) {
            saved += 1;
            total += bytes;
        }
    }
    if used_changed || saved > 0 {
        match write_atomically(&dir.join(EARLY_FILE), used.join("\n").as_bytes()) {
            // Only if no key arrived while the list was being written: those
            // are not in it, and the next save has to write them.
            Ok(()) => {
                if USED_GENERATION.load(Ordering::Relaxed) == used_generation {
                    USED_CHANGED.store(false, Ordering::Relaxed);
                }
            }
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
        // SAFETY: a plain size query on a live library.
        let size = unsafe { l.library.GetSerializedSize() };
        let Some(mut bytes) = zeroed(size) else {
            // Left for a later save: the pipelines stay counted as added.
            tracing::warn!(bytes = size, shard, "no memory to serialise the host pipeline library shard; saved later");
            return None;
        };
        // SAFETY: a buffer of the size the library asks for.
        let bytes = unsafe { l.library.Serialize(&mut bytes).map(|()| bytes) };
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

/// `n` zero bytes, or `None` when the host has no memory for them. A shard
/// holds each pipeline's machine code (about 1.7 GB in all on NVIDIA after
/// `--cache --pipelines`), and an infallible allocation of that size aborted
/// the whole process mid-run ("memory allocation of 2650101862 bytes failed",
/// RTX 4050 Laptop, 32 GB).
fn zeroed(n: usize) -> Option<Vec<u8>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).ok()?;
    v.resize(n, 0);
    Some(v)
}

/// Writes `bytes` to a temporary file beside `path`, then renames it over
/// `path`: a reader never sees a part. Serialised so concurrent worker and
/// shutdown saves do not race on the same staging file or target rename.
fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
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
    /// The list as saved: one key a line, most recent run first (a list in
    /// no order, as earlier builds saved it, reads as well).
    fn read(listed: &str) -> Used {
        let mut before = HashMap::new();
        for (place, key) in listed.lines().filter(|l| !l.is_empty()).enumerate() {
            before.entry(key.to_owned()).or_insert(place);
        }
        Used { before, now: HashSet::new() }
    }

    /// The list to keep: this run's keys, then earlier runs' others, most
    /// recent first, at most `EARLY_MAX` (the oldest go once it is full -
    /// a set gave no order to drop by, so any earlier keys made room).
    fn list(&self) -> Vec<String> {
        let mut earlier: Vec<_> = self.before.iter().filter(|(k, _)| !self.now.contains(*k)).collect();
        earlier.sort_unstable_by_key(|&(_, &place)| place);
        let earlier = earlier.into_iter().map(|(k, _)| k);
        self.now.iter().chain(earlier).take(EARLY_MAX).cloned().collect()
    }
}

static USED: Mutex<Option<Used>> = Mutex::new(None);
static USED_CHANGED: AtomicBool = AtomicBool::new(false);
/// Bumped whenever a title pipeline key is added, so a save knows whether the
/// early list it wrote is still the whole list.
static USED_GENERATION: AtomicU64 = AtomicU64::new(0);

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

/// Reads the early list now if not yet, so a save keeps it (a pre-build
/// asks for no title pipeline).
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
        USED_GENERATION.fetch_add(1, Ordering::Relaxed);
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

    /// This run's keys stay in the list; earlier runs' fill it up to the cap,
    /// most recent first - once it is full the *oldest* runs' keys make room.
    #[test]
    fn the_early_list_keeps_this_run_first() {
        let keys = |r: std::ops::Range<usize>| r.map(|i| format!("{i:032x}")).collect::<Vec<_>>();
        // Earlier runs' keys, place 0 the most recent.
        let before = keys(0..EARLY_MAX).into_iter().enumerate().map(|(place, k)| (k, place)).collect();
        let used = Used { before, now: keys(EARLY_MAX - 10..EARLY_MAX + 10).into_iter().collect() };
        let list = used.list();
        assert_eq!(list.len(), EARLY_MAX);
        assert!(used.now.iter().all(|k| list.contains(k)));
        assert_eq!(list.iter().collect::<HashSet<_>>().len(), EARLY_MAX, "no key twice");
        // Ten earlier-only keys do not fit: the ten oldest (the highest
        // places) are the ones left out, not any ten.
        assert!(list.contains(&format!("{:032x}", 0)), "the most recent earlier key is kept");
        assert!(
            !list.contains(&format!("{:032x}", EARLY_MAX - 11)),
            "the oldest earlier key made room"
        );
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

    /// A stored pipeline loads while a save holds its shard's lock (a save
    /// serialises the whole shard, for seconds in the world).
    #[test]
    fn a_pipeline_loads_while_a_save_holds_the_library() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::core::s;
        let (Some(device), Some(shards)) = (
            crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()),
            libraries(),
        ) else {
            return; // no host GPU here
        };
        let shard = &shards[shard_of("test: a pipeline loads while a save holds the library")];
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
        let saving = shard.state.lock().unwrap_or_else(|e| e.into_inner());
        let (done, loaded) = std::sync::mpsc::channel();
        std::thread::scope(|s| {
            let made = &made;
            s.spawn(move || done.send(made.get(false).is_ok()));
            let loaded = loaded.recv_timeout(std::time::Duration::from_secs(30));
            drop(saving);
            assert_eq!(loaded, Ok(true), "the load waited for the save");
        });
    }

    /// A key the pre-build records while a shard's files are being written
    /// was not in them: the shard must stay dirty, or the next save skips it
    /// and the published list never gains the key.
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

#[cfg(test)]
mod keyed_tests {
    use super::*;

    /// Pruning keeps only directories this wrote: a `backup/` (or any
    /// directory without a hash name) in the cache directory survives,
    /// with what is in it, even when it is the oldest thing there.
    #[test]
    fn a_translation_with_only_a_later_shard_is_recognized() {
        let root = tempfile::tempdir().unwrap();
        let fixtures = [
            ("0000000000000001", "host-pipelines.bin".to_owned(), true),
            ("0000000000000002", shard_file(0), true),
            ("0000000000000003", shard_file(SHARDS - 1), true),
            ("0000000000000004", shard_file(SHARDS), false),
            ("backup", shard_file(SHARDS - 1), false),
        ];
        for (name, file, expected) in fixtures {
            let dir = root.path().join(name);
            std::fs::create_dir(&dir).unwrap();
            std::fs::write(dir.join(&file), b"synthetic shard").unwrap();
            assert_eq!(is_translation(&dir), expected, "{name}/{file}");
        }
    }

    #[test]
    fn pruning_leaves_foreign_directories_alone() {
        let dir = std::env::temp_dir().join(format!("victory-keyed-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // An old directory of ours, an old one of someone else's, and the
        // one this build uses.
        for (name, own) in [("00112233aabbccdd", true), ("backup", false)] {
            let at = dir.join(name);
            std::fs::create_dir_all(&at).unwrap();
            std::fs::write(at.join("sentinel"), b"keep me").unwrap();
            if own {
                std::fs::write(at.join("host-pipelines.bin"), b"library").unwrap();
            }
        }
        let current = dir.join(victory_shader::SOURCE_HASH);
        std::fs::create_dir_all(&current).unwrap();
        std::fs::write(current.join("host-pipelines.bin"), b"library").unwrap();

        assert_eq!(keyed(&dir), current);
        assert!(dir.join("backup").join("sentinel").is_file(), "a foreign directory was deleted");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

