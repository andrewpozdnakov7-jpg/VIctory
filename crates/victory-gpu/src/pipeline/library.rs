//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early (decision 0007, "Pipeline cache").
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The library is saved under the cache
//!    folder; the host refuses a library of another adapter or driver, and
//!    an entry whose description differs, so neither is ever used.
//! 2. The title pipelines a run drew or dispatched with are listed by a key
//!    known when the title creates them (their shader containers and
//!    graphics state); the next run hands those to the workers at
//!    creation, during loading.
//!
//! All library calls are serialised: the interface does not promise that
//! stores, loads and serialising may overlap.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, mpsc};

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
}

// SAFETY: D3D12 objects are free-threaded; every call is made under the
// mutex that holds the library.
unsafe impl Send for Library {}

const LIBRARY_FILE: &str = "host-pipelines.bin";
const EARLY_FILE: &str = "early-pipelines.txt";
/// Title pipelines listed for an early start, at most (sized for full
/// open-world pipeline counts while keeping memory bounded).
const EARLY_MAX: usize = 65_536;
/// Set whenever a new pipeline is stored into the library since the last save.
static LIBRARY_DIRTY: AtomicBool = AtomicBool::new(false);
/// Prevents overlapping periodic saves from queuing redundant work.
static SAVE_IN_FLIGHT: AtomicBool = AtomicBool::new(false);

fn library() -> Option<&'static Mutex<Library>> {
    static LIBRARY: OnceLock<Option<Mutex<Library>>> = OnceLock::new();
    LIBRARY
        .get_or_init(|| {
            let device = crate::host::host()?.device.cast::<ID3D12Device1>().ok()?;
            let blob = DIR
                .get()
                .and_then(|d| std::fs::read(d.join(LIBRARY_FILE)).ok())
                .unwrap_or_default();
            let library = open(&device, blob);
            if library.is_none() {
                tracing::warn!("no host pipeline library: pipelines are not kept");
            }
            library.map(Mutex::new)
        })
        .as_ref()
}

/// The library in `blob`, or an empty one if the host rejects the blob (an
/// adapter or driver change, damage).
fn open(device: &ID3D12Device1, blob: Vec<u8>) -> Option<Library> {
    let make = |bytes: &[u8]| -> windows::core::Result<ID3D12PipelineLibrary1> {
        // SAFETY: the bytes outlive the library (kept beside it).
        unsafe {
            device
                .CreatePipelineLibrary::<ID3D12PipelineLibrary>(bytes)?
                .cast()
        }
    };
    if !blob.is_empty() {
        match make(&blob) {
            Ok(library) => {
                tracing::info!(bytes = blob.len(), "host pipeline library loaded");
                return Some(Library {
                    library,
                    _blob: blob,
                    added: 0,
                });
            }
            Err(e) => {
                tracing::info!(error = %e, "host pipeline library rejected; starting an empty one")
            }
        }
    } else {
        tracing::info!("no host pipeline library yet; starting an empty one");
    }
    let library = make(&[]).ok()?;
    Some(Library {
        library,
        _blob: Vec::new(),
        added: 0,
    })
}

/// A pipeline's name in the library: the hash of what its host
/// description holds.
pub(crate) fn name(kind: &str, root: &[u8], code: &[&[u8]], state: &str) -> String {
    let mut h = blake3::Hasher::new();
    for part in [kind.as_bytes(), root]
        .into_iter()
        .chain(code.iter().copied())
        .chain([state.as_bytes()])
    {
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
    let Some(library) = library() else {
        MADE.fetch_add(1, Ordering::Relaxed);
        return create();
    };
    if let Ok(pso) = load(
        &library.lock().unwrap_or_else(|e| e.into_inner()).library,
        PCWSTR(wide.as_ptr()),
    ) {
        LOADED.fetch_add(1, Ordering::Relaxed);
        return Ok(pso);
    }
    // Compiled outside the lock: compiles take up to seconds.
    let pso = create()?;
    MADE.fetch_add(1, Ordering::Relaxed);
    let mut l = library.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: a live pipeline; a name already stored (a description that
    // differs) is refused, which leaves the library as it was.
    if unsafe { l.library.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }.is_ok() {
        l.added += 1;
        LIBRARY_DIRTY.store(true, Ordering::Relaxed);
    }
    Ok(pso)
}

struct SaveAck {
    done: Mutex<bool>,
    cv: Condvar,
}

impl SaveAck {
    fn new() -> Self {
        Self {
            done: Mutex::new(false),
            cv: Condvar::new(),
        }
    }

    fn signal(&self) {
        let mut done = self.done.lock().unwrap_or_else(|e| e.into_inner());
        *done = true;
        self.cv.notify_all();
    }

    fn wait(&self) {
        let mut done = self.done.lock().unwrap_or_else(|e| e.into_inner());
        while !*done {
            done = self.cv.wait(done).unwrap_or_else(|e| e.into_inner());
        }
    }
}

struct SaveRequest {
    dir: PathBuf,
    ack: Option<Arc<SaveAck>>,
}

/// Dedicated background saver thread so periodic saves never block PSO
/// compilation workers on serialization or disk I/O, while shutdown (`now =
/// true`) can still wait for a clean flush.
fn saver() -> Option<&'static mpsc::Sender<SaveRequest>> {
    static SAVER: OnceLock<Option<mpsc::Sender<SaveRequest>>> = OnceLock::new();
    SAVER
        .get_or_init(|| {
            let (tx, rx) = mpsc::channel::<SaveRequest>();
            let spawned = std::thread::Builder::new()
                .name("victory-pipeline-save".into())
                .spawn(move || {
                    // SAFETY: plain priority call on our own background thread.
                    unsafe {
                        use windows::Win32::System::Threading::*;
                        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
                    }
                    let mut buf = Vec::new();
                    while let Ok(req) = rx.recv() {
                        perform_save(&req.dir, &mut buf);
                        SAVE_IN_FLIGHT.store(false, Ordering::Release);
                        if let Some(ack) = req.ack {
                            ack.signal();
                        }
                    }
                });
            match spawned {
                Ok(_) => Some(tx),
                Err(e) => {
                    tracing::warn!(error = %e, "no pipeline saver thread");
                    None
                }
            }
        })
        .as_ref()
}

/// Serializes the library into `buf` (growing `buf` outside `library.lock()` so
/// multi-megabyte page faults never stall PSO workers), returning the serialized
/// byte length and count of newly stored pipelines.
fn serialize_library(library: &Mutex<Library>, buf: &mut Vec<u8>) -> Option<(usize, u64)> {
    let needed = {
        let l = library.lock().unwrap_or_else(|e| e.into_inner());
        if l.added == 0 {
            LIBRARY_DIRTY.store(false, Ordering::Relaxed);
            return None;
        }
        // SAFETY: plain size query on a live pipeline library under its mutex.
        unsafe { l.library.GetSerializedSize() }
    };
    if needed == 0 {
        return None;
    }
    if buf.len() < needed {
        buf.resize(needed.saturating_add(needed / 16).max(needed), 0);
    }
    loop {
        let mut l = library.lock().unwrap_or_else(|e| e.into_inner());
        if l.added == 0 {
            LIBRARY_DIRTY.store(false, Ordering::Relaxed);
            return None;
        }
        // SAFETY: plain size query on a live pipeline library under its mutex.
        let exact = unsafe { l.library.GetSerializedSize() };
        if exact > buf.len() {
            drop(l);
            buf.resize(exact.saturating_add(exact / 16).max(exact), 0);
            continue;
        }
        // SAFETY: `buf[..exact]` has the exact size requested by `GetSerializedSize`.
        match unsafe { l.library.Serialize(&mut buf[..exact]) } {
            Ok(()) => {
                let added = std::mem::take(&mut l.added);
                LIBRARY_DIRTY.store(false, Ordering::Relaxed);
                return Some((exact, added));
            }
            Err(e) => {
                tracing::warn!(error = %e, "host pipeline library not serialised");
                return None;
            }
        }
    }
}

/// Performs the actual snapshot + unlocked disk I/O for `host-pipelines.bin`
/// and `early-pipelines.txt`.
fn perform_save(dir: &Path, buf: &mut Vec<u8>) {
    let Some(library) = library() else { return };
    let serialized = serialize_library(library, buf);
    let used_snapshot = if USED_CHANGED.swap(false, Ordering::AcqRel) {
        used().as_ref().map(Used::list)
    } else {
        None
    };
    if serialized.is_none() && used_snapshot.is_none() {
        return;
    }

    if let Err(e) = std::fs::create_dir_all(dir) {
        if let Some((_, added)) = serialized {
            library.lock().unwrap_or_else(|e| e.into_inner()).added += added;
            LIBRARY_DIRTY.store(true, Ordering::Relaxed);
        }
        if used_snapshot.is_some() {
            USED_CHANGED.store(true, Ordering::Relaxed);
        }
        tracing::warn!(error = %e, "host pipeline library directory not created");
        return;
    }

    let mut lib_bytes = 0usize;
    let mut ok = true;
    if let Some((len, added)) = serialized {
        if let Err(e) = write_atomically(&dir.join(LIBRARY_FILE), &buf[..len]) {
            library.lock().unwrap_or_else(|e| e.into_inner()).added += added;
            LIBRARY_DIRTY.store(true, Ordering::Relaxed);
            tracing::warn!(error = %e, "host pipeline library not saved");
            ok = false;
        } else {
            lib_bytes = len;
        }
    }
    let mut early_count = 0usize;
    if let Some(used) = used_snapshot {
        early_count = used.len();
        if let Err(e) = write_atomically(&dir.join(EARLY_FILE), used.join("\n").as_bytes()) {
            USED_CHANGED.store(true, Ordering::Relaxed);
            tracing::warn!(error = %e, "early pipeline list not saved");
            ok = false;
        }
    }
    if ok {
        tracing::info!(
            bytes = lib_bytes,
            early = early_count,
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipeline library saved"
        );
    }
}

/// Writes the library and the early list if pipelines were added since the
/// last save and `SAVE_EVERY` has passed (or `now`). Periodic saves run
/// asynchronously on the dedicated saver thread; `now = true` waits for disk
/// completion.
pub(crate) fn save(now: bool) {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    const SAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(20);
    let Some(dir) = DIR.get() else { return };
    if !now && !LIBRARY_DIRTY.load(Ordering::Relaxed) && !USED_CHANGED.load(Ordering::Relaxed) {
        return;
    }
    if !now && SAVE_IN_FLIGHT.load(Ordering::Acquire) {
        return;
    }
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if !now && last.is_some_and(|t| t.elapsed() < SAVE_EVERY) {
            return;
        }
        if !now && SAVE_IN_FLIGHT.swap(true, Ordering::AcqRel) {
            return;
        }
        *last = Some(std::time::Instant::now());
    }
    if let Some(tx) = saver() {
        let ack = now.then(|| Arc::new(SaveAck::new()));
        let req = SaveRequest {
            dir: dir.clone(),
            ack: ack.clone(),
        };
        if tx.send(req).is_ok() {
            if let Some(ack) = ack {
                ack.wait();
            }
            return;
        }
    }
    // Fallback if no background saver thread could be created.
    let mut buf = Vec::new();
    perform_save(dir, &mut buf);
    SAVE_IN_FLIGHT.store(false, Ordering::Release);
}

/// Writes `bytes` to a unique temporary file beside `path`, then renames it
/// over `path`, removing the temporary file if either write or rename fails.
fn write_atomically(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("cache");
    let tmp = path.with_file_name(format!(".{file_name}.tmp-{}-{n}", std::process::id()));
    if let Err(err) = std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, path)) {
        let _ = std::fs::remove_file(&tmp);
        return Err(err);
    }
    Ok(())
}

/// Title pipeline keys drawn or dispatched with: in earlier runs (the
/// list as loaded), and in this one.
#[derive(Default)]
struct Used {
    /// Each earlier key with its place in the list as loaded: the list is
    /// kept most recent run first, so a lower place is a more recent run.
    before: HashMap<String, usize>,
    /// This run's retained keys and their order of first use.
    now: HashMap<String, usize>,
}

impl Used {
    /// The list as saved: one key a line, most recent run first (a list
    /// in no order, as earlier builds saved it, reads as well).
    fn read(listed: &str) -> Used {
        let mut before = HashMap::new();
        for (place, key) in listed.lines().filter(|l| !l.is_empty()).take(EARLY_MAX).enumerate() {
            before.entry(key.to_owned()).or_insert(place);
        }
        Used { before, now: HashMap::new() }
    }

    /// Keeps the earliest uses in this run so the next run can warm them.
    fn note(&mut self, key: &str) -> bool {
        let place = self.now.len();
        if place == EARLY_MAX || self.now.contains_key(key) {
            return false;
        }
        self.now.insert(key.to_owned(), place);
        true
    }

    /// The list to keep: first uses in this run, then earlier runs' other
    /// keys, most recent run first, at most `EARLY_MAX`.
    fn list(&self) -> Vec<String> {
        let mut retained = vec![String::new(); self.now.len()];
        for (key, &place) in &self.now {
            retained[place] = key.clone();
        }
        if retained.len() == EARLY_MAX {
            return retained;
        }

        let mut earlier: Vec<_> = self.before.iter().filter(|(k, _)| !self.now.contains_key(*k)).collect();
        earlier.sort_unstable_by_key(|&(_, &place)| place);
        retained.extend(earlier.into_iter().take(EARLY_MAX - retained.len()).map(|(k, _)| k.clone()));
        retained
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

/// Whether the title pipeline with `key` has been used in this or an earlier run.
pub(crate) fn wanted_early(key: &str) -> bool {
    used().as_ref().is_some_and(|u| u.before.contains_key(key) || u.now.contains_key(key))
}

/// The title pipeline with `key` was drawn or dispatched with.
pub(crate) fn note_used(key: &str) {
    if used().get_or_insert_with(Used::default).note(key) {
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

    #[test]
    fn a_rejected_blob_gives_an_empty_library() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok())
        else {
            return; // no host GPU here
        };
        let library =
            open(&device, b"not a pipeline library".repeat(16)).expect("an empty library");
        // SAFETY: a plain query on a live library.
        assert!(unsafe { library.library.GetSerializedSize() } > 0);
        assert!(library._blob.is_empty());
    }

    #[test]
    fn early_list_records_and_deduplicates_above_old_8192_cap() {
        const { assert!(EARLY_MAX >= 65_536) };
        let mut used = Used::default();
        for i in (0..EARLY_MAX).rev() {
            let key = format!("{i:032x}");
            assert!(used.note(&key));
            assert!(!used.note(&key), "duplicates are deduplicated");
        }
        assert!(!used.note("late"));
        let saved = used.list();
        assert_eq!(saved.len(), EARLY_MAX);
        let expected: Vec<_> = (0..EARLY_MAX).rev().map(|i| format!("{i:032x}")).collect();
        assert_eq!(saved, expected);
    }

    #[test]
    fn fresh_pipeline_uses_can_displace_keys_from_a_full_previous_run() {
        let key = |i: usize| format!("{i:032x}");
        let older: Vec<_> = (0..EARLY_MAX).map(key).collect();
        let mut used = Used::read(&older.join("\n"));
        let fresh = key(EARLY_MAX);
        assert!(used.note(&fresh));
        assert!(used.note(&older[2]));

        let saved = used.list();
        assert_eq!(saved.len(), EARLY_MAX);
        assert_eq!(&saved[..4], &[fresh.clone(), older[2].clone(), older[0].clone(), older[1].clone()]);
        assert!(!saved.contains(&older[EARLY_MAX - 1]));
        let restored = Used::read(&saved.join("\n"));
        assert_eq!(restored.before[&fresh], 0);
        assert_eq!(restored.before[&older[2]], 1);
    }

    #[test]
    fn write_atomically_overwrites_and_cleans_up_temp_file_on_error() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join(LIBRARY_FILE);
        write_atomically(&target, b"first").unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"first");
        write_atomically(&target, b"second").unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"second");

        // Trigger a rename failure by placing a directory at the destination path.
        let blocked = dir.path().join("blocked.bin");
        std::fs::create_dir(&blocked).unwrap();
        assert!(write_atomically(&blocked, b"payload").is_err());

        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp-"))
            .collect();
        assert!(
            leftovers.is_empty(),
            "temporary staging files leaked on error: {leftovers:?}"
        );
    }
}
