//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early (decision 0007, "Pipeline cache").
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The library is saved under the cache
//!    folder; the host refuses a library of another adapter or driver, and
//!    an entry whose description differs, so neither is ever used. A kind
//!    of pipeline the library refuses to load right after storing it (as
//!    D3D12's does mesh pipelines) is kept no more: each is compiled when
//!    made.
//! 2. The title pipelines a run drew or dispatched with are listed by a key
//!    known when the title creates them (their shader containers and
//!    graphics state); the next run hands those to the workers at
//!    creation, during loading.
//!
//! All library calls are serialised: the interface does not promise that
//! stores, loads and serialising may overlap.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
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
}

// SAFETY: D3D12 objects are free-threaded; every call is made under the
// mutex that holds the library.
unsafe impl Send for Library {}

const LIBRARY_FILE: &str = "host-pipelines.bin";
const EARLY_FILE: &str = "early-pipelines.txt";
/// Title pipelines listed for an early start, at most.
const EARLY_MAX: usize = 8192;

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
            device.CreatePipelineLibrary::<ID3D12PipelineLibrary>(bytes)?.cast()
        }
    };
    if !blob.is_empty() {
        match make(&blob) {
            Ok(library) => {
                tracing::info!(bytes = blob.len(), "host pipeline library loaded");
                return Some(Library { library, _blob: blob, added: 0 });
            }
            Err(e) => tracing::info!(error = %e, "host pipeline library rejected; starting an empty one"),
        }
    } else {
        tracing::info!("no host pipeline library yet; starting an empty one");
    }
    let library = make(&[]).ok()?;
    Some(Library { library, _blob: Vec::new(), added: 0 })
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

/// Whether the library keeps a kind of pipeline ("compute", "graphics",
/// "mesh"), as the first of the kind stored shows when loaded back at once.
const UNKNOWN: u8 = 0;
const YES: u8 = 1;
const NO: u8 = 2;
static KEPT: [AtomicU8; 3] = [
    AtomicU8::new(UNKNOWN),
    AtomicU8::new(UNKNOWN),
    AtomicU8::new(UNKNOWN),
];

fn kind_index(kind: &str) -> usize {
    match kind {
        "compute" => 0,
        "graphics" => 1,
        _ => 2,
    }
}

/// The pipeline named `name`, of `kind`: loaded from the library by `load`
/// if it is there with the same description, else made by `create` and
/// stored. The first of a kind stored is loaded back at once: refused, the
/// library gets no more of the kind (and says so once).
pub(crate) fn pipeline(
    name: &str,
    kind: &str,
    load: impl Fn(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    pipeline_in(library(), &KEPT[kind_index(kind)], name, kind, load, create)
}

fn pipeline_in(
    library: Option<&Mutex<Library>>,
    kept: &AtomicU8,
    name: &str,
    kind: &str,
    load: impl Fn(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    let wide = HSTRING::from(name);
    let library = library.filter(|_| kept.load(Ordering::Relaxed) != NO);
    let Some(library) = library else {
        MADE.fetch_add(1, Ordering::Relaxed);
        return create();
    };
    if let Ok(pso) = load(&library.lock().unwrap_or_else(|e| e.into_inner()).library, PCWSTR(wide.as_ptr())) {
        LOADED.fetch_add(1, Ordering::Relaxed);
        return Ok(pso);
    }
    // Compiled outside the lock: compiles take up to seconds.
    let pso = create()?;
    MADE.fetch_add(1, Ordering::Relaxed);
    let mut l = library.lock().unwrap_or_else(|e| e.into_inner());
    // A different worker can disable this kind while we compile.
    if kept.load(Ordering::Relaxed) == NO {
        return Ok(pso);
    }
    // SAFETY: a live pipeline; a name already stored (a description that
    // differs) is refused, which leaves the library as it was.
    if unsafe { l.library.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }.is_err() {
        return Ok(pso);
    }
    l.added += 1;
    if kept.load(Ordering::Relaxed) == UNKNOWN {
        let again = load(&l.library, PCWSTR(wide.as_ptr()));
        kept.store(if again.is_ok() { YES } else { NO }, Ordering::Relaxed);
        if again.is_err() {
            tracing::warn!(
                kind,
                "host pipeline library refuses a {kind} pipeline it just stored: {kind} pipelines are compiled every run, none kept"
            );
        }
    }
    Ok(pso)
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
    let blob = {
        let mut l = library.lock().unwrap_or_else(|e| e.into_inner());
        if l.added == 0 && !USED_CHANGED.swap(false, Ordering::Relaxed) {
            return;
        }
        // SAFETY: a buffer of the size the library asks for.
        let bytes = unsafe {
            let mut bytes = vec![0u8; l.library.GetSerializedSize()];
            l.library.Serialize(&mut bytes).map(|()| bytes)
        };
        match bytes {
            Ok(bytes) => {
                l.added = 0;
                bytes
            }
            Err(e) => {
                tracing::warn!(error = %e, "host pipeline library not serialised");
                return;
            }
        }
    };
    let used: Vec<String> = USED.lock().unwrap_or_else(|e| e.into_inner()).iter().flatten().cloned().collect();
    let written = std::fs::create_dir_all(dir)
        .and_then(|()| write_atomically(&dir.join(LIBRARY_FILE), &blob))
        .and_then(|()| write_atomically(&dir.join(EARLY_FILE), used.join("\n").as_bytes()));
    match written {
        Ok(()) => tracing::info!(
            bytes = blob.len(),
            early = used.len(),
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipeline library saved"
        ),
        Err(e) => tracing::warn!(error = %e, "host pipeline library not saved"),
    }
}

/// Writes `bytes` to a temporary file beside `path`, then renames it over
/// `path`: a reader never sees a part.
fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

/// Title pipeline keys drawn or dispatched with (this run and the last).
static USED: Mutex<Option<HashSet<String>>> = Mutex::new(None);
static USED_CHANGED: AtomicBool = AtomicBool::new(false);

fn used() -> std::sync::MutexGuard<'static, Option<HashSet<String>>> {
    let mut used = USED.lock().unwrap_or_else(|e| e.into_inner());
    if used.is_none() {
        let listed = DIR
            .get()
            .and_then(|d| std::fs::read_to_string(d.join(EARLY_FILE)).ok())
            .unwrap_or_default();
        let set: HashSet<String> = listed.lines().filter(|l| !l.is_empty()).map(str::to_owned).collect();
        tracing::info!(pipelines = set.len(), "title pipelines to start early");
        *used = Some(set);
    }
    used
}

/// Whether the title pipeline with `key` was drawn with in an earlier run.
pub(crate) fn wanted_early(key: &str) -> bool {
    used().as_ref().is_some_and(|s| s.contains(key))
}

/// The title pipeline with `key` was drawn or dispatched with.
pub(crate) fn note_used(key: &str) {
    let mut used = used();
    let set = used.get_or_insert_with(HashSet::new);
    if set.len() < EARLY_MAX && set.insert(key.to_owned()) {
        USED_CHANGED.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::{Fxc::D3DCompile, ID3DBlob};
    use windows::core::s;

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

    /// Each kind of pipeline has a verdict of its own.
    #[test]
    fn kinds_are_told_apart() {
        let kinds = ["compute", "graphics", "mesh"].map(kind_index);
        assert_eq!(kinds, [0, 1, 2]);
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
    fn in_flight_compile_does_not_store_a_rejected_kind() {
        use std::sync::Barrier;
        use windows::Win32::Foundation::E_INVALIDARG;

        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok())
        else {
            return; // no host GPU here
        };
        let library = Mutex::new(open(&device, Vec::new()).expect("pipeline library"));
        let kept = AtomicU8::new(UNKNOWN);
        let shader = compute_shader(&device, 1);
        let desc = compute_desc(&shader);
        // SAFETY: the shader's blob and root signature live through the call.
        let pso: ID3D12PipelineState = unsafe { device.CreateComputePipelineState(&desc) }.unwrap();
        let ready = Barrier::new(2);
        let resume = Barrier::new(2);
        let refuse = |_: &ID3D12PipelineLibrary1, _: PCWSTR| {
            Err(windows::core::Error::from_hresult(E_INVALIDARG))
        };
        std::thread::scope(|scope| {
            scope.spawn(|| {
                pipeline_in(Some(&library), &kept, "second", "mesh", refuse, || {
                    ready.wait();
                    resume.wait();
                    Ok(pso.clone())
                })
                .unwrap();
            });
            ready.wait();
            // This worker gets a successful store and a refused load while
            // the other one is still compiling outside the library lock.
            let stored = pipeline_in(Some(&library), &kept, "first", "mesh", refuse, || {
                Ok(pso.clone())
            });
            resume.wait();
            stored.unwrap();
        });
        assert_eq!(kept.load(Ordering::Relaxed), NO);
        assert_eq!(library.lock().unwrap().added, 1);
    }

    #[test]
    fn stale_name_does_not_disable_pipeline_caching() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok())
        else {
            return; // no host GPU here
        };
        let old = compute_shader(&device, 1);
        let current = compute_shader(&device, 2);
        let old_desc = compute_desc(&old);
        let desc = compute_desc(&current);
        // SAFETY: the two shader descriptions point to live blobs and signatures.
        let old_pso: ID3D12PipelineState =
            unsafe { device.CreateComputePipelineState(&old_desc) }.unwrap();
        let library = Mutex::new(open(&device, Vec::new()).expect("pipeline library"));
        let kept = AtomicU8::new(UNKNOWN);
        let name = HSTRING::from("stale");
        let load = |library: &ID3D12PipelineLibrary1, name| {
            // SAFETY: the name lives through the call, as do the code and root.
            unsafe { library.LoadComputePipeline(name, &desc) }
        };
        // SAFETY: the name and PSO live through the call.
        unsafe { library.lock().unwrap().library.StorePipeline(PCWSTR(name.as_ptr()), &old_pso) }
            .unwrap();
        assert!(load(&library.lock().unwrap().library, PCWSTR(name.as_ptr())).is_err());
        // SAFETY: the current shader's blob and signature are live.
        let make = || unsafe { device.CreateComputePipelineState(&desc) };
        pipeline_in(Some(&library), &kept, "stale", "compute", load, make).unwrap();
        assert_eq!(kept.load(Ordering::Relaxed), UNKNOWN);

        pipeline_in(Some(&library), &kept, "new", "compute", load, make).unwrap();
        assert_eq!(kept.load(Ordering::Relaxed), YES);
        let library = library.lock().unwrap();
        assert_eq!(library.added, 1);
        let name = HSTRING::from("new");
        assert!(load(&library.library, PCWSTR(name.as_ptr())).is_ok());
    }

    fn compute_shader(device: &ID3D12Device1, threads: u32) -> (ID3DBlob, ID3D12RootSignature) {
        let source = format!(r#"[RootSignature("")] [numthreads({threads},1,1)] void main() {{}}"#);
        let mut code = None;
        // SAFETY: valid source and out pointer within this call.
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("library-test"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )
        }
        .unwrap();
        let code = code.unwrap();
        // SAFETY: the bytes belong to the live compiler blob.
        let root = unsafe {
            let bytes =
                std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            device.CreateRootSignature(0, bytes).unwrap()
        };
        (code, root)
    }

    /// The caller holds the blob and root for every use of the description.
    fn compute_desc(shader: &(ID3DBlob, ID3D12RootSignature)) -> D3D12_COMPUTE_PIPELINE_STATE_DESC {
        let (code, root) = shader;
        // SAFETY: copy only the borrowed COM interface and the blob's metadata.
        unsafe {
            D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: code.GetBufferPointer(),
                    BytecodeLength: code.GetBufferSize(),
                },
                ..Default::default()
            }
        }
    }
}
