//! The pipeline pre-build, the warm-up's second half: every pipeline of the
//! title's shader pack made ahead of a run into the host pipeline library
//! the run reads (`<cache>/pipelines`), so that a first run, and a first
//! visit to any place, compiles nothing the pack lists.
//!
//! Each pack record becomes the creation stream the engine would hand the
//! device for that pipeline (`victory_game::pipelines`), which goes through
//! the run's own code (`victory_gpu::pipeline::Prebuild`): a pipeline is
//! only ever used if its host description is the run's, byte for byte.
//! Records whose states cannot be named, or whose shaders are refused, are
//! counted by cause and skipped. The library is saved as it grows: an
//! interrupted pre-build keeps its work, and the next one finds those
//! pipelines there.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use anyhow::Result;
use victory_game::pipelines::{self, Named, Unnamed};
use victory_game::{sgd5, shaderpack};
use victory_gpu::pipeline::{self, Prebuild};
use victory_shader::{Container, FourCC, HwStage, native_programs};

use crate::shader::{MAX_FILE, Source, sources};

/// Where a native program's container is: file, source, its stage, and its
/// 16-char BLAKE3 hex hash (matching `victory_gpu::pipeline::Shader::new`).
type Found = (usize, usize, HwStage, String);

/// The pack identifies the compute or vertex/mesh container by its database hash.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Primary {
    Compute(u64),
    Graphics(u64),
}

/// One pipeline to make: a graphics or compute record of the pack.
#[derive(Clone, Copy, Debug)]
enum Job {
    Graphics(usize),
    Compute(usize),
}

pub fn run(paths: &[PathBuf], threads: usize) -> Result<()> {
    let started = Instant::now();
    pipeline::use_pipeline_cache(victory_core::Config::load()?.cache_subdir("pipelines")?);
    // The files that hold shaders (or the pack itself).
    let files: Vec<(PathBuf, Vec<u8>)> = paths
        .iter()
        .flat_map(|root| {
            walkdir::WalkDir::new(root)
                .into_iter()
                .filter_map(Result::ok)
        })
        .filter(|e| e.file_type().is_file() && e.metadata().is_ok_and(|m| m.len() <= MAX_FILE))
        .filter_map(|e| Some((e.path().to_owned(), std::fs::read(e.path()).ok()?)))
        .filter(|(_, d)| d.starts_with(shaderpack::MAGIC) || !sources(d).is_empty())
        .collect();
    // It ships beside the shader databases.
    let found = (files.iter())
        .filter(|(_, d)| d.starts_with(shaderpack::MAGIC) || sgd5::is_sgd5(d))
        .find_map(|(p, d)| Some((p, shaderpack::find(d)?)));
    let Some((pack_path, pack)) = found else {
        println!("pipelines    no shader pack among the files: nothing to pre-build");
        return Ok(());
    };
    let pack = shaderpack::Pack::parse(&pack).map_err(anyhow::Error::msg)?;
    let graphics: Vec<shaderpack::Graphics> = pack.graphics().collect();
    let compute: Vec<shaderpack::Compute> = pack.compute().collect();

    // Every container by the hash instructions its native program starts
    // with, which the pack's programs start with too.
    let sources: Vec<Vec<Source>> = files.iter().map(|(_, d)| sources(d)).collect();
    let mut programs: HashMap<[u8; 24], Found> = HashMap::new();
    let mut by_engine = HashMap::new();
    let mut databases = Vec::new();
    for (f, list) in sources.iter().enumerate() {
        let dbs = sgd5::databases(&files[f].1);
        let hashes: HashMap<_, _> = dbs
            .iter()
            .flat_map(|db| db.shaders.iter().flatten())
            .map(|shader| (shader.record.offset, shader.hash))
            .collect();
        databases.extend(dbs);

        let next = AtomicUsize::new(0);
        let indexed: Vec<_> = (0..list.len()).map(|_| OnceLock::new()).collect();
        std::thread::scope(|s| {
            for _ in 0..threads.min(list.len()) {
                s.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(source) = list.get(i) else { break };
                        let Some(bytes) = source.bytes() else {
                            continue;
                        };
                        let Some(trimmed) = container_bytes(&bytes) else {
                            continue;
                        };
                        let ids = program_ids(trimmed);
                        if !ids.is_empty() {
                            let hash = blake3::hash(trimmed).to_hex()[..16].to_owned();
                            indexed[i].get_or_init(|| (ids, hash));
                        }
                    }
                });
            }
        });
        // Keep the input order for programs with no matching database hash,
        // independently of which worker finished unpacking a container first.
        for (i, entry) in indexed.into_iter().enumerate() {
            let hash = match &list[i] {
                Source::Packed(r) => hashes.get(&r.offset),
                Source::Raw { .. } => None,
            };
            let Some((ids, container_hash)) = entry.into_inner() else {
                continue;
            };
            for (id, stage) in ids {
                let found = (f, i, stage, container_hash.clone());
                programs.entry(id).or_insert_with(|| found.clone());
                if let Some(&hash) = hash {
                    let primary = match stage {
                        HwStage::Cs => Primary::Compute(hash),
                        HwStage::Vs | HwStage::Ms => Primary::Graphics(hash),
                        _ => continue,
                    };
                    by_engine.entry((primary, id)).or_insert(found);
                }
            }
        }
    }
    let named = Named::find(&pack, &databases);
    eprintln!(
        "{}: shader pack, {} graphics and {} compute pipelines; states named: {} depth-stencil, {} rasteriser, {} blend",
        pack_path.display(),
        graphics.len(),
        compute.len(),
        named.depth.len(),
        named.raster.len(),
        named.blend.len(),
    );

    // Distinct containers can carry the same native program. The pack names
    // the compute or vertex/mesh container by its database hash as well.
    let find_container = |code: &[u8], primary: Option<Primary>| -> Option<&Found> {
        let id = shaderpack::program_id(code)?;
        primary
            .and_then(|p| by_engine.get(&(p, id)))
            .or_else(|| programs.get(&id))
    };
    let container = |code: &[u8], primary: Option<Primary>| -> Option<(Vec<u8>, HwStage)> {
        let (f, i, stage, _) = find_container(code, primary)?;
        let bytes = sources[*f][*i].bytes()?;
        Some((container_bytes(&bytes)?.to_vec(), *stage))
    };
    let lookup_hash = |code: &[u8], primary: Option<Primary>| -> Option<(HwStage, &str)> {
        let (_, _, stage, hash) = find_container(code, primary)?;
        Some((*stage, hash.as_str()))
    };
    // A job's pre-build key without inflating, copying, or re-hashing its
    // shader containers. Matches `prepare(job).map(|p| p.key())` identically.
    let key_of = |job: Job| -> Result<String, String> {
        let g = match job {
            Job::Graphics(i) => &graphics[i],
            Job::Compute(i) => {
                let c = &compute[i];
                let (_, hash) = lookup_hash(c.stage.program, Some(Primary::Compute(c.shader_hash)))
                    .ok_or("shader not found")?;
                return Ok(Prebuild::compute_key(hash));
            }
        };
        let [ps, vs, gs, ..] = g.stages;
        if g.stages[3..].iter().any(Option::is_some) {
            return Err("stages other than vertex, mesh and pixel".into());
        }
        let (stage, first_hash) = vs
            .or(gs)
            .and_then(|s| lookup_hash(s.program, Some(Primary::Graphics(g.shader_hash))))
            .ok_or("shader not found")?;
        let pixel_hash = match ps {
            Some(ps) => Some(lookup_hash(ps.program, None).ok_or("shader not found")?.1),
            None => None,
        };
        let (mesh, first_stage) = match stage {
            HwStage::Ms => (true, "mesh"),
            HwStage::Vs => (false, "vertex"),
            _ => return Err("first stage not a vertex or mesh shader".into()),
        };
        let stream = pipelines::stream(g, &named, &[], mesh, None).map_err(|u| {
            match u {
                Unnamed::DepthStencil => "depth-stencil state not named",
                Unnamed::Rasterizer => "rasteriser state not named",
                Unnamed::Blend => "blend state not named",
                Unnamed::Metadata => "metadata not read",
            }
            .to_owned()
        })?;
        let mut stages: [(&'static str, &str); 2] = [(first_stage, first_hash), ("pixel", "")];
        let count = match pixel_hash {
            Some(h) => {
                stages[1].1 = h;
                2
            }
            None => 1,
        };
        // SAFETY: the stream points at its own live input elements; shader
        // pointers are not read by `graphics_key`.
        Ok(unsafe { Prebuild::graphics_key(stream.bytes(), &stages[..count]) })
    };
    // A job's pipeline, or why it has none.
    let prepare = |job: Job| -> Result<Prebuild, String> {
        let g = match job {
            Job::Graphics(i) => &graphics[i],
            Job::Compute(i) => {
                let c = &compute[i];
                let (cs, _) = container(c.stage.program, Some(Primary::Compute(c.shader_hash)))
                    .ok_or("shader not found")?;
                return Ok(Prebuild::compute(&cs));
            }
        };
        let [ps, vs, gs, ..] = g.stages;
        if g.stages[3..].iter().any(Option::is_some) {
            return Err("stages other than vertex, mesh and pixel".into());
        }
        let (first, stage) = vs
            .or(gs)
            .and_then(|s| container(s.program, Some(Primary::Graphics(g.shader_hash))))
            .ok_or("shader not found")?;
        let pixel = match ps {
            Some(ps) => Some(container(ps.program, None).ok_or("shader not found")?.0),
            None => None,
        };
        let mesh = match stage {
            HwStage::Ms => true,
            HwStage::Vs => false,
            _ => return Err("first stage not a vertex or mesh shader".into()),
        };
        let stream = pipelines::stream(g, &named, &first, mesh, pixel.as_deref()).map_err(|u| {
            match u {
                Unnamed::DepthStencil => "depth-stencil state not named",
                Unnamed::Rasterizer => "rasteriser state not named",
                Unnamed::Blend => "blend state not named",
                Unnamed::Metadata => "metadata not read",
            }
            .to_owned()
        })?;
        // SAFETY: the stream points at `first`, `pixel` and its own input
        // elements, all alive here.
        Ok(unsafe { Prebuild::graphics(stream.bytes()) })
    };

    // A job's shaders as a run names them (diagnostics).
    let shaders_of = |job: Job| {
        let stages: Vec<_> = match job {
            Job::Compute(i) => vec![(
                "compute",
                compute[i].stage.program,
                Some(Primary::Compute(compute[i].shader_hash)),
            )],
            Job::Graphics(i) => {
                let [ps, vs, gs, ..] = graphics[i].stages;
                let hash = Some(Primary::Graphics(graphics[i].shader_hash));
                let first = vs
                    .map(|s| ("vertex", s.program, hash))
                    .or(gs.map(|s| ("mesh", s.program, hash)));
                first
                    .into_iter()
                    .chain(ps.map(|s| ("pixel", s.program, None)))
                    .collect()
            }
        };
        let names: Vec<String> = (stages.iter())
            .map(|(stage, code, primary)| match lookup_hash(code, *primary) {
                Some((_, h)) => format!("{stage} {h}"),
                None => format!("{stage} ?"),
            })
            .collect();
        format!("[{}]", names.join(", "))
    };
    let mut jobs: Vec<Job> = (0..compute.len()).map(Job::Compute).collect();
    jobs.extend((0..graphics.len()).map(Job::Graphics));
    // Diagnostics, `VICTORY_PREBUILD_KEYS=<file>`: make nothing, list each
    // record's key (what a run lists pipelines by) with its shaders and
    // state as a run's `VICTORY_PIPELINE_LOG` has them, or why it is
    // skipped; and the records each source named states for.
    let keys_only = std::env::var_os("VICTORY_PREBUILD_KEYS");
    if keys_only.is_some() {
        let mut by: BTreeMap<(&str, &str), usize> = BTreeMap::new();
        for g in &graphics {
            for (kind, h) in [
                ("depth-stencil", g.depth_stencil_hash),
                ("rasteriser", g.rasterizer_hash),
                ("blend", g.blend_hash),
            ] {
                *by.entry((kind, named.source.get(&h).copied().unwrap_or("not named")))
                    .or_default() += 1;
            }
        }
        for ((kind, source), n) in by {
            println!("  {kind:<14} {source:<12} {n:>6} records");
        }
    }
    let skipped: Mutex<BTreeMap<String, usize>> = Mutex::new(BTreeMap::new());
    let skip = |why: String| *lock(&skipped).entry(why).or_default() += 1;
    // First every record's key, which needs no translation: what a
    // pre-build put in the library is skipped, what an earlier run drew
    // with is made first.
    let lines: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let todo: Mutex<Vec<(bool, Job)>> = Mutex::new(Vec::new());
    let listed = AtomicUsize::new(0);
    if keys_only.is_some() {
        each(&jobs, threads, |job| match prepare(job) {
            Ok(p) => lock(&lines).push(format!("{}\t{job:?}\t{}", p.key(), p.describe())),
            Err(why) => {
                lock(&lines).push(format!("-\t{job:?}\t{}\t{why}", shaders_of(job)));
                skip(why);
            }
        });
    } else {
        each(&jobs, threads, |job| match key_of(job) {
            Ok(key) if pipeline::prebuilt(&key) => drop(listed.fetch_add(1, Ordering::Relaxed)),
            Ok(key) => lock(&todo).push((!pipeline::wanted_early(&key), job)),
            Err(why) => skip(why),
        });
    }
    let listed = listed.into_inner();
    if let Some(file) = keys_only {
        std::fs::write(file, lock(&lines).join("\n"))?;
    }
    let mut todo = todo.into_inner().unwrap_or_else(|e| e.into_inner());
    todo.sort_by_key(|&(later, _)| later);
    let todo: Vec<Job> = todo.into_iter().map(|(_, job)| job).collect();

    let next = AtomicUsize::new(0);
    let (made0, loaded0) = pipeline::pipeline_counts();
    let counts = || {
        let (made, loaded) = pipeline::pipeline_counts();
        (made - made0, loaded - loaded0)
    };
    std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads.min(todo.len()))
            .map(|_| {
                s.spawn(|| {
                    while let Some(&job) = todo.get(next.fetch_add(1, Ordering::Relaxed)) {
                        if let Err(why) = prepare(job).and_then(|p| p.make()) {
                            skip(why);
                        }
                    }
                })
            })
            .collect();
        let (n, what) = (todo.len(), "pipelines to make");
        crate::translate::report(pack_path, n, what, &next, started, &workers, || {
            let (made, loaded) = counts();
            format!("; {made} made, {loaded} found in the library")
        });
    });
    pipeline::save_pipeline_cache();
    let (made, loaded) = counts();
    println!(
        "pipelines    {} in the pack, {:.1} s: {made} made, {} already in the library",
        jobs.len(),
        started.elapsed().as_secs_f64(),
        listed + loaded as usize,
    );
    for (why, n) in lock(&skipped).iter() {
        println!("  {n:>7}  skipped: {why}");
    }
    Ok(())
}

/// Runs `f` on every job, on `threads` threads.
fn each(jobs: &[Job], threads: usize, f: impl Fn(Job) + Sync) {
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads.min(jobs.len()) {
            s.spawn(|| {
                while let Some(&job) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                    f(job);
                }
            });
        }
    });
}

/// The exact bytes supplied to D3D12 and hashed for its prebuild key.
fn container_bytes(bytes: &[u8]) -> Option<&[u8]> {
    let size = u32::from_le_bytes(bytes.get(0x18..0x1c)?.try_into().ok()?) as usize;
    bytes.get(..size).filter(|bytes| !bytes.is_empty())
}

/// The hash instructions each native program of a container starts with.
fn program_ids(bytes: &[u8]) -> Vec<([u8; 24], HwStage)> {
    let Ok(c) = Container::parse(bytes) else {
        return Vec::new();
    };
    let Some(part) = c.part(FourCC::DXIL).or(c.part(FourCC::ACCC)) else {
        return Vec::new();
    };
    native_programs(part.data)
        .map(|programs| {
            (programs.iter())
                .filter_map(|p| Some((shaderpack::program_id(&p.code)?, p.stage)))
                .collect()
        })
        .unwrap_or_default()
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}
