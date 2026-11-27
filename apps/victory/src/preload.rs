//! Resumable offline warm-up supervisor.
//!
//! `translate --cache --pipelines` already keeps completed work in the cache.
//! This module runs it in a child process and retries after a crash: each
//! child starts from the library the previous one managed to save, so a
//! compiler failure does not force the whole warm-up to begin again.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail, ensure};

/// Stop after this many consecutive attempts with no new pipeline keys.
const STAGNANT_LIMIT: usize = 2;

/// Run the existing warm-up until it finishes or makes no further progress.
pub fn run(paths: &[PathBuf], jobs: Option<usize>, attempts: usize) -> Result<()> {
    ensure!(attempts > 0, "--attempts must be greater than zero");

    let exe = std::env::current_exe().context("cannot find the victory executable")?;
    let progress = victory_core::Config::load()?
        .cache_subdir("pipelines")?
        .join("prebuilt-pipelines.txt");
    let mut stagnant = 0;

    for attempt in 1..=attempts {
        let before = count_keys(&progress);
        eprintln!("preload: attempt {attempt}/{attempts}");

        let mut child = Command::new(&exe);
        child.arg("translate").arg("--cache").arg("--pipelines");
        if let Some(jobs) = jobs {
            child.arg("--jobs").arg(jobs.to_string());
        }
        child.args(paths);

        let status = child
            .status()
            .with_context(|| format!("cannot start {}", exe.display()))?;
        if status.success() {
            return Ok(());
        }

        let after = count_keys(&progress);
        stagnant = stagnant_after(before, after, stagnant);
        eprintln!("preload: child exited {status}; pipeline keys {before} -> {after}");
        if stagnant >= STAGNANT_LIMIT {
            bail!(
                "preload stopped after {attempt} attempt(s) with no new pipeline keys; \
                 try --jobs 1 to isolate a failing translation"
            );
        }
    }

    bail!("preload did not finish after {attempts} attempt(s)")
}

/// Complete pipeline keys in the saved list. Its first line is a cache tag,
/// not a pipeline: `library.rs` writes the tag followed by the keys, without
/// a trailing newline.
fn count_keys(path: &Path) -> usize {
    match std::fs::read(path) {
        Ok(bytes) => String::from_utf8_lossy(&bytes)
            .lines()
            .count()
            .saturating_sub(1),
        Err(_) => 0,
    }
}

/// Consecutive attempts without progress, and whether to stop now.
fn stagnant_after(before: usize, after: usize, previous: usize) -> usize {
    if after > before { 0 } else { previous + 1 }
}

#[cfg(test)]
mod tests {
    use super::{STAGNANT_LIMIT, count_keys, run, stagnant_after};

    fn temp_file(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("victory-preload-{}-{name}", std::process::id()))
    }

    #[test]
    fn zero_attempts_are_refused_before_any_child_starts() {
        assert!(run(&[], None, 0).is_err());
    }

    #[test]
    fn a_missing_list_has_no_keys() {
        let path = temp_file("missing");
        let _ = std::fs::remove_file(&path);
        assert_eq!(count_keys(&path), 0);
    }

    #[test]
    fn a_tag_only_list_has_no_keys() {
        let path = temp_file("tag-only");
        std::fs::write(&path, b"cache-tag").unwrap();
        assert_eq!(count_keys(&path), 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn progress_counts_keys_without_a_trailing_newline() {
        let path = temp_file("keys");
        std::fs::write(&path, b"cache-tag\nalpha\nbeta").unwrap();
        assert_eq!(count_keys(&path), 2);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn new_keys_reset_the_stagnant_streak() {
        assert_eq!(stagnant_after(1, 3, 1), 0);
    }

    #[test]
    fn two_attempts_without_new_keys_stop_the_supervisor() {
        let mut stagnant = 0;
        stagnant = stagnant_after(2, 2, stagnant);
        assert!(stagnant < STAGNANT_LIMIT);
        stagnant = stagnant_after(2, 2, stagnant);
        assert_eq!(stagnant, STAGNANT_LIMIT);
    }
}
