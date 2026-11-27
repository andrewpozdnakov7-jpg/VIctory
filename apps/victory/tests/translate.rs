use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn translate_rejects_zero_workers() {
    let input = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    for extra_args in [&[][..], &["--cache"][..]] {
        let output = Command::new(env!("CARGO_BIN_EXE_victory"))
            .arg("translate")
            .arg(&input)
            .args(["--jobs", "0"])
            .args(extra_args)
            .output()
            .expect("run victory translate");

        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.code(), Some(2), "{extra_args:?}: {stderr}");
        assert!(stderr.contains("--jobs"), "{extra_args:?}: {stderr}");
    }
}

#[test]
fn warm_up_rejects_missing_input_before_creating_cache() {
    let valid = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let scratch = std::env::temp_dir().join(format!(
        "victory-warm-input-{}-{}",
        std::process::id(),
        timestamp.as_nanos()
    ));
    fs::create_dir(&scratch).unwrap();
    let missing = scratch.join("missing-shader-input");
    let cache = scratch.join("cache");
    let data = scratch.join("data");
    let results: Vec<_> = [&[][..], &["--pipelines"][..]]
        .into_iter()
        .map(|extra_args| {
            let output = Command::new(env!("CARGO_BIN_EXE_victory"))
                .current_dir(&scratch)
                .env_remove("VICTORY_CONFIG")
                .env("VICTORY_CACHE_DIR", &cache)
                .env("VICTORY_DATA_DIR", &data)
                .args(["translate", "--cache"])
                .args(extra_args)
                .args([&valid, &missing])
                .output()
                .expect("run shader warm-up");
            (extra_args, output, cache.exists())
        })
        .collect();
    fs::remove_dir_all(&scratch).unwrap();

    for (extra_args, output, cache_created) in results {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "{extra_args:?}: {output:?}");
        assert!(
            stderr.contains("cannot access shader warm-up input")
                && stderr.contains("missing-shader-input"),
            "{extra_args:?}: {stderr}"
        );
        assert!(!cache_created, "{extra_args:?}: cache was created");
    }
}
