# Linux and Wine (experimental)

VIctory is a Windows program. On Linux we cross-compile the Windows build and run it
under Wine, with vkd3d-proton supplying D3D12 and DXVK supplying DXGI. This is a tester
setup, not a supported release: on an RTX 5090 the opening scene renders lit at about
1 fps, while some compute passes are still refused on 32-lane NVIDIA GPUs. See
[architecture.md](architecture.md) for how
the translation works. Game files stay outside the repository.

## What we test with

| Part | What works for us |
|------|-------------------|
| Wine | 11.22, in a dedicated prefix (`target/wine-prefix`) |
| D3D12 | vkd3d-proton 3.1-83 (`v3.1-83-gcae8353ed`) from GE-Proton11-14 |
| DXGI | DXVK's `dxgi.dll` from GE-Proton11-14; Wine's own DXGI crashes in `wined3d` on the first present |
| Host GPU | NVIDIA RTX 5090; a Vulkan ICD and a driver that supports shader model 6.6 and mesh shaders |
| Runner | Headless gamescope for runs with no window on the desktop |

Older vkd3d-proton builds (Proton 11.0's, for example) crash while creating the first
world pipelines; use the GE-Proton11-14 DLLs.

## Building on Linux

Install clang-cl, lld-link and llvm-lib, make an `xwin` splat of the MSVC CRT and
Windows SDK (`xwin --accept-license splat --output ~/.xwin`: this accepts Microsoft's
licence), and use the pinned Rust toolchain, whose `rust-toolchain.toml` names the
`x86_64-pc-windows-msvc` target (`rustup target add x86_64-pc-windows-msvc` adds it to a
toolchain that lacks it). The splat is only a build input; no host
or game paths belong in the repository. Point the usual target variables at it:

```sh
X=$HOME/.xwin  # or the folder given to xwin splat
export CC_x86_64_pc_windows_msvc=clang-cl CXX_x86_64_pc_windows_msvc=clang-cl
export AR_x86_64_pc_windows_msvc=llvm-lib
export CFLAGS_x86_64_pc_windows_msvc="--target=x86_64-pc-windows-msvc -Wno-unused-command-line-argument /imsvc$X/crt/include /imsvc$X/sdk/include/ucrt /imsvc$X/sdk/include/um /imsvc$X/sdk/include/shared"
export CXXFLAGS_x86_64_pc_windows_msvc="$CFLAGS_x86_64_pc_windows_msvc"
export CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=lld-link
export CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS="-Lnative=$X/crt/lib/x86_64 -Lnative=$X/sdk/lib/um/x86_64 -Lnative=$X/sdk/lib/ucrt/x86_64"
cargo build --profile diag -p victory --features diagnostics --target x86_64-pc-windows-msvc
```

`CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUNNER=wine` is useful when running tests by hand.
The `wine-test` xtask below sets these variables itself; it comes from `#1218`, which is
not merged.

## Prefix setup

Make a prefix of your own, turn off crash dialogs (otherwise a crashed test leaves
`winedbg` waiting forever), and put the GE-Proton11-14 DLLs in it:

```sh
export WINEPREFIX=$PWD/target/wine-prefix
wineboot -i
wine reg add 'HKCU\Software\Wine\WineDbg' /v ShowCrashDialog /t REG_DWORD /d 0 /f
cp "$GE/files/lib/wine/vkd3d-proton/x86_64-windows/d3d12.dll" \
  "$GE/files/lib/wine/vkd3d-proton/x86_64-windows/d3d12core.dll" \
  "$WINEPREFIX/drive_c/windows/system32/"
cp "$GE/files/lib/wine/dxvk/x86_64-windows/dxgi.dll" "$WINEPREFIX/drive_c/windows/system32/"
export WINEDLLOVERRIDES="d3d12,d3d12core,dxgi=n"
```

`$GE` is the GE-Proton11-14 folder. `#1218`'s test runner does the same prefix setup
with `--vkd3d-proton <dir>` and `--dxvk <dir>`.

## Running the title

The title's code needs AVX2 ([the README's requirements](../README.md#getting-started));
`victory run --execute` refuses to start without it. Pick the GPU with its Vulkan ICD
(`VK_DRIVER_FILES`) and, for DXVK, `DXVK_FILTER_DEVICE_NAME`. Set `VICTORY_CACHE_DIR` and
`VICTORY_DATA_DIR` to writable folders outside the repository. Keep the title's
executable path in Wine's `Z:` form, and start from a working directory outside the
repository too: vkd3d-proton writes `vkd3d-proton.cache` there. Check the paths with
`wine "$VICTORY_EXE" config` (it exits nonzero when one violates the asset policy), then
run. In this example, the launch stops if the directory setup or path check fails:

```sh
(
  set -e
  mkdir -p ~/victory-run
  cd ~/victory-run   # vkd3d-proton's cache lands here
  VICTORY_EXE='Z:\path\to\repo\target\x86_64-pc-windows-msvc\diag\victory.exe'
  GAME_DIR='Z:\path\to\game'
  TITLE_EXE="$GAME_DIR\\<the title's exe>"
  export VICTORY_GAME_DIR="$GAME_DIR"
  export VICTORY_CACHE_DIR='Z:\path\to\cache' VICTORY_DATA_DIR='Z:\path\to\data'
  export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
  export DXVK_FILTER_DEVICE_NAME="RTX 5090" WINEDEBUG=-all
  wine "$VICTORY_EXE" config
  gamescope --backend headless -W 1920 -H 1080 -w 1920 -h 1080 \
    --prefer-vk-device 10de:2b85 -- \
    sh -c 'wine "$0" run "$1" --execute' "$VICTORY_EXE" "$TITLE_EXE"
)
```

`--prefer-vk-device` takes your GPU's PCI id; omit it if the ICD selects the right
device. `Z:` is Wine's view of the host filesystem. A headless gamescope run has no
window on the desktop and can be captured and logged in its own directory.

## Caches (important on NVIDIA)

**Never set `VKD3D_SHADER_CACHE_PATH=0`.** Leave it unset: vkd3d-proton writes
`vkd3d-proton.cache` (and `vkd3d-proton.cache.write`) in the current working
directory by default, so launch from a stable directory. A custom directory is a
normal host path. With the cache off, every pipeline load becomes a driver compile
(median about 245 ms in our runs): one run had about 1,000 pipelines still waiting
at frame 1400 and started with roughly five minutes of backlog. Draws are dropped
until the backlog drains. Whether an early frame looks lit or dark in two runs of
the same build came down to this cache being warm, not to a separate lighting bug.

NVIDIA's own shader disk cache also evicted title pipelines at its default size. Give
it room and keep it:

```sh
export __GL_SHADER_DISK_CACHE_SIZE=12884901888
export __GL_SHADER_DISK_CACHE_SKIP_CLEANUP=1
```

The observed cache size grew to about 700 MB on this setup.

## NVIDIA specifics

* The RTX 5090 runs 32-lane waves. `#439` handles one console wave sharing LDS
  (groups up to 64 threads) in the console flow; pair it with `#2118` so a partly
  built BVH structure cannot run. The separate `#2176` → `#2711` → `#2925` stack is
  optional for groups of more than 64 threads: it uses `VICTORY_WAVE32_APPROX=1`,
  which approximates some cross-half operations. Combined with `#439`, it needs care:
  the `#2925` phase plan must skip programs the console flow already handles, or it
  rejects many of them (see the report on `#2925`).
* NVIDIA's view heap is capped ([issue #747](https://github.com/Marvisak/VIctory/issues/747)):
  with `2^16` dynamic slots, world tables run out and root views are dropped. The
  issue measured `2^18` slots removing those "no dynamic slots" drops (RTX 4070 Ti:
  about 1,460 → 2,690 draws per frame).
* Do not run a GPU test while the title is running. They share the GPU, and an Xid
  from a test can take down the game run as well.

## Unmerged PRs needed on NVIDIA/Wine (2026-11-26)

These were all **open** upstream on that date. This is the stack behind our working
NVIDIA/Wine build, not plain `main`; re-check their state before copying the list.

| PR | Title | Without it |
|----|-------|------------|
| [#2723](https://github.com/Marvisak/VIctory/pull/2723) | Shader: key every heap handle cache by element type | A float-typed handle is reused for an atomic; the resulting invalid SPIR-V crashes NVIDIA's `vkCreateComputePipelines` |
| [#439](https://github.com/Marvisak/VIctory/pull/439) | Shaders: one console wave sharing LDS runs in the console flow on 32-lane hosts | Programs that share LDS inside one console wave stay refused, including BVH-builder stages; needs `#2118` with it |
| [#2118](https://github.com/Marvisak/VIctory/pull/2118) | GPU: the BVH builder runs whole or not at all | Missing builder stages write wrong bounds into title memory and the title can crash |
| [#2703](https://github.com/Marvisak/VIctory/pull/2703) | GPU: a root constant's dword 4 is a value, not a view offset | Builder dispatches are refused with "no root view, or outside title buffers" |
| [#2101](https://github.com/Marvisak/VIctory/pull/2101) | GPU: castable formats the host has | One format vkd3d-proton does not know makes it refuse every castable list, so textures of one memory get separate host textures and copies |
| [#2312](https://github.com/Marvisak/VIctory/pull/2312) | GPU, Shaders: spread compute dispatches past 65,535 groups in x across z | Direct and indirect compute dispatches wider than 65,535 groups in x are cut short |
| [#2969](https://github.com/Marvisak/VIctory/pull/2969) | GPU: shaders of our own that Wine's compiler builds too | Wine's compiler rejects `RWByteAddressBuffer` methods and ignores `[RootSignature]`, so the indirect clamp and depth conversion fail and indirect draws are dropped |
| [#1722](https://github.com/Marvisak/VIctory/pull/1722) | GPU: the host pipeline library works under Wine | A non-null empty library blob is rejected; no `host-pipelines.bin` is written and no pipeline is started early on the next run |
| [#2968](https://github.com/Marvisak/VIctory/pull/2968) | Shaders: f32 floor, ceil and fract flush denormal operands | vkd3d-proton on NVIDIA keeps denormals where the console flushes them; `rounding_matches_the_hardware` fails and floor/ceil/fract DXIL differs |

The tested base also carries unrelated fixes; this table lists only the ones
specific to NVIDIA/Wine.

## Tests and known gaps

`cargo xtask wine-test` (PR `#1218`, open) cross-builds and runs the Windows test suite
under Wine in `target/wine-prefix`; it leaves the host `.cargo/config.toml` and the
user's `~/.wine` alone. With both DLL sets:

```sh
cargo xtask wine-test \
  --vkd3d-proton "$GE/files/lib/wine/vkd3d-proton/x86_64-windows" \
  --dxvk "$GE/files/lib/wine/dxvk/x86_64-windows" \
  -- -p victory-gpu --lib resource:: -- \
  --skip resource::tests::a_list_waiting_for_another_queues_write_reads_what_it_wrote
```

Arguments after `--` replace the defaults (`--workspace --exclude xtask --no-fail-fast`);
the module filter and the skip above go straight to `cargo test`. The runner also builds
the loader's `hle-probe` fixture for the test that cannot start cargo from inside Wine.

DXIL validation needs a Windows SDK's `dxil.dll`, which a Wine prefix does not have:
point `VICTORY_DXIL` at an x64 copy in `Z:` form ([the README's Windows shader-test
validator](../README.md#windows-shader-test-validator)). Without it `translate --validate`
cannot load a validator and the tests that use one skip their validation; the translated
shaders do not need it at runtime.

Known Wine/vkd3d-proton gaps and the fixes to look for (states as of 2026-11-26):

| Gap | PR |
|-----|----|
| The indirect clamp fails under Wine's compiler: `no argument clamp` and `not bounded` drops | `#2969` (open) |
| `modules::net::tests::only_loopback_is_reachable` | `#3091` (open) |
| Empty or rejected pipeline library (`a_rejected_blob_gives_an_empty_library`, `pipelines_load_on_many_threads_at_once`) | `#1722` (open) |
| `emit::tests::stub_pipeline_is_created` (checksum assertions are Windows-only) | `#1266` (closed) |
| `emit::semantics_tests::rounding_matches_the_hardware` (denormal floor/ceil/fract) | `#2968` (open) |
| `hosts_that_do_not_fuse_are_named`, `a_heap_buffer_read_where_no_buffer_view_is_reads_zeros` | `#2973` (open) |
| Page-table, NTFS hard-link and capped-view-heap tests, skipped or adapted under Wine | `#2973` (open) |

* Run `victory-gpu` one top-level module per process: the whole suite together can take
  the headless X server down after about two minutes. Skip
  `resource::tests::a_list_waiting_for_another_queues_write_reads_what_it_wrote`, which
  hangs under Wine on both `main` and `#2969`.

These are Wine results; the Windows suite remains authoritative.
