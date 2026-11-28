//! Debugging aid: `VICTORY_AUDIO_CAPTURE=<file.wav>` records what the title
//! plays, 48 kHz stereo 16-bit, through process loopback: our own process
//! tree only, never the rest of the host's mix. Stretches where nothing was
//! played are written as silence, so the file keeps real time.
//! The first packet log carries the WAV start as Unix time in 100 ns units
//! when the packet has a usable timestamp.
//!
//! The recorder runs in a child copy of this executable: an audio client
//! in the title's process raises that process's timer resolution, which
//! changes how the title's own waits behave.

use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{E_ILLEGAL_METHOD_CALL, E_POINTER, HANDLE, WAIT_OBJECT_0, WAIT_TIMEOUT};
use windows::Win32::Media::Audio::{
    AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY, AUDCLNT_BUFFERFLAGS_SILENT,
    AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR, AUDCLNT_SHAREMODE_SHARED,
    AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM, AUDCLNT_STREAMFLAGS_LOOPBACK,
    AUDIOCLIENT_ACTIVATION_PARAMS, AUDIOCLIENT_ACTIVATION_PARAMS_0,
    AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK, AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS,
    ActivateAudioInterfaceAsync, IActivateAudioInterfaceAsyncOperation,
    IActivateAudioInterfaceCompletionHandler, IActivateAudioInterfaceCompletionHandler_Impl,
    IAudioCaptureClient, IAudioClient, PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE,
    VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK, WAVE_FORMAT_PCM, WAVEFORMATEX,
};
use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx};
use windows::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
use windows::Win32::System::SystemInformation::GetSystemTimePreciseAsFileTime;
use windows::Win32::System::Threading::{
    OpenProcess, PROCESS_SYNCHRONIZE, Sleep, WaitForSingleObject,
};
use windows::core::{Error, HRESULT, IUnknown, Interface, Ref, implement};

const RATE: u32 = 48_000;
const FRAME_BYTES: usize = 4;

/// Activation completes asynchronously; the recording thread polls for it.
#[implement(IActivateAudioInterfaceCompletionHandler)]
struct Activated;

impl IActivateAudioInterfaceCompletionHandler_Impl for Activated_Impl {
    fn ActivateCompleted(
        &self,
        _: Ref<IActivateAudioInterfaceAsyncOperation>,
    ) -> windows::core::Result<()> {
        Ok(())
    }
}

/// Names the process to record, in the recorder's environment.
const TARGET: &str = "VICTORY_AUDIO_CAPTURE_OF";

/// Starts the recorder if `VICTORY_AUDIO_CAPTURE` names a file. With this
/// module's debug log, also logs once a second what a `Sleep(1)` takes in
/// this process (the title's millisecond waits get the same).
pub fn start() {
    if tracing::enabled!(tracing::Level::DEBUG) {
        let _ = std::thread::Builder::new()
            .name("victory timing probe".into())
            .spawn(|| {
                loop {
                    std::thread::sleep(Duration::from_secs(1));
                    let at = Instant::now();
                    // SAFETY: plain wait.
                    unsafe { Sleep(1) };
                    let ms = at.elapsed().as_secs_f32() * 1000.0;
                    tracing::debug!(ms, "Sleep(1) took");
                }
            });
    }
    if std::env::var_os("VICTORY_AUDIO_CAPTURE").is_some()
        && let Err(e) = std::env::current_exe().and_then(|exe| {
            std::process::Command::new(exe)
                .env(TARGET, std::process::id().to_string())
                .spawn()
        })
    {
        tracing::warn!("audio capture not started: {e}");
    }
}

/// In the recorder (a child started by [`start`]): records until the
/// target exits and returns true; false in any other process.
pub fn child() -> bool {
    let (Some(pid), Some(path)) = (
        std::env::var(TARGET).ok().and_then(|p| p.parse().ok()),
        std::env::var_os("VICTORY_AUDIO_CAPTURE"),
    ) else {
        return false;
    };
    if let Err(e) = record(pid, Path::new(&path)) {
        tracing::warn!("audio capture stopped: {e}");
    }
    true
}

fn header(frames: u64) -> [u8; 44] {
    let data = u32::try_from(frames * FRAME_BYTES as u64).unwrap_or(u32::MAX - 36);
    let mut h = [0u8; 44];
    let fields: [(&[u8], usize); 11] = [
        (b"RIFF", 0),
        (&(36 + data).to_le_bytes(), 4),
        (b"WAVEfmt ", 8),
        (&16u32.to_le_bytes(), 16),
        (&1u16.to_le_bytes(), 20),
        (&2u16.to_le_bytes(), 22),
        (&RATE.to_le_bytes(), 24),
        (&(RATE * FRAME_BYTES as u32).to_le_bytes(), 28),
        (&(FRAME_BYTES as u16).to_le_bytes(), 32),
        (&16u16.to_le_bytes(), 34),
        (b"data", 36),
    ];
    for (bytes, at) in fields {
        h[at..at + bytes.len()].copy_from_slice(bytes);
    }
    h[40..].copy_from_slice(&data.to_le_bytes());
    h
}

/// Convert the packet's WASAPI QPC timestamp to Unix time in 100 ns units.
fn packet_unix_100ns(qpc: u64, before: u64, after: u64, filetime: u64) -> Option<u64> {
    const UNIX_EPOCH: u64 = 116_444_736_000_000_000;
    let sample_span = after.checked_sub(before)?;
    let packet_age = (before + sample_span / 2).checked_sub(qpc)?;
    filetime.checked_sub(UNIX_EPOCH)?.checked_sub(packet_age)
}

fn log_first_packet(qpc: u64, flags: u32) {
    if flags & AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR.0 as u32 != 0 {
        tracing::info!("audio capture: first packet has no reliable timestamp");
        return;
    }
    match sample_packet_time(qpc) {
        Ok(Some((unix_100ns, sample_span))) => tracing::info!(
            wav_start_unix_100ns = unix_100ns,
            clock_sample_100ns = sample_span,
            "audio capture: first packet"
        ),
        Ok(None) => tracing::info!("audio capture: first packet has no reliable timestamp"),
        Err(error) => tracing::debug!(%error, "audio capture: first packet clock sample failed"),
    }
}

fn sample_packet_time(qpc: u64) -> windows::core::Result<Option<(u64, u64)>> {
    const TICKS_PER_SECOND: i128 = 10_000_000;
    let (mut frequency, mut before, mut after) = (0i64, 0i64, 0i64);
    // Bracket the wall-clock read to estimate its QPC and log the sample span.
    // SAFETY: QPC functions write one value to each valid out pointer.
    let time = unsafe {
        QueryPerformanceFrequency(&mut frequency)?;
        QueryPerformanceCounter(&mut before)?;
        let time = GetSystemTimePreciseAsFileTime();
        QueryPerformanceCounter(&mut after)?;
        time
    };
    if frequency <= 0 || before < 0 || after < before {
        return Ok(None);
    }
    let to_ticks =
        |counter| (i128::from(counter) * TICKS_PER_SECOND / i128::from(frequency)) as u64;
    let (before, after) = (to_ticks(before), to_ticks(after));
    let filetime = (u64::from(time.dwHighDateTime) << 32) | u64::from(time.dwLowDateTime);
    Ok(packet_unix_100ns(qpc, before, after, filetime).map(|unix| (unix, after - before)))
}

/// Waits for the client, or returns early when the target exits.
///
/// # Safety
/// The target process handle must stay open during the wait.
unsafe fn wait_for_client(
    operation: &IActivateAudioInterfaceAsyncOperation,
    target: HANDLE,
) -> windows::core::Result<Option<IAudioClient>> {
    loop {
        let (mut hr, mut client) = (HRESULT(0), None::<IUnknown>);
        // SAFETY: valid storage for the activation result and interface.
        match unsafe { operation.GetActivateResult(&mut hr, &mut client) } {
            Ok(()) => {
                hr.ok()?;
                let client = client.ok_or_else(|| Error::new(E_POINTER, "no audio client"))?;
                return client.cast().map(Some);
            }
            Err(error) if error.code() == E_ILLEGAL_METHOD_CALL => {
                // SAFETY: the caller keeps the target process handle open.
                match unsafe { WaitForSingleObject(target, 10) } {
                    WAIT_OBJECT_0 => return Ok(None),
                    WAIT_TIMEOUT => {}
                    _ => return Err(Error::from_thread()),
                }
            }
            Err(error) => return Err(error),
        }
    }
}

fn record(pid: u32, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // SAFETY: COM calls with parameters built here; buffers are used only
    // between `GetBuffer` and `ReleaseBuffer`.
    unsafe {
        CoInitializeEx(None, COINIT_MULTITHREADED).ok()?;
        let target = OpenProcess(PROCESS_SYNCHRONIZE, false, pid)?;
        let params = AUDIOCLIENT_ACTIVATION_PARAMS {
            ActivationType: AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
            Anonymous: AUDIOCLIENT_ACTIVATION_PARAMS_0 {
                ProcessLoopbackParams: AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS {
                    TargetProcessId: pid,
                    ProcessLoopbackMode: PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE,
                },
            },
        };
        /// A `PROPVARIANT` holding a blob (`VT_BLOB`).
        #[repr(C)]
        struct Blob {
            vt: u16,
            reserved: [u16; 3],
            size: u32,
            data: *const AUDIOCLIENT_ACTIVATION_PARAMS,
        }
        let blob = Blob {
            vt: 0x41,
            reserved: [0; 3],
            size: size_of_val(&params) as u32,
            data: &params,
        };
        let handler: IActivateAudioInterfaceCompletionHandler = Activated.into();
        let operation = ActivateAudioInterfaceAsync(
            VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
            &IAudioClient::IID,
            Some((&raw const blob).cast()),
            &handler,
        )?;
        let Some(client) = wait_for_client(&operation, target)? else {
            return Ok(());
        };
        let format = WAVEFORMATEX {
            wFormatTag: WAVE_FORMAT_PCM as u16,
            nChannels: 2,
            nSamplesPerSec: RATE,
            nAvgBytesPerSec: RATE * FRAME_BYTES as u32,
            nBlockAlign: FRAME_BYTES as u16,
            wBitsPerSample: 16,
            cbSize: 0,
        };
        client.Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM,
            10_000_000,
            0,
            &format,
            None,
        )?;
        let capture: IAudioCaptureClient = client.GetService()?;
        let mut file = File::create(path)?;
        file.write_all(&header(0))?;
        client.Start()?;
        tracing::info!(path = %path.display(), "recording the title's audio");

        let (mut written, mut first) = (0u64, None);
        let mut patched = Instant::now();
        loop {
            std::thread::sleep(Duration::from_millis(10));
            let ended = WaitForSingleObject(target, 0) == WAIT_OBJECT_0;
            while capture.GetNextPacketSize()? > 0 {
                let (mut data, mut frames, mut flags, mut qpc) = (std::ptr::null_mut(), 0, 0, 0);
                capture.GetBuffer(&mut data, &mut frames, &mut flags, None, Some(&mut qpc))?;
                if frames == 0 {
                    break;
                }
                let first_packet = written == 0;
                // Where this packet belongs, from its time stamp (100 ns).
                let at = (qpc - *first.get_or_insert(qpc)) * u64::from(RATE) / 10_000_000;
                if at > written + u64::from(RATE / 500) {
                    tracing::debug!(
                        ms = (at - written) * 1000 / u64::from(RATE),
                        "audio capture: nothing played"
                    );
                    let silence = vec![0u8; (at - written) as usize * FRAME_BYTES];
                    file.write_all(&silence)?;
                    written = at;
                }
                let bytes = frames as usize * FRAME_BYTES;
                if flags & AUDCLNT_BUFFERFLAGS_SILENT.0 as u32 != 0 {
                    file.write_all(&vec![0u8; bytes])?;
                } else {
                    file.write_all(std::slice::from_raw_parts(data, bytes))?;
                }
                if flags & AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY.0 as u32 != 0 && written > 0 {
                    tracing::debug!("audio capture: discontinuity");
                }
                written += u64::from(frames);
                capture.ReleaseBuffer(frames)?;
                if first_packet {
                    log_first_packet(qpc, flags);
                }
            }
            if ended || patched.elapsed() >= Duration::from_secs(1) {
                // Valid sizes at any time: runs end by being killed.
                file.seek(SeekFrom::Start(0))?;
                file.write_all(&header(written))?;
                file.seek(SeekFrom::End(0))?;
                patched = Instant::now();
            }
            if ended {
                return Ok(());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::packet_unix_100ns;

    #[test]
    fn converts_audio_packet_qpc_to_unix_ticks() {
        let filetime = 116_444_736_000_001_000;
        assert_eq!(packet_unix_100ns(90, 100, 104, filetime), Some(988));
    }

    #[test]
    fn rejects_inconsistent_clock_samples() {
        let filetime = 116_444_736_000_000_100;
        assert_eq!(packet_unix_100ns(110, 100, 104, filetime), None);
        assert_eq!(packet_unix_100ns(90, 104, 100, filetime), None);
        assert_eq!(packet_unix_100ns(0, 1000, 1004, filetime), None);
    }
}

#[cfg(test)]
mod activation_tests {
    use super::*;
    use windows::Win32::Foundation::{CloseHandle, E_FAIL};
    use windows::Win32::Media::Audio::IActivateAudioInterfaceAsyncOperation_Impl;
    use windows::Win32::System::Threading::CreateEventW;
    use windows::core::{IUnknownImpl, OutRef, PCWSTR};

    #[implement(IActivateAudioInterfaceAsyncOperation)]
    struct ActivationError(HRESULT);

    impl IActivateAudioInterfaceAsyncOperation_Impl for ActivationError_Impl {
        fn GetActivateResult(
            &self,
            _: *mut HRESULT,
            _: OutRef<IUnknown>,
        ) -> windows::core::Result<()> {
            Err(Error::from_hresult(self.get_impl().0))
        }
    }

    #[test]
    fn activation_wait_stops_on_target_exit_or_method_failure() {
        // SAFETY: a signaled event acts like an already exited target.
        let target = unsafe { CreateEventW(None, true, true, PCWSTR::null()) }.unwrap();
        let pending: IActivateAudioInterfaceAsyncOperation =
            ActivationError(E_ILLEGAL_METHOD_CALL).into();
        let broken: IActivateAudioInterfaceAsyncOperation = ActivationError(E_FAIL).into();
        // SAFETY: the event stays open until both waits finish.
        let (ended, failed) = unsafe {
            let ended = wait_for_client(&pending, target);
            let failed = wait_for_client(&broken, target);
            CloseHandle(target).unwrap();
            (ended, failed)
        };
        assert!(ended.unwrap().is_none());
        assert_eq!(failed.unwrap_err().code(), E_FAIL);
    }
}
