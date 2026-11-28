//! The console's audio DSP (`chad.dll`): streams that convolve blocks of
//! float samples with an impulse response (reverb), or run other block
//! transforms, on a coprocessor.
//!
//! A title connects once with a memory block for the DSP, activates
//! streams, and submits one command per block (gains, an input and an
//! output buffer). Activation hands back a status block whose second word is
//! the ticket of the last completed command and whose fourth is its result;
//! the title checks it one block later and treats a late command as an
//! overrun. Commands complete when submitted here.
//!
//! Convolution runs on the host with the console's partitioned
//! frequency-domain impulse responses (see [`super::dsp_convolve`]), so
//! reverb returns are wet. The FFT and IFFT stream transforms remain silent:
//! their inputs and outputs are spectra, not audio a title can play.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;
use std::sync::Mutex;

use super::dsp_convolve::Convolver;
use super::xasync::E_INVALIDARG;
use super::xgameruntime::{HRESULT, S_OK};
use crate::{HleModule, hle_export};

const E_POINTER: HRESULT = 0x8000_4003_u32 as i32;
const E_OUTOFMEMORY: HRESULT = 0x8007_000E_u32 as i32;

const CONVOLUTION: u32 = 1;
/// XDspActivationOptions: a per-channel IR and deinterleaved blocks both
/// need a stereo stream; polar format is for FFT/IFFT only.
const DEINTERLEAVED: u32 = 2;
const STEREO_IR: u32 = 8;
const STEREO_OPTIONS: u32 = STEREO_IR | DEINTERLEAVED;
const NOT_FOR_CONVOLUTION: u32 = 4;

pub struct Dsp {
    max_streams: u32,
    active: Mutex<u32>,
}

/// What activation describes.
#[repr(C, packed)]
pub struct StreamDesc {
    kind: u32,
    options: u32,
    frames: u32,
    channels: u32,
    ir_floats: u32,
    ir: *const f32,
}

/// The status block a stream's owner polls.
#[repr(C)]
pub struct Status {
    submitted: u32,
    completed: u32,
    reserved: u32,
    result: HRESULT,
}

pub struct Stream {
    dsp: *mut Dsp,
    frames: u32,
    channels: u32,
    status: Box<Status>,
    /// `None` for the FFT/IFFT transforms, which we do not compute.
    convolver: Option<Convolver>,
    deinterleaved: bool,
}

/// One block of work.
#[repr(C)]
pub struct Command {
    gains: [f32; 2],
    input: *const f32,
    output: *mut f32,
    reserved: u64,
}

pub extern "system" fn XDspConnectWithMaximumStreamLimit(
    base: *mut c_void,
    base_length: u32,
    ir_seconds: u32,
    max_streams: u32,
    max_channels: u32,
    out: *mut *mut Dsp,
) -> HRESULT {
    if base.is_null()
        || !(0x1_0000..=0x1F40_0000).contains(&base_length)
        || ir_seconds > 0x80
        || max_streams > 0x100
        || max_channels > 2
    {
        return E_INVALIDARG;
    }
    if out.is_null() {
        return E_POINTER;
    }
    tracing::info!(
        base_length,
        ir_seconds,
        max_streams,
        max_channels,
        "audio DSP connected"
    );
    let dsp = Box::new(Dsp {
        max_streams,
        active: Mutex::new(0),
    });
    // SAFETY: checked non-null.
    unsafe { out.write(Box::into_raw(dsp)) };
    S_OK
}

pub extern "system" fn XDspDisconnect(dsp: *mut Dsp) -> HRESULT {
    if dsp.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: made by `XDspConnectWithMaximumStreamLimit`; disconnected
    // once, after its streams.
    drop(unsafe { Box::from_raw(dsp) });
    S_OK
}

fn valid(desc: &StreamDesc) -> bool {
    let (kind, options, frames, channels) = (desc.kind, desc.options, desc.frames, desc.channels);
    let shape = if kind == CONVOLUTION {
        (1..=2).contains(&channels)
            && matches!(frames, 0x200 | 0x400)
            && options & NOT_FOR_CONVOLUTION == 0
            // The console's IR is complex pairs, one filter block at least.
            && !desc.ir.is_null()
            && desc.ir_floats >= 2 * frames
    } else {
        channels == 1 && matches!(frames, 0x200 | 0x400 | 0x800)
    };
    shape && (options & STEREO_OPTIONS == 0 || channels == 2)
}

pub extern "system" fn XDspActivate(
    dsp: *mut Dsp,
    desc: *const StreamDesc,
    status: *mut *const Status,
    stream: *mut *mut Stream,
) -> HRESULT {
    if status.is_null() || stream.is_null() {
        return E_POINTER;
    }
    if dsp.is_null() || desc.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null; the title passes its description.
    let desc = unsafe { &*desc };
    if !valid(desc) {
        return E_INVALIDARG;
    }
    let (kind, options, frames, channels) = (desc.kind, desc.options, desc.frames, desc.channels);
    tracing::debug!(
        kind,
        options,
        frames,
        channels,
        "audio DSP stream activated"
    );
    let convolver = if kind == CONVOLUTION {
        // `ir_floats` is one channel's length; a stereo IR holds one per
        // channel (JMAN730/project-flamingo 645abbeb).
        let stereo_ir = options & STEREO_IR != 0;
        let len = desc.ir_floats as usize * if stereo_ir { channels as usize } else { 1 };
        // SAFETY: `valid` checked the pointer; the title's IR holds
        // `ir_floats` floats per IR channel and stays alive in title memory
        // for the stream's lifetime (the console copies it here).
        let ir = unsafe { std::slice::from_raw_parts(desc.ir, len) };
        match Convolver::new(
            ir,
            desc.ir_floats as usize,
            frames as usize,
            channels as usize,
            stereo_ir,
        ) {
            Some(convolver) => Some(convolver),
            None => {
                tracing::warn!(ir_floats = { desc.ir_floats }, frames, channels, options, "audio DSP convolution stream refused");
                return E_INVALIDARG;
            }
        }
    } else {
        None
    };
    // Reserve the stream only once it can be made, so a refused activation
    // does not hold a slot.
    {
        // SAFETY: a connection from `XDspConnectWithMaximumStreamLimit`.
        let dsp = unsafe { &*dsp };
        let mut active = dsp.active.lock().unwrap_or_else(|e| e.into_inner());
        if *active >= dsp.max_streams {
            return E_OUTOFMEMORY;
        }
        *active += 1;
    }
    let new = Box::new(Stream {
        dsp,
        frames: desc.frames,
        channels: desc.channels,
        status: Box::new(Status {
            submitted: 0,
            completed: 0,
            reserved: 0,
            result: S_OK,
        }),
        convolver,
        deinterleaved: options & DEINTERLEAVED != 0,
    });
    // SAFETY: checked non-null.
    unsafe {
        status.write(&*new.status);
        stream.write(Box::into_raw(new));
    }
    S_OK
}

pub extern "system" fn XDspDeactivate(stream: *mut Stream) -> HRESULT {
    if stream.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: made by `XDspActivate`; deactivated once.
    let stream = unsafe { Box::from_raw(stream) };
    // SAFETY: the connection outlives its streams.
    let dsp = unsafe { &*stream.dsp };
    *dsp.active.lock().unwrap_or_else(|e| e.into_inner()) -= 1;
    S_OK
}

/// Completes the command at once. Convolution writes wet output; the FFT
/// and IFFT transforms still return silence.
pub extern "system" fn XDspSubmitCommand(
    stream: *mut Stream,
    command: *const Command,
    ticket: *mut u32,
) -> HRESULT {
    if stream.is_null() || command.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: a stream from `XDspActivate`, the title's command.
    let (stream, command) = unsafe { (&mut *stream, &*command) };
    if !command.output.is_null() {
        let samples = (stream.frames * stream.channels) as usize;
        let mut written = false;
        if let Some(convolver) = &mut stream.convolver
            && !command.input.is_null()
        {
            // SAFETY: the title's input and output buffers hold one block.
            let (input, output) = unsafe {
                (
                    std::slice::from_raw_parts(command.input, samples),
                    std::slice::from_raw_parts_mut(command.output, samples),
                )
            };
            written = convolver.process(
                input,
                output,
                command.gains,
                stream.deinterleaved,
                stream.deinterleaved,
            );
            if written {
                // shortcut: an IR in a layout not seen yet must not blast the
                // mix (one NaN in it makes every sample infinite). NaN and
                // infinity become silence, the rest is held to [-1, 1].
                for s in output.iter_mut() {
                    *s = if s.is_finite() { s.clamp(-1.0, 1.0) } else { 0.0 };
                }
            }
        }
        if !written {
            // SAFETY: the title's output buffer holds one block.
            unsafe { std::ptr::write_bytes(command.output, 0, samples) };
        }
    }
    let status = &mut *stream.status;
    let next = status.submitted.wrapping_add(1);
    // SAFETY: the status block is read by title code on other threads.
    unsafe {
        std::ptr::write_volatile(&mut status.submitted, next);
        std::ptr::write_volatile(&mut status.result, S_OK);
        std::ptr::write_volatile(&mut status.completed, next);
    }
    if !ticket.is_null() {
        // SAFETY: optional out parameter.
        unsafe { ticket.write(next) };
    }
    S_OK
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "dsp",
        dlls: &["chad.dll"],
        exports: vec![
            hle_export!(XDspConnectWithMaximumStreamLimit),
            hle_export!(XDspDisconnect),
            hle_export!(XDspActivate),
            hle_export!(XDspDeactivate),
            hle_export!(XDspSubmitCommand),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolution_stream_as_the_title_drives_it() {
        let mut base = vec![0u8; 0x10_0000];
        let mut dsp = std::ptr::null_mut();
        assert_eq!(
            XDspConnectWithMaximumStreamLimit(
                base.as_mut_ptr().cast(),
                0x10_0000,
                24,
                0x80,
                1,
                &mut dsp
            ),
            S_OK
        );
        let ir = vec![0f32; 0x800];
        // Stereo convolution with option 2, as the title asks for.
        let desc = StreamDesc {
            kind: CONVOLUTION,
            options: 2,
            frames: 0x400,
            channels: 2,
            ir_floats: 0x800,
            ir: ir.as_ptr(),
        };
        let (mut status, mut stream) = (std::ptr::null(), std::ptr::null_mut());
        assert_eq!(XDspActivate(dsp, &desc, &mut status, &mut stream), S_OK);
        let mut output = vec![1f32; 0x800];
        let input = vec![0.5f32; 0x800];
        let command = Command {
            gains: [1.0, 1.0],
            input: input.as_ptr(),
            output: output.as_mut_ptr(),
            reserved: 0,
        };
        let mut ticket = 0;
        assert_eq!(XDspSubmitCommand(stream, &command, &mut ticket), S_OK);
        // SAFETY: the stream's status block.
        let (completed, result) = unsafe { ((*status).completed, (*status).result) };
        assert_eq!((completed, result), (ticket, S_OK));
        assert!(output.iter().all(|&s| s == 0.0));
        assert_eq!(XDspDeactivate(stream), S_OK);

        // Mono streams refuse stereo options; transforms are mono only.
        let bad = StreamDesc {
            channels: 1,
            ..desc
        };
        assert_eq!(
            XDspActivate(dsp, &bad, &mut status, &mut stream),
            E_INVALIDARG
        );
        let transform = StreamDesc {
            kind: 2,
            options: 0,
            channels: 1,
            ir_floats: 0,
            ir: std::ptr::null(),
            ..desc
        };
        assert_eq!(
            XDspActivate(dsp, &transform, &mut status, &mut stream),
            S_OK
        );
        assert_eq!(XDspDeactivate(stream), S_OK);
        assert_eq!(XDspDisconnect(dsp), S_OK);
    }

    #[test]
    fn convolution_through_the_api_returns_wet_audio() {
        const BLOCK: usize = 0x200;
        let mut base = vec![0u8; 0x10_0000];
        let mut dsp = std::ptr::null_mut();
        assert_eq!(
            XDspConnectWithMaximumStreamLimit(
                base.as_mut_ptr().cast(),
                0x10_0000,
                24,
                4,
                2,
                &mut dsp
            ),
            S_OK
        );
        // A unity impulse in the console's frequency-domain layout.
        let mut time = vec![0f32; BLOCK];
        time[0] = 1.0;
        let ir = crate::modules::dsp_convolve::encode_ir(&time, BLOCK);
        let desc = StreamDesc {
            kind: CONVOLUTION,
            options: 0,
            frames: BLOCK as u32,
            channels: 1,
            ir_floats: ir.len() as u32,
            ir: ir.as_ptr(),
        };
        let (mut status, mut stream) = (std::ptr::null(), std::ptr::null_mut());
        assert_eq!(XDspActivate(dsp, &desc, &mut status, &mut stream), S_OK);
        let input = vec![0.75f32; BLOCK];
        let mut output = vec![0f32; BLOCK];
        let command = Command {
            gains: [1.0, 1.0],
            input: input.as_ptr(),
            output: output.as_mut_ptr(),
            reserved: 0,
        };
        let mut ticket = 0;
        assert_eq!(XDspSubmitCommand(stream, &command, &mut ticket), S_OK);
        assert_eq!(ticket, 1);
        // SAFETY: the stream's status block.
        let (completed, result) = unsafe { ((*status).completed, (*status).result) };
        assert_eq!((completed, result), (ticket, S_OK));
        let energy: f32 = output.iter().map(|s| s * s).sum::<f32>() / BLOCK as f32;
        assert!(energy > 0.5, "wet return, energy {energy}");
        assert_eq!(XDspDeactivate(stream), S_OK);
        assert_eq!(XDspDisconnect(dsp), S_OK);
    }

    #[test]
    fn convolution_gain_matches_direct_reference() {
        const BLOCK: usize = 1024;
        const GAINS: [[f32; 2]; 4] = [[0.25, 0.75], [0.75, 1.0], [1.0, 1.0], [1.0, 0.25]];
        let taps = [
            (0, 0.375),
            (1, 0.375),
            (1022, 0.125),
            (1023, 0.125),
            (1024, 0.125),
            (1025, 0.125),
            (2046, -0.0625),
            (2047, -0.0625),
        ];
        let mut ir = vec![0.0; BLOCK * 2];
        for (index, value) in taps {
            ir[index] = value;
        }
        let encoded = crate::modules::dsp_convolve::title_ir(&ir, BLOCK);

        let mut scratch = vec![0u8; 0x1_0000];
        let mut dsp = std::ptr::null_mut();
        assert_eq!(
            XDspConnectWithMaximumStreamLimit(
                scratch.as_mut_ptr().cast(),
                0x1_0000,
                24,
                4,
                1,
                &mut dsp
            ),
            S_OK,
        );
        let mut outcomes = Vec::new();
        for (level, amplitude) in [("quiet", 0.125), ("loud", 0.75)] {
            let input: Vec<f32> = (0..BLOCK * GAINS.len())
                .map(|index| amplitude * [1.0, 0.875, 1.0, 0.75, 1.0][index % 5])
                .collect();
            for (mode, command_gains) in [
                ("unity", [[1.0; 2]; GAINS.len()]),
                ("ramp", GAINS),
                ("double-start", GAINS.map(|[start, _]| [2.0 * start; 2])),
            ] {
                let desc = StreamDesc {
                    kind: CONVOLUTION,
                    options: 0,
                    frames: BLOCK as u32,
                    channels: 1,
                    ir_floats: encoded.len() as u32,
                    ir: encoded.as_ptr(),
                };
                let (mut status, mut stream) = (std::ptr::null(), std::ptr::null_mut());
                assert_eq!(XDspActivate(dsp, &desc, &mut status, &mut stream), S_OK);
                let mut actual = Vec::new();
                let mut expected = Vec::new();
                let mut clipped_reference = 0;
                for (block, gains) in command_gains.into_iter().enumerate() {
                    let start = block * BLOCK;
                    let mut output = vec![f32::NAN; BLOCK];
                    let command = Command {
                        gains,
                        input: input[start..start + BLOCK].as_ptr(),
                        output: output.as_mut_ptr(),
                        reserved: 0,
                    };
                    let mut ticket = 0;
                    assert_eq!(XDspSubmitCommand(stream, &command, &mut ticket), S_OK);
                    assert_eq!(ticket, block as u32 + 1);
                    assert_eq!(
                        unsafe { ((*status).completed, (*status).result) },
                        (ticket, S_OK)
                    );
                    actual.extend(output);
                    for offset in 0..BLOCK {
                        let wet: f32 = taps
                            .iter()
                            .filter_map(|&(delay, coefficient)| {
                                (start + offset)
                                    .checked_sub(delay)
                                    .map(|i| input[i] * coefficient)
                            })
                            .sum();
                        let gain = gains[0] + (gains[1] - gains[0]) * offset as f32 / BLOCK as f32;
                        let scaled = wet * gain;
                        clipped_reference += usize::from(scaled > 1.00005);
                        expected.push(scaled.clamp(-1.0, 1.0));
                    }
                }
                assert_eq!(XDspDeactivate(stream), S_OK);
                assert!(actual.iter().all(|sample| sample.is_finite()));
                let peak = actual.iter().copied().map(f32::abs).fold(0.0, f32::max);
                let rms = (actual
                    .iter()
                    .map(|&sample| f64::from(sample).powi(2))
                    .sum::<f64>()
                    / actual.len() as f64)
                    .sqrt();
                let clipped = actual.iter().filter(|&&sample| sample == 1.0).count();
                let max_error = actual
                    .iter()
                    .zip(expected)
                    .map(|(a, e)| (a - e).abs())
                    .fold(0.0, f32::max);
                println!(
                    "{level} {mode}: {} samples, peak {peak:.6}, rms {rms:.6}, clipped {clipped}/{clipped_reference}, max_error {max_error:.8}",
                    actual.len()
                );
                outcomes.push((level, mode, clipped, clipped_reference, max_error));
            }
        }
        assert_eq!(XDspDisconnect(dsp), S_OK);
        for (level, mode, clipped, expected_clipped, max_error) in outcomes {
            assert!(max_error < 0.00005, "{level} {mode}: max_error {max_error}");
            assert_eq!(clipped, expected_clipped, "{level} {mode}");
            assert_eq!(clipped > 0, level == "loud" && mode == "double-start");
        }
    }
}
