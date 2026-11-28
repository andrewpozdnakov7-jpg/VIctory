//! A read-back target's texels as numbers: decoding, health and previews.

use crate::capture::{half, small_float};

/// The bytes of one texel of DXGI format `format`, if decoded here.
pub(super) fn texel_size(format: u32) -> Option<usize> {
    Some(match format {
        1..=4 => 16,
        9..=16 | 19..=22 => 8,
        23..=32 | 33..=47 | 67 | 87..=93 => 4,
        48..=59 => 2,
        60..=65 => 1,
        _ => return None,
    })
}

/// How a format's values read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    /// Normalized: 0 to 1 as they are.
    Unit,
    /// Floats: light, compressed for the preview.
    Light,
    /// One channel, or integers: shown from the lowest to the highest.
    Single,
}

pub(super) fn kind(format: u32) -> Kind {
    match format {
        1 | 2 | 9 | 10 | 15 | 16 | 26 | 33 | 34 | 67 => Kind::Light,
        3 | 4 | 12 | 14 | 19..=22 | 25 | 30 | 32 | 36 | 38..=47 | 50 | 52..=60 | 62..=64 => {
            Kind::Single
        }
        _ => Kind::Unit,
    }
}

/// One texel of `format` as RGBA floats (one channel: in R, G and B).
pub(super) fn texel(format: u32, b: &[u8]) -> Option<[f32; 4]> {
    if b.len() < texel_size(format)? {
        return None;
    }
    let u16_at = |i: usize| u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
    let u32_at =
        |i: usize| u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
    let f32_at = |i: usize| f32::from_bits(u32_at(i));
    let one = |v: f32| [v, v, v, 1.0];
    let unorm8 = |i: usize| f32::from(b[i]) / 255.0;
    let snorm8 = |i: usize| (f32::from(b[i] as i8) / 127.0).max(-1.0);
    let unorm16 = |i: usize| f32::from(u16_at(i)) / 65535.0;
    let snorm16 = |i: usize| (f32::from(u16_at(i) as i16) / 32767.0).max(-1.0);
    Some(match format {
        1 | 2 => [f32_at(0), f32_at(1), f32_at(2), f32_at(3)],
        3 => [0, 1, 2, 3].map(|i| u32_at(i) as f32),
        4 => [0, 1, 2, 3].map(|i| u32_at(i) as i32 as f32),
        9 | 10 => [0, 1, 2, 3].map(|i| half(u16_at(i))),
        11 => [0, 1, 2, 3].map(unorm16),
        12 => [0, 1, 2, 3].map(|i| f32::from(u16_at(i))),
        13 => [0, 1, 2, 3].map(snorm16),
        14 => [0, 1, 2, 3].map(|i| f32::from(u16_at(i) as i16)),
        15 | 16 => [f32_at(0), f32_at(1), 0.0, 1.0],
        19..=22 => one(f32_at(0)),
        23 | 24 | 89 => {
            let v = u32_at(0);
            let ten = |shift: u32| (v >> shift & 0x3ff) as f32 / 1023.0;
            [ten(0), ten(10), ten(20), (v >> 30) as f32 / 3.0]
        }
        25 => {
            let v = u32_at(0);
            let ten = |shift: u32| (v >> shift & 0x3ff) as f32;
            [ten(0), ten(10), ten(20), (v >> 30) as f32]
        }
        26 => {
            let v = u32_at(0);
            [
                small_float(v & 0x7ff, 6),
                small_float(v >> 11 & 0x7ff, 6),
                small_float(v >> 22, 5),
                1.0,
            ]
        }
        27..=29 => [0, 1, 2, 3].map(unorm8),
        30 => [0, 1, 2, 3].map(|i| f32::from(b[i])),
        31 => [0, 1, 2, 3].map(snorm8),
        32 => [0, 1, 2, 3].map(|i| f32::from(b[i] as i8)),
        33 | 34 => [half(u16_at(0)), half(u16_at(1)), 0.0, 1.0],
        35 => [unorm16(0), unorm16(1), 0.0, 1.0],
        36 => [f32::from(u16_at(0)), f32::from(u16_at(1)), 0.0, 1.0],
        37 => [snorm16(0), snorm16(1), 0.0, 1.0],
        38 => [
            f32::from(u16_at(0) as i16),
            f32::from(u16_at(1) as i16),
            0.0,
            1.0,
        ],
        39..=41 => one(f32_at(0)),
        42 => one(u32_at(0) as f32),
        43 => one(u32_at(0) as i32 as f32),
        44..=46 => one((u32_at(0) & 0xff_ffff) as f32 / 16_777_215.0),
        47 => one((u32_at(0) >> 24) as f32),
        48 | 49 => [unorm8(0), unorm8(1), 0.0, 1.0],
        50 => [f32::from(b[0]), f32::from(b[1]), 0.0, 1.0],
        51 => [snorm8(0), snorm8(1), 0.0, 1.0],
        52 => [f32::from(b[0] as i8), f32::from(b[1] as i8), 0.0, 1.0],
        53 | 54 => one(half(u16_at(0))),
        55 | 56 => one(unorm16(0)),
        57 => one(f32::from(u16_at(0))),
        58 => one(snorm16(0)),
        59 => one(f32::from(u16_at(0) as i16)),
        60 | 61 | 65 => one(unorm8(0)),
        62 => one(f32::from(b[0])),
        63 => one(snorm8(0)),
        64 => one(f32::from(b[0] as i8)),
        67 => {
            let v = u32_at(0);
            let scale = 2f32.powi((v >> 27) as i32 - 24);
            let nine = |shift: u32| (v >> shift & 0x1ff) as f32 * scale;
            [nine(0), nine(9), nine(18), 1.0]
        }
        87 | 88 | 90..=93 => [unorm8(2), unorm8(1), unorm8(0), unorm8(3)],
        _ => return None,
    })
}

/// Red and blue lit, green well under both.
fn magenta(p: [f32; 4]) -> bool {
    p[0] > 0.02 && p[2] > 0.02 && p[1] < 0.5 * p[0].min(p[2])
}

fn luma(p: [f32; 4]) -> f32 {
    0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]
}

/// A target's pixel health.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Health {
    pub pixels: u64,
    pub black: u64,
    pub nan: u64,
    pub inf: u64,
    pub magenta: u64,
    /// Of the finite pixels.
    pub luma_mean: f64,
    /// The lowest and highest finite channel value (R, G, B).
    pub low: f32,
    pub high: f32,
}

impl Health {
    pub(super) fn percent(&self, n: u64) -> f64 {
        if self.pixels == 0 {
            0.0
        } else {
            n as f64 * 100.0 / self.pixels as f64
        }
    }
}

/// Measures `values`. One-channel targets (`single`) are never magenta.
pub(super) fn health(values: &[[f32; 4]], single: bool) -> Health {
    let mut h = Health {
        pixels: values.len() as u64,
        low: f32::MAX,
        high: f32::MIN,
        ..Health::default()
    };
    let (mut sum, mut finite) = (0f64, 0u64);
    for &p in values {
        let rgb = [p[0], p[1], p[2]];
        if rgb.iter().any(|c| c.is_nan()) {
            h.nan += 1;
            continue;
        }
        if rgb.iter().any(|c| c.is_infinite()) {
            h.inf += 1;
            continue;
        }
        if rgb.iter().all(|c| c.abs() < 1e-5) {
            h.black += 1;
        }
        if !single && magenta(p) {
            h.magenta += 1;
        }
        for c in rgb {
            h.low = h.low.min(c);
            h.high = h.high.max(c);
        }
        sum += f64::from(luma(p));
        finite += 1;
    }
    if finite == 0 {
        (h.low, h.high) = (0.0, 0.0);
    } else {
        h.luma_mean = sum / finite as f64;
    }
    h
}

/// A preview of `values` (`w` x `h`), at most `max_w` wide: its size and its
/// PNG rows (filter byte 0, then RGB). NaN and infinite texels are cyan.
pub(super) fn preview(
    values: &[[f32; 4]],
    (w, h): (u32, u32),
    kind: Kind,
    health: &Health,
    max_w: u32,
) -> (u32, u32, Vec<u8>) {
    let (pw, ph) = if w <= max_w {
        (w.max(1), h.max(1))
    } else {
        let ph = (u64::from(h) * u64::from(max_w) / u64::from(w)) as u32;
        (max_w, ph.max(1))
    };
    let byte = |v: f32| {
        let u = match kind {
            Kind::Unit => v,
            Kind::Light => (v.max(0.0) / (1.0 + v.max(0.0))).powf(1.0 / 2.2),
            Kind::Single => (v - health.low) / (health.high - health.low).max(1e-20),
        };
        (u.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
    };
    let mut rows = Vec::with_capacity((pw as usize * 3 + 1) * ph as usize);
    for y in 0..ph {
        rows.push(0);
        let sy = (u64::from(y) * u64::from(h) / u64::from(ph)) as usize;
        for x in 0..pw {
            let sx = (u64::from(x) * u64::from(w) / u64::from(pw)) as usize;
            let p = values[sy * w as usize + sx];
            if p[..3].iter().any(|c| !c.is_finite()) {
                rows.extend([0, 255, 255]);
            } else {
                rows.extend([byte(p[0]), byte(p[1]), byte(p[2])]);
            }
        }
    }
    (pw, ph, rows)
}

/// Which of the targets (sizes, in first-use order) a snapshot reads: those
/// shaped like the largest screen-like one (1.2 to 2.5 wide per high) and at
/// least 45 % of its width, at most `max`, spread over the order. Without a
/// screen-like one, the largest. Also that largest screen-like size.
pub(super) fn pick(sizes: &[(u32, u32)], max: usize) -> (Vec<usize>, Option<(u32, u32)>) {
    let area = |(w, h): (u32, u32)| u64::from(w) * u64::from(h);
    let aspect = |(w, h): (u32, u32)| f64::from(w) / f64::from(h.max(1));
    let screen = sizes
        .iter()
        .copied()
        .filter(|&s| (1.2..=2.5).contains(&aspect(s)))
        .max_by_key(|&s| area(s));
    let mut chosen: Vec<usize> = match screen {
        Some(r) => (0..sizes.len())
            .filter(|&i| {
                (aspect(sizes[i]) / aspect(r) - 1.0).abs() <= 0.12
                    && f64::from(sizes[i].0) >= 0.45 * f64::from(r.0)
            })
            .collect(),
        None => {
            let mut all: Vec<usize> = (0..sizes.len()).collect();
            all.sort_by_key(|&i| std::cmp::Reverse(area(sizes[i])));
            all.truncate(max);
            all.sort_unstable();
            all
        }
    };
    if chosen.len() > max {
        let n = chosen.len();
        chosen = (0..max)
            .map(|k| chosen[if max == 1 { 0 } else { k * (n - 1) / (max - 1) }])
            .collect();
    }
    let reference = screen.or_else(|| sizes.iter().copied().max_by_key(|&s| area(s)));
    (chosen, reference)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texels_decode_by_format() {
        assert_eq!(texel(28, &[255, 0, 51, 255]), Some([1.0, 0.0, 0.2, 1.0]));
        assert_eq!(texel(87, &[51, 0, 255, 255]), Some([1.0, 0.0, 0.2, 1.0]));
        let floats = [1.5f32, -2.0, 0.0, 1.0];
        let bytes: Vec<u8> = floats.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(texel(2, &bytes), Some(floats));
        assert_eq!(
            texel(10, &[0, 0x3c, 0, 0, 0, 0, 0, 0x3c]),
            Some([1.0, 0.0, 0.0, 1.0])
        );
        // R11G11B10: 1.0 in each.
        let ones = 15u32 << 6 | (15u32 << 6) << 11 | (15u32 << 5) << 22;
        assert_eq!(texel(26, &ones.to_le_bytes()), Some([1.0; 4]));
        assert_eq!(texel(45, &0x00ff_ffffu32.to_le_bytes()), Some([1.0; 4]));
        assert_eq!(texel(28, &[1, 2]), None);
        assert_eq!(texel(999, &[0; 16]), None);
        for format in 1..=93 {
            if let Some(n) = texel_size(format) {
                assert!(texel(format, &vec![0; n]).is_some(), "{format}");
            }
        }
    }

    #[test]
    fn health_counts_black_nan_inf_and_magenta() {
        let values = [
            [0.0, 0.0, 0.0, 1.0],
            [f32::NAN, 0.0, 0.0, 1.0],
            [0.0, f32::INFINITY, 0.0, 1.0],
            [0.8, 0.1, 0.7, 1.0],
        ];
        let h = health(&values, false);
        assert_eq!(
            (h.pixels, h.black, h.nan, h.inf, h.magenta),
            (4, 1, 1, 1, 1)
        );
        assert_eq!(h.percent(h.black), 25.0);
        assert_eq!((h.low, h.high), (0.0, 0.8));
        assert_eq!(health(&values, true).magenta, 0);
        assert_eq!(health(&[], false).percent(0), 0.0);
        let nothing = health(&[[f32::NAN; 4]], false);
        assert_eq!(
            (nothing.low, nothing.high, nothing.luma_mean),
            (0.0, 0.0, 0.0)
        );
    }

    #[test]
    fn single_channel_snorm_previews_distinguish_negative_zero_and_positive() {
        let r16 = [-i16::MAX, 0, i16::MAX].map(i16::to_le_bytes).concat();
        let r8 = [-i8::MAX, 0, i8::MAX].map(|v| v as u8);
        for (format, size, data) in [(58, 2, r16.as_slice()), (63, 1, r8.as_slice())] {
            let values: Vec<_> = data
                .chunks_exact(size)
                .map(|bytes| texel(format, bytes).unwrap())
                .collect();
            let measured = health(&values, true);
            assert_eq!((measured.low, measured.high, measured.black), (-1.0, 1.0, 1));
            let (_, _, rows) = preview(&values, (3, 1), kind(format), &measured, 960);
            assert_eq!(rows, [0, 0, 0, 0, 128, 128, 128, 255, 255, 255], "format {format}");
        }
    }

    #[test]
    fn previews_shrink_and_paint_not_finite_texels_cyan() {
        let values = [[f32::NAN, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]];
        let h = health(&values, false);
        let (w, rows_high, rows) = preview(&values, (2, 1), Kind::Unit, &h, 960);
        assert_eq!((w, rows_high), (2, 1));
        assert_eq!(rows, [0, 0, 255, 255, 255, 0, 0]);
        let big = vec![[0.5f32; 4]; 2000 * 10];
        let (w, rows_high, rows) = preview(&big, (2000, 10), Kind::Unit, &h, 960);
        assert_eq!((w, rows_high, rows.len()), (960, 4, (960 * 3 + 1) * 4));
    }

    #[test]
    fn screen_like_targets_are_picked_and_spread() {
        let sizes = [
            (2048, 2048),
            (2560, 1440),
            (1280, 720),
            (64, 36),
            (2560, 1440),
            (1920, 1080),
        ];
        assert_eq!(pick(&sizes, 12), (vec![1, 2, 4, 5], Some((2560, 1440))));
        assert_eq!(pick(&sizes, 2).0, [1, 5]);
        // Nothing screen-like: the largest.
        assert_eq!(
            pick(&[(512, 512), (2048, 2048), (16, 16)], 2),
            (vec![0, 1], Some((2048, 2048)))
        );
        assert_eq!(pick(&[], 12), (vec![], None));
    }
}
