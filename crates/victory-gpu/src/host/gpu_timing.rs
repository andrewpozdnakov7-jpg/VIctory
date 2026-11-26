use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

const REPORT_INTERVAL: Duration = Duration::from_secs(5);

#[derive(Default)]
struct Samples {
    by_pipeline: HashMap<String, (f64, u64)>,
    wall_ms: f64,
    gpu_ms: f64,
    commands: u64,
    lists: u64,
}

struct Window {
    since: Instant,
    frames: u64,
    samples: Samples,
}

impl Window {
    fn new(now: Instant) -> Self {
        Self {
            since: now,
            frames: 0,
            samples: Samples::default(),
        }
    }

    fn record(&mut self, wall: Duration, commands: &[(f64, String)]) {
        self.samples.lists += 1;
        self.samples.wall_ms += wall.as_secs_f64() * 1000.0;
        self.samples.commands += commands.len() as u64;
        for (ms, note) in commands {
            // Keep the command kind and the pipeline hashes, independent
            // of the draw's size and per-invocation addresses.
            let head = note.split(" | ").next().unwrap_or(note);
            let kind = head.split(' ').next().unwrap_or_default();
            let key = match head.find('[') {
                Some(at) => format!("{kind} {}", &head[at..]),
                None => kind.to_owned(),
            };
            let entry = self.samples.by_pipeline.entry(key).or_default();
            entry.0 += ms;
            entry.1 += 1;
            self.samples.gpu_ms += ms;
        }
    }

    fn present(&mut self, now: Instant) -> Option<Report> {
        self.frames += 1;
        if now.saturating_duration_since(self.since) < REPORT_INTERVAL {
            return None;
        }
        // A present closes the window. Both counters move together, even
        // when no timed list ran in this window.
        self.since = now;
        let frames = std::mem::take(&mut self.frames);
        let samples = std::mem::take(&mut self.samples);
        (samples.lists != 0).then_some(Report { frames, samples })
    }
}

pub(super) struct Report {
    frames: u64,
    samples: Samples,
}

impl Report {
    fn per_frame(&self, value: f64) -> f64 {
        value / self.frames as f64
    }

    #[cfg(test)]
    pub(super) fn command_count(&self) -> u64 {
        self.samples.commands
    }

    pub(super) fn log(&self) {
        let samples = &self.samples;
        let commands = samples.commands;
        let frames = self.frames;
        tracing::info!(
            "GPU timing: lists {:.1} ms/frame of wall time, their commands {:.1} ms/frame \
             ({commands} commands in {frames} frames, {:.2} commands/frame)",
            self.per_frame(samples.wall_ms),
            self.per_frame(samples.gpu_ms),
            self.per_frame(commands as f64),
        );
        let mut by: Vec<_> = samples.by_pipeline.iter().collect();
        by.sort_by(|a, b| b.1.0.total_cmp(&a.1.0).then(a.0.cmp(b.0)));
        for (key, (ms, n)) in by.into_iter().take(25) {
            tracing::info!(
                "  GPU {:.2} ms/frame in {:.2} commands/frame: {key}",
                self.per_frame(*ms),
                self.per_frame(*n as f64),
            );
        }
    }
}

fn window(now: Instant) -> MutexGuard<'static, Window> {
    static WINDOW: OnceLock<Mutex<Window>> = OnceLock::new();
    WINDOW
        .get_or_init(|| Mutex::new(Window::new(now)))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Adds one completed list. The lock also defines the present boundary,
/// so a list falls entirely on one side of a report.
pub(super) fn record(wall: Duration, commands: &[(f64, String)]) {
    window(Instant::now()).record(wall, commands);
}

pub(super) fn present(now: Instant) -> Option<Report> {
    window(now).present(now)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_fps_reports_use_the_present_count_and_clear_both_windows() {
        let start = Instant::now();
        let mut window = Window::new(start);
        let commands = [0, 1].map(|_| (10.0, "dispatch 1x1x1 [compute first]".to_owned()));
        for seconds in 0..6 {
            window.record(Duration::from_millis(30), &commands);
            let report = window.present(start + Duration::from_secs(seconds));
            if seconds < 5 {
                assert!(report.is_none());
                continue;
            }
            let report = report.unwrap();
            assert_eq!(report.frames, 6);
            assert_eq!(report.samples.lists, 6);
            assert_eq!(report.command_count(), 12);
            assert_eq!(report.per_frame(report.samples.gpu_ms), 20.0);
            assert_eq!(report.per_frame(report.command_count() as f64), 2.0);
            assert_eq!(report.per_frame(report.samples.wall_ms), 30.0);
        }
        let commands = [(4.0, "draw 3x1 [vertex second]".to_owned())];
        for seconds in [7, 9, 11] {
            window.record(Duration::from_millis(7), &commands);
            let report = window.present(start + Duration::from_secs(seconds));
            if seconds < 11 {
                assert!(report.is_none());
                continue;
            }
            let report = report.unwrap();
            assert_eq!(report.frames, 3);
            assert_eq!(report.samples.lists, 3);
            assert_eq!(report.command_count(), 3);
            assert_eq!(report.per_frame(report.samples.gpu_ms), 4.0);
            assert_eq!(report.per_frame(report.command_count() as f64), 1.0);
            assert_eq!(report.per_frame(report.samples.wall_ms), 7.0);
            assert_eq!(
                report.samples.by_pipeline,
                HashMap::from([("draw [vertex second]".to_owned(), (12.0, 3))])
            );
        }
    }
}
