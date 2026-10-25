//! End-to-end check of the threaded app pipeline without a camera: synthetic 30 fps frames
//! (head rolling ±15° at 0.5 Hz, then leaving and coming back) go through the real tracker and
//! output threads, and the opentrack UDP packets are captured and analysed for rate,
//! smoothness, lag and lost-face behaviour.
//!
//! Usage: cargo run --release --example e2e_udp -- <face photo>
#[path = "common/synth.rs"]
mod synth;
use mytracker::camera::FrameSlot;
use mytracker::config::Config;
use mytracker::image_ops::Frame;
use mytracker::output::{self, OutputState};
use mytracker::tracker::{self, Pipeline, Shared};
use parking_lot::Mutex;
use std::net::UdpSocket;
use std::sync::Arc;
use std::time::{Duration, Instant};
use synth::{render, H, W};

const AMP: f32 = 15.0;
const FREQ: f32 = 0.5;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: e2e_udp <face photo>");
    let img = image::open(&path)?.to_rgb8();
    let src = Frame { width: img.width(), height: img.height(), rgb: Arc::new(img.into_raw()), captured_at: Instant::now(), seq: 0 };

    // Locate the face in the photo.
    let mut p = Pipeline::new()?;
    let probe = render(&src, src.width as f32 / 2.0, src.height as f32 / 2.0, W as f32 / 2.0, H as f32 / 2.0, W as f32 / src.width as f32, 0.0, 1e9);
    let r = p.step(&probe, &Config::default()).expect("no face in the photo");
    let k = src.width as f32 / W as f32;
    let (fx, fy) = ((r.roi.cx - W as f32 / 2.0) * k + src.width as f32 / 2.0, (r.roi.cy - H as f32 / 2.0) * k + src.height as f32 / 2.0);
    let face = r.roi.size * k / 1.5;
    let (cx, cy, s, mask) = (W as f32 / 2.0, H as f32 / 2.0, 220.0 / face, face * 1.3);

    // Pre-render one oscillation period at 30 fps, plus an empty frame.
    let fps = 30.0;
    let n = (fps / FREQ) as usize;
    let cycle: Vec<Frame> = (0..n).map(|i| render(&src, fx, fy, cx, cy, s, AMP * (std::f32::consts::TAU * i as f32 / n as f32).sin(), mask)).collect();
    let still = render(&src, fx, fy, cx, cy, s, 0.0, mask);
    let empty = render(&src, fx, fy, -5000.0, cy, s, 0.0, mask);

    let rx = UdpSocket::bind("127.0.0.1:0")?;
    rx.set_read_timeout(Some(Duration::from_millis(100)))?;
    let mut cfg = Config::default();
    cfg.udp.port = rx.local_addr()?.port();
    cfg.freetrack.enabled = false;
    cfg.show_preview = false;
    let hold = cfg.hold_on_lost_secs as f64;
    let shared = Shared::new(cfg);
    let slot = Arc::new(FrameSlot::default());
    tracker::spawn(shared.clone(), slot.clone());
    output::spawn(shared.clone(), Arc::new(Mutex::new(OutputState::default())));

    // Phases (seconds): still 0-2, rolling 2-8, gone 8-12, back 12-14.
    let t0 = Instant::now();
    let feeder = {
        let slot = slot.clone();
        std::thread::spawn(move || {
            let mut seq = 0u64;
            let start = Instant::now();
            loop {
                let t = start.elapsed().as_secs_f32();
                if t > 14.0 {
                    break;
                }
                let base = if !(2.0..12.0).contains(&t) {
                    &still
                } else if t < 8.0 {
                    &cycle[(((t - 2.0) * fps) as usize) % n]
                } else {
                    &empty
                };
                seq += 1;
                slot.put(Frame { captured_at: Instant::now(), seq, ..base.clone() });
                let next = start + Duration::from_secs_f32(seq as f32 / fps);
                std::thread::sleep(next.saturating_duration_since(Instant::now()));
            }
        })
    };

    let mut samples: Vec<(f64, f64)> = Vec::new();
    let mut buf = [0u8; 64];
    while t0.elapsed() < Duration::from_secs(14) {
        if let Ok(48) = rx.recv(&mut buf) {
            let roll = f64::from_le_bytes(buf[40..48].try_into()?);
            samples.push((t0.elapsed().as_secs_f64(), roll));
        }
    }
    feeder.join().ok();
    shared.quit.store(true, std::sync::atomic::Ordering::SeqCst);

    let in_range = |a: f64, b: f64| samples.iter().filter(move |(t, _)| *t >= a && *t < b);
    let rate = in_range(1.0, 13.0).count() as f64 / 12.0;
    println!("UDP packet rate: {rate:.0} Hz (target 120)");

    // Rolling phase: compare with the truth (image rotation +θ is head roll −θ), find the lag.
    let truth = |t: f64, lag: f64| -(AMP as f64) * (std::f64::consts::TAU * FREQ as f64 * (t - 2.0 - lag)).sin();
    let roll: Vec<_> = in_range(3.0, 8.0).copied().collect();
    let (mut best_lag, mut best_err) = (0.0, f64::MAX);
    for l in 0..200 {
        let lag = l as f64 * 0.001;
        let err = (roll.iter().map(|(t, v)| (v - truth(*t, lag)).powi(2)).sum::<f64>() / roll.len() as f64).sqrt();
        if err < best_err {
            best_err = err;
            best_lag = lag;
        }
    }
    let max_step = roll.windows(2).map(|w| (w[1].1 - w[0].1).abs()).fold(0.0, f64::max);
    let peak = roll.iter().map(|(_, v)| v.abs()).fold(0.0, f64::max);
    let frame_step = AMP as f64 * std::f64::consts::TAU * FREQ as f64 / fps as f64;
    println!("Rolling ±{AMP}°: output peak {peak:.1}°, lag {:.0} ms after capture, RMS error at that lag {best_err:.2}°", best_lag * 1000.0);
    println!("Largest step between packets {max_step:.2}° (a raw 30 fps signal would step up to {frame_step:.2}°)");

    // Still phase jitter.
    let still_v: Vec<f64> = in_range(1.0, 2.0).map(|(_, v)| *v).collect();
    let mean = still_v.iter().sum::<f64>() / still_v.len().max(1) as f64;
    let sd = (still_v.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / still_v.len().max(1) as f64).sqrt();
    println!("Holding still: roll jitter σ = {sd:.3}°");

    // Lost face: hold, then ease back to centre; then re-acquire.
    let at = |t: f64| samples.iter().min_by(|a, b| (a.0 - t).abs().total_cmp(&(b.0 - t).abs())).map(|s| s.1).unwrap_or(f64::NAN);
    println!(
        "Face gone at 8.0 s: roll at 8.0 {:.1}°, {:.1} s later {:.1}° (held), {:.1} s later {:.1}° (easing to centre), at 12 s {:.1}°",
        at(8.0),
        hold - 0.3,
        at(8.0 + hold - 0.3),
        hold + 1.5,
        at(8.0 + hold + 1.5),
        at(11.95)
    );
    let back: Vec<_> = in_range(12.0, 14.0).copied().collect();
    let back_step = back.windows(2).map(|w| (w[1].1 - w[0].1).abs()).fold(0.0, f64::max);
    println!("Face back at 12 s: roll at 13.5 s {:.2}°, largest step while re-acquiring {back_step:.2}°", at(13.5));
    Ok(())
}
