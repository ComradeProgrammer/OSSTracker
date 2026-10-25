//! Output thread: runs at a fixed rate (default 120 Hz), independent of the camera frame rate.
//! Interpolates/predicts between tracker measurements, applies hold/pause/recentre behaviour,
//! response curves, and sends to opentrack (UDP) and FreeTrack (Windows shared memory).

pub mod freetrack;
pub mod udp;

use crate::filter::{apply_invert, map_pose};
use crate::tracker::Shared;
use parking_lot::Mutex;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone, Default)]
pub struct OutputState {
    /// Curve-mapped values sent to the game (deg / cm).
    pub mapped: [f64; 6],
    /// Smoothed head pose before curves (deg / cm).
    pub smoothed: [f64; 6],
    pub udp_error: Option<String>,
    pub freetrack_status: String,
}

pub fn spawn(shared: Arc<Shared>, out_state: Arc<Mutex<OutputState>>) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("output".into())
        .spawn(move || {
            while !shared.quit.load(Ordering::SeqCst) {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&shared, &out_state)));
                if r.is_err() {
                    log::error!("output thread panicked; restarting");
                    std::thread::sleep(Duration::from_millis(200));
                }
            }
        })
        .expect("spawn output thread")
}

fn run(shared: &Shared, out_state: &Mutex<OutputState>) {
    let mut udp = udp::UdpSender::default();
    #[allow(clippy::default_constructed_unit_structs)]
    let mut ft = freetrack::FreeTrack::default();
    let mut y = [0.0f64; 6];
    let mut held = [0.0f64; 6];
    let mut last = Instant::now();
    let mut was_tracking = false;
    let mut last_epoch = 0;
    let mut glide_until = Instant::now();
    let mut next_tick = Instant::now();
    let mut frozen: Option<[f64; 6]> = None;

    while !shared.quit.load(Ordering::SeqCst) {
        let cfg = shared.config.read().clone();
        let period = Duration::from_secs_f64(1.0 / cfg.output_hz.max(30) as f64);
        next_tick += period;
        let now = Instant::now();
        if next_tick > now {
            std::thread::sleep(next_tick - now);
        } else {
            next_tick = now; // fell behind (e.g. machine suspended): don't try to catch up
        }
        let now = Instant::now();
        let dt = now.duration_since(last).as_secs_f64().clamp(1e-4, 0.25);
        last = now;

        let st = shared.state.lock().clone();
        let s = &cfg.smoothing;

        // Target pose for this tick.
        let target = if st.tracking {
            let age = now.saturating_duration_since(st.measured_at).as_secs_f64();
            // Extrapolate at most ~1.5 camera frames ahead; beyond that, hold.
            let max_pred = if st.fps > 1.0 { 1.5 / st.fps as f64 } else { 0.05 };
            let lead = age.min(max_pred) * s.prediction as f64;
            let mut t = [0.0; 6];
            for i in 0..6 {
                t[i] = st.filtered[i] + st.velocity[i] * lead;
            }
            held = st.filtered;
            t
        } else {
            let lost_for = st.lost_since.map(|t| now.saturating_duration_since(t).as_secs_f64()).unwrap_or(0.0);
            let hold = cfg.hold_on_lost_secs as f64;
            if lost_for <= hold {
                held
            } else {
                // Ease back to centre with a ~0.6 s time constant.
                let k = (-(lost_for - hold) / 0.6).exp();
                held.map(|v| v * k)
            }
        };

        // Glide (slower smoothing) right after re-acquiring or recentring, so the view never snaps.
        if (st.tracking && !was_tracking) || st.center_epoch != last_epoch {
            glide_until = now + Duration::from_millis(400);
        }
        was_tracking = st.tracking;
        last_epoch = st.center_epoch;
        let mut tau = s.output_smoothing_ms as f64 / 1000.0;
        if now < glide_until {
            tau = tau.max(0.12);
        }
        let a = if tau <= 1e-4 { 1.0 } else { 1.0 - (-dt / tau).exp() };
        for i in 0..6 {
            if target[i].is_finite() {
                y[i] += (target[i] - y[i]) * a;
            }
        }

        let mut mapped = map_pose(&y, &cfg.axes);
        let mut raw = apply_invert(&y, &cfg.axes);
        // Pause freezes the view where it is (like TrackIR's pause key).
        if shared.paused.load(Ordering::SeqCst) {
            let f = *frozen.get_or_insert(mapped);
            mapped = f;
            raw = apply_invert(&held, &cfg.axes);
        } else {
            frozen = None;
        }

        let udp_err = if cfg.udp.enabled {
            let v = if cfg.udp.send_mapped { &mapped } else { &raw };
            udp.send(&cfg.udp.host, cfg.udp.port, v).err()
        } else {
            None
        };
        let ft_status = if cfg.freetrack.enabled { ft.send(&mapped, &raw, &cfg.freetrack) } else { ft.disable() };

        let mut o = out_state.lock();
        o.mapped = mapped;
        o.smoothed = y;
        o.udp_error = udp_err;
        o.freetrack_status = ft_status;
    }
}
