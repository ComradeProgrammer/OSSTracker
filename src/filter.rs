//! Smoothing and response curves.
//!
//! Two smoothing stages remove jitter without adding much lag:
//! 1. A One-Euro filter on every camera measurement: heavy smoothing while the head is still,
//!    light smoothing while it moves.
//! 2. A high-rate output smoother that interpolates between camera frames (with a little velocity
//!    prediction), so a 30 fps camera still produces smooth 120 Hz output instead of steps.

use crate::config::{AxisConfig, SmoothingConfig};

#[derive(Clone, Debug)]
pub struct OneEuro {
    x: Option<f64>,
    dx: f64,
}

fn alpha(cutoff: f64, dt: f64) -> f64 {
    let tau = 1.0 / (2.0 * std::f64::consts::PI * cutoff.max(1e-3));
    1.0 / (1.0 + tau / dt)
}

impl OneEuro {
    pub fn new() -> Self {
        Self { x: None, dx: 0.0 }
    }

    pub fn reset(&mut self) {
        self.x = None;
        self.dx = 0.0;
    }

    /// Filter `v` sampled `dt` seconds after the previous sample. Returns (value, velocity per second).
    pub fn filter(&mut self, v: f64, dt: f64, min_cutoff: f64, beta: f64) -> (f64, f64) {
        let dt = dt.clamp(1e-4, 0.5);
        let Some(prev) = self.x else {
            self.x = Some(v);
            self.dx = 0.0;
            return (v, 0.0);
        };
        let raw_dx = (v - prev) / dt;
        let ad = alpha(1.0, dt);
        self.dx += ad * (raw_dx - self.dx);
        let cutoff = min_cutoff + beta * self.dx.abs();
        let a = alpha(cutoff, dt);
        let x = prev + a * (v - prev);
        self.x = Some(x);
        (x, self.dx)
    }
}

impl Default for OneEuro {
    fn default() -> Self {
        Self::new()
    }
}

/// One-Euro filters for all six axes.
#[derive(Default)]
pub struct PoseFilter {
    f: [OneEuro; 6],
}

impl PoseFilter {
    pub fn reset(&mut self) {
        for f in &mut self.f {
            f.reset();
        }
    }

    pub fn filter(&mut self, v: &[f64; 6], dt: f64, s: &SmoothingConfig) -> ([f64; 6], [f64; 6]) {
        let mut out = [0.0; 6];
        let mut vel = [0.0; 6];
        for i in 0..6 {
            let (mc, b) = if i < 3 { (s.rot_min_cutoff, s.rot_beta) } else { (s.pos_min_cutoff, s.pos_beta) };
            let (x, dx) = self.f[i].filter(v[i], dt, mc as f64, b as f64);
            out[i] = x;
            vel[i] = dx;
        }
        (out, vel)
    }
}

/// Map one axis through deadzone + response curve + limits. Input and output share units.
pub fn map_axis(x: f64, a: &AxisConfig) -> f64 {
    if !a.enabled || !x.is_finite() {
        return 0.0;
    }
    let dz = a.deadzone as f64;
    let lim = (a.input_limit as f64).max(dz + 1e-3);
    let mag = ((x.abs() - dz) / (lim - dz)).clamp(0.0, 1.0);
    let e = (a.curve as f64).clamp(0.0, 1.0);
    let shaped = (1.0 - e) * mag + e * mag * mag * mag;
    x.signum() * shaped * a.output_limit as f64
}

/// Apply invert flags (the user's convention fix-ups) without any curve.
pub fn apply_invert(v: &[f64; 6], axes: &[AxisConfig; 6]) -> [f64; 6] {
    let mut o = *v;
    for i in 0..6 {
        if axes[i].invert {
            o[i] = -o[i];
        }
        if !axes[i].enabled {
            o[i] = 0.0;
        }
    }
    o
}

pub fn map_pose(v: &[f64; 6], axes: &[AxisConfig; 6]) -> [f64; 6] {
    let mut o = [0.0; 6];
    for i in 0..6 {
        let x = if axes[i].invert { -v[i] } else { v[i] };
        o[i] = map_axis(x, &axes[i]);
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_euro_converges_and_smooths() {
        let mut f = OneEuro::new();
        let mut last = 0.0;
        for i in 0..300 {
            let noise = if i % 2 == 0 { 0.3 } else { -0.3 };
            last = f.filter(10.0 + noise, 1.0 / 30.0, 0.6, 0.08).0;
        }
        assert!((last - 10.0).abs() < 0.1, "{last}");
    }

    #[test]
    fn curve_is_monotonic_odd_and_clamped() {
        let a = AxisConfig { input_limit: 40.0, output_limit: 180.0, curve: 0.4, deadzone: 1.0, ..Default::default() };
        let mut prev = -1.0;
        for i in 0..=100 {
            let x = i as f64 * 0.5;
            let y = map_axis(x, &a);
            assert!(y >= prev);
            assert!((map_axis(-x, &a) + y).abs() < 1e-9);
            prev = y;
        }
        assert_eq!(map_axis(0.5, &a), 0.0);
        assert!((map_axis(40.0, &a) - 180.0).abs() < 1e-9);
        assert!((map_axis(90.0, &a) - 180.0).abs() < 1e-9);
        assert_eq!(map_axis(f64::NAN, &a), 0.0);
    }
}
