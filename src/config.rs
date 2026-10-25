//! User settings, persisted as JSON in the platform config directory.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AxisConfig {
    pub enabled: bool,
    pub invert: bool,
    /// Head movement (deg or cm) that maps to `output_limit`.
    pub input_limit: f32,
    /// Output (deg or cm) at `input_limit`; also the clamp.
    pub output_limit: f32,
    /// 0 = linear, 1 = fully cubic (fine control around centre).
    pub curve: f32,
    /// Movement around centre that is ignored (deg or cm).
    pub deadzone: f32,
}

impl Default for AxisConfig {
    fn default() -> Self {
        Self { enabled: true, invert: false, input_limit: 30.0, output_limit: 30.0, curve: 0.0, deadzone: 0.0 }
    }
}

impl AxisConfig {
    fn new(input_limit: f32, output_limit: f32, curve: f32, deadzone: f32) -> Self {
        Self { input_limit, output_limit, curve, deadzone, ..Default::default() }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SmoothingConfig {
    /// One-Euro minimum cutoff (Hz) for rotation: lower = steadier when still.
    pub rot_min_cutoff: f32,
    /// One-Euro speed coefficient for rotation: higher = less lag when moving fast.
    pub rot_beta: f32,
    pub pos_min_cutoff: f32,
    pub pos_beta: f32,
    /// Output interpolation time constant (ms). Hides the camera frame steps.
    pub output_smoothing_ms: f32,
    /// 0..1: how much velocity extrapolation is used to cancel the interpolation delay.
    pub prediction: f32,
}

impl Default for SmoothingConfig {
    fn default() -> Self {
        Self {
            rot_min_cutoff: 0.6,
            rot_beta: 0.08,
            pos_min_cutoff: 0.5,
            pos_beta: 0.3,
            output_smoothing_ms: 25.0,
            prediction: 0.5,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct UdpConfig {
    pub enabled: bool,
    pub host: String,
    pub port: u16,
    /// Send curve-mapped values (when the game reads UDP directly) instead of raw head pose
    /// (when opentrack does the mapping).
    pub send_mapped: bool,
}

impl Default for UdpConfig {
    fn default() -> Self {
        Self { enabled: true, host: "127.0.0.1".into(), port: 4242, send_mapped: false }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct FreetrackConfig {
    /// Write the FreeTrack shared memory (Windows only). DCS reads it through NPClient64.dll.
    pub enabled: bool,
    /// Point the TrackIR/FreeTrack registry keys at `npclient_dir`.
    pub set_registry: bool,
    /// Folder that contains NPClient64.dll (e.g. opentrack's `modules` folder). Empty = auto-detect.
    pub npclient_dir: String,
}

impl Default for FreetrackConfig {
    fn default() -> Self {
        Self { enabled: cfg!(windows), set_registry: true, npclient_dir: String::new() }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct JoyBinding {
    pub device: String,
    pub code: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Config {
    /// Camera name as reported by the OS; empty = first camera.
    pub camera_name: String,
    /// Requested capture mode; None = automatic (full sensor field of view, highest frame rate).
    pub camera_mode: Option<(u32, u32, u32)>,
    /// Horizontal field of view of the camera in degrees (used for perspective-correct pose).
    pub camera_hfov: f32,
    pub mirror_preview: bool,
    pub show_preview: bool,
    pub show_landmarks: bool,
    pub smoothing: SmoothingConfig,
    /// Yaw, pitch, roll, x, y, z.
    pub axes: [AxisConfig; 6],
    /// Seconds to hold the last pose after the face is lost before easing back to centre.
    pub hold_on_lost_secs: f32,
    pub udp: UdpConfig,
    pub freetrack: FreetrackConfig,
    pub hotkey_recenter: String,
    pub hotkey_pause: String,
    pub joy_recenter: Option<JoyBinding>,
    pub joy_pause: Option<JoyBinding>,
    /// Output updates per second.
    pub output_hz: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            camera_name: String::new(),
            camera_mode: None,
            camera_hfov: 70.0,
            mirror_preview: true,
            show_preview: true,
            show_landmarks: true,
            smoothing: SmoothingConfig::default(),
            axes: [
                AxisConfig::new(40.0, 180.0, 0.35, 0.5), // yaw: look all the way back with ~40° of head turn
                AxisConfig::new(25.0, 90.0, 0.35, 0.5),  // pitch
                AxisConfig::new(30.0, 30.0, 0.0, 1.0),   // roll
                AxisConfig::new(15.0, 15.0, 0.0, 0.2),   // x
                AxisConfig::new(15.0, 15.0, 0.0, 0.2),   // y
                AxisConfig::new(20.0, 20.0, 0.0, 0.3),   // z
            ],
            hold_on_lost_secs: 2.0,
            udp: UdpConfig::default(),
            freetrack: FreetrackConfig::default(),
            hotkey_recenter: "F12".into(),
            hotkey_pause: "Shift+F12".into(),
            joy_recenter: None,
            joy_pause: None,
            output_hz: 120,
        }
    }
}

pub fn config_path() -> Option<PathBuf> {
    // Portable mode: a config next to the executable wins.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join("mytracker.json");
            if p.exists() {
                return Some(p);
            }
        }
    }
    directories::ProjectDirs::from("", "", "MyTracker").map(|d| d.config_dir().join("config.json"))
}

impl Config {
    /// Load the config; any problem (missing, corrupt, old format) falls back to defaults.
    pub fn load() -> Self {
        let Some(path) = config_path() else { return Self::default() };
        match std::fs::read_to_string(&path) {
            Ok(s) => match serde_json::from_str::<Config>(&s) {
                Ok(c) => c.sanitized(),
                Err(e) => {
                    log::warn!("config {} unreadable ({e}); using defaults", path.display());
                    let _ = std::fs::rename(&path, path.with_extension("json.bad"));
                    Self::default()
                }
            },
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self) {
        let Some(path) = config_path() else { return };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        match serde_json::to_string_pretty(self) {
            Ok(s) => {
                let tmp = path.with_extension("json.tmp");
                if std::fs::write(&tmp, s).is_ok() {
                    let _ = std::fs::rename(&tmp, &path);
                }
            }
            Err(e) => log::warn!("config save failed: {e}"),
        }
    }

    /// Clamp values so a hand-edited config cannot produce NaNs or divide by zero.
    pub fn sanitized(mut self) -> Self {
        self.camera_hfov = finite_or(self.camera_hfov, 70.0).clamp(30.0, 150.0);
        for a in &mut self.axes {
            a.input_limit = finite_or(a.input_limit, 30.0).max(0.1);
            a.output_limit = finite_or(a.output_limit, 30.0).max(0.0);
            a.curve = finite_or(a.curve, 0.0).clamp(0.0, 1.0);
            a.deadzone = finite_or(a.deadzone, 0.0).clamp(0.0, a.input_limit * 0.9);
        }
        let s = &mut self.smoothing;
        s.rot_min_cutoff = finite_or(s.rot_min_cutoff, 0.6).clamp(0.01, 30.0);
        s.pos_min_cutoff = finite_or(s.pos_min_cutoff, 0.5).clamp(0.01, 30.0);
        s.rot_beta = finite_or(s.rot_beta, 0.08).clamp(0.0, 10.0);
        s.pos_beta = finite_or(s.pos_beta, 0.3).clamp(0.0, 10.0);
        s.output_smoothing_ms = finite_or(s.output_smoothing_ms, 25.0).clamp(0.0, 500.0);
        s.prediction = finite_or(s.prediction, 0.5).clamp(0.0, 1.0);
        self.hold_on_lost_secs = finite_or(self.hold_on_lost_secs, 2.0).clamp(0.0, 60.0);
        self.output_hz = self.output_hz.clamp(30, 500);
        self
    }
}

fn finite_or(v: f32, d: f32) -> f32 {
    if v.is_finite() { v } else { d }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_config_fills_defaults() {
        let c: Config = serde_json::from_str(r#"{"camera_hfov": 90.0, "udp": {"port": 5555}}"#).unwrap();
        assert_eq!(c.camera_hfov, 90.0);
        assert_eq!(c.udp.port, 5555);
        assert_eq!(c.udp.host, "127.0.0.1");
        assert_eq!(c.axes[0].output_limit, 180.0);
    }

    #[test]
    fn sanitize_fixes_nonsense() {
        let mut c = Config::default();
        c.axes[0].input_limit = 0.0;
        c.camera_hfov = f32::NAN;
        let c = c.sanitized();
        assert!(c.axes[0].input_limit > 0.0);
        assert_eq!(c.camera_hfov, 70.0);
    }
}
