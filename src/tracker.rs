//! The tracking thread: frame -> face region -> landmarks -> head pose -> filtered 6DoF.
//!
//! Design notes:
//! * The face region is tracked from the previous frame's landmarks, MediaPipe-style, so the
//!   detector is only needed to (re)acquire. The landmark model keeps working at large yaw
//!   where frontal detectors fail, which is exactly when you check six.
//! * Crops are affine warps that read black outside the frame: a face that is huge (camera on
//!   the monitor bezel) or half out of frame never produces an invalid ROI or a crash.
//! * Losing the face is a normal state: the last pose is held and tracking resumes when the
//!   detector finds the face again. Nothing in this loop can terminate the program.

use crate::camera::FrameSlot;
use crate::config::Config;
use crate::face_model::{model_points, EYE_LEFT_IMG, EYE_RIGHT_IMG, HEAD_CENTER, RIGID_LANDMARKS};
use crate::filter::PoseFilter;
use crate::image_ops::{preview_rgba, Affine2, Frame};
use crate::models::{Detection, FaceDetector, LandmarkModel, LM_SIZE};
use crate::pose::{relative_pose6, solve_pnp, umeyama, Intrinsics, Pose};
use nalgebra::{Vector2, Vector3};
use parking_lot::{Mutex, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Below this the face is too small to track reliably (pixels of ROI side).
const MIN_ROI_PX: f32 = 48.0;
/// Landmark model presence threshold.
const PRESENCE_MIN: f32 = 0.5;

#[derive(Clone, Default)]
pub struct Preview {
    pub width: usize,
    pub height: usize,
    pub rgba: Arc<Vec<u8>>,
    /// Normalised (0..1) preview coordinates, already mirrored if the preview is mirrored.
    pub landmarks: Vec<(f32, f32)>,
    pub rigid: Vec<(f32, f32)>,
    pub roi: Option<[(f32, f32); 4]>,
    /// Head axes: origin, x (right), y (up), z (forward) tips.
    pub axes: Option<[(f32, f32); 4]>,
    pub seq: u64,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Timings {
    pub detect_ms: f32,
    pub landmark_ms: f32,
    pub pose_ms: f32,
    pub total_ms: f32,
}

#[derive(Clone)]
pub struct TrackState {
    pub tracking: bool,
    pub message: String,
    /// Unfiltered pose relative to centre (opentrack convention, deg / cm).
    pub raw: [f64; 6],
    /// One-Euro filtered pose and its velocity (per second).
    pub filtered: [f64; 6],
    pub velocity: [f64; 6],
    pub measured_at: Instant,
    pub lost_since: Option<Instant>,
    /// Bumped on every recentre so the output stage can glide instead of jumping.
    pub center_epoch: u64,
    pub fps: f32,
    pub timings: Timings,
    pub reproj_px: f32,
    pub distance_cm: f32,
}

impl Default for TrackState {
    fn default() -> Self {
        Self {
            tracking: false,
            message: "Starting…".into(),
            raw: [0.0; 6],
            filtered: [0.0; 6],
            velocity: [0.0; 6],
            measured_at: Instant::now(),
            lost_since: Some(Instant::now()),
            center_epoch: 0,
            fps: 0.0,
            timings: Timings::default(),
            reproj_px: 0.0,
            distance_cm: 0.0,
        }
    }
}

/// State shared between the tracker, output, hotkey and UI threads.
pub struct Shared {
    pub config: RwLock<Config>,
    pub state: Mutex<TrackState>,
    pub preview: Mutex<Preview>,
    pub recenter: AtomicBool,
    pub paused: AtomicBool,
    pub quit: AtomicBool,
    /// Called when new data is ready for the UI (e.g. request a repaint).
    pub notify: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
}

impl Shared {
    pub fn new(config: Config) -> Arc<Self> {
        Arc::new(Self {
            config: RwLock::new(config),
            state: Mutex::new(TrackState::default()),
            preview: Mutex::new(Preview::default()),
            recenter: AtomicBool::new(false),
            paused: AtomicBool::new(false),
            quit: AtomicBool::new(false),
            notify: Mutex::new(None),
        })
    }

    pub fn request_recenter(&self) {
        self.recenter.store(true, Ordering::SeqCst);
    }

    pub fn toggle_pause(&self) {
        self.paused.fetch_xor(true, Ordering::SeqCst);
    }

    fn notify(&self) {
        if let Some(f) = self.notify.lock().as_ref() {
            f();
        }
    }
}

/// Square, rotated face crop in frame coordinates.
#[derive(Clone, Copy, Debug)]
pub struct Roi {
    pub cx: f32,
    pub cy: f32,
    pub size: f32,
    /// Rotation of the crop's x axis in image coordinates (radians).
    pub angle: f32,
}

impl Roi {
    /// Crop pixel -> frame pixel.
    pub fn affine(&self) -> Affine2 {
        let k = self.size / LM_SIZE as f32;
        let (s, c) = self.angle.sin_cos();
        let h = LM_SIZE as f32 * 0.5;
        Affine2 { m: [c * k, -s * k, self.cx - h * k * (c - s), s * k, c * k, self.cy - h * k * (s + c)] }
    }

    pub fn corners(&self) -> [(f32, f32); 4] {
        let a = self.affine();
        let n = LM_SIZE as f32;
        [a.apply(0.0, 0.0), a.apply(n, 0.0), a.apply(n, n), a.apply(0.0, n)]
    }

    fn from_detection(d: &Detection, frame: &Frame) -> Self {
        let (l, r) = (d.keypoints[0], d.keypoints[1]);
        let angle = (r.1 - l.1).atan2(r.0 - l.0);
        Self { cx: d.cx, cy: d.cy, size: d.w.max(d.h) * 1.5, angle }.clamped(frame)
    }

    /// ROI for the next frame from this frame's landmarks (frame pixels).
    fn from_landmarks(pts: &[[f32; 3]], frame: &Frame) -> Self {
        let l = pts[EYE_LEFT_IMG];
        let r = pts[EYE_RIGHT_IMG];
        let angle = (r[1] - l[1]).atan2(r[0] - l[0]);
        let (s, c) = angle.sin_cos();
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for p in pts.iter().take(468) {
            // Rotate into the face-aligned frame.
            let u = c * p[0] + s * p[1];
            let v = -s * p[0] + c * p[1];
            x0 = x0.min(u);
            x1 = x1.max(u);
            y0 = y0.min(v);
            y1 = y1.max(v);
        }
        let (mu, mv) = ((x0 + x1) * 0.5, (y0 + y1) * 0.5);
        let cx = c * mu - s * mv;
        let cy = s * mu + c * mv;
        Self { cx, cy, size: (x1 - x0).max(y1 - y0) * 1.5, angle }.clamped(frame)
    }

    fn clamped(mut self, frame: &Frame) -> Self {
        let max = frame.width.max(frame.height) as f32 * 2.5;
        self.size = self.size.clamp(1.0, max);
        self
    }

    fn valid(&self) -> bool {
        self.size >= MIN_ROI_PX && self.cx.is_finite() && self.cy.is_finite() && self.angle.is_finite()
    }

    /// Keep the previous crop when the change is tiny. A stable crop gives the landmark model
    /// identical input for a still face, which removes most of the frame-to-frame jitter.
    fn with_hysteresis(self, prev: &Roi) -> Roi {
        let dc = ((self.cx - prev.cx).powi(2) + (self.cy - prev.cy).powi(2)).sqrt();
        let ds = (self.size / prev.size - 1.0).abs();
        let mut da = (self.angle - prev.angle).abs();
        if da > std::f32::consts::PI {
            da = 2.0 * std::f32::consts::PI - da;
        }
        if dc < 0.02 * prev.size && ds < 0.03 && da < 1.5f32.to_radians() {
            *prev
        } else {
            self
        }
    }
}

struct Models {
    detector: FaceDetector,
    landmarks: LandmarkModel,
}

pub fn spawn(shared: Arc<Shared>, frames: Arc<FrameSlot>) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("tracker".into())
        .spawn(move || {
            let mut models = None;
            while !shared.quit.load(Ordering::SeqCst) {
                if models.is_none() {
                    set_message(&shared, "Loading face models…");
                    match FaceDetector::new().and_then(|d| Ok(Models { detector: d, landmarks: LandmarkModel::new()? })) {
                        Ok(m) => models = Some(m),
                        Err(e) => {
                            set_message(&shared, &format!("Cannot load face models: {e:#}"));
                            std::thread::sleep(Duration::from_secs(5));
                            continue;
                        }
                    }
                }
                let m = models.as_mut().expect("models loaded");
                let mut tracker = Tracker::new();
                // Belt and braces: a panic in one frame's processing restarts the loop, it never exits the app.
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tracker.run(&shared, &frames, m)));
                if r.is_err() {
                    log::error!("tracker panicked; restarting");
                    set_message(&shared, "Tracker error; restarting…");
                    std::thread::sleep(Duration::from_millis(200));
                }
            }
        })
        .expect("spawn tracker thread")
}

/// The tracking pipeline without threads, for offline replay and tests.
pub struct Pipeline {
    tracker: Tracker,
    models: Models,
}

pub struct StepResult {
    /// Pose relative to the first tracked frame (opentrack convention, unfiltered).
    pub pose: [f64; 6],
    pub reproj_px: f64,
    pub distance_cm: f64,
    pub roi: Roi,
}

impl Pipeline {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self { tracker: Tracker::new(), models: Models { detector: FaceDetector::new()?, landmarks: LandmarkModel::new()? } })
    }

    pub fn timings(&self) -> Timings {
        self.tracker.timings
    }

    pub fn step(&mut self, frame: &Frame, cfg: &Config) -> Option<StepResult> {
        let m = self.tracker.measure(frame, cfg, &mut self.models)?;
        let center = *self.tracker.center.get_or_insert(m.pose);
        let pivot = Vector3::new(HEAD_CENTER[0], HEAD_CENTER[1], HEAD_CENTER[2]);
        Some(StepResult {
            pose: relative_pose6(&center, &m.pose, &pivot).v,
            reproj_px: m.rms,
            distance_cm: m.pose.t.norm(),
            roi: m.roi,
        })
    }
}

fn set_message(shared: &Shared, msg: &str) {
    let mut s = shared.state.lock();
    s.message = msg.to_string();
    if s.tracking {
        s.tracking = false;
        s.lost_since = Some(Instant::now());
    }
}

struct Tracker {
    roi: Option<Roi>,
    center: Option<Pose>,
    last_pose: Option<(Pose, Instant)>,
    glitches: u32,
    filter: PoseFilter,
    last_measure: Option<Instant>,
    /// Where the face was last seen, to re-acquire the same person after a brief loss.
    last_seen: Option<(Roi, Instant)>,
    last_frame_at: Option<Instant>,
    fps: f32,
    timings: Timings,
    model: Vec<Vector3<f64>>,
}

struct Measurement {
    pose: Pose,
    rms: f64,
    points: Vec<[f32; 3]>,
    roi: Roi,
}

impl Tracker {
    fn new() -> Self {
        Self {
            roi: None,
            center: None,
            last_pose: None,
            glitches: 0,
            filter: PoseFilter::default(),
            last_measure: None,
            last_seen: None,
            last_frame_at: None,
            fps: 0.0,
            timings: Timings::default(),
            model: model_points(RIGID_LANDMARKS),
        }
    }

    fn run(&mut self, shared: &Shared, frames: &FrameSlot, m: &mut Models) {
        let mut last_seq = 0;
        let mut waiting_since = Instant::now();
        while !shared.quit.load(Ordering::SeqCst) {
            let Some(frame) = frames.wait_newer(last_seq, Duration::from_millis(250)) else {
                if waiting_since.elapsed() > Duration::from_secs(2) {
                    set_message(shared, "Waiting for camera frames…");
                }
                continue;
            };
            waiting_since = Instant::now();
            last_seq = frame.seq;
            if !frame.is_valid() {
                continue;
            }
            let cfg = shared.config.read().clone();
            let t0 = Instant::now();
            let meas = self.measure(&frame, &cfg, m);
            self.timings.total_ms = ema(self.timings.total_ms, ms(t0));
            if let Some(prev) = self.last_frame_at {
                let dt = frame.captured_at.duration_since(prev).as_secs_f32();
                if dt > 0.0 {
                    self.fps = if self.fps == 0.0 { 1.0 / dt } else { self.fps * 0.9 + 0.1 / dt };
                }
            }
            self.last_frame_at = Some(frame.captured_at);
            self.publish(shared, &frame, &cfg, meas);
        }
    }

    /// Locate the face and solve its pose. Returns None when no reliable face is in this frame.
    fn measure(&mut self, frame: &Frame, cfg: &Config, m: &mut Models) -> Option<Measurement> {
        let mut detected_now = false;
        if self.roi.is_none() {
            self.roi = self.detect(frame, m);
            detected_now = true;
        }
        let roi = self.roi?;
        let mut res = self.landmarks(frame, &roi, m);
        if res.is_none() && !detected_now {
            // Lost the face between frames (fast move): re-detect on this same frame.
            self.roi = self.detect(frame, m);
            res = self.roi.and_then(|r| self.landmarks(frame, &r, m));
        }
        let Some((points, used_roi)) = res else {
            self.roi = None;
            return None;
        };

        let next = Roi::from_landmarks(&points, frame);
        self.roi = if next.valid() { Some(next.with_hysteresis(&used_roi)) } else { None };
        self.last_seen = Some((used_roi, frame.captured_at));

        let t = Instant::now();
        let k = Intrinsics::from_hfov(frame.width, frame.height, cfg.camera_hfov as f64);
        let obs: Vec<_> = RIGID_LANDMARKS.iter().map(|&i| Vector2::new(points[i][0] as f64, points[i][1] as f64)).collect();
        let img3: Vec<_> = RIGID_LANDMARKS
            .iter()
            .map(|&i| Vector3::new(points[i][0] as f64, points[i][1] as f64, points[i][2] as f64))
            .collect();
        // Closed-form weak-perspective guess, then perspective-correct refinement.
        let (s, rot, tr) = umeyama(&self.model, &img3)?;
        let z = k.f / s;
        let init = Pose { rot, t: Vector3::new((tr.x - k.cx) * z / k.f, (tr.y - k.cy) * z / k.f, z) };
        let face_px = used_roi.size as f64 / 1.5;
        let res = solve_pnp(&self.model, &obs, &k, init, (face_px * 0.02).max(1.0))?;
        if log::log_enabled!(log::Level::Debug) {
            let e = |r: &nalgebra::Rotation3<f64>| { let (a, b, c) = r.euler_angles(); format!("[{:.1} {:.1} {:.1}]", a.to_degrees(), b.to_degrees(), c.to_degrees()) };
            log::debug!("init t {:.1?} R {} | pnp t {:.1?} R {} rms {:.1} | roi c=({:.0},{:.0}) s={:.0} a={:.1}",
                init.t.as_slice(), e(&init.rot), res.pose.t.as_slice(), e(&res.pose.rot), res.rms, used_roi.cx, used_roi.cy, used_roi.size, used_roi.angle.to_degrees());
        }
        self.timings.pose_ms = ema(self.timings.pose_ms, ms(t));

        let sane = res.rms < face_px * 0.12 && res.pose.t.z > 3.0 && res.pose.t.z < 400.0;
        if !sane {
            return None;
        }
        Some(Measurement { pose: res.pose, rms: res.rms, points, roi: used_roi })
    }

    fn detect(&mut self, frame: &Frame, m: &mut Models) -> Option<Roi> {
        let t = Instant::now();
        let dets = match m.detector.detect(frame) {
            Ok(d) => d,
            Err(e) => {
                log::warn!("detector: {e:#}");
                return None;
            }
        };
        self.timings.detect_ms = ema(self.timings.detect_ms, ms(t));
        // Just lost the face: take the detection nearest to where it was, so someone walking
        // behind you can't steal the tracking. Otherwise the user is normally the biggest
        // face (closest to the camera).
        let recent = self.last_seen.filter(|(_, t)| frame.captured_at.saturating_duration_since(*t) < Duration::from_secs(2));
        let near = recent.and_then(|(r, _)| {
            let dist = |d: &Detection| ((d.cx - r.cx).powi(2) + (d.cy - r.cy).powi(2)).sqrt() / r.size;
            dets.iter().filter(|d| dist(d) < 0.8).min_by(|a, b| dist(a).total_cmp(&dist(b)))
        });
        let best = match near {
            Some(d) => d,
            None => dets.iter().max_by(|a, b| (a.w * a.h * a.score).total_cmp(&(b.w * b.h * b.score)))?,
        };
        let roi = Roi::from_detection(best, frame);
        roi.valid().then_some(roi)
    }

    fn landmarks(&mut self, frame: &Frame, roi: &Roi, m: &mut Models) -> Option<(Vec<[f32; 3]>, Roi)> {
        if !roi.valid() {
            return None;
        }
        let t = Instant::now();
        let crop = roi.affine();
        let res = match m.landmarks.run(frame, &crop) {
            Ok(r) => r,
            Err(e) => {
                log::warn!("landmarks: {e:#}");
                return None;
            }
        };
        self.timings.landmark_ms = ema(self.timings.landmark_ms, ms(t));
        if res.presence < PRESENCE_MIN || res.points.len() < 468 {
            return None;
        }
        let k = crop.scale_factor();
        let pts: Vec<[f32; 3]> = res
            .points
            .iter()
            .map(|p| {
                let (x, y) = crop.apply(p[0], p[1]);
                [x, y, p[2] * k]
            })
            .collect();
        if pts.iter().any(|p| !p.iter().all(|v| v.is_finite())) {
            return None;
        }
        Some((pts, *roi))
    }

    fn publish(&mut self, shared: &Shared, frame: &Frame, cfg: &Config, meas: Option<Measurement>) {
        let now = frame.captured_at;
        let mut accepted = None;
        if let Some(ms_) = meas {
            // Reject single-frame glitches (a sudden >45° flip), but never more than 3 in a row.
            let glitch = match self.last_pose {
                Some((p, t)) if now.duration_since(t) < Duration::from_millis(200) => {
                    (p.rot.inverse() * ms_.pose.rot).angle() > 45f64.to_radians()
                }
                _ => false,
            };
            if glitch && self.glitches < 3 {
                self.glitches += 1;
            } else {
                self.glitches = 0;
                self.last_pose = Some((ms_.pose, now));
                accepted = Some(ms_);
            }
        }

        let recenter = shared.recenter.load(Ordering::SeqCst);
        let mut st = shared.state.lock();
        st.fps = self.fps;
        st.timings = self.timings;
        match &accepted {
            Some(ms_) => {
                if self.center.is_none() || recenter {
                    self.center = Some(ms_.pose);
                    self.filter.reset();
                    shared.recenter.store(false, Ordering::SeqCst);
                    st.center_epoch += 1;
                }
                let center = self.center.as_ref().expect("center set");
                let pivot = Vector3::new(HEAD_CENTER[0], HEAD_CENTER[1], HEAD_CENTER[2]);
                let rel = relative_pose6(center, &ms_.pose, &pivot).v;
                let dt = self.last_measure.map(|t| now.duration_since(t).as_secs_f64()).unwrap_or(1.0 / 30.0);
                // After a gap the velocity estimate is meaningless; restart the filter.
                if dt > 0.5 {
                    self.filter.reset();
                }
                let (f, v) = self.filter.filter(&rel, dt, &cfg.smoothing);
                self.last_measure = Some(now);
                st.raw = rel;
                st.filtered = f;
                st.velocity = v;
                st.measured_at = now;
                st.tracking = true;
                st.lost_since = None;
                st.reproj_px = ms_.rms as f32;
                st.distance_cm = ms_.pose.t.norm() as f32;
                st.message = if recenter { "Recentred".into() } else { "Tracking".into() };
            }
            None => {
                if st.tracking {
                    st.lost_since = Some(now);
                }
                st.tracking = false;
                st.velocity = [0.0; 6];
                st.message = if self.roi.is_some() { "Face unclear…".into() } else { "Searching for face…".into() };
            }
        }
        drop(st);

        if cfg.show_preview {
            self.make_preview(shared, frame, cfg, accepted.as_ref());
        }
        shared.notify();
    }

    fn make_preview(&self, shared: &Shared, frame: &Frame, cfg: &Config, meas: Option<&Measurement>) {
        let (w, h, rgba) = preview_rgba(frame, 640, cfg.mirror_preview);
        let (fw, fh) = (frame.width as f32, frame.height as f32);
        let norm = |x: f32, y: f32| {
            let u = x / fw;
            (if cfg.mirror_preview { 1.0 - u } else { u }, y / fh)
        };
        let mut p = Preview { width: w, height: h, rgba: Arc::new(rgba), ..Default::default() };
        if let Some(ms_) = meas {
            if cfg.show_landmarks {
                p.landmarks = ms_.points.iter().take(468).map(|q| norm(q[0], q[1])).collect();
                p.rigid = RIGID_LANDMARKS.iter().map(|&i| norm(ms_.points[i][0], ms_.points[i][1])).collect();
            }
            let c = ms_.roi.corners();
            p.roi = Some([norm(c[0].0, c[0].1), norm(c[1].0, c[1].1), norm(c[2].0, c[2].1), norm(c[3].0, c[3].1)]);
            let k = Intrinsics::from_hfov(frame.width, frame.height, cfg.camera_hfov as f64);
            let o = Vector3::new(HEAD_CENTER[0], HEAD_CENTER[1], HEAD_CENTER[2] + 12.0);
            // Model +x is the subject's left; draw the "right" axis as -x.
            let tips = [o, o + Vector3::new(-6.0, 0.0, 0.0), o + Vector3::new(0.0, 6.0, 0.0), o + Vector3::new(0.0, 0.0, 6.0)];
            let proj: Vec<_> = tips.iter().filter_map(|t| k.project(&ms_.pose.transform(t))).collect();
            if proj.len() == 4 {
                let n = |v: &Vector2<f64>| norm(v.x as f32, v.y as f32);
                p.axes = Some([n(&proj[0]), n(&proj[1]), n(&proj[2]), n(&proj[3])]);
            }
        } else if let Some(r) = self.roi {
            let c = r.corners();
            p.roi = Some([norm(c[0].0, c[0].1), norm(c[1].0, c[1].1), norm(c[2].0, c[2].1), norm(c[3].0, c[3].1)]);
        }
        let mut slot = shared.preview.lock();
        p.seq = slot.seq + 1;
        *slot = p;
    }
}

fn ms(t: Instant) -> f32 {
    t.elapsed().as_secs_f32() * 1000.0
}

fn ema(prev: f32, v: f32) -> f32 {
    if prev == 0.0 { v } else { prev * 0.9 + v * 0.1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: u32, h: u32) -> Frame {
        Frame { width: w, height: h, rgb: Arc::new(vec![0; (w * h * 3) as usize]), captured_at: Instant::now(), seq: 1 }
    }

    #[test]
    fn roi_affine_maps_crop_centre_to_roi_centre() {
        let r = Roi { cx: 300.0, cy: 200.0, size: 256.0, angle: 0.3 };
        let (x, y) = r.affine().apply(128.0, 128.0);
        assert!((x - 300.0).abs() < 1e-3 && (y - 200.0).abs() < 1e-3);
        // The crop's x axis follows the ROI angle.
        let (x2, y2) = r.affine().apply(228.0, 128.0);
        assert!(((y2 - y).atan2(x2 - x) - 0.3).abs() < 1e-4);
    }

    #[test]
    fn roi_from_landmarks_is_square_around_points() {
        let f = frame(640, 480);
        let mut pts = vec![[0.0f32; 3]; 478];
        for (i, p) in pts.iter_mut().enumerate() {
            let a = i as f32 / 478.0 * std::f32::consts::TAU;
            *p = [320.0 + 50.0 * a.cos(), 240.0 + 80.0 * a.sin(), 0.0];
        }
        pts[EYE_LEFT_IMG] = [290.0, 220.0, 0.0];
        pts[EYE_RIGHT_IMG] = [350.0, 220.0, 0.0];
        let r = Roi::from_landmarks(&pts, &f);
        assert!(r.angle.abs() < 1e-6);
        assert!((r.cx - 320.0).abs() < 1.0 && (r.cy - 240.0).abs() < 1.0);
        assert!((r.size - 160.0 * 1.5).abs() < 2.0);
    }

    #[test]
    fn hysteresis_keeps_tiny_changes_only() {
        let a = Roi { cx: 100.0, cy: 100.0, size: 200.0, angle: 0.0 };
        let small = Roi { cx: 101.0, ..a };
        let big = Roi { cx: 120.0, ..a };
        assert_eq!(small.with_hysteresis(&a).cx, 100.0);
        assert_eq!(big.with_hysteresis(&a).cx, 120.0);
    }
}
