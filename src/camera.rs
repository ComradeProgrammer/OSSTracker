//! Camera capture on its own thread.
//!
//! The capture thread never takes the app down: open failures, unplugged cameras and decode
//! errors are reported in [`CameraStatus`] and retried. Only the newest frame is kept, so a slow
//! tracker never builds up latency.

use crate::image_ops::Frame;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution};
use nokhwa::Camera;
use parking_lot::{Condvar, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CameraRequest {
    /// Device name; empty = first device.
    pub name: String,
    /// (width, height, fps); None = automatic.
    pub mode: Option<(u32, u32, u32)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Mode {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{} @ {} fps", self.width, self.height, self.fps)
    }
}

#[derive(Clone, Debug, Default)]
pub struct CameraStatus {
    pub devices: Vec<String>,
    pub modes: Vec<Mode>,
    pub active: Option<(Mode, String)>,
    pub error: Option<String>,
    pub fps: f32,
    pub decode_ms: f32,
}

#[derive(Default)]
pub struct FrameSlot {
    frame: Mutex<Option<Frame>>,
    cv: Condvar,
}

impl FrameSlot {
    pub fn put(&self, f: Frame) {
        *self.frame.lock() = Some(f);
        self.cv.notify_all();
    }

    /// Wait for a frame newer than `after_seq`.
    pub fn wait_newer(&self, after_seq: u64, timeout: Duration) -> Option<Frame> {
        let mut g = self.frame.lock();
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(f) = g.as_ref() {
                if f.seq > after_seq {
                    return Some(f.clone());
                }
            }
            if self.cv.wait_until(&mut g, deadline).timed_out() {
                return None;
            }
        }
    }
}

pub struct CameraHandle {
    pub frames: Arc<FrameSlot>,
    pub status: Arc<Mutex<CameraStatus>>,
    generation: Arc<AtomicU64>,
    seq: Arc<AtomicU64>,
}

impl CameraHandle {
    pub fn new() -> Self {
        Self {
            frames: Arc::new(FrameSlot::default()),
            status: Arc::new(Mutex::new(CameraStatus::default())),
            generation: Arc::new(AtomicU64::new(0)),
            seq: Arc::new(AtomicU64::new(0)),
        }
    }

    /// (Re)start capture with `req`. Any previous capture thread stops at its next frame.
    pub fn start(&self, req: CameraRequest) {
        let gen = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        let ctx = Ctx {
            frames: self.frames.clone(),
            status: self.status.clone(),
            generation: self.generation.clone(),
            seq: self.seq.clone(),
            gen,
        };
        let r = std::thread::Builder::new().name("camera".into()).spawn(move || {
            // A panic inside a camera backend must not kill the app; report it and retry.
            loop {
                let ctx2 = ctx.clone();
                let req2 = req.clone();
                let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || capture_loop(&ctx2, &req2)));
                if res.is_ok() || !ctx.alive() {
                    return;
                }
                ctx.set_error("Camera driver crashed; restarting…".into());
                std::thread::sleep(Duration::from_secs(1));
            }
        });
        if let Err(e) = r {
            self.status.lock().error = Some(format!("cannot start camera thread: {e}"));
        }
    }

    pub fn stop(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
    }
}

impl Default for CameraHandle {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
struct Ctx {
    frames: Arc<FrameSlot>,
    status: Arc<Mutex<CameraStatus>>,
    generation: Arc<AtomicU64>,
    seq: Arc<AtomicU64>,
    gen: u64,
}

impl Ctx {
    fn alive(&self) -> bool {
        self.generation.load(Ordering::SeqCst) == self.gen
    }
    fn set_error(&self, e: String) {
        if self.alive() {
            let mut s = self.status.lock();
            if s.error.as_deref() != Some(e.as_str()) {
                log::warn!("camera: {e}");
            }
            s.error = Some(e);
            s.active = None;
            s.fps = 0.0;
        }
    }
    /// Sleep in small steps so a restart request is honoured quickly.
    fn sleep(&self, d: Duration) {
        let end = Instant::now() + d;
        while self.alive() && Instant::now() < end {
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

fn capture_loop(ctx: &Ctx, req: &CameraRequest) {
    while ctx.alive() {
        match open_camera(ctx, req) {
            Ok(mut cam) => {
                stream(ctx, &mut cam);
                let _ = cam.stop_stream();
            }
            Err(e) => ctx.set_error(e),
        }
        ctx.sleep(Duration::from_secs(2));
    }
}

fn open_camera(ctx: &Ctx, req: &CameraRequest) -> Result<Camera, String> {
    #[cfg(target_os = "macos")]
    if !nokhwa::nokhwa_check() {
        return Err("Camera access not granted. Allow it in System Settings > Privacy & Security > Camera (retrying automatically).".into());
    }
    let devices = nokhwa::query(ApiBackend::Auto).map_err(|e| format!("cannot list cameras: {e}"))?;
    let names: Vec<String> = devices.iter().map(|d| d.human_name()).collect();
    if ctx.alive() {
        ctx.status.lock().devices = names.clone();
    }
    if devices.is_empty() {
        return Err("No camera found. Plug in a camera; MyTracker will pick it up automatically.".into());
    }
    let info = devices
        .iter()
        .find(|d| !req.name.is_empty() && d.human_name() == req.name)
        .unwrap_or(&devices[0]);
    let index: CameraIndex = info.index().clone();

    let mut cam = Camera::new(index.clone(), RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate))
        .or_else(|_| Camera::new(index.clone(), RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestResolution)))
        .map_err(|e| format!("cannot open \"{}\": {e} (is another app using it?)", info.human_name()))?;

    let formats = list_formats(&mut cam);
    let mut modes: Vec<Mode> = formats.iter().map(mode_of).collect();
    modes.sort_by(|a, b| b.cmp(a));
    modes.dedup();

    let wanted = req
        .mode
        .and_then(|(w, h, fps)| best_format_for(&formats, w, h, fps))
        .or_else(|| choose_auto(&formats));
    if let Some(fmt) = wanted {
        if fmt != cam.camera_format() {
            if let Err(e) = cam.set_camera_requset(RequestedFormat::new::<RgbFormat>(RequestedFormatType::Exact(fmt))) {
                log::warn!("camera: could not switch to {fmt}: {e}; keeping {}", cam.camera_format());
            }
        }
    }
    cam.open_stream().map_err(|e| format!("cannot start \"{}\": {e} (is another app using it?)", info.human_name()))?;
    let active = cam.camera_format();
    log::info!("camera: {} opened at {active}", info.human_name());
    if ctx.alive() {
        let mut s = ctx.status.lock();
        s.modes = modes;
        s.active = Some((mode_of(&active), format!("{}", active.format())));
        s.error = None;
    }
    Ok(cam)
}

fn stream(ctx: &Ctx, cam: &mut Camera) {
    let mut errors = 0;
    let mut last = Instant::now();
    let mut fps = 0.0f32;
    let mut decode_ms = 0.0f32;
    while ctx.alive() {
        let buf = match cam.frame() {
            Ok(b) => b,
            Err(e) => {
                errors += 1;
                if errors > 30 {
                    ctx.set_error(format!("camera stopped delivering frames: {e}"));
                    return;
                }
                std::thread::sleep(Duration::from_millis(10));
                continue;
            }
        };
        let captured_at = Instant::now();
        let t = Instant::now();
        let (w, h, rgb) = match decode(&buf) {
            Ok(v) => v,
            Err(e) => {
                errors += 1;
                if errors > 30 {
                    ctx.set_error(format!("cannot decode camera frames: {e}"));
                    return;
                }
                continue;
            }
        };
        errors = 0;
        if w == 0 || h == 0 || rgb.len() < (w * h * 3) as usize {
            continue;
        }
        let dt = captured_at.duration_since(last).as_secs_f32();
        last = captured_at;
        if dt > 0.0 {
            fps = if fps == 0.0 { 1.0 / dt } else { fps * 0.9 + 0.1 / dt };
        }
        decode_ms = decode_ms * 0.9 + 0.1 * t.elapsed().as_secs_f32() * 1000.0;
        let seq = ctx.seq.fetch_add(1, Ordering::SeqCst) + 1;
        ctx.frames.put(Frame { width: w, height: h, rgb: Arc::new(rgb), captured_at, seq });
        if seq.is_multiple_of(10) {
            let mut s = ctx.status.lock();
            s.fps = fps;
            s.decode_ms = decode_ms;
        }
    }
}

/// Decode a camera buffer to packed RGB8. MJPEG goes through zune-jpeg (pure Rust, handles
/// webcam frames without Huffman tables); raw formats use nokhwa's converters.
fn decode(buf: &nokhwa::Buffer) -> Result<(u32, u32, Vec<u8>), String> {
    if buf.source_frame_format() == FrameFormat::MJPEG {
        use zune_core::colorspace::ColorSpace;
        use zune_core::options::DecoderOptions;
        let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB).set_max_width(8192).set_max_height(8192);
        let mut d = zune_jpeg::JpegDecoder::new_with_options(buf.buffer(), opts);
        let px = d.decode().map_err(|e| format!("{e:?}"))?;
        let (w, h) = d.dimensions().ok_or("jpeg without dimensions")?;
        if px.len() != w * h * 3 {
            return Err(format!("jpeg decoded to {} bytes for {w}x{h}", px.len()));
        }
        Ok((w as u32, h as u32, px))
    } else {
        let img = buf.decode_image::<RgbFormat>().map_err(|e| e.to_string())?;
        Ok((img.width(), img.height(), img.into_raw()))
    }
}

fn mode_of(f: &CameraFormat) -> Mode {
    Mode { width: f.resolution().width(), height: f.resolution().height(), fps: f.frame_rate() }
}

const DECODABLE: [FrameFormat; 5] = [FrameFormat::MJPEG, FrameFormat::YUYV, FrameFormat::NV12, FrameFormat::RAWRGB, FrameFormat::GRAY];

/// All formats the camera reports. Some backends (notably macOS) only answer one of the
/// two query styles, so both are merged.
fn list_formats(cam: &mut Camera) -> Vec<CameraFormat> {
    let mut out: Vec<CameraFormat> = cam.compatible_camera_formats().unwrap_or_default();
    for ff in DECODABLE {
        if let Ok(map) = cam.compatible_list_by_resolution(ff) {
            for (res, rates) in map {
                for r in rates {
                    let f = CameraFormat::new(res, ff, r);
                    if !out.contains(&f) {
                        out.push(f);
                    }
                }
            }
        }
    }
    out.retain(|f| DECODABLE.contains(&f.format()) && f.resolution().width() > 0 && f.resolution().height() > 0);
    if out.is_empty() {
        out.push(cam.camera_format());
    }
    out
}

fn format_bonus(f: FrameFormat, height: u32) -> f64 {
    match f {
        // Uncompressed avoids JPEG artefacts and decode time when the bandwidth allows it.
        FrameFormat::YUYV | FrameFormat::NV12 if height <= 720 => 3.0,
        FrameFormat::MJPEG => 1.0,
        _ => 0.0,
    }
}

/// Automatic mode choice. Forcing a 4:3 mode such as 640x480 crops a 16:9 sensor, so:
/// keep the sensor's full field of view (aspect ratio of its largest mode), then prefer
/// a high frame rate, then a resolution around 720p (enough detail, cheap to decode).
pub fn choose_auto(formats: &[CameraFormat]) -> Option<CameraFormat> {
    let largest = formats.iter().max_by_key(|f| f.resolution().width() as u64 * f.resolution().height() as u64)?;
    let native_aspect = largest.resolution().width() as f64 / largest.resolution().height() as f64;
    formats
        .iter()
        .max_by(|a, b| score(a, native_aspect).total_cmp(&score(b, native_aspect)))
        .copied()
}

fn score(f: &CameraFormat, native_aspect: f64) -> f64 {
    let (w, h) = (f.resolution().width() as f64, f.resolution().height() as f64);
    let aspect_penalty = if ((w / h) - native_aspect).abs() > 0.03 { 1000.0 } else { 0.0 };
    let fps = (f.frame_rate() as f64).min(90.0);
    let fps_score = if fps >= 25.0 { 300.0 + fps * 2.0 } else { fps * 10.0 };
    let res_penalty = if h < 480.0 { (480.0 - h) * 0.5 } else if h > 1080.0 { (h - 1080.0) * 0.3 } else { (h - 720.0).abs() * 0.05 };
    fps_score - aspect_penalty - res_penalty + format_bonus(f.format(), h as u32)
}

fn best_format_for(formats: &[CameraFormat], w: u32, h: u32, fps: u32) -> Option<CameraFormat> {
    formats
        .iter()
        .filter(|f| f.resolution() == Resolution::new(w, h) && f.frame_rate() == fps)
        .max_by(|a, b| format_bonus(a.format(), h).total_cmp(&format_bonus(b.format(), h)))
        .copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(w: u32, h: u32, ff: FrameFormat, fps: u32) -> CameraFormat {
        CameraFormat::new(Resolution::new(w, h), ff, fps)
    }

    #[test]
    fn auto_prefers_full_fov_over_cropped_4_3() {
        let fmts = [f(640, 480, FrameFormat::YUYV, 30), f(1280, 720, FrameFormat::MJPEG, 30), f(1920, 1080, FrameFormat::MJPEG, 30)];
        let c = choose_auto(&fmts).unwrap();
        assert_eq!(c.resolution(), Resolution::new(1280, 720));
    }

    #[test]
    fn auto_prefers_60fps_and_avoids_slow_uncompressed() {
        let fmts = [
            f(1920, 1080, FrameFormat::YUYV, 5),
            f(1280, 720, FrameFormat::YUYV, 10),
            f(1280, 720, FrameFormat::MJPEG, 60),
            f(1920, 1080, FrameFormat::MJPEG, 30),
        ];
        let c = choose_auto(&fmts).unwrap();
        assert_eq!((c.resolution(), c.frame_rate(), c.format()), (Resolution::new(1280, 720), 60, FrameFormat::MJPEG));
    }

    #[test]
    fn auto_keeps_4_3_sensor() {
        let fmts = [f(640, 480, FrameFormat::YUYV, 30), f(320, 240, FrameFormat::YUYV, 60), f(1600, 1200, FrameFormat::MJPEG, 15)];
        let c = choose_auto(&fmts).unwrap();
        assert_eq!(c.resolution(), Resolution::new(640, 480));
    }
}

#[cfg(test)]
mod decode_tests {
    use super::*;

    #[test]
    fn mjpeg_frames_decode_to_rgb() {
        let img = image::RgbImage::from_fn(64, 48, |x, y| image::Rgb([(x * 4) as u8, (y * 5) as u8, 128]));
        let mut jpg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpg, 95).encode_image(&img).unwrap();
        let buf = nokhwa::Buffer::new(Resolution::new(64, 48), &jpg, FrameFormat::MJPEG);
        let (w, h, px) = decode(&buf).unwrap();
        assert_eq!((w, h, px.len()), (64, 48, 64 * 48 * 3));
        let i = (20 * 64 + 30) * 3;
        assert!((px[i] as i32 - 120).abs() < 8 && (px[i + 1] as i32 - 100).abs() < 8);
        // Garbage must be an error, not a panic.
        let bad = nokhwa::Buffer::new(Resolution::new(64, 48), &[0xFF, 0xD8, 1, 2, 3], FrameFormat::MJPEG);
        assert!(decode(&bad).is_err());
    }
}
