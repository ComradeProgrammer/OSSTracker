//! Neural network wrappers: BlazeFace (face detection) and MediaPipe Face Mesh V2 (478 landmarks).
//!
//! Both models are embedded into the binary, so a missing asset file can never stop the app from starting.

use crate::image_ops::{warp_to_tensor, Affine2, Frame};
use anyhow::{anyhow, Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::sync::OnceLock;

static DETECTOR_ONNX: &[u8] = include_bytes!("../models/blaze_face_short_range.onnx");
static DETECTOR_ANCHORS: &str = include_str!("../models/anchors.csv");
static LANDMARK_ONNX: &[u8] = include_bytes!("../models/face_landmarks.onnx");

const DET_SIZE: usize = 128;
pub const LM_SIZE: usize = 256;
pub const NUM_LANDMARKS: usize = 478;

fn environment() -> Result<&'static ort::environment::Environment> {
    static ENV: OnceLock<std::result::Result<ort::environment::Environment, String>> = OnceLock::new();
    ENV.get_or_init(|| ort::init().with_name("mytracker").build().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| anyhow!("ONNX Runtime init failed: {e}"))
}

fn load(bytes: &[u8], threads: usize) -> Result<Session> {
    let e = |e: ort::Error<ort::session::builder::SessionBuilder>| anyhow!("{e}");
    Session::builder(environment()?)
        .map_err(|e| anyhow!("{e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(e)?
        .with_intra_threads(threads)
        .map_err(e)?
        .commit_from_memory(bytes)
        .map_err(|e| anyhow!("{e}"))
}

/// Run `session` on one NHWC f32 image and return every output flattened.
fn run(session: &mut Session, input: &[f32], size: usize) -> Result<Vec<Vec<f32>>> {
    let tensor = Tensor::from_array(([1usize, size, size, 3], input.to_vec().into_boxed_slice()))
        .map_err(|e| anyhow!("{e}"))?;
    let out = session.run(ort::inputs![tensor]).map_err(|e| anyhow!("{e}"))?;
    let mut res = Vec::with_capacity(out.len());
    for (_, v) in out.iter() {
        let (_, data) = v.try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        res.push(data.to_vec());
    }
    Ok(res)
}

#[derive(Clone, Debug)]
pub struct Detection {
    pub score: f32,
    /// Box centre in frame pixels.
    pub cx: f32,
    pub cy: f32,
    /// Box width/height in frame pixels.
    pub w: f32,
    pub h: f32,
    /// Keypoints in frame pixels: 0 right eye, 1 left eye, 2 nose, 3 mouth, 4 right ear, 5 left ear
    /// ("right"/"left" from the subject's point of view, so keypoint 0 is on the image left).
    pub keypoints: [(f32, f32); 6],
}

pub struct FaceDetector {
    session: Session,
    anchors: Vec<(f32, f32)>,
    input: Vec<f32>,
    pub min_score: f32,
}

impl FaceDetector {
    pub fn new() -> Result<Self> {
        let session = load(DETECTOR_ONNX, 1).context("loading face detector")?;
        let anchors: Vec<(f32, f32)> = DETECTOR_ANCHORS
            .lines()
            .filter_map(|l| {
                let mut it = l.split(',').map(|v| v.trim().parse::<f32>());
                match (it.next(), it.next()) {
                    (Some(Ok(x)), Some(Ok(y))) => Some((x, y)),
                    _ => None,
                }
            })
            .collect();
        if anchors.len() != 896 {
            return Err(anyhow!("expected 896 detector anchors, got {}", anchors.len()));
        }
        Ok(Self { session, anchors, input: vec![0.0; DET_SIZE * DET_SIZE * 3], min_score: 0.5 })
    }

    /// Detect faces anywhere in the frame. Tries the normal letterboxed view first, then a
    /// zoomed-out view, which catches faces too large for the first pass (very close to the camera).
    pub fn detect(&mut self, frame: &Frame) -> Result<Vec<Detection>> {
        let mut dets = self.detect_at(frame, 1.0)?;
        if dets.is_empty() {
            dets = self.detect_at(frame, 0.6)?;
        }
        Ok(dets)
    }

    fn detect_at(&mut self, frame: &Frame, zoom: f32) -> Result<Vec<Detection>> {
        let (w, h) = (frame.width as f32, frame.height as f32);
        // Letterbox: keep aspect ratio, never crop the frame.
        let s = DET_SIZE as f32 * zoom / w.max(h);
        let ou = DET_SIZE as f32 * 0.5 - w * s * 0.5;
        let ov = DET_SIZE as f32 * 0.5 - h * s * 0.5;
        let dst_to_src = Affine2 { m: [1.0 / s, 0.0, -ou / s, 0.0, 1.0 / s, -ov / s] };
        warp_to_tensor(frame, &dst_to_src, DET_SIZE, DET_SIZE, 1.0 / 127.5, -1.0, &mut self.input);

        let out = run(&mut self.session, &self.input, DET_SIZE)?;
        if out.len() < 2 {
            return Err(anyhow!("detector returned {} outputs", out.len()));
        }
        let (reg, cls) = if out[0].len() == 896 * 16 { (&out[0], &out[1]) } else { (&out[1], &out[0]) };
        if reg.len() != 896 * 16 || cls.len() != 896 {
            return Err(anyhow!("unexpected detector output sizes {} / {}", reg.len(), cls.len()));
        }

        let to_frame = |u: f32, v: f32| ((u * DET_SIZE as f32 - ou) / s, (v * DET_SIZE as f32 - ov) / s);
        let mut cands = Vec::new();
        for i in 0..896 {
            let score = sigmoid(cls[i].clamp(-100.0, 100.0));
            if score < self.min_score || !score.is_finite() {
                continue;
            }
            let r = &reg[i * 16..i * 16 + 16];
            let (ax, ay) = self.anchors[i];
            let n = DET_SIZE as f32;
            let (cx, cy) = to_frame(ax + r[0] / n, ay + r[1] / n);
            let bw = r[2] / n * DET_SIZE as f32 / s;
            let bh = r[3] / n * DET_SIZE as f32 / s;
            let mut keypoints = [(0.0, 0.0); 6];
            for (k, kp) in keypoints.iter_mut().enumerate() {
                *kp = to_frame(ax + r[4 + 2 * k] / n, ay + r[5 + 2 * k] / n);
            }
            if bw > 1.0 && bh > 1.0 && cx.is_finite() && cy.is_finite() {
                cands.push(Detection { score, cx, cy, w: bw, h: bh, keypoints });
            }
        }
        Ok(weighted_nms(cands, 0.3))
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn iou(a: &Detection, b: &Detection) -> f32 {
    let (ax0, ay0, ax1, ay1) = (a.cx - a.w / 2.0, a.cy - a.h / 2.0, a.cx + a.w / 2.0, a.cy + a.h / 2.0);
    let (bx0, by0, bx1, by1) = (b.cx - b.w / 2.0, b.cy - b.h / 2.0, b.cx + b.w / 2.0, b.cy + b.h / 2.0);
    let iw = (ax1.min(bx1) - ax0.max(bx0)).max(0.0);
    let ih = (ay1.min(by1) - ay0.max(by0)).max(0.0);
    let inter = iw * ih;
    let union = a.w * a.h + b.w * b.h - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// MediaPipe-style weighted non-maximum suppression: overlapping boxes are averaged, weighted by score.
fn weighted_nms(mut cands: Vec<Detection>, thresh: f32) -> Vec<Detection> {
    cands.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut out = Vec::new();
    while !cands.is_empty() {
        let top = cands[0].clone();
        let (cluster, rest): (Vec<_>, Vec<_>) = cands.into_iter().partition(|c| iou(&top, c) > thresh);
        cands = rest;
        let total: f32 = cluster.iter().map(|c| c.score).sum();
        if cluster.is_empty() || total <= 0.0 {
            out.push(top);
            continue;
        }
        let mut d = Detection { score: top.score, cx: 0.0, cy: 0.0, w: 0.0, h: 0.0, keypoints: [(0.0, 0.0); 6] };
        for c in &cluster {
            let wgt = c.score / total;
            d.cx += c.cx * wgt;
            d.cy += c.cy * wgt;
            d.w += c.w * wgt;
            d.h += c.h * wgt;
            for k in 0..6 {
                d.keypoints[k].0 += c.keypoints[k].0 * wgt;
                d.keypoints[k].1 += c.keypoints[k].1 * wgt;
            }
        }
        out.push(d);
    }
    out
}

pub struct LandmarkResult {
    /// 478 landmarks in crop pixels (0..256): x right, y down, z depth (same scale as x, + away from camera).
    pub points: Vec<[f32; 3]>,
    /// Face presence probability (0..1).
    pub presence: f32,
}

pub struct LandmarkModel {
    session: Session,
    input: Vec<f32>,
}

impl LandmarkModel {
    pub fn new() -> Result<Self> {
        let session = load(LANDMARK_ONNX, 2).context("loading face landmark model")?;
        Ok(Self { session, input: vec![0.0; LM_SIZE * LM_SIZE * 3] })
    }

    /// Run the landmark model on the crop described by `crop_to_frame` (crop pixel -> frame pixel).
    pub fn run(&mut self, frame: &Frame, crop_to_frame: &Affine2) -> Result<LandmarkResult> {
        warp_to_tensor(frame, crop_to_frame, LM_SIZE, LM_SIZE, 1.0 / 255.0, 0.0, &mut self.input);
        let out = run(&mut self.session, &self.input, LM_SIZE)?;
        let lm = out
            .iter()
            .find(|t| t.len() == NUM_LANDMARKS * 3)
            .ok_or_else(|| anyhow!("landmark output not found"))?;
        let presence = out
            .get(1)
            .filter(|t| t.len() == 1)
            .map(|t| sigmoid(t[0].clamp(-100.0, 100.0)))
            .unwrap_or(0.0);
        let points = lm.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        Ok(LandmarkResult { points, presence })
    }
}
