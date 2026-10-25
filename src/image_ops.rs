//! Minimal image helpers: an RGB frame type and affine crop/resample into NN input tensors.
//!
//! Every sample outside the source frame reads as black instead of failing. That is what lets
//! the tracker follow a face that is partly out of frame or very close to the camera.

use std::sync::Arc;
use std::time::Instant;

/// A packed RGB8 camera frame.
#[derive(Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgb: Arc<Vec<u8>>,
    pub captured_at: Instant,
    pub seq: u64,
}

impl Frame {
    pub fn is_valid(&self) -> bool {
        self.width > 0 && self.height > 0 && self.rgb.len() >= (self.width * self.height * 3) as usize
    }
}

/// 2D affine transform mapping (x, y) -> (a*x + b*y + c, d*x + e*y + f).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine2 {
    pub m: [f32; 6],
}

impl Affine2 {
    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        let m = &self.m;
        (m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5])
    }

    /// Linear part only (no translation), for directions and lengths.
    pub fn apply_vec(&self, x: f32, y: f32) -> (f32, f32) {
        let m = &self.m;
        (m[0] * x + m[1] * y, m[3] * x + m[4] * y)
    }

    pub fn scale_factor(&self) -> f32 {
        let m = &self.m;
        (m[0] * m[4] - m[1] * m[3]).abs().sqrt()
    }
}

/// Resample `frame` into an NHWC f32 tensor of `out_w` x `out_h`.
/// `dst_to_src` maps output pixel centres to source pixel coordinates.
/// Values are `pixel * scale + offset` (e.g. scale 1/255, offset 0 for [0, 1]).
pub fn warp_to_tensor(
    frame: &Frame,
    dst_to_src: &Affine2,
    out_w: usize,
    out_h: usize,
    scale: f32,
    offset: f32,
    out: &mut [f32],
) {
    debug_assert!(out.len() >= out_w * out_h * 3);
    let w = frame.width as i32;
    let h = frame.height as i32;
    let src = frame.rgb.as_slice();
    let stride = (w * 3) as usize;
    let black = offset;
    let m = dst_to_src.m;

    for v in 0..out_h {
        let vy = v as f32 + 0.5;
        // Source position for the row start; step by the first column of the transform.
        let mut sx = m[0] * 0.5 + m[1] * vy + m[2] - 0.5;
        let mut sy = m[3] * 0.5 + m[4] * vy + m[5] - 0.5;
        let row = &mut out[v * out_w * 3..(v + 1) * out_w * 3];
        for px in row.chunks_exact_mut(3) {
            let x0 = sx.floor();
            let y0 = sy.floor();
            let fx = sx - x0;
            let fy = sy - y0;
            let x0 = x0 as i32;
            let y0 = y0 as i32;
            if x0 >= 0 && y0 >= 0 && x0 + 1 < w && y0 + 1 < h {
                let i00 = y0 as usize * stride + x0 as usize * 3;
                let i10 = i00 + 3;
                let i01 = i00 + stride;
                let i11 = i01 + 3;
                let w00 = (1.0 - fx) * (1.0 - fy);
                let w10 = fx * (1.0 - fy);
                let w01 = (1.0 - fx) * fy;
                let w11 = fx * fy;
                for c in 0..3 {
                    let p = src[i00 + c] as f32 * w00
                        + src[i10 + c] as f32 * w10
                        + src[i01 + c] as f32 * w01
                        + src[i11 + c] as f32 * w11;
                    px[c] = p * scale + offset;
                }
            } else if x0 >= -1 && y0 >= -1 && x0 < w && y0 < h {
                // Border: clamp-sample the nearest valid pixel so edges don't get a black halo.
                let cx = (sx.round() as i32).clamp(0, w - 1) as usize;
                let cy = (sy.round() as i32).clamp(0, h - 1) as usize;
                let i = cy * stride + cx * 3;
                for c in 0..3 {
                    px[c] = src[i + c] as f32 * scale + offset;
                }
            } else {
                px[0] = black;
                px[1] = black;
                px[2] = black;
            }
            sx += m[0];
            sy += m[3];
        }
    }
}

/// Downscale `frame` to RGBA for the UI preview (nearest neighbour; it's only a preview).
pub fn preview_rgba(frame: &Frame, max_width: u32, mirror: bool) -> (usize, usize, Vec<u8>) {
    let step = (frame.width as f32 / max_width as f32).max(1.0);
    let ow = ((frame.width as f32 / step) as usize).max(1);
    let oh = ((frame.height as f32 / step) as usize).max(1);
    let mut out = vec![255u8; ow * oh * 4];
    let src = frame.rgb.as_slice();
    let stride = frame.width as usize * 3;
    for y in 0..oh {
        let sy = ((y as f32 + 0.5) * step) as usize;
        let sy = sy.min(frame.height as usize - 1);
        for x in 0..ow {
            let sx = ((x as f32 + 0.5) * step) as usize;
            let sx = sx.min(frame.width as usize - 1);
            let si = sy * stride + sx * 3;
            let dx = if mirror { ow - 1 - x } else { x };
            let di = (y * ow + dx) * 4;
            out[di..di + 3].copy_from_slice(&src[si..si + 3]);
        }
    }
    (ow, oh, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 3]) -> Frame {
        let mut rgb = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                rgb.extend_from_slice(&f(x, y));
            }
        }
        Frame { width: w, height: h, rgb: Arc::new(rgb), captured_at: Instant::now(), seq: 0 }
    }

    #[test]
    fn identity_warp_copies_pixels() {
        let f = frame(8, 6, |x, y| [(x * 10) as u8, (y * 10) as u8, 7]);
        let mut out = vec![0.0; 8 * 6 * 3];
        let id = Affine2 { m: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0] };
        warp_to_tensor(&f, &id, 8, 6, 1.0, 0.0, &mut out);
        for y in 0..6 {
            for x in 0..8 {
                let i = (y * 8 + x) * 3;
                assert!((out[i] - (x * 10) as f32).abs() < 1e-3, "x at {x},{y}: {}", out[i]);
                assert!((out[i + 1] - (y * 10) as f32).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn outside_is_black_not_panic() {
        let f = frame(4, 4, |_, _| [200, 200, 200]);
        let mut out = vec![9.0; 4 * 4 * 3];
        let far = Affine2 { m: [1.0, 0.0, 1000.0, 0.0, 1.0, -1000.0] };
        warp_to_tensor(&f, &far, 4, 4, 1.0 / 255.0, -0.5, &mut out);
        assert!(out.iter().all(|v| (*v + 0.5).abs() < 1e-6));
    }
}
