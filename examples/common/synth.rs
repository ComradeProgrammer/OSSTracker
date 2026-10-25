//! Synthetic camera frames from a still photo, shared by the offline examples.
use mytracker::image_ops::{warp_to_tensor, Affine2, Frame};
use std::sync::Arc;
use std::time::Instant;

pub const W: u32 = 1280;
pub const H: u32 = 720;

/// Render a frame: the photo's face centre (fx, fy) is placed at (tx, ty), scaled by `s`, rotated by `deg`.
/// Everything further than `mask` photo pixels from the face centre is blanked.
#[allow(clippy::too_many_arguments)]
pub fn render(src: &Frame, fx: f32, fy: f32, tx: f32, ty: f32, s: f32, deg: f32, mask: f32) -> Frame {
    let (sn, cs) = deg.to_radians().sin_cos();
    // out -> src: src = R^-1 (out - t) / s + f
    let m = [cs / s, sn / s, 0.0, -sn / s, cs / s, 0.0];
    let a = Affine2 { m: [m[0], m[1], fx - (m[0] * tx + m[1] * ty), m[3], m[4], fy - (m[3] * tx + m[4] * ty)] };
    let mut buf = vec![0f32; (W * H * 3) as usize];
    warp_to_tensor(src, &a, W as usize, H as usize, 1.0, 0.0, &mut buf);
    // Keep only a disc around the face so other people in the photo don't interfere.
    let r2 = (mask * s).powi(2);
    for y in 0..H as usize {
        for x in 0..W as usize {
            if (x as f32 - tx).powi(2) + (y as f32 - ty).powi(2) > r2 {
                let i = (y * W as usize + x) * 3;
                buf[i..i + 3].fill(40.0);
            }
        }
    }
    let rgb = buf.iter().map(|v| v.clamp(0.0, 255.0) as u8).collect();
    Frame { width: W, height: H, rgb: Arc::new(rgb), captured_at: Instant::now(), seq: 0 }
}

