//! Offline stress test of the full tracking pipeline on a still photo.
//! Synthesises camera frames with head roll, moving very close to the camera, sliding half out
//! of frame and disappearing, and prints what the tracker reports for each frame.
//!
//! Usage: cargo run --release --example replay -- <face photo> [out_dir]
#[path = "common/synth.rs"]
mod synth;
use mytracker::config::Config;
use mytracker::image_ops::Frame;
use mytracker::tracker::Pipeline;
use std::sync::Arc;
use std::time::Instant;
use synth::{render, H, W};

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: replay <face photo>");
    let img = image::open(&path)?.to_rgb8();
    let src = Frame { width: img.width(), height: img.height(), rgb: Arc::new(img.into_raw()), captured_at: Instant::now(), seq: 0 };
    let cfg = Config::default();
    let mut p = Pipeline::new()?;

    // Find the face in the photo first to know where to centre the synthetic camera.
    let probe = render(&src, src.width as f32 / 2.0, src.height as f32 / 2.0, W as f32 / 2.0, H as f32 / 2.0, W as f32 / src.width as f32, 0.0, 1e9);
    let r = p.step(&probe, &cfg).expect("no face in the photo");
    let k = src.width as f32 / W as f32;
    let (fx, fy) = ((r.roi.cx - W as f32 / 2.0) * k + src.width as f32 / 2.0, (r.roi.cy - H as f32 / 2.0) * k + src.height as f32 / 2.0);
    let face_src = r.roi.size * k / 1.5;
    println!("face in photo at ({fx:.0},{fy:.0}) ~{face_src:.0}px");

    let mask = face_src * 1.3;
    let base = 220.0 / face_src; // face ~220 px wide: normal sitting distance
    let mut p = Pipeline::new()?;
    let mut frames: Vec<(String, f32, f32, f32, f32)> = Vec::new(); // label, tx, ty, scale, roll
    let (cx, cy) = (W as f32 / 2.0, H as f32 / 2.0);
    for i in 0..5 {
        frames.push(("baseline".into(), cx, cy, base, 0.0));
        let _ = i;
    }
    for d in (0..=40).step_by(5) {
        frames.push((format!("roll {d}°"), cx, cy, base, d as f32));
    }
    for d in (0..=40).rev().step_by(10) {
        frames.push((format!("roll back {d}°"), cx, cy, base, d as f32));
    }
    let mut s = base;
    while s < base * 4.5 {
        frames.push((format!("closer x{:.1} (face {:.0}px)", s / base, s * face_src), cx, cy, s, 0.0));
        s *= 1.15;
    }
    let close = s;
    for i in 0..=8 {
        let tx = cx + i as f32 * 60.0;
        frames.push((format!("very close, sliding right +{}px", i * 60), tx, cy, close, 0.0));
    }
    for _ in 0..3 {
        frames.push(("face gone".into(), -5000.0, cy, base, 0.0));
    }
    for _ in 0..3 {
        frames.push(("face back".into(), cx + 200.0, cy - 50.0, base * 1.2, 0.0));
    }

    let mut ok = 0;
    let t = Instant::now();
    for (i, (label, tx, ty, sc, roll)) in frames.iter().enumerate() {
        let mut f = render(&src, fx, fy, *tx, *ty, *sc, *roll, mask);
        f.seq = i as u64 + 1;
        match p.step(&f, &cfg) {
            Some(r) => {
                ok += 1;
                let v = r.pose;
                println!(
                    "{label:<42} yaw {:>6.1} pitch {:>6.1} roll {:>6.1} | x {:>6.1} y {:>6.1} z {:>6.1} cm | dist {:>5.1} cm fit {:.1}px",
                    v[0], v[1], v[2], v[3], v[4], v[5], r.distance_cm, r.reproj_px
                );
            }
            None => println!("{label:<42} -- no face --"),
        }
    }
    let tm = p.timings();
    println!(
        "{ok}/{} frames tracked in {:?}; avg landmark {:.1} ms, detect {:.1} ms, pose {:.2} ms",
        frames.len(),
        t.elapsed(),
        tm.landmark_ms,
        tm.detect_ms,
        tm.pose_ms
    );
    Ok(())
}
