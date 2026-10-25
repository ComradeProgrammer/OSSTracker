//! Head pose from 2D landmarks: closed-form initial guess (Umeyama) refined by robust
//! Levenberg-Marquardt PnP, plus conversion to yaw/pitch/roll relative to a centre pose.

use nalgebra::{Matrix3, Matrix6, Rotation3, UnitQuaternion, Vector2, Vector3, Vector6};

#[derive(Clone, Copy, Debug)]
pub struct Intrinsics {
    pub f: f64,
    pub cx: f64,
    pub cy: f64,
}

impl Intrinsics {
    /// Pinhole intrinsics from the horizontal field of view; principal point at the image centre.
    pub fn from_hfov(width: u32, height: u32, hfov_deg: f64) -> Self {
        let hfov = hfov_deg.clamp(20.0, 150.0).to_radians();
        let f = width as f64 * 0.5 / (hfov * 0.5).tan();
        Self { f, cx: width as f64 * 0.5, cy: height as f64 * 0.5 }
    }

    pub fn project(&self, p: &Vector3<f64>) -> Option<Vector2<f64>> {
        if p.z <= 1e-6 {
            return None;
        }
        Some(Vector2::new(self.f * p.x / p.z + self.cx, self.f * p.y / p.z + self.cy))
    }
}

/// Rigid transform model -> camera (OpenCV camera axes: x right, y down, z forward).
#[derive(Clone, Copy, Debug)]
pub struct Pose {
    pub rot: Rotation3<f64>,
    pub t: Vector3<f64>,
}

impl Pose {
    pub fn transform(&self, p: &Vector3<f64>) -> Vector3<f64> {
        self.rot * p + self.t
    }
}

/// Least-squares similarity transform with dst ≈ s * R * src + t (Umeyama 1991).
pub fn umeyama(src: &[Vector3<f64>], dst: &[Vector3<f64>]) -> Option<(f64, Rotation3<f64>, Vector3<f64>)> {
    let n = src.len();
    if n < 3 || n != dst.len() {
        return None;
    }
    let mu_s = src.iter().sum::<Vector3<f64>>() / n as f64;
    let mu_d = dst.iter().sum::<Vector3<f64>>() / n as f64;
    let mut cov = Matrix3::zeros();
    let mut var_s = 0.0;
    for (s, d) in src.iter().zip(dst) {
        let ds = s - mu_s;
        cov += (d - mu_d) * ds.transpose();
        var_s += ds.norm_squared();
    }
    cov /= n as f64;
    var_s /= n as f64;
    if var_s < 1e-12 {
        return None;
    }
    let svd = cov.svd(true, true);
    let (u, vt) = (svd.u?, svd.v_t?);
    let mut s_mat = Matrix3::identity();
    if (u.determinant() * vt.determinant()) < 0.0 {
        s_mat[(2, 2)] = -1.0;
    }
    let r = u * s_mat * vt;
    let d = svd.singular_values;
    let trace = d[0] * s_mat[(0, 0)] + d[1] * s_mat[(1, 1)] + d[2] * s_mat[(2, 2)];
    let scale = trace / var_s;
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let t = mu_d - scale * r * mu_s;
    Some((scale, Rotation3::from_matrix_unchecked(r), t))
}

fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Robust (Huber) reprojection cost.
fn cost(model: &[Vector3<f64>], obs: &[Vector2<f64>], k: &Intrinsics, pose: &Pose, delta: f64) -> f64 {
    let mut c = 0.0;
    for (x, o) in model.iter().zip(obs) {
        let p = pose.transform(x);
        let e = match k.project(&p) {
            Some(u) => (u - o).norm(),
            None => 1e3 * delta,
        };
        c += if e <= delta { 0.5 * e * e } else { delta * (e - 0.5 * delta) };
    }
    c
}

pub struct PnpResult {
    pub pose: Pose,
    /// RMS reprojection error in pixels.
    pub rms: f64,
}

/// Refine `init` by minimizing the robust reprojection error of `model` points onto `obs`.
pub fn solve_pnp(model: &[Vector3<f64>], obs: &[Vector2<f64>], k: &Intrinsics, init: Pose, huber_px: f64) -> Option<PnpResult> {
    if model.len() < 4 || model.len() != obs.len() {
        return None;
    }
    let delta = huber_px.max(0.5);
    let mut pose = init;
    let mut cur = cost(model, obs, k, &pose, delta);
    let mut lambda = 1e-3;
    for _ in 0..20 {
        let mut h = Matrix6::<f64>::zeros();
        let mut g = Vector6::<f64>::zeros();
        for (x, o) in model.iter().zip(obs) {
            let rx = pose.rot * x;
            let p = rx + pose.t;
            if p.z <= 1e-6 {
                continue;
            }
            let iz = 1.0 / p.z;
            let u = Vector2::new(k.f * p.x * iz + k.cx, k.f * p.y * iz + k.cy);
            let e = u - o;
            let en = e.norm();
            let w = if en <= delta { 1.0 } else { delta / en };
            // d(proj)/dP
            let jp = nalgebra::Matrix2x3::new(k.f * iz, 0.0, -k.f * p.x * iz * iz, 0.0, k.f * iz, -k.f * p.y * iz * iz);
            let jr = jp * (-skew(&rx));
            let mut j = nalgebra::Matrix2x6::<f64>::zeros();
            j.fixed_view_mut::<2, 3>(0, 0).copy_from(&jr);
            j.fixed_view_mut::<2, 3>(0, 3).copy_from(&jp);
            h += w * j.transpose() * j;
            g += w * j.transpose() * e;
        }
        let mut improved = false;
        for _ in 0..6 {
            let mut hd = h;
            for i in 0..6 {
                hd[(i, i)] += lambda * h[(i, i)].max(1e-9);
            }
            let Some(step) = hd.cholesky().map(|c| c.solve(&(-g))) else {
                lambda *= 10.0;
                continue;
            };
            let drot = Vector3::new(step[0], step[1], step[2]);
            let cand = Pose {
                rot: Rotation3::new(drot) * pose.rot,
                t: pose.t + Vector3::new(step[3], step[4], step[5]),
            };
            let c = cost(model, obs, k, &cand, delta);
            if c.is_finite() && c < cur {
                let small = drot.norm() < 1e-7 && step.fixed_rows::<3>(3).norm() < 1e-6;
                pose = cand;
                cur = c;
                lambda = (lambda * 0.3).max(1e-7);
                improved = true;
                if small {
                    return finish(model, obs, k, pose);
                }
                break;
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }
    finish(model, obs, k, pose)
}

fn finish(model: &[Vector3<f64>], obs: &[Vector2<f64>], k: &Intrinsics, pose: Pose) -> Option<PnpResult> {
    let mut sum = 0.0;
    for (x, o) in model.iter().zip(obs) {
        let u = k.project(&pose.transform(x))?;
        sum += (u - o).norm_squared();
    }
    let rms = (sum / model.len() as f64).sqrt();
    if !rms.is_finite() || !pose.t.iter().all(|v| v.is_finite()) || pose.t.z <= 0.0 {
        return None;
    }
    Some(PnpResult { pose, rms })
}

/// 6DoF in the opentrack convention: degrees and centimetres.
/// yaw + = look right, pitch + = look up, roll + = tilt right (clockwise as seen by the user),
/// x + = move right, y + = move up, z + = move back (away from the screen).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pose6 {
    pub v: [f64; 6],
}

pub const YAW: usize = 0;
pub const PITCH: usize = 1;
pub const ROLL: usize = 2;
pub const TX: usize = 3;
pub const TY: usize = 4;
pub const TZ: usize = 5;
pub const AXIS_NAMES: [&str; 6] = ["Yaw", "Pitch", "Roll", "X", "Y", "Z"];

/// Head pose in model axes (x = subject's left, y = up, z = out of the face) relative to a centre pose.
/// `pivot` is the point (model coordinates) whose motion is reported as translation.
pub fn relative_pose6(center: &Pose, cur: &Pose, pivot: &Vector3<f64>) -> Pose6 {
    let r_rel = center.rot.inverse() * cur.rot;
    let m = r_rel.matrix();
    // R = Ry(a) * Rx(b) * Rz(c): a yaw about up, b pitch about the lateral axis, c roll about forward.
    let b = (-m[(1, 2)]).clamp(-1.0, 1.0).asin();
    let a = m[(0, 2)].atan2(m[(2, 2)]);
    let c = m[(1, 0)].atan2(m[(1, 1)]);
    let p_cur = cur.transform(pivot);
    let p_ctr = center.transform(pivot);
    let d = center.rot.inverse() * (p_cur - p_ctr);
    Pose6 {
        v: [
            -a.to_degrees(), // a > 0 turns toward the subject's left
            -b.to_degrees(), // b > 0 pitches the nose down
            c.to_degrees(),  // c > 0 tilts the top of the head to the subject's right
            -d.x,            // model +x is the subject's left
            d.y,
            -d.z, // model +z points at the camera; moving back is -z
        ],
    }
}

/// Model-to-camera rotation for a face looking straight into the camera.
pub fn frontal_rotation() -> Rotation3<f64> {
    Rotation3::from_matrix_unchecked(Matrix3::new(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0))
}

pub fn quat(p: &Pose) -> UnitQuaternion<f64> {
    UnitQuaternion::from_rotation_matrix(&p.rot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::face_model::{model_points, RIGID_LANDMARKS};

    fn synth_pose(yaw: f64, pitch: f64, roll: f64, t: Vector3<f64>) -> Pose {
        // Build the user-facing rotation the same way relative_pose6 decomposes it.
        let a = (-yaw).to_radians();
        let b = (-pitch).to_radians();
        let c = roll.to_radians();
        let r_rel = Rotation3::from_axis_angle(&Vector3::y_axis(), a)
            * Rotation3::from_axis_angle(&Vector3::x_axis(), b)
            * Rotation3::from_axis_angle(&Vector3::z_axis(), c);
        Pose { rot: frontal_rotation() * r_rel, t }
    }

    #[test]
    fn euler_signs_round_trip() {
        let center = synth_pose(0.0, 0.0, 0.0, Vector3::new(0.0, 0.0, 50.0));
        let piv = Vector3::zeros();
        for &(y, p, r) in &[(30.0, 0.0, 0.0), (0.0, 20.0, 0.0), (0.0, 0.0, -15.0), (40.0, -10.0, 12.0)] {
            let cur = synth_pose(y, p, r, Vector3::new(0.0, 0.0, 50.0));
            let out = relative_pose6(&center, &cur, &piv);
            assert!((out.v[YAW] - y).abs() < 1e-6, "{:?}", out);
            assert!((out.v[PITCH] - p).abs() < 1e-6, "{:?}", out);
            assert!((out.v[ROLL] - r).abs() < 1e-6, "{:?}", out);
        }
    }

    #[test]
    fn yaw_right_moves_nose_to_image_left() {
        // When the user looks to their right, the nose tip moves toward the user's right,
        // which is the image LEFT for a non-mirrored camera.
        let k = Intrinsics::from_hfov(1280, 720, 70.0);
        let nose = Vector3::new(0.0, -0.46, 7.59);
        let c = synth_pose(0.0, 0.0, 0.0, Vector3::new(0.0, 0.0, 50.0));
        let r = synth_pose(30.0, 0.0, 0.0, Vector3::new(0.0, 0.0, 50.0));
        let u0 = k.project(&c.transform(&nose)).unwrap();
        let u1 = k.project(&r.transform(&nose)).unwrap();
        assert!(u1.x < u0.x);
        // Looking up moves the nose up in the image (smaller y).
        let up = synth_pose(0.0, 20.0, 0.0, Vector3::new(0.0, 0.0, 50.0));
        assert!(k.project(&up.transform(&nose)).unwrap().y < u0.y);
        // Moving right (+x out) moves the head toward image left.
        let piv = Vector3::zeros();
        let moved = Pose { rot: c.rot, t: c.t + Vector3::new(-5.0, 0.0, 0.0) };
        let out = relative_pose6(&c, &moved, &piv);
        assert!((out.v[TX] - 5.0).abs() < 1e-9, "{:?}", out);
        let back = Pose { rot: c.rot, t: c.t + Vector3::new(0.0, 0.0, 10.0) };
        assert!((relative_pose6(&c, &back, &piv).v[TZ] - 10.0).abs() < 1e-9);
        let upw = Pose { rot: c.rot, t: c.t + Vector3::new(0.0, -3.0, 0.0) };
        assert!((relative_pose6(&c, &upw, &piv).v[TY] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn pnp_recovers_pose_with_noise_and_close_range() {
        let model = model_points(RIGID_LANDMARKS);
        for &(dist, yaw, pitch, roll, tx) in &[(60.0, 0.0, 0.0, 0.0, 0.0), (25.0, 50.0, -20.0, 10.0, 8.0), (20.0, -70.0, 15.0, -25.0, -6.0)] {
            let k = Intrinsics::from_hfov(1280, 720, 78.0);
            let truth = synth_pose(yaw, pitch, roll, Vector3::new(tx, -2.0, dist));
            let mut seed = 12345u64;
            let mut noise = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed % 1000) as f64 / 1000.0 - 0.5
            };
            let obs: Vec<_> = model
                .iter()
                .map(|x| k.project(&truth.transform(x)).unwrap() + Vector2::new(noise(), noise()) * 1.5)
                .collect();
            // Start from a poor frontal guess, like the first frame would.
            let init = Pose { rot: frontal_rotation(), t: Vector3::new(0.0, 0.0, dist * 1.3) };
            let res = solve_pnp(&model, &obs, &k, init, 3.0).expect("pnp");
            let center = synth_pose(0.0, 0.0, 0.0, Vector3::new(0.0, 0.0, 50.0));
            let est = relative_pose6(&center, &res.pose, &Vector3::zeros());
            let exp = relative_pose6(&center, &truth, &Vector3::zeros());
            for i in 0..3 {
                assert!((est.v[i] - exp.v[i]).abs() < 2.0, "axis {i}: est {:?} exp {:?} rms {}", est, exp, res.rms);
            }
            assert!((res.pose.t - truth.t).norm() < 2.0, "t est {:?} truth {:?}", res.pose.t, truth.t);
        }
    }

    #[test]
    fn umeyama_recovers_similarity() {
        let model = model_points(RIGID_LANDMARKS);
        let r = Rotation3::from_euler_angles(0.3, -0.5, 0.2);
        let t = Vector3::new(100.0, 50.0, 3.0);
        let dst: Vec<_> = model.iter().map(|p| 7.0 * (r * p) + t).collect();
        let (s, r2, t2) = umeyama(&model, &dst).unwrap();
        assert!((s - 7.0).abs() < 1e-9);
        assert!((r2.matrix() - r.matrix()).norm() < 1e-9);
        assert!((t2 - t).norm() < 1e-6);
    }
}
