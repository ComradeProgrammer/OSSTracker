//! 3D face model used for pose estimation.

use nalgebra::Vector3;

#[path = "face_model_data.rs"]
mod data;
pub use data::CANONICAL_VERTICES;

/// Landmarks on bone (forehead, nose bridge, eye corners, cheekbones, temples) that barely move
/// with facial expressions. Mouth, jaw, eyelids and eyebrows are deliberately excluded so that
/// talking or blinking does not shake the view.
pub const RIGID_LANDMARKS: &[usize] = &[
    // nose bridge and tip
    1, 4, 5, 6, 168, 197, 195, 45, 275, 51, 281,
    // eye corners
    33, 133, 263, 362,
    // forehead
    10, 151, 9, 108, 337, 109, 338, 67, 297, 69, 299,
    // temples / face side
    127, 356, 234, 454, 162, 389, 21, 251,
    // cheekbones and upper cheeks
    116, 345, 117, 346, 118, 347, 123, 352, 50, 280, 111, 340,
];

/// Eye outer corners (image left / image right for a frontal face), used for the crop rotation.
pub const EYE_LEFT_IMG: usize = 33;
pub const EYE_RIGHT_IMG: usize = 263;

/// Approximate centre of the head in model coordinates (cm). Translation is reported for this
/// point, so turning the head does not leak into sideways movement.
pub const HEAD_CENTER: [f64; 3] = [0.0, 0.0, -4.0];

pub fn model_points(indices: &[usize]) -> Vec<Vector3<f64>> {
    indices
        .iter()
        .map(|&i| {
            let v = CANONICAL_VERTICES[i];
            Vector3::new(v[0] as f64, v[1] as f64, v[2] as f64)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indices_valid_and_unique() {
        let mut seen = std::collections::HashSet::new();
        for &i in RIGID_LANDMARKS {
            assert!(i < 468);
            assert!(seen.insert(i), "duplicate {i}");
        }
        // Subject's right eye is on the model's -x side.
        assert!(CANONICAL_VERTICES[EYE_LEFT_IMG][0] < 0.0);
        assert!(CANONICAL_VERTICES[EYE_RIGHT_IMG][0] > 0.0);
    }
}
