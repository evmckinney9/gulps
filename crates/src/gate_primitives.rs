//! Basis matrices, two-qubit invariants, coordinate transforms, and the
//! SU(2) parameterization.

use crate::linalg::{eigenphases_4x4, normalize_quat};
use crate::{Mat2, Mat4, C0, C1, C64, CI, CM1};
use std::f64::consts::{FRAC_1_SQRT_2, PI};

/// Project a 4×4 unitary onto SU(4), returning it and the phase removed.
pub fn project_su4(u: &Mat4) -> (Mat4, f64) {
    let det = u.determinant();
    let quarter_phase = -det.arg() * 0.25;
    let phase = C64::new(quarter_phase.cos(), quarter_phase.sin());
    (u * phase, det.arg() / 4.0)
}

/// Magic basis, normalized by 1/√2.
pub fn magic() -> Mat4 {
    let s = C64::new(FRAC_1_SQRT_2, 0.0);
    let si = C64::new(0.0, FRAC_1_SQRT_2);
    Mat4::new(s, C0, C0, si, C0, si, s, C0, C0, si, -s, C0, s, C0, C0, -si)
}

/// MAGIC†.
pub fn magic_dag() -> Mat4 {
    magic().adjoint()
}

/// σ_y ⊗ σ_y.
pub fn sysy() -> Mat4 {
    Mat4::new(
        C0, C0, C0, CM1, C0, C0, C1, C0, C0, C1, C0, C0, CM1, C0, C0, C0,
    )
}

/// Makhlin invariants [Re(G1), Im(G1), Re(G2)] of a 4×4 unitary.
pub fn makhlin_invariants(u: &Mat4) -> [f64; 3] {
    let um = magic_dag() * u * magic();
    let det_um = um.determinant();
    let det_norm = det_um.norm();
    let det_phase = if det_norm > 1e-30 {
        det_um / det_norm
    } else {
        C1
    };
    let m = um.transpose() * um; // transpose, NOT adjoint
    let t1 = m.trace();
    let t1s = t1 * t1;
    // tr(M^2) without forming M*M
    let mut tr_m2 = C0;
    for i in 0..4 {
        for j in 0..4 {
            tr_m2 += m[(i, j)] * m[(j, i)];
        }
    }
    let g1 = t1s / (det_phase * 16.0);
    let g2 = (t1s - tr_m2) / (det_phase * 4.0);
    [g1.re, g1.im, g2.re]
}

/// Weyl chamber coordinates (c1, c2, c3) in units of π, folded to
/// c1 ≥ c2 ≥ c3 ≥ 0. c1 may exceed 0.5 when c3 > 0.
pub fn weyl_coordinates(u: &Mat4) -> [f64; 3] {
    let (u_su4, _) = project_su4(u);

    let sy = sysy();
    let u_tilde = sy * u_su4.transpose() * sy;

    let product = u_su4 * u_tilde;
    let two_s = eigenphases_4x4(&product);

    let mut s = two_s.map(|x| x / 2.0);
    s.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // alcove normalization
    let sum: f64 = s.iter().sum();
    let n = sum.round() as usize;
    for item in s.iter_mut().take(n.min(4)) {
        *item -= 1.0;
    }

    let shift = n % 4;
    let rolled = [
        s[shift % 4],
        s[(1 + shift) % 4],
        s[(2 + shift) % 4],
        s[(3 + shift) % 4],
    ];

    let c1 = rolled[0] + rolled[1];
    let c2 = rolled[0] + rolled[2];
    let c3 = rolled[1] + rolled[2];

    // c3 < 0 ↦ (1−c1, c2, −c3); tolerance avoids the c3≡0 oscillation
    // between c1 and 1−c1 from floating-point negative zero.
    let (c1, c3) = if c3 < -1e-15 {
        (1.0 - c1, -c3)
    } else {
        (c1, c3)
    };

    [c1, c2.max(0.0), c3.max(0.0)]
}

/// Monodromy to Weyl coordinates.
pub fn weyl_from_monodromy(m: &[f64; 3]) -> [f64; 3] {
    [m[0] + m[1], m[0] + m[2], m[1] + m[2]]
}

/// Weyl to monodromy coordinates (inverse of weyl_from_monodromy).
pub fn monodromy_from_weyl(c1: f64, c2: f64, c3: f64) -> [f64; 3] {
    [
        0.5 * (c1 + c2 - c3),
        0.5 * (c1 - c2 + c3),
        0.5 * (-c1 + c2 + c3),
    ]
}

/// Canonical two-qubit gate from Weyl coordinates `(c1, c2, c3) ∈ [0,1]` (radians = c·π/2).
pub fn canonical_matrix(c1: f64, c2: f64, c3: f64) -> Mat4 {
    let a = PI / 2.0 * c1;
    let b = PI / 2.0 * c2;
    let g = PI / 2.0 * c3;
    let eig = C64::new(g.cos(), g.sin());
    let eig_c = eig.conj();
    let cam = (a - b).cos();
    let sam = (a - b).sin();
    let cap = (a + b).cos();
    let sap = (a + b).sin();
    Mat4::new(
        eig * cam,
        C0,
        C0,
        CI * eig * sam,
        C0,
        eig_c * cap,
        CI * eig_c * sap,
        C0,
        C0,
        CI * eig_c * sap,
        eig_c * cap,
        C0,
        CI * eig * sam,
        C0,
        C0,
        eig * cam,
    )
}

/// Min Weyl residual over direct and ρ-reflected branches.
pub fn weyl_res_both_branches(c: &[f64; 3], target: &[f64; 3]) -> f64 {
    let refl = [1.0 - target[0], target[1], -target[2]];
    let direct = (c[0] - target[0])
        .abs()
        .max((c[1] - target[1]).abs())
        .max((c[2] - target[2]).abs());
    let reflected = (c[0] - refl[0])
        .abs()
        .max((c[1] - refl[1]).abs())
        .max((c[2] - refl[2]).abs());
    direct.min(reflected)
}

/// Normalized quaternion (w,x,y,z) to a 2×2 SU(2) matrix.
fn quat_to_su2(q: &[f64]) -> Mat2 {
    let ([w, x, y, z], _) = normalize_quat(q);
    let a = C64::new(w, z);
    let b = C64::new(x, y);
    Mat2::new(a, b, -b.conj(), a.conj())
}

/// 8 real params → (u0, u1) SU(2) pair.
pub fn params_to_unitaries(params: &[f64; 8]) -> (Mat2, Mat2) {
    (quat_to_su2(&params[0..4]), quat_to_su2(&params[4..8]))
}

/// Kronecker product kron(a, b) for 2×2 → 4×4.
pub fn kron_2x2(a: &Mat2, b: &Mat2) -> Mat4 {
    let (a00, a01, a10, a11) = (a[(0, 0)], a[(0, 1)], a[(1, 0)], a[(1, 1)]);
    Mat4::new(
        a00 * b[(0, 0)],
        a00 * b[(0, 1)],
        a01 * b[(0, 0)],
        a01 * b[(0, 1)],
        a00 * b[(1, 0)],
        a00 * b[(1, 1)],
        a01 * b[(1, 0)],
        a01 * b[(1, 1)],
        a10 * b[(0, 0)],
        a10 * b[(0, 1)],
        a11 * b[(0, 0)],
        a11 * b[(0, 1)],
        a10 * b[(1, 0)],
        a10 * b[(1, 1)],
        a11 * b[(1, 0)],
        a11 * b[(1, 1)],
    )
}
