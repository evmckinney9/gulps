//! Gate primitive definitions for two-qubit invariants and coordinates.
//!
//! Mathematical objects: basis matrices, invariant functions, coordinate
//! transforms, and SU(2) parameterization.  All functions operate on
//! stack-allocated nalgebra matrices with no heap allocation.

use faer::Mat;
use nalgebra::{Complex, Matrix2, Matrix4};
use std::f64::consts::{FRAC_1_SQRT_2, PI};

pub type C64 = Complex<f64>;
pub type Mat4 = Matrix4<C64>;
pub type Mat2 = Matrix2<C64>;

// Frequently used complex constants.
pub const C0: C64 = C64::new(0.0, 0.0);
pub const C1: C64 = C64::new(1.0, 0.0);

// ---------------------------------------------------------------------------
// Basis matrices (constructed per call - compiler constant-folds these)
// ---------------------------------------------------------------------------

/// Magic basis Q = (H⊗I)·CX·(I⊗S†), normalized by 1/√2.
pub fn magic() -> Mat4 {
    let s = C64::new(FRAC_1_SQRT_2, 0.0);
    let si = C64::new(0.0, FRAC_1_SQRT_2);
    Mat4::new(s, C0, C0, si, C0, si, s, C0, C0, si, -s, C0, s, C0, C0, -si)
}

/// MAGIC†  (conjugate transpose of magic basis).
pub fn magic_dag() -> Mat4 {
    magic().adjoint()
}

/// σ_y ⊗ σ_y  (real, symmetric, unitary, self-inverse).
pub fn sysy() -> Mat4 {
    let neg = C64::new(-1.0, 0.0);
    Mat4::new(
        C0, C0, C0, neg, C0, C0, C1, C0, C0, C1, C0, C0, neg, C0, C0, C0,
    )
}

// ---------------------------------------------------------------------------
// Invariant functions
// ---------------------------------------------------------------------------

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
    // tr(M²) = Σᵢⱼ M[i,j]·M[j,i] - 16 multiply-adds, avoids full matmul
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

/// Extract eigenvalue phases / π from a 4×4 unitary matrix.
///
/// Uses faer's eigenvalue decomposition, which handles degenerate eigenvalues
/// correctly (LAPACK-grade QR with proper deflation).
fn eigenphases_4x4(m: &Mat4) -> [f64; 4] {
    // nalgebra Mat4<C64> → faer Mat<C64> (C64 IS num_complex::Complex<f64>)
    let fm = Mat::<C64>::from_fn(4, 4, |i, j| m[(i, j)]);
    let eigvals = fm.eigenvalues().expect("eigenvalue decomposition failed");

    let mut phases = [0.0f64; 4];
    for (i, ev) in eigvals.iter().enumerate() {
        let mut p = ev.arg() / PI;
        if p < -0.5 + 1e-12 {
            p += 2.0;
        }
        phases[i] = p;
    }
    phases
}

/// Eigendecomposition of a 4×4 matrix returning eigenvalues, phases, and
/// right eigenvectors.
///
/// Returns (eigenvalues, phases/π, eigenvectors) where eigenvectors[i] is
/// the right eigenvector for eigenvalue i.
pub fn eigendecomp_4x4(m: &Mat4) -> ([C64; 4], [f64; 4], [[C64; 4]; 4]) {
    let fm = Mat::<C64>::from_fn(4, 4, |i, j| m[(i, j)]);
    let evd = fm.eigen().expect("eigendecomposition failed");

    let mut eigvals = [C0; 4];
    let mut phases = [0.0f64; 4];
    let mut vecs = [[C0; 4]; 4];

    for (i, ev) in evd.S().column_vector().iter().enumerate() {
        eigvals[i] = *ev;
        let mut p = ev.arg() / PI;
        if p < -0.5 + 1e-12 {
            p += 2.0;
        }
        phases[i] = p;
    }
    let u = evd.U();
    for i in 0..4 {
        for j in 0..4 {
            vecs[i][j] = u[(j, i)]; // column i, row j
        }
    }

    (eigvals, phases, vecs)
}

/// Weyl chamber coordinates (c1, c2, c3) in units of π.
///
/// Folded to canonical chamber: c1 ≥ c2 ≥ c3 ≥ 0, c1 ≤ 0.5 on the c3=0
/// face. c1 can exceed 0.5 when c3 > 0 (distinct local equivalence class).
pub fn weyl_coordinates(u: &Mat4) -> [f64; 3] {
    // SU(4) projection: multiply by det(U)^{-1/4}
    // For unitary U, |det|=1, so det=e^{iθ} and det^{-1/4}=e^{-iθ/4}.
    // One atan2 + sincos instead of complex log + exp.
    let det = u.determinant();
    let quarter_phase = -det.arg() * 0.25;
    let phase = C64::new(quarter_phase.cos(), quarter_phase.sin());
    let u_su4 = u * phase;

    // Ũ = (σ_y⊗σ_y) Uᵀ (σ_y⊗σ_y)
    let sy = sysy();
    let u_tilde = sy * u_su4.transpose() * sy;

    // Eigenvalue phases of U·Ũ (bounded Schur with quartic fallback)
    let product = u_su4 * u_tilde;
    let two_s = eigenphases_4x4(&product);

    // Sort descending, halve
    let mut s = [
        two_s[0] / 2.0,
        two_s[1] / 2.0,
        two_s[2] / 2.0,
        two_s[3] / 2.0,
    ];
    s.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Alcove normalization
    let sum: f64 = s.iter().sum();
    let n = sum.round() as usize;
    for item in s.iter_mut().take(n.min(4)) {
        *item -= 1.0;
    }

    // Circular shift left by n
    let shift = n % 4;
    let rolled = [
        s[shift % 4],
        s[(1 + shift) % 4],
        s[(2 + shift) % 4],
        s[(3 + shift) % 4],
    ];

    // M_WEYL @ rolled[:3]  →  c = (s0+s1, s0+s2, s1+s2)
    let c1 = rolled[0] + rolled[1];
    let c2 = rolled[0] + rolled[2];
    let c3 = rolled[1] + rolled[2];

    // Canonical folding: c3 < 0 ↦ (1−c1, c2, −c3).
    // Uses tolerance to avoid spurious folds on floating-point negative zero,
    // which causes CX-face (c3≡0) coordinates to oscillate between c1 and 1−c1.
    let (c1, c3) = if c3 < -1e-15 { (1.0 - c1, -c3) } else { (c1, c3) };

    [c1, c2.max(0.0), c3.max(0.0)]
}

// ---------------------------------------------------------------------------
// Coordinate transforms
// ---------------------------------------------------------------------------

/// Monodromy → Weyl: c = M_WEYL @ m[:3] where M_WEYL = [[1,1,0],[1,0,1],[0,1,1]].
pub fn weyl_from_monodromy(m: &[f64; 3]) -> [f64; 3] {
    [m[0] + m[1], m[0] + m[2], m[1] + m[2]]
}

/// Weyl → Monodromy (inverse of M_WEYL).
///
/// M_WEYL⁻¹ = 0.5 × [[1,1,-1],[1,-1,1],[-1,1,1]].
pub fn monodromy_from_weyl(c1: f64, c2: f64, c3: f64) -> [f64; 3] {
    [
        0.5 * (c1 + c2 - c3),
        0.5 * (c1 - c2 + c3),
        0.5 * (-c1 + c2 + c3),
    ]
}

/// Canonical unitary exp(-i(a·XX + b·YY + c·ZZ)) from Weyl coordinates.
///
/// c1, c2, c3 in normalized [0,1] units (multiply by π/2 for radians).
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
    let ci = C64::new(0.0, 1.0);
    Mat4::new(
        eig * cam,
        C0,
        C0,
        ci * eig * sam,
        C0,
        eig_c * cap,
        ci * eig_c * sap,
        C0,
        C0,
        ci * eig_c * sap,
        eig_c * cap,
        C0,
        ci * eig * sam,
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

// ---------------------------------------------------------------------------
// Parameterization: 8D quaternion pairs → SU(2) × SU(2)
// ---------------------------------------------------------------------------

/// Normalized quaternion (w,x,y,z) → 2×2 SU(2) matrix.
fn quat_to_su2(q: &[f64]) -> Mat2 {
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        .sqrt()
        .max(1e-12);
    let w = q[0] / norm;
    let x = q[1] / norm;
    let y = q[2] / norm;
    let z = q[3] / norm;
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
