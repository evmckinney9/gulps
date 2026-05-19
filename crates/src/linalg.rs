//! Fixed-size numerical kernels for the solver hot path.

use crate::{Mat4, C0, C64};
use faer::Mat;
use std::f64::consts::PI;

/// Unit-normalize a length-4 quaternion slice; norm floored at 1e-12.
#[inline]
pub(crate) fn normalize_quat(q: &[f64]) -> ([f64; 4], f64) {
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        .sqrt()
        .max(1e-12);
    ([q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm], norm)
}

/// Matrix-vector product `m * v` for a 4×4 matrix and a length-4 vector.
#[inline]
pub(crate) fn mat4_vec(m: &Mat4, v: &[C64; 4]) -> [C64; 4] {
    let mut out = [C0; 4];
    for j in 0..4 {
        for l in 0..4 {
            out[j] += m[(j, l)] * v[l];
        }
    }
    out
}

/// tr(M^2) without forming M*M.
#[inline]
pub(crate) fn trace_of_square(m: &Mat4) -> C64 {
    let mut s = C0;
    for i in 0..4 {
        for j in 0..4 {
            s += m[(i, j)] * m[(j, i)];
        }
    }
    s
}

/// S · M^T · S where S = σ_y⊗σ_y is the signed permutation (π: 0↔3, 1↔2;
/// signs -,+,+,-), so the result is sign[i]·sign[j]·M[π(j),π(i)] without two matmuls.
#[inline]
#[rustfmt::skip] // 4×4 grid layout mirrors the matrix it builds
pub(crate) fn sysy_transpose_sysy(m: &Mat4) -> Mat4 {
    Mat4::new(
        m[(3, 3)], -m[(2, 3)], -m[(1, 3)],  m[(0, 3)],
       -m[(3, 2)],  m[(2, 2)],  m[(1, 2)], -m[(0, 2)],
       -m[(3, 1)],  m[(2, 1)],  m[(1, 1)], -m[(0, 1)],
        m[(3, 0)], -m[(2, 0)], -m[(1, 0)],  m[(0, 0)],
    )
}

/// L∞ norm of a 3-vector.
#[inline]
pub(crate) fn linf_3(r: &[f64; 3]) -> f64 {
    r[0].abs().max(r[1].abs()).max(r[2].abs())
}

/// Gram matrix J·Jᵀ + λI for a 3×8 Jacobian.
#[inline]
pub(crate) fn gram_3x3(jac: &[[f64; 8]; 3], damping: f64) -> [[f64; 3]; 3] {
    let mut g = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in i..3 {
            let mut s = 0.0;
            for k in 0..8 {
                s += jac[i][k] * jac[j][k];
            }
            g[i][j] = s;
            g[j][i] = s;
        }
        g[i][i] += damping;
    }
    g
}

/// 3×3 SPD linear solve via Cholesky. Returns None if A is not positive definite.
#[inline]
pub(crate) fn solve_3x3(a: &[[f64; 3]; 3], b: &[f64; 3]) -> Option<[f64; 3]> {
    let l00 = a[0][0].sqrt();
    if l00 < 1e-30 {
        return None;
    }
    let l10 = a[1][0] / l00;
    let l20 = a[2][0] / l00;
    let l11_sq = a[1][1] - l10 * l10;
    if l11_sq <= 0.0 {
        return None;
    }
    let l11 = l11_sq.sqrt();
    let l21 = (a[2][1] - l20 * l10) / l11;
    let l22_sq = a[2][2] - l20 * l20 - l21 * l21;
    if l22_sq <= 0.0 {
        return None;
    }
    let l22 = l22_sq.sqrt();

    // Forward substitution: L·y = b
    let y0 = b[0] / l00;
    let y1 = (b[1] - l10 * y0) / l11;
    let y2 = (b[2] - l20 * y0 - l21 * y1) / l22;

    // Back substitution: Lᵀ·x = y
    let x2 = y2 / l22;
    let x1 = (y1 - l21 * x2) / l11;
    let x0 = (y0 - l10 * x1 - l20 * x2) / l00;

    Some([x0, x1, x2])
}

/// Apply Jᵀ·v to get the 8-vector GN step direction.
#[inline]
pub(crate) fn jt_times_v(jac: &[[f64; 8]; 3], v: &[f64; 3]) -> [f64; 8] {
    let mut out = [0.0f64; 8];
    for k in 0..8 {
        out[k] = jac[0][k] * v[0] + jac[1][k] * v[1] + jac[2][k] * v[2];
    }
    out
}

/// Eigenvalue phases (in units of π) of a 4×4 matrix, via faer.
pub(crate) fn eigenphases_4x4(m: &Mat4) -> [f64; 4] {
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

/// Eigenvalues, phases (units of π), and right eigenvectors of a 4×4 matrix; `vecs[i] ↔ eigvals[i]`.
pub(crate) fn eigendecomp_4x4(m: &Mat4) -> ([C64; 4], [f64; 4], [[C64; 4]; 4]) {
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
            vecs[i][j] = u[(j, i)];
        }
    }

    (eigvals, phases, vecs)
}
