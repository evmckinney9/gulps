//! Local equivalence recovery via KAK decomposition.
//!
//! Rust port of `recover_equiv.py` - uses Qiskit's `TwoQubitWeylDecomposition`
//! directly, avoiding Python round-trips.

use faer_ext::IntoFaer;
use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;
use std::f64::consts::FRAC_PI_2;

use qiskit_synthesis::two_qubit_decompose::TwoQubitWeylDecomposition;

use crate::gate_primitives::{kron_2x2, Mat2, Mat4, C0, C1};

const WEYL_MATCH_TOL: f64 = 2e-5;

// Pauli matrices as 2×2 nalgebra
fn pauli_x() -> Mat2 {
    Mat2::new(C0, C1, C1, C0)
}
fn pauli_y() -> Mat2 {
    let ci = Complex64::new(0.0, 1.0);
    Mat2::new(C0, -ci, ci, C0)
}
fn pauli_z() -> Mat2 {
    let neg = Complex64::new(-1.0, 0.0);
    Mat2::new(C1, C0, C0, neg)
}

// ---------------------------------------------------------------------------
// ndarray ↔ nalgebra conversions (2×2)
// ---------------------------------------------------------------------------

fn ndarray_to_mat2(a: &ArrayView2<Complex64>) -> Mat2 {
    Mat2::new(a[[0, 0]], a[[0, 1]], a[[1, 0]], a[[1, 1]])
}

fn mat4_to_ndarray(m: &Mat4) -> Array2<Complex64> {
    Array2::from_shape_fn((4, 4), |(r, c)| m[(r, c)])
}

// ---------------------------------------------------------------------------
// Closest unitary via polar decomposition (SVD)
// ---------------------------------------------------------------------------

fn closest_unitary(a: &Array2<Complex64>) -> Array2<Complex64> {
    let fa = a.view().into_faer();
    let svd = fa.svd().expect("SVD failed");
    let u = svd.U();
    let v = svd.V();
    // V @ Wh = U @ V†
    let result = u * v.adjoint();
    Array2::from_shape_fn((4, 4), |(r, c)| result[(r, c)])
}

// ---------------------------------------------------------------------------
// KAK struct - extracted K-factors from TwoQubitWeylDecomposition
// ---------------------------------------------------------------------------

struct KakFactors {
    a: f64,
    b: f64,
    c: f64,
    global_phase: f64,
    k1l: Mat2,
    k1r: Mat2,
    k2l: Mat2,
    k2r: Mat2,
}

fn decompose(u: &Array2<Complex64>) -> Result<KakFactors, String> {
    let decomp = TwoQubitWeylDecomposition::new_inner(u.view(), Some(1.0), None)
        .map_err(|e| format!("KAK decomposition failed: {e}"))?;

    Ok(KakFactors {
        a: decomp.a(),
        b: decomp.b(),
        c: decomp.c(),
        global_phase: decomp.global_phase,
        k1l: ndarray_to_mat2(&decomp.k1l_view()),
        k1r: ndarray_to_mat2(&decomp.k1r_view()),
        k2l: ndarray_to_mat2(&decomp.k2l_view()),
        k2r: ndarray_to_mat2(&decomp.k2r_view()),
    })
}

// ---------------------------------------------------------------------------
// Branch corrections
// ---------------------------------------------------------------------------

/// Direct branch: K-factor ratios.
fn direct_corrections(t: &KakFactors, b: &KakFactors) -> (Mat2, Mat2, Mat2, Mat2, f64) {
    let k4 = t.k1l * b.k1l.adjoint();
    let k3 = t.k1r * b.k1r.adjoint();
    let k2 = b.k2l.adjoint() * t.k2l;
    let k1 = b.k2r.adjoint() * t.k2r;
    (k1, k2, k3, k4, t.global_phase - b.global_phase)
}

/// Rho-reflected branch: inserts Pauli corrections.
fn rho_corrections(t: &KakFactors, b: &KakFactors) -> (Mat2, Mat2, Mat2, Mat2, f64) {
    let py = pauli_y();
    let pz = pauli_z();
    let px = pauli_x();
    let k4 = t.k1l * py * b.k1l.adjoint();
    let k3 = t.k1r * b.k1r.adjoint();
    let k2 = b.k2l.adjoint() * pz * t.k2l;
    let k1 = b.k2r.adjoint() * px * t.k2r;
    (k1, k2, k3, k4, t.global_phase - b.global_phase)
}

/// Frobenius reconstruction error: ‖U_target - recon‖_F
fn recon_error(
    u_target: &Mat4,
    u_basis: &Mat4,
    k1: &Mat2,
    k2: &Mat2,
    k3: &Mat2,
    k4: &Mat2,
    gphase: f64,
) -> f64 {
    let phase = Complex64::new(gphase.cos(), gphase.sin());
    let left = kron_2x2(k4, k3);
    let right = kron_2x2(k2, k1);
    let recon = left * u_basis * right * phase;
    let diff = u_target - recon;
    // Frobenius norm = sqrt(sum |d_ij|^2)
    let mut sum = 0.0f64;
    for r in 0..4 {
        for c in 0..4 {
            sum += diff[(r, c)].norm_sqr();
        }
    }
    sum.sqrt()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Result of local equivalence recovery.
pub struct RecoveryResult {
    pub k1: Mat2,
    pub k2: Mat2,
    pub k3: Mat2,
    pub k4: Mat2,
    pub global_phase: f64,
}

/// Recover local equivalence corrections between two locally-equivalent unitaries.
///
/// Finds k1, k2, k3, k4 ∈ SU(2) and global_phase such that:
///   U_target ≈ exp(i·global_phase) · kron(k4, k3) · U_basis · kron(k2, k1)
pub fn recover_local_equivalence(
    u_target: &Mat4,
    u_basis: &Mat4,
) -> Result<RecoveryResult, String> {
    let target_nd = mat4_to_ndarray(u_target);
    let basis_nd = mat4_to_ndarray(u_basis);

    let t = decompose(&target_nd)?;
    let b = match decompose(&basis_nd) {
        Ok(b) => b,
        Err(_) => {
            // Closest unitary fallback
            let basis_closest = closest_unitary(&basis_nd);
            decompose(&basis_closest)?
        }
    };

    let (a1, b1, c1) = (t.a, t.b, t.c);
    let (a2, b2, c2) = (b.a, b.b, b.c);

    // Case 1: Direct Weyl match
    if (a1 - a2).abs() < WEYL_MATCH_TOL
        && (b1 - b2).abs() < WEYL_MATCH_TOL
        && (c1 - c2).abs() < WEYL_MATCH_TOL
    {
        let (k1, k2, k3, k4, gp) = direct_corrections(&t, &b);
        return Ok(RecoveryResult {
            k1,
            k2,
            k3,
            k4,
            global_phase: gp,
        });
    }

    // Case 2: Rho-reflected match
    if (a1 - (FRAC_PI_2 - a2)).abs() < WEYL_MATCH_TOL
        && (b1 - b2).abs() < WEYL_MATCH_TOL
        && (c1 + c2).abs() < WEYL_MATCH_TOL
    {
        let (k1, k2, k3, k4, gp) = rho_corrections(&t, &b);
        return Ok(RecoveryResult {
            k1,
            k2,
            k3,
            k4,
            global_phase: gp,
        });
    }

    // Case 3: Frobenius fallback - try both, pick lowest error
    let (dk1, dk2, dk3, dk4, dgp) = direct_corrections(&t, &b);
    let (rk1, rk2, rk3, rk4, rgp) = rho_corrections(&t, &b);

    let err_d = recon_error(u_target, u_basis, &dk1, &dk2, &dk3, &dk4, dgp);
    let err_r = recon_error(u_target, u_basis, &rk1, &rk2, &rk3, &rk4, rgp);

    let best_err = err_d.min(err_r);
    if best_err < 0.1 {
        if err_d <= err_r {
            return Ok(RecoveryResult {
                k1: dk1,
                k2: dk2,
                k3: dk3,
                k4: dk4,
                global_phase: dgp,
            });
        } else {
            return Ok(RecoveryResult {
                k1: rk1,
                k2: rk2,
                k3: rk3,
                k4: rk4,
                global_phase: rgp,
            });
        }
    }

    Err(format!(
        "Cannot recover local equivalence; Weyl diffs ({:.2e}, {:.2e}, {:.2e}), best_err={:.2e}",
        (a1 - a2).abs(),
        (b1 - b2).abs(),
        (c1 - c2).abs(),
        best_err,
    ))
}

// ---------------------------------------------------------------------------
// Stitch: P accumulation + intermediate recovery in one call
// ---------------------------------------------------------------------------

/// Intermediate recovery correction for one segment.
pub struct IntermediateCorrection {
    pub k3: Mat2,
    pub k4: Mat2,
    pub global_phase: f64,
}

/// Result of stitching all segments.
pub struct StitchResult {
    /// Intermediate corrections (one per inner segment, empty for last).
    pub corrections: Vec<IntermediateCorrection>,
    /// Final accumulated unitary P.
    pub final_p: Mat4,
}

/// Accumulate P through all segments and compute intermediate recoveries.
///
/// For each inner segment i:
///   P = basis[i] @ kron(u1[i], u0[i]) @ P
///   if not last: recover(target_canonical[i], P) → (k3, k4, gphase), then P = kron(k4, k3) @ P
pub fn stitch_segments(
    initial_p: &Mat4,
    u0s: &[Mat2],
    u1s: &[Mat2],
    basis_matrices: &[Mat4],
    target_canonical_matrices: &[Mat4],
) -> Result<StitchResult, String> {
    let n = u0s.len();
    let mut p = *initial_p;
    let mut corrections = Vec::with_capacity(n.saturating_sub(1));

    for i in 0..n {
        // P = basis[i] @ kron(u1[i], u0[i]) @ P
        let u1u0 = kron_2x2(&u1s[i], &u0s[i]);
        p = basis_matrices[i] * u1u0 * p;

        // Intermediate recovery (skip for final segment)
        if i < n - 1 {
            let result = recover_local_equivalence(&target_canonical_matrices[i], &p)?;
            let k4k3 = kron_2x2(&result.k4, &result.k3);
            p = k4k3 * p;
            corrections.push(IntermediateCorrection {
                k3: result.k3,
                k4: result.k4,
                global_phase: result.global_phase,
            });
        }
    }

    Ok(StitchResult {
        corrections,
        final_p: p,
    })
}
