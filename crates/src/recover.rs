//! Local equivalence recovery via KAK decomposition.
//!
//! The KAK decomposition algorithm in this module is derived from
//! Qiskit's `TwoQubitWeylDecomposition` implementation.
//! Original source: https://github.com/Qiskit/qiskit/blob/main/crates/synthesis/src/two_qubit_decompose/weyl_decomposition.rs
// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
// NOTE: In earlier version, I directly imported TwoQubitWeylDecomposition, but ported it here to (1) simplify the Cargo dependency stack and build process, and (2) modify the canonicalized conventions to agree with the rest of my invariant representations.

use faer::{Mat, Side};
use rand::prelude::*;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use crate::gate_primitives::{kron_2x2, project_su4, Mat2, Mat4, C0, C1, C64, CI, CM1};

const WEYL_MATCH_TOL: f64 = 2e-5;
const TWO_PI: f64 = 2.0 * PI;
const PI32: f64 = 3.0 * FRAC_PI_2;

// ---------------------------------------------------------------------------
// Magic basis (Qiskit's B_NON_NORMALIZED convention)
// ---------------------------------------------------------------------------
//
// decompose() uses Qiskit's B_NON_NORMALIZED (entries of magnitude 1, B†B=2I).
// This is NOT the same as gate_primitives::magic() (Q-basis, = B/√2 with a
// column permutation). The KAK Weyl flip logic was derived for B-ordering.

/// Qiskit's non-normalized magic basis B.
fn b_nonnorm() -> Mat4 {
    Mat4::new(
        C1, CI, C0, C0, C0, C0, CI, C1, C0, C0, CI, CM1, C1, -CI, C0, C0,
    )
}

/// B† (conjugate transpose of B_NON_NORMALIZED).
fn b_nonnorm_dag() -> Mat4 {
    b_nonnorm().adjoint()
}

// ---------------------------------------------------------------------------
// iPauli matrices for Weyl chamber flipping
// ---------------------------------------------------------------------------

fn ipx() -> Mat2 {
    Mat2::new(C0, CI, CI, C0)
}

fn ipy() -> Mat2 {
    Mat2::new(C0, C1, CM1, C0)
}

fn ipz() -> Mat2 {
    Mat2::new(CI, C0, C0, -CI)
}

// ---------------------------------------------------------------------------
// Pauli matrices (for branch corrections)
// ---------------------------------------------------------------------------

fn pauli_x() -> Mat2 {
    Mat2::new(C0, C1, C1, C0)
}
fn pauli_y() -> Mat2 {
    Mat2::new(C0, -CI, CI, C0)
}
fn pauli_z() -> Mat2 {
    Mat2::new(C1, C0, C0, CM1)
}

// ---------------------------------------------------------------------------
// Closest unitary via polar decomposition (SVD)
// ---------------------------------------------------------------------------

fn closest_unitary(a: &Mat4) -> Mat4 {
    let fm = Mat::<C64>::from_fn(4, 4, |i, j| a[(i, j)]);
    let svd = fm.svd().expect("SVD failed");
    let result = svd.U() * svd.V().adjoint();
    Mat4::from_fn(|r, c| result[(r, c)])
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Sort indices by ascending value.
fn arg_sort(data: &[f64; 3]) -> [usize; 3] {
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&a, &b| {
        data[a]
            .partial_cmp(&data[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx
}

/// Decompose a 4×4 unitary that is kron(L, R) into its L and R factors.
///
/// Returns (L, R, phase) where L, R ∈ SU(2) and U ≈ exp(i·phase) · kron(L, R).
fn decompose_two_qubit_product_gate(u: &Mat4) -> Result<(Mat2, Mat2, f64), String> {
    // Try R = top-left 2×2 block
    let mut r = Mat2::new(u[(0, 0)], u[(0, 1)], u[(1, 0)], u[(1, 1)]);
    let mut det_r = r.determinant();
    if det_r.norm() < 0.1 {
        // Fall back to bottom-left 2×2 block
        r = Mat2::new(u[(2, 0)], u[(2, 1)], u[(3, 0)], u[(3, 1)]);
        det_r = r.determinant();
    }
    if det_r.norm() < 0.1 {
        return Err("decompose_two_qubit_product_gate: detR < 0.1".into());
    }
    // Normalize R to SU(2): R /= sqrt(det(R))
    r /= det_r.sqrt();

    // temp = U @ kron(I, R†)
    let r_dag = r.adjoint();
    let ir_dag = kron_2x2(&Mat2::identity(), &r_dag);
    let temp = u * ir_dag;

    // L = temp[::2, ::2] (rows {0,2}, cols {0,2})
    let l_unnorm = Mat2::new(temp[(0, 0)], temp[(0, 2)], temp[(2, 0)], temp[(2, 2)]);
    let det_l = l_unnorm.determinant();
    if det_l.norm() < 0.9 {
        return Err("decompose_two_qubit_product_gate: detL < 0.9".into());
    }
    let l = l_unnorm / det_l.sqrt();
    let phase = det_l.arg() / 2.0;

    Ok((l, r, phase))
}

// ---------------------------------------------------------------------------
// KAK decomposition
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

/// Diagonalize a complex-symmetric 4×4 matrix via random real linear combination.
///
/// Returns (P, d) where P is real-orthogonal (promoted to complex) and d are the
/// complex eigenvalues, satisfying M² = P · diag(d) · P^T.
fn diagonalize_m2(m2: &Mat4) -> Result<(Mat4, [C64; 4]), String> {
    let mut rng = rand::rng();
    for i in 0..100 {
        let (rand_a, rand_b) = if i == 0 {
            // Fixed seeds matching Qiskit for determinism
            (1.2602066112249388, 0.22317849046722027)
        } else {
            (rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0))
        };

        let m2_real =
            Mat::<f64>::from_fn(4, 4, |r, c| rand_a * m2[(r, c)].re + rand_b * m2[(r, c)].im);

        let evd = match m2_real.self_adjoint_eigen(Side::Lower) {
            Ok(evd) => evd,
            Err(_) => continue,
        };

        let p_ref = evd.U();
        let p = Mat4::from_fn(|r, c| C64::new(p_ref[(r, c)], 0.0));

        let ptm2p = p.transpose() * m2 * &p;
        let d = [ptm2p[(0, 0)], ptm2p[(1, 1)], ptm2p[(2, 2)], ptm2p[(3, 3)]];

        // Validate: P · diag(d) · P^T ≈ M²
        let recon = &p
            * Mat4::from_diagonal(&nalgebra::Vector4::new(d[0], d[1], d[2], d[3]))
            * p.transpose();
        let max_diff = (0..4)
            .flat_map(|r| (0..4).map(move |c| (r, c)))
            .map(|(r, c)| (recon[(r, c)] - m2[(r, c)]).norm())
            .fold(0.0f64, f64::max);

        if max_diff < 1.0e-13 {
            return Ok((p, d));
        }
    }
    Err("KAK decomposition: failed to diagonalize M² after 100 attempts".into())
}

/// Full KAK decomposition of a 4×4 unitary.
///
/// Decomposes U = exp(i·φ) · kron(K1l, K1r) · Can(a,b,c) · kron(K2l, K2r)
/// with (a,b,c) in the canonical Weyl chamber.
fn decompose(u: &Mat4) -> Result<KakFactors, String> {
    let b_mat = b_nonnorm();
    let b_dag = b_nonnorm_dag();

    // Step 1: Determinant normalization → SU(4)
    let (u_su4, mut global_phase) = project_su4(u);

    // Step 2: Magic basis transform: U' = B† @ U_su4 @ B
    let u_p = &b_dag * &u_su4 * &b_mat;

    // Step 3: M² = U'^T @ U' (transpose, NOT adjoint)
    let m2 = u_p.transpose() * &u_p;

    // Step 4: Diagonalize M² via random real linear combination
    // M² is complex-symmetric: M² = A + iB where A, B real-symmetric and commuting.
    // A random real combination rand_a·A + rand_b·B is real-symmetric with the same
    // eigenvectors (generically non-degenerate), allowing standard self-adjoint eigendecomp.
    let (mut p, d_eig) = diagonalize_m2(&m2)?;

    // Step 5: Extract phases
    let mut d = [0.0f64; 4];
    for i in 0..4 {
        d[i] = -d_eig[i].arg() / 2.0;
    }
    d[3] = -d[0] - d[1] - d[2];

    // cs[i] = (d[i] + d[3]) / 2 mod 2π
    let mut cs = [0.0f64; 3];
    for i in 0..3 {
        cs[i] = ((d[i] + d[3]) / 2.0).rem_euclid(TWO_PI);
    }

    // Sort by distance to Weyl chamber boundary (π/4)
    let cstemp = [
        cs[0]
            .rem_euclid(FRAC_PI_2)
            .min(FRAC_PI_2 - cs[0].rem_euclid(FRAC_PI_2)),
        cs[1]
            .rem_euclid(FRAC_PI_2)
            .min(FRAC_PI_2 - cs[1].rem_euclid(FRAC_PI_2)),
        cs[2]
            .rem_euclid(FRAC_PI_2)
            .min(FRAC_PI_2 - cs[2].rem_euclid(FRAC_PI_2)),
    ];
    let order = arg_sort(&cstemp);
    // Cyclic reorder: (order[1], order[2], order[0])
    let reorder = [order[1], order[2], order[0]];

    let cs_old = cs;
    let d_old = d;
    cs = [cs_old[reorder[0]], cs_old[reorder[1]], cs_old[reorder[2]]];
    d[0] = d_old[reorder[0]];
    d[1] = d_old[reorder[1]];
    d[2] = d_old[reorder[2]];
    // d[3] unchanged

    // Reorder P columns
    let p_orig = p;
    p = Mat4::from_fn(|r, c| {
        if c < 3 {
            p_orig[(r, reorder[c])]
        } else {
            p_orig[(r, 3)]
        }
    });

    // Ensure det(P) > 0 (P ∈ SO(4))
    if p.determinant().re < 0.0 {
        for r in 0..4 {
            p[(r, 3)] = -p[(r, 3)];
        }
    }

    // Step 6: Extract K-matrices
    // K1 = B @ U' @ P @ diag(exp(i·d)) @ B†
    let mut diag_exp = Mat4::zeros();
    for i in 0..4 {
        diag_exp[(i, i)] = (CI * d[i]).exp();
    }
    let k1 = &b_mat * (&u_p * &p * &diag_exp) * &b_dag;
    // K2 = B @ P^T @ B†
    let k2 = &b_mat * p.transpose() * &b_dag;

    let (mut k1l, mut k1r, phase_l) = decompose_two_qubit_product_gate(&k1)?;
    let (k2l, mut k2r, phase_r) = decompose_two_qubit_product_gate(&k2)?;
    global_phase += phase_l + phase_r;

    // Step 7: Flip into Weyl chamber
    let ipx_m = ipx();
    let ipy_m = ipy();
    let ipz_m = ipz();

    if cs[0] > FRAC_PI_2 {
        cs[0] -= PI32;
        k1l = k1l * ipy_m;
        k1r = k1r * ipy_m;
        global_phase += FRAC_PI_2;
    }
    if cs[1] > FRAC_PI_2 {
        cs[1] -= PI32;
        k1l = k1l * ipx_m;
        k1r = k1r * ipx_m;
        global_phase += FRAC_PI_2;
    }
    let mut conjs = 0;
    if cs[0] > FRAC_PI_4 {
        cs[0] = FRAC_PI_2 - cs[0];
        k1l = k1l * ipy_m;
        k2r = ipy_m * k2r;
        conjs += 1;
        global_phase -= FRAC_PI_2;
    }
    if cs[1] > FRAC_PI_4 {
        cs[1] = FRAC_PI_2 - cs[1];
        k1l = k1l * ipx_m;
        k2r = ipx_m * k2r;
        conjs += 1;
        global_phase += FRAC_PI_2;
        if conjs == 1 {
            global_phase -= PI;
        }
    }
    if cs[2] > FRAC_PI_2 {
        cs[2] -= PI32;
        k1l = k1l * ipz_m;
        k1r = k1r * ipz_m;
        global_phase += FRAC_PI_2;
        if conjs == 1 {
            global_phase -= PI;
        }
    }
    if conjs == 1 {
        cs[2] = FRAC_PI_2 - cs[2];
        k1l = k1l * ipz_m;
        k2r = ipz_m * k2r;
        global_phase += FRAC_PI_2;
    }
    if cs[2] > FRAC_PI_4 {
        cs[2] -= FRAC_PI_2;
        k1l = k1l * ipz_m;
        k1r = k1r * ipz_m;
        global_phase -= FRAC_PI_2;
    }

    // Step 8: [a, b, c] = [cs[1], cs[0], cs[2]]
    Ok(KakFactors {
        a: cs[1],
        b: cs[0],
        c: cs[2],
        global_phase,
        k1l,
        k1r,
        k2l,
        k2r,
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
    let phase = C64::new(gphase.cos(), gphase.sin());
    let left = kron_2x2(k4, k3);
    let right = kron_2x2(k2, k1);
    let recon = left * u_basis * right * phase;
    let diff = u_target - recon;
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

impl From<(Mat2, Mat2, Mat2, Mat2, f64)> for RecoveryResult {
    fn from((k1, k2, k3, k4, global_phase): (Mat2, Mat2, Mat2, Mat2, f64)) -> Self {
        Self {
            k1,
            k2,
            k3,
            k4,
            global_phase,
        }
    }
}

/// Recover local equivalence corrections between two locally-equivalent unitaries.
///
/// Finds k1, k2, k3, k4 in SU(2) and global_phase such that:
///   U_target ~ exp(i*global_phase) * kron(k4, k3) * U_basis * kron(k2, k1)
pub fn recover_local_equivalence(
    u_target: &Mat4,
    u_basis: &Mat4,
) -> Result<RecoveryResult, String> {
    let t = decompose(u_target)?;
    let b = match decompose(u_basis) {
        Ok(b) => b,
        Err(_) => {
            let basis_closest = closest_unitary(u_basis);
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
        return Ok(direct_corrections(&t, &b).into());
    }

    // Case 2: Rho-reflected match
    if (a1 - (FRAC_PI_2 - a2)).abs() < WEYL_MATCH_TOL
        && (b1 - b2).abs() < WEYL_MATCH_TOL
        && (c1 + c2).abs() < WEYL_MATCH_TOL
    {
        return Ok(rho_corrections(&t, &b).into());
    }

    // Case 3: Frobenius fallback - try both, pick lowest error
    let direct = direct_corrections(&t, &b);
    let rho = rho_corrections(&t, &b);

    let err_d = recon_error(
        u_target, u_basis, &direct.0, &direct.1, &direct.2, &direct.3, direct.4,
    );
    let err_r = recon_error(u_target, u_basis, &rho.0, &rho.1, &rho.2, &rho.3, rho.4);

    let best_err = err_d.min(err_r);
    if best_err < 0.1 {
        return Ok(if err_d <= err_r { direct } else { rho }.into());
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
///
/// The target canonical matrices are REQUIRED because the solver computed (u0, u1)
/// assuming the prefix is in canonical_matrix(w) form (Q-basis). The recovery maps
/// P toward this specific form so that subsequent segments see the expected K-factors.
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::f64::consts::PI;

    /// Generate a random SU(4) matrix via polar decomposition of a random complex matrix.
    fn random_su4(rng: &mut impl Rng) -> Mat4 {
        let a = Mat4::from_fn(|_, _| {
            C64::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0))
        });
        let u = closest_unitary(&a);
        let (su4, _) = project_su4(&u);
        su4
    }

    /// Verify KAK roundtrip: U = exp(ig) kron(K1l,K1r) Can(a,b,c) kron(K2l,K2r).
    #[test]
    fn kak_roundtrip() {
        let mut rng = StdRng::seed_from_u64(456);
        let mut max_err = 0.0f64;

        for i in 0..200 {
            let u = random_su4(&mut rng);
            let kak = decompose(&u).expect(&format!("decompose failed on matrix {i}"));

            let can = crate::gate_primitives::canonical_matrix(
                kak.a * 2.0 / PI,
                kak.b * 2.0 / PI,
                kak.c * 2.0 / PI,
            );
            let k1 = kron_2x2(&kak.k1l, &kak.k1r);
            let k2 = kron_2x2(&kak.k2l, &kak.k2r);
            let phase = C64::new(kak.global_phase.cos(), kak.global_phase.sin());
            let recon = k1 * can * k2 * phase;

            let (u_su4, _) = project_su4(&u);
            let err: f64 = (0..4)
                .flat_map(|r| (0..4).map(move |c| (r, c)))
                .map(|(r, c)| (recon[(r, c)] - u_su4[(r, c)]).norm())
                .fold(0.0, f64::max);
            max_err = max_err.max(err);

            assert!(
                err < 1e-10,
                "Matrix {i}: reconstruction error {:.2e}",
                err,
            );
        }
        eprintln!("Max reconstruction error: {:.2e}", max_err);
    }

    /// Verify known gates (Identity, CX, iSWAP, SWAP).
    #[test]
    fn known_gates() {
        let tol = 1e-10;

        let id = Mat4::identity();
        let kak = decompose(&id).unwrap();
        assert!(kak.a.abs() < tol && kak.b.abs() < tol && kak.c.abs() < tol);

        let cx = Mat4::new(
            C1, C0, C0, C0, C0, C1, C0, C0, C0, C0, C0, C1, C0, C0, C1, C0,
        );
        let kak = decompose(&cx).unwrap();
        assert!((kak.a - FRAC_PI_4).abs() < tol && kak.b.abs() < tol && kak.c.abs() < tol);

        let iswap = Mat4::new(
            C1, C0, C0, C0, C0, C0, CI, C0, C0, CI, C0, C0, C0, C0, C0, C1,
        );
        let kak = decompose(&iswap).unwrap();
        assert!(
            (kak.a - FRAC_PI_4).abs() < tol
                && (kak.b - FRAC_PI_4).abs() < tol
                && kak.c.abs() < tol
        );

        let swap = Mat4::new(
            C1, C0, C0, C0, C0, C0, C1, C0, C0, C1, C0, C0, C0, C0, C0, C1,
        );
        let kak = decompose(&swap).unwrap();
        assert!(
            (kak.a - FRAC_PI_4).abs() < tol
                && (kak.b - FRAC_PI_4).abs() < tol
                && (kak.c - FRAC_PI_4).abs() < tol
        );
    }

    /// Verify recover_local_equivalence on constructed locally-equivalent pairs.
    #[test]
    fn recovery_roundtrip() {
        let mut rng = StdRng::seed_from_u64(2025);

        for _ in 0..100 {
            let u_basis = random_su4(&mut rng);

            // Random SU(2) local unitaries via polar decomposition
            let random_su2 = |rng: &mut StdRng| -> Mat2 {
                let a = Mat2::from_fn(|_, _| {
                    C64::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0))
                });
                let fm = faer::Mat::<C64>::from_fn(2, 2, |i, j| a[(i, j)]);
                let svd = fm.svd().expect("SVD failed");
                let u = svd.U() * svd.V().adjoint();
                Mat2::from_fn(|r, c| u[(r, c)])
            };

            let u_target = kron_2x2(&random_su2(&mut rng), &random_su2(&mut rng))
                * &u_basis
                * kron_2x2(&random_su2(&mut rng), &random_su2(&mut rng));

            let result = recover_local_equivalence(&u_target, &u_basis).unwrap();
            let err = recon_error(
                &u_target,
                &u_basis,
                &result.k1,
                &result.k2,
                &result.k3,
                &result.k4,
                result.global_phase,
            );
            assert!(err < 1e-6, "Recovery error: {err:.2e}");
        }
    }

}
