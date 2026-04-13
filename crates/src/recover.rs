//! Local equivalence recovery and segment stitching.
//!
//! KAK decomposition algorithm derived from Qiskit's `TwoQubitWeylDecomposition`.
//! Original: https://github.com/Qiskit/qiskit/blob/main/crates/synthesis/src/two_qubit_decompose/weyl_decomposition.rs
// (C) Copyright IBM 2026.
// Licensed under the Apache License, Version 2.0.
// http://www.apache.org/licenses/LICENSE-2.0
// Modified: ported and adapted for gulps conventions.

use faer::{Mat, Side};
use rand::prelude::*;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use crate::gate_primitives::{kron_2x2, project_su4, Mat2, Mat4, C0, C1, C64, CI, CM1};

const WEYL_MATCH_TOL: f64 = 2e-5;
const TWO_PI: f64 = 2.0 * PI;
const PI32: f64 = 3.0 * FRAC_PI_2;

// ---------------------------------------------------------------------------
// Fixed basis matrices
// ---------------------------------------------------------------------------

// Qiskit's B_NON_NORMALIZED (entries of magnitude 1, B†B = 2I).
// NOT the same as gate_primitives::magic() (Q-basis = B/√2 with column permutation).
fn b_nonnorm() -> Mat4 {
    Mat4::new(
        C1, CI, C0, C0, C0, C0, CI, C1, C0, C0, CI, CM1, C1, -CI, C0, C0,
    )
}

fn b_nonnorm_dag() -> Mat4 {
    b_nonnorm().adjoint()
}

fn ipx() -> Mat2 { Mat2::new(C0, CI, CI, C0) }
fn ipy() -> Mat2 { Mat2::new(C0, C1, CM1, C0) }
fn ipz() -> Mat2 { Mat2::new(CI, C0, C0, -CI) }
fn pauli_x() -> Mat2 { Mat2::new(C0, C1, C1, C0) }
fn pauli_y() -> Mat2 { Mat2::new(C0, -CI, CI, C0) }
fn pauli_z() -> Mat2 { Mat2::new(C1, C0, C0, CM1) }

// ---------------------------------------------------------------------------
// KAK decomposition internals
// ---------------------------------------------------------------------------

/// Closest unitary via polar decomposition (SVD).
fn closest_unitary(a: &Mat4) -> Mat4 {
    let fm = Mat::<C64>::from_fn(4, 4, |i, j| a[(i, j)]);
    let svd = fm.svd().expect("SVD failed");
    let result = svd.U() * svd.V().adjoint();
    Mat4::from_fn(|r, c| result[(r, c)])
}

fn arg_sort(data: &[f64; 3]) -> [usize; 3] {
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&a, &b| data[a].partial_cmp(&data[b]).unwrap_or(std::cmp::Ordering::Equal));
    idx
}

/// Factor kron(L, R) into (L, R, phase) with L, R in SU(2).
fn decompose_product_gate(u: &Mat4) -> Result<(Mat2, Mat2, f64), String> {
    let mut r = Mat2::new(u[(0, 0)], u[(0, 1)], u[(1, 0)], u[(1, 1)]);
    let mut det_r = r.determinant();
    if det_r.norm() < 0.1 {
        r = Mat2::new(u[(2, 0)], u[(2, 1)], u[(3, 0)], u[(3, 1)]);
        det_r = r.determinant();
    }
    if det_r.norm() < 0.1 {
        return Err("decompose_product_gate: detR < 0.1".into());
    }
    r /= det_r.sqrt();

    let ir_dag = kron_2x2(&Mat2::identity(), &r.adjoint());
    let temp = u * ir_dag;
    let l_unnorm = Mat2::new(temp[(0, 0)], temp[(0, 2)], temp[(2, 0)], temp[(2, 2)]);
    let det_l = l_unnorm.determinant();
    if det_l.norm() < 0.9 {
        return Err("decompose_product_gate: detL < 0.9".into());
    }
    Ok((l_unnorm / det_l.sqrt(), r, det_l.arg() / 2.0))
}

/// Diagonalize complex-symmetric M² via random real linear combination.
///
/// M² = A + iB where A, B are real-symmetric and commute. A random real
/// combination aA + bB shares the eigenvectors, allowing standard real
/// eigendecomposition. Retries with fresh random seeds if degenerate.
fn diagonalize_m2(m2: &Mat4) -> Result<(Mat4, [C64; 4]), String> {
    let mut rng = rand::rng();
    for i in 0..100 {
        let (ra, rb) = if i == 0 {
            (1.2602066112249388, 0.22317849046722027) // Qiskit fixed seed
        } else {
            (rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0))
        };

        let m2r = Mat::<f64>::from_fn(4, 4, |r, c| ra * m2[(r, c)].re + rb * m2[(r, c)].im);
        let evd = match m2r.self_adjoint_eigen(Side::Lower) {
            Ok(evd) => evd,
            Err(_) => continue,
        };

        let p_ref = evd.U();
        let p = Mat4::from_fn(|r, c| C64::new(p_ref[(r, c)], 0.0));
        let ptm2p = p.transpose() * m2 * &p;
        let d = [ptm2p[(0, 0)], ptm2p[(1, 1)], ptm2p[(2, 2)], ptm2p[(3, 3)]];

        let recon = &p
            * Mat4::from_diagonal(&nalgebra::Vector4::new(d[0], d[1], d[2], d[3]))
            * p.transpose();
        let max_diff = (0..4)
            .flat_map(|r| (0..4).map(move |c| (r, c)))
            .map(|(r, c)| (recon[(r, c)] - m2[(r, c)]).norm())
            .fold(0.0f64, f64::max);

        if max_diff < 1e-13 {
            return Ok((p, d));
        }
    }
    Err("diagonalize_m2: failed after 100 attempts".into())
}

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

/// Full KAK decomposition: U = e^{ig} kron(K1l,K1r) Can(a,b,c) kron(K2l,K2r).
fn decompose(u: &Mat4) -> Result<KakFactors, String> {
    let b_mat = b_nonnorm();
    let b_dag = b_nonnorm_dag();

    let (u_su4, mut global_phase) = project_su4(u);
    let u_p = &b_dag * &u_su4 * &b_mat;
    let m2 = u_p.transpose() * &u_p;
    let (mut p, d_eig) = diagonalize_m2(&m2)?;

    // Extract and reorder phases
    let mut d = [0.0f64; 4];
    for i in 0..4 {
        d[i] = -d_eig[i].arg() / 2.0;
    }
    d[3] = -d[0] - d[1] - d[2];

    let mut cs = [0.0f64; 3];
    for i in 0..3 {
        cs[i] = ((d[i] + d[3]) / 2.0).rem_euclid(TWO_PI);
    }

    // Sort by distance to pi/4 boundary, cyclic reorder
    let cstemp = [
        cs[0].rem_euclid(FRAC_PI_2).min(FRAC_PI_2 - cs[0].rem_euclid(FRAC_PI_2)),
        cs[1].rem_euclid(FRAC_PI_2).min(FRAC_PI_2 - cs[1].rem_euclid(FRAC_PI_2)),
        cs[2].rem_euclid(FRAC_PI_2).min(FRAC_PI_2 - cs[2].rem_euclid(FRAC_PI_2)),
    ];
    let order = arg_sort(&cstemp);
    let reorder = [order[1], order[2], order[0]];

    let (cs_old, d_old) = (cs, d);
    cs = [cs_old[reorder[0]], cs_old[reorder[1]], cs_old[reorder[2]]];
    d[0] = d_old[reorder[0]];
    d[1] = d_old[reorder[1]];
    d[2] = d_old[reorder[2]];

    let p_orig = p;
    p = Mat4::from_fn(|r, c| if c < 3 { p_orig[(r, reorder[c])] } else { p_orig[(r, 3)] });
    if p.determinant().re < 0.0 {
        for r in 0..4 { p[(r, 3)] = -p[(r, 3)]; }
    }

    // Extract K-matrices
    let mut diag_exp = Mat4::zeros();
    for i in 0..4 { diag_exp[(i, i)] = (CI * d[i]).exp(); }
    let k1 = &b_mat * (&u_p * &p * &diag_exp) * &b_dag;
    let k2 = &b_mat * p.transpose() * &b_dag;

    let (mut k1l, mut k1r, phase_l) = decompose_product_gate(&k1)?;
    let (k2l, mut k2r, phase_r) = decompose_product_gate(&k2)?;
    global_phase += phase_l + phase_r;

    // Flip into canonical Weyl chamber
    let (ipx_m, ipy_m, ipz_m) = (ipx(), ipy(), ipz());
    if cs[0] > FRAC_PI_2 {
        cs[0] -= PI32; k1l = k1l * ipy_m; k1r = k1r * ipy_m; global_phase += FRAC_PI_2;
    }
    if cs[1] > FRAC_PI_2 {
        cs[1] -= PI32; k1l = k1l * ipx_m; k1r = k1r * ipx_m; global_phase += FRAC_PI_2;
    }
    let mut conjs = 0;
    if cs[0] > FRAC_PI_4 {
        cs[0] = FRAC_PI_2 - cs[0];
        k1l = k1l * ipy_m; k2r = ipy_m * k2r;
        conjs += 1; global_phase -= FRAC_PI_2;
    }
    if cs[1] > FRAC_PI_4 {
        cs[1] = FRAC_PI_2 - cs[1];
        k1l = k1l * ipx_m; k2r = ipx_m * k2r;
        conjs += 1; global_phase += FRAC_PI_2;
        if conjs == 1 { global_phase -= PI; }
    }
    if cs[2] > FRAC_PI_2 {
        cs[2] -= PI32; k1l = k1l * ipz_m; k1r = k1r * ipz_m;
        global_phase += FRAC_PI_2;
        if conjs == 1 { global_phase -= PI; }
    }
    if conjs == 1 {
        cs[2] = FRAC_PI_2 - cs[2];
        k1l = k1l * ipz_m; k2r = ipz_m * k2r; global_phase += FRAC_PI_2;
    }
    if cs[2] > FRAC_PI_4 {
        cs[2] -= FRAC_PI_2; k1l = k1l * ipz_m; k1r = k1r * ipz_m; global_phase -= FRAC_PI_2;
    }

    Ok(KakFactors { a: cs[1], b: cs[0], c: cs[2], global_phase, k1l, k1r, k2l, k2r })
}

/// Decompose the basis, falling back to closest_unitary if needed.
fn decompose_basis(u_basis: &Mat4) -> Result<KakFactors, String> {
    match decompose(u_basis) {
        Ok(b) => Ok(b),
        Err(_) => decompose(&closest_unitary(u_basis)),
    }
}

// ---------------------------------------------------------------------------
// Recovery: find local corrections between locally-equivalent unitaries
// ---------------------------------------------------------------------------

pub struct RecoveryResult {
    pub k1: Mat2,
    pub k2: Mat2,
    pub k3: Mat2,
    pub k4: Mat2,
    pub global_phase: f64,
}

impl From<(Mat2, Mat2, Mat2, Mat2, f64)> for RecoveryResult {
    fn from((k1, k2, k3, k4, global_phase): (Mat2, Mat2, Mat2, Mat2, f64)) -> Self {
        Self { k1, k2, k3, k4, global_phase }
    }
}

fn direct_corrections(t: &KakFactors, b: &KakFactors) -> (Mat2, Mat2, Mat2, Mat2, f64) {
    (
        b.k2r.adjoint() * t.k2r,
        b.k2l.adjoint() * t.k2l,
        t.k1r * b.k1r.adjoint(),
        t.k1l * b.k1l.adjoint(),
        t.global_phase - b.global_phase,
    )
}

fn rho_corrections(t: &KakFactors, b: &KakFactors) -> (Mat2, Mat2, Mat2, Mat2, f64) {
    (
        b.k2r.adjoint() * pauli_x() * t.k2r,
        b.k2l.adjoint() * pauli_z() * t.k2l,
        t.k1r * b.k1r.adjoint(),
        t.k1l * pauli_y() * b.k1l.adjoint(),
        t.global_phase - b.global_phase,
    )
}

fn recon_error(
    u_target: &Mat4, u_basis: &Mat4,
    k1: &Mat2, k2: &Mat2, k3: &Mat2, k4: &Mat2, gphase: f64,
) -> f64 {
    let phase = C64::new(gphase.cos(), gphase.sin());
    let recon = kron_2x2(k4, k3) * u_basis * kron_2x2(k2, k1) * phase;
    let diff = u_target - recon;
    let mut sum = 0.0f64;
    for r in 0..4 { for c in 0..4 { sum += diff[(r, c)].norm_sqr(); } }
    sum.sqrt()
}

/// General recovery: U_target ~ e^{ig} kron(k4,k3) U_basis kron(k2,k1).
///
/// Tries direct Weyl match, rho-reflected match, then Frobenius fallback.
pub fn recover_local_equivalence(
    u_target: &Mat4,
    u_basis: &Mat4,
) -> Result<RecoveryResult, String> {
    let t = decompose(u_target)?;
    let b = decompose_basis(u_basis)?;

    // Direct match
    if (t.a - b.a).abs() < WEYL_MATCH_TOL
        && (t.b - b.b).abs() < WEYL_MATCH_TOL
        && (t.c - b.c).abs() < WEYL_MATCH_TOL
    {
        return Ok(direct_corrections(&t, &b).into());
    }

    // Rho-reflected match
    if (t.a - (FRAC_PI_2 - b.a)).abs() < WEYL_MATCH_TOL
        && (t.b - b.b).abs() < WEYL_MATCH_TOL
        && (t.c + b.c).abs() < WEYL_MATCH_TOL
    {
        return Ok(rho_corrections(&t, &b).into());
    }

    // Frobenius fallback: try both, pick lowest error
    let direct = direct_corrections(&t, &b);
    let rho = rho_corrections(&t, &b);
    let err_d = recon_error(u_target, u_basis, &direct.0, &direct.1, &direct.2, &direct.3, direct.4);
    let err_r = recon_error(u_target, u_basis, &rho.0, &rho.1, &rho.2, &rho.3, rho.4);
    let best_err = err_d.min(err_r);
    if best_err < 0.1 {
        return Ok(if err_d <= err_r { direct } else { rho }.into());
    }

    Err(format!(
        "recover: Weyl diffs ({:.2e}, {:.2e}, {:.2e}), best_err={:.2e}",
        (t.a - b.a).abs(), (t.b - b.b).abs(), (t.c - b.c).abs(), best_err,
    ))
}

/// Optimized recovery when the target is a canonical matrix.
///
/// Canonical matrices exp(-i(aXX+bYY+cZZ)) have identity K-factors in the
/// KAK decomposition, so we only decompose the basis (halving the cost).
fn recover_canonical_target(
    target_weyl: [f64; 3],
    u_basis: &Mat4,
) -> Result<RecoveryResult, String> {
    let b = decompose_basis(u_basis)?;
    let (a1, b1, c1) = (target_weyl[0] * FRAC_PI_2, target_weyl[1] * FRAC_PI_2, target_weyl[2] * FRAC_PI_2);

    // Direct match (target K-factors = I)
    if (a1 - b.a).abs() < WEYL_MATCH_TOL
        && (b1 - b.b).abs() < WEYL_MATCH_TOL
        && (c1 - b.c).abs() < WEYL_MATCH_TOL
    {
        return Ok(RecoveryResult {
            k1: b.k2r.adjoint(), k2: b.k2l.adjoint(),
            k3: b.k1r.adjoint(), k4: b.k1l.adjoint(),
            global_phase: -b.global_phase,
        });
    }

    // Rho-reflected match (target K-factors = I, with Pauli insertions)
    if (a1 - (FRAC_PI_2 - b.a)).abs() < WEYL_MATCH_TOL
        && (b1 - b.b).abs() < WEYL_MATCH_TOL
        && (c1 + b.c).abs() < WEYL_MATCH_TOL
    {
        return Ok(RecoveryResult {
            k1: b.k2r.adjoint() * pauli_x(), k2: b.k2l.adjoint() * pauli_z(),
            k3: b.k1r.adjoint(), k4: pauli_y() * b.k1l.adjoint(),
            global_phase: -b.global_phase,
        });
    }

    // Frobenius fallback: construct canonical matrix and identity KakFactors
    let u_target = crate::gate_primitives::canonical_matrix(target_weyl[0], target_weyl[1], target_weyl[2]);
    let t_id = KakFactors {
        a: a1, b: b1, c: c1, global_phase: 0.0,
        k1l: Mat2::identity(), k1r: Mat2::identity(),
        k2l: Mat2::identity(), k2r: Mat2::identity(),
    };
    let direct = direct_corrections(&t_id, &b);
    let rho = rho_corrections(&t_id, &b);
    let err_d = recon_error(&u_target, u_basis, &direct.0, &direct.1, &direct.2, &direct.3, direct.4);
    let err_r = recon_error(&u_target, u_basis, &rho.0, &rho.1, &rho.2, &rho.3, rho.4);
    let best_err = err_d.min(err_r);
    if best_err < 0.1 {
        return Ok(if err_d <= err_r { direct } else { rho }.into());
    }

    Err(format!(
        "recover_canonical: Weyl diffs ({:.2e}, {:.2e}, {:.2e}), best_err={:.2e}",
        (a1 - b.a).abs(), (b1 - b.b).abs(), (c1 - b.c).abs(), best_err,
    ))
}

// ---------------------------------------------------------------------------
// Stitch: P accumulation + recovery + fusion in one call
// ---------------------------------------------------------------------------

pub struct StitchResult {
    pub fused_u0s: Vec<Mat2>,
    pub fused_u1s: Vec<Mat2>,
    pub intermediate_gphase: f64,
    pub final_recovery: RecoveryResult,
}

/// Full stitch pipeline: accumulate P, recover intermediates (fusing corrections
/// into next segment's u0/u1), and perform final recovery against the target.
pub fn stitch_segments(
    initial_p: &Mat4,
    u0s: &[Mat2],
    u1s: &[Mat2],
    basis_matrices: &[Mat4],
    target_weyl_coords: &[[f64; 3]],
    final_target: &Mat4,
) -> Result<StitchResult, String> {
    let n = u0s.len();
    let mut p = *initial_p;
    let mut fused_u0s = Vec::with_capacity(n);
    let mut fused_u1s = Vec::with_capacity(n);
    let mut intermediate_gphase = 0.0;

    if n > 0 {
        fused_u0s.push(u0s[0]);
        fused_u1s.push(u1s[0]);
    }

    for i in 0..n {
        let u1u0 = kron_2x2(&u1s[i], &u0s[i]);
        p = basis_matrices[i] * u1u0 * p;

        if i < n - 1 {
            let result = recover_canonical_target(target_weyl_coords[i], &p)?;
            p = kron_2x2(&result.k4, &result.k3) * p;
            intermediate_gphase += result.global_phase;
            fused_u0s.push(u0s[i + 1] * result.k3);
            fused_u1s.push(u1s[i + 1] * result.k4);
        }
    }

    let final_recovery = recover_local_equivalence(final_target, &p)?;
    Ok(StitchResult { fused_u0s, fused_u1s, intermediate_gphase, final_recovery })
}
