//! GN restart-loop solver for two-qubit gate segment synthesis.
//!
//! Two stages:
//! 1. Random-restart GN on Makhlin invariants (fast, ~6.8μs/iter)
//! 2. Weyl LM polish for fidelity saturation (128 iters, ~9μs/iter)
//!
//! Stage 1 features:
//! - Analytic Jacobian for Makhlin residual (exact chain rule)
//! - Best-tracking per restart
//! - Tiered stagnation abort (j≥8: 3e-1, j≥16: 5e-2, j≥32: 1e-3)
//! - Weyl-coordinate early-exit in the outer restart loop
//!
//! Stage 2 (Weyl polish) features:
//! - Analytic Jacobian via eigenvalue perturbation theory (1 eigendecomp
//!   w/ eigenvectors + chain rule, replacing 9 eigenvalue-only calls)
//! - Trial acceptance still uses eigenvalues-only (2 eigendecomps/iter total)
//! - Adaptive LM damping, stale-exit after 4 non-improving iters
//! - Generic targets: ~17 iters to 1e-16. Degenerate: stalls in ~4 iters.
//! - In-loop polish attempts were tested and rejected (see CLAUDE.md):
//!   early exit prevents finding the best Makhlin solution from later restarts.

use crate::gate_primitives::*;
use rand::prelude::*;
use rand::rngs::SmallRng;
use std::f64::consts::PI;

const MAXITER: usize = 512;
const MAX_RESTARTS: usize = 512;
const DAMPING: f64 = 1e-14;
const POLISH_FLOOR: f64 = 1e-12;
const POLISH_BUDGET: usize = 128; // post-loop polish budget (fidelity saturation)

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Solve one segment: find (u0, u1) ∈ SU(2)×SU(2) such that
/// basis_gate ⊗ kron(u1, u0) ⊗ prefix_op is locally equivalent to target_mat.
///
/// Returns (u0, u1, weyl_residual, makhlin_residual).
pub fn solve(
    seed: u64,
    prefix_op: &Mat4,
    basis_gate: &Mat4,
    target_mat: &Mat4,
    makhlin_tol: f64,
    weyl_tol: f64,
) -> (Mat2, Mat2, f64, f64) {
    let target_makhlin = makhlin_invariants(target_mat);
    let target_weyl = weyl_coordinates(target_mat);
    let (prefix_magic, magic_basis, target_packed) =
        precompute_makhlin_args(prefix_op, basis_gate, &target_makhlin);

    // Makhlin GN restart loop — find the best Makhlin solution.
    let (mut best_params, best_res) = gn_restart_loop(
        seed,
        &prefix_magic,
        &magic_basis,
        &target_packed,
        makhlin_tol,
        prefix_op,
        basis_gate,
        &target_weyl,
        weyl_tol,
    );

    // Weyl LM polish: close the Makhlin→Weyl gap for fidelity.
    // ~17 iters to 1e-16 at generic targets, stale-exit in ~4 iters at degenerate.
    let weyl_res = weyl_polish(&mut best_params, prefix_op, basis_gate, &target_weyl, POLISH_BUDGET);
    let (u0, u1) = params_to_unitaries(&best_params);
    (u0, u1, weyl_res, best_res)
}

// ---------------------------------------------------------------------------
// GN restart loop
// ---------------------------------------------------------------------------

/// Returns (best_params, makhlin_res).
///
/// Early exit via Makhlin convergence or raw Weyl check.
/// The post-loop polish handles fidelity — this loop maximizes Makhlin quality.
fn gn_restart_loop(
    seed: u64,
    prefix_magic: &Mat4,
    magic_basis: &Mat4,
    target_packed: &[f64; 5],
    makhlin_tol: f64,
    prefix_op: &Mat4,
    basis_gate: &Mat4,
    target_weyl: &[f64; 3],
    weyl_tol: f64,
) -> ([f64; 8], f64) {
    let mut best_params = [0.0f64; 8];
    let mut best_res = f64::INFINITY;

    for restart in 0..MAX_RESTARTS {
        // Deterministic per-restart seed via PCG-style LCG mixing.
        let rseed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
            .wrapping_add(restart as u64);
        let mut rng = SmallRng::seed_from_u64(rseed);
        let mut init = [0.0f64; 8];
        for v in &mut init {
            *v = rng.random_range(-0.1..0.1);
        }

        // GN inner loop
        let (final_x, final_res) =
            gn_inner_loop(&init, prefix_magic, magic_basis, target_packed, makhlin_tol);

        // Update global best
        if final_res < best_res && final_x.iter().all(|v| v.is_finite()) {
            best_params = final_x;
            best_res = final_res;

            // Early exit: Makhlin converged OR Weyl acceptable
            if best_res <= makhlin_tol {
                break;
            }
            if weyl_check(&best_params, prefix_op, basis_gate, target_weyl) <= weyl_tol {
                break;
            }
        }
    }

    (best_params, best_res)
}

// ---------------------------------------------------------------------------
// GN inner loop (one restart)
// ---------------------------------------------------------------------------

fn gn_inner_loop(
    init: &[f64; 8],
    prefix_magic: &Mat4,
    magic_basis: &Mat4,
    target_packed: &[f64; 5],
    tol: f64,
) -> ([f64; 8], f64) {
    let mut x = *init;
    let mut best_x = *init;
    let mut best_norm = f64::INFINITY;
    let mut prev_norm = f64::INFINITY;

    for j in 0..MAXITER {
        // Stagnation abort (three tiers)
        let stagnated = (j >= 8 && prev_norm > 3e-1)
            || (j >= 16 && prev_norm > 5e-2)
            || (j >= 32 && prev_norm > 1e-3);
        if best_norm <= tol || stagnated {
            break;
        }

        // Residual + Jacobian (fused: single forward pass)
        let (r, jac) = makhlin_residual_and_jacobian(&x, prefix_magic, magic_basis, target_packed);
        let curr_norm = linf_3(&r);

        // Best tracking
        if curr_norm < best_norm {
            best_x = x;
            best_norm = curr_norm;
        }

        // GN step: x_new = x + Jᵀ (J Jᵀ + λI)⁻¹ (-r)
        let gram = gram_3x3(&jac, DAMPING);
        if let Some(sol) = solve_3x3(&gram, &[-r[0], -r[1], -r[2]]) {
            let delta = jt_times_v(&jac, &sol);
            let mut x_new = x;
            for k in 0..8 {
                x_new[k] += delta[k];
            }
            if x_new.iter().all(|v| v.is_finite()) {
                x = x_new;
            }
        }
        prev_norm = curr_norm;
    }

    // Evaluate final iterate (GN step taken but not yet evaluated)
    let r_last = makhlin_residual_fused(&x, prefix_magic, magic_basis, target_packed);
    let last_norm = linf_3(&r_last);
    if last_norm < best_norm && x.iter().all(|v| v.is_finite()) {
        (x, last_norm)
    } else {
        (best_x, best_norm)
    }
}

// ---------------------------------------------------------------------------
// Weyl LM polish
// ---------------------------------------------------------------------------

/// Single-pass LM polish on Weyl coordinates.
///
/// Closes the quadratic Makhlin→Weyl gap at generic targets (~17 iters to 1e-16).
/// At degenerate faces (c1≈c2), the Weyl Jacobian is ill-conditioned and the
/// polish stalls harmlessly.
///
/// Uses analytic Jacobian via eigenvalue perturbation theory: one eigendecomp
/// per iteration (with eigenvectors), then chain-rule through kron and quaternion
/// parameterization. ~3x faster than forward-difference (1 vs 9 eigendecomps).
fn weyl_polish(
    params: &mut [f64; 8],
    prefix_op: &Mat4,
    basis_gate: &Mat4,
    target_weyl: &[f64; 3],
    max_iters: usize,
) -> f64 {
    // Initial Weyl check — skip if already saturated
    let (u0, u1) = params_to_unitaries(params);
    let k = kron_2x2(&u1, &u0);
    let u = basis_gate * k * prefix_op;
    let c = weyl_coordinates(&u);
    let init_res = weyl_res_both_branches(&c, target_weyl);
    if init_res <= POLISH_FLOOR {
        return init_res;
    }

    // Select branch target (direct or ρ-reflected) — fixed throughout polish
    let refl = [1.0 - target_weyl[0], target_weyl[1], -target_weyl[2]];
    let direct = linf_3(&[c[0] - target_weyl[0], c[1] - target_weyl[1], c[2] - target_weyl[2]]);
    let reflected = linf_3(&[c[0] - refl[0], c[1] - refl[1], c[2] - refl[2]]);
    let branch = if reflected < direct { refl } else { *target_weyl };

    // Precompute for the sensitivity chain
    let basis_t = basis_gate.transpose();

    let mut x = *params;
    let mut best_x = *params;
    let mut best_res = init_res;
    let mut damping = 1e-6;
    let mut stale = 0u32;

    for _ in 0..max_iters {
        if best_res <= POLISH_FLOOR || stale >= 4 {
            break;
        }

        // Compute residual + analytic Jacobian in one pass (1 eigendecomp w/ eigvecs)
        let (r0, jac) = weyl_residual_and_jacobian_analytic(
            &x, prefix_op, basis_gate, &basis_t, &branch,
        );
        let r0_norm = linf_3(&r0);

        // LM step: Jᵀ(JJᵀ + λI)⁻¹(-r)
        let gram = gram_3x3(&jac, damping);
        if let Some(sol) = solve_3x3(&gram, &[-r0[0], -r0[1], -r0[2]]) {
            let delta = jt_times_v(&jac, &sol);
            let mut x_new = x;
            for i in 0..8 {
                x_new[i] += delta[i];
            }
            if x_new.iter().all(|v| v.is_finite()) {
                // Trial evaluation (eigenvalues only, no eigenvectors)
                let r_new = weyl_residual_vec(&x_new, prefix_op, basis_gate, &branch);
                let new_norm = linf_3(&r_new);
                if new_norm < r0_norm {
                    x = x_new;
                    damping = (damping * 0.5_f64).max(1e-12);
                    if new_norm < best_res {
                        best_x = x;
                        best_res = new_norm;
                        stale = 0;
                    } else {
                        stale += 1;
                    }
                } else {
                    damping = (damping * 4.0).min(1e6);
                    stale += 1;
                }
            } else {
                stale += 1;
            }
        } else {
            stale += 1;
        }
    }

    *params = best_x;
    best_res
}

/// Analytic Weyl residual + Jacobian via eigenvalue perturbation theory.
///
/// The key insight: since u0, u1 ∈ SU(2), det(kron(u1,u0)) = 1 always,
/// so the SU(4) phase α = det(U)^{-1/4} is constant w.r.t. params.
/// This eliminates the det-projection derivative entirely.
///
/// For eigenvalue perturbation: dλ_i = v_i^H · dP · v_i, where v_i are
/// right eigenvectors of the product P = U_su4 · Ũ. The sensitivity of
/// each eigenvalue to K = kron(u1,u0) factors into a 4×4 matrix Λ_i that
/// reuses the existing contract_kron + quat_sensitivity machinery.
fn weyl_residual_and_jacobian_analytic(
    x: &[f64; 8],
    prefix_op: &Mat4,
    basis_gate: &Mat4,
    basis_t: &Mat4,
    branch_target: &[f64; 3],
) -> ([f64; 3], [[f64; 8]; 3]) {
    // Forward pass: params → U → V (U_su4) → P → eigendecomp
    let (u0, u1) = params_to_unitaries(x);
    let kk = kron_2x2(&u1, &u0);
    let u = basis_gate * kk * prefix_op;

    // SU(4) projection (α is constant w.r.t. params, but needed for V and Jacobian)
    let det_u = u.determinant();
    let quarter_phase = -det_u.arg() * 0.25;
    let alpha = C64::new(quarter_phase.cos(), quarter_phase.sin());
    let v = u * alpha;

    // Product P = V · Ũ where Ũ = S · V^T · S, S = σ_y⊗σ_y.
    // S is a signed permutation: rows/cols (0↔3, 1↔2) with signs (-,+,+,-).
    // (S·M·S)[i,j] = sign[i]·sign[j]·M[π(i),π(j)] where π: 0↔3, 1↔2.
    // For M = V^T: (S·V^T·S)[i,j] = sign[i]·sign[j]·V[π(j),π(i)].
    let v_tilde = sysy_transpose_sysy(&v);
    let p = v * v_tilde;

    // Eigendecomposition with eigenvectors (one call replaces 9 eigenvalue-only calls)
    let (eigvals, raw_phases, vecs) = eigendecomp_4x4(&p);

    // Sort descending + record permutation
    let mut perm = [0usize, 1, 2, 3];
    perm.sort_by(|&a, &b| {
        raw_phases[b]
            .partial_cmp(&raw_phases[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut s_phases = [0.0f64; 4];
    for i in 0..4 {
        s_phases[i] = raw_phases[perm[i]] / 2.0;
    }

    // Alcove normalization
    let sum: f64 = s_phases.iter().sum();
    let n = sum.round() as usize;
    for item in s_phases.iter_mut().take(n.min(4)) {
        *item -= 1.0;
    }
    let shift = n % 4;
    let rolled = [
        s_phases[shift % 4],
        s_phases[(1 + shift) % 4],
        s_phases[(2 + shift) % 4],
        s_phases[(3 + shift) % 4],
    ];

    // Weyl coordinates
    let c1_raw = rolled[0] + rolled[1];
    let c2_raw = rolled[0] + rolled[2];
    let c3_raw = rolled[1] + rolled[2];
    let folded = c3_raw < 0.0;
    let c = [
        if folded { 1.0 - c1_raw } else { c1_raw },
        c2_raw.max(0.0),
        if folded { -c3_raw } else { c3_raw.max(0.0) },
    ];

    // Residual
    let r = [
        c[0] - branch_target[0],
        c[1] - branch_target[1],
        c[2] - branch_target[2],
    ];

    // --- Analytic Jacobian via eigenvalue perturbation theory ---
    //
    // For each eigenpair (λ_i, v_i) of P:
    //   dλ_i = v_i^H · dP · v_i
    //
    // where dP = dV · Ũ + V · S · dV^T · S and dV = B · dK · P_op · α
    // (α constant since K ∈ SU(4)).
    //
    // This gives: dλ_i = α · Σ_{m,n} Λ_i[m,n] · dK[m,n]
    // where Λ_i is a 4×4 sensitivity matrix that contracts with kron derivatives
    // via the same machinery as the Makhlin Jacobian.

    // Precompute per-eigenvector quantities
    let basis_h = basis_gate.adjoint();
    let v_h = v.adjoint();

    // Quaternion normalization (same as Makhlin Jacobian)
    let norm0 = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3])
        .sqrt()
        .max(1e-12);
    let norm1 = (x[4] * x[4] + x[5] * x[5] + x[6] * x[6] + x[7] * x[7])
        .sqrt()
        .max(1e-12);
    let p0 = [x[0] / norm0, x[1] / norm0, x[2] / norm0, x[3] / norm0];
    let p1 = [x[4] / norm1, x[5] / norm1, x[6] / norm1, x[7] / norm1];

    let mut jac = [[0.0f64; 8]; 3];

    for ei in 0..4 {
        let vi = &vecs[ei];

        // w_i = Ũ · v_i  (for term 1: v_i^H · dV · Ũ · v_i)
        let mut wi = [C0; 4];
        for j in 0..4 {
            for l in 0..4 {
                wi[j] += v_tilde[(j, l)] * vi[l];
            }
        }

        // f1_i = P_op · w_i  (propagate through U = B · K · P_op)
        let mut f1i = [C0; 4];
        for j in 0..4 {
            for l in 0..4 {
                f1i[j] += prefix_op[(j, l)] * wi[l];
            }
        }

        // b_i = B^H · v_i  (left factor from v_i^H · B)
        let mut bi = [C0; 4];
        for j in 0..4 {
            for l in 0..4 {
                bi[j] += basis_h[(j, l)] * vi[l];
            }
        }

        // For term 2: v_i^H · V · S · dV^T · S · v_i
        // q_i = V^H · v_i
        let mut qi = [C0; 4];
        for j in 0..4 {
            for l in 0..4 {
                qi[j] += v_h[(j, l)] * vi[l];
            }
        }

        // y_i = S · v_i  where S = σ_y⊗σ_y is a signed permutation:
        //   S = [[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]]
        let yi = [-vi[3], vi[2], vi[1], -vi[0]];

        // r_i = S · q_i  (same signed permutation)
        let ri = [-qi[3], qi[2], qi[1], -qi[0]];

        // a_i = P_op · conj(r_i)  (for second outer product factor)
        let mut ai = [C0; 4];
        for j in 0..4 {
            for l in 0..4 {
                ai[j] += prefix_op[(j, l)] * ri[l].conj();
            }
        }

        // d_i = B^T · y_i  (transpose, NOT adjoint)
        let mut di = [C0; 4];
        for j in 0..4 {
            for l in 0..4 {
                di[j] += basis_t[(j, l)] * yi[l];
            }
        }

        // Build 4×4 sensitivity matrix Λ_i[m,n] = α · (conj(b_i[m]) · f1_i[n] + d_i[m] · a_i[n])
        let mut lambda_i = Mat4::zeros();
        for m in 0..4 {
            for nn in 0..4 {
                lambda_i[(m, nn)] = alpha * (bi[m].conj() * f1i[nn] + di[m] * ai[nn]);
            }
        }

        // Contract with kron derivatives (same as Makhlin Jacobian)
        let g0 = contract_kron_u0(&lambda_i, &u1);
        let g1 = contract_kron_u1(&lambda_i, &u0);

        // Quaternion sensitivity
        let (e0, s0_qs) = quat_sensitivity(&g0, &p0);
        let (e1, s1_qs) = quat_sensitivity(&g1, &p1);

        // d(phase_i)/dp_k = Im(conj(λ_i) · dλ_i) / π
        // dλ_i for u0 params: (e0[j] - s0 * p0[j]) * 2/norm0
        let inv_norm0 = 1.0 / norm0;
        let inv_norm1 = 1.0 / norm1;
        // We loop over ORIGINAL eigenvalue index ei. vecs[ei] is the
        // eigenvector for eigvals[ei]. The sort permutation is applied later
        // when mapping to sorted phase positions.
        let conj_eigval = eigvals[ei].conj();

        let mut d_phase_u0 = [0.0f64; 4];
        for j in 0..4 {
            let dlambda = (e0[j] - s0_qs * p0[j]) * (2.0 * inv_norm0);
            d_phase_u0[j] = (conj_eigval * dlambda).im / PI;
        }

        let mut d_phase_u1 = [0.0f64; 4];
        for j in 0..4 {
            let dlambda = (e1[j] - s1_qs * p1[j]) * (2.0 * inv_norm1);
            d_phase_u1[j] = (conj_eigval * dlambda).im / PI;
        }

        // Accumulate into Jacobian:
        // d(sorted_phase_i)/dp = d(raw_phase_{perm[i]})/dp / 2
        // We're computing d(raw_phase_{ei})/dp here.
        // Find which sorted position this original eigenvalue maps to.
        let sorted_pos = perm.iter().position(|&p| p == ei).unwrap();

        // d(s_i) = d(raw_phase) / 2  (the halving)
        // Alcove: d(s_i) unchanged (n = const, shift = const locally)
        // After circular shift: rolled[shifted_pos]
        let rolled_pos = (sorted_pos + 4 - shift) % 4;

        // Skip eigenvalue 3 (only first 3 contribute to Weyl via M_WEYL)
        // M_WEYL: c1 = rolled[0]+rolled[1], c2 = rolled[0]+rolled[2], c3 = rolled[1]+rolled[2]
        let half = 0.5; // phases are halved
        let dc1_contrib = if rolled_pos == 0 || rolled_pos == 1 {
            half
        } else {
            0.0
        };
        let dc2_contrib = if rolled_pos == 0 || rolled_pos == 2 {
            half
        } else {
            0.0
        };
        let dc3_contrib = if rolled_pos == 1 || rolled_pos == 2 {
            half
        } else {
            0.0
        };

        // Canonical fold derivative
        let (dc1_f, dc3_f) = if folded {
            (-dc1_contrib, -dc3_contrib)
        } else {
            (dc1_contrib, dc3_contrib)
        };
        // c2 = max(c2_raw, 0): derivative is 0 if c2_raw < 0
        let dc2_f = if c2_raw >= 0.0 { dc2_contrib } else { 0.0 };

        for j in 0..4 {
            jac[0][j] += dc1_f * d_phase_u0[j];
            jac[1][j] += dc2_f * d_phase_u0[j];
            jac[2][j] += dc3_f * d_phase_u0[j];
            jac[0][j + 4] += dc1_f * d_phase_u1[j];
            jac[1][j + 4] += dc2_f * d_phase_u1[j];
            jac[2][j + 4] += dc3_f * d_phase_u1[j];
        }
    }

    (r, jac)
}

/// Compute Weyl coordinates from params (shared by residual and check).
fn params_to_weyl(
    params: &[f64; 8],
    prefix_op: &Mat4,
    basis_gate: &Mat4,
) -> [f64; 3] {
    let (u0, u1) = params_to_unitaries(params);
    let k = kron_2x2(&u1, &u0);
    let u = basis_gate * k * prefix_op;
    weyl_coordinates(&u)
}

/// Signed Weyl residual vector against a specific branch target.
fn weyl_residual_vec(
    params: &[f64; 8],
    prefix_op: &Mat4,
    basis_gate: &Mat4,
    target: &[f64; 3],
) -> [f64; 3] {
    let c = params_to_weyl(params, prefix_op, basis_gate);
    [c[0] - target[0], c[1] - target[1], c[2] - target[2]]
}

/// Weyl residual over both direct and ρ-reflected branches.
fn weyl_check(
    params: &[f64; 8],
    prefix_op: &Mat4,
    basis_gate: &Mat4,
    target_weyl: &[f64; 3],
) -> f64 {
    let c = params_to_weyl(params, prefix_op, basis_gate);
    weyl_res_both_branches(&c, target_weyl)
}

// ---------------------------------------------------------------------------
// Fused residual functions (precomputed magic-basis transforms)
// ---------------------------------------------------------------------------

/// Absorb MAGIC into prefix/basis and precompute determinant phase.
///
/// Returns (prefix_magic, magic_basis, target_packed) where
/// target_packed = [Re(G1), Im(G1), Re(G2), Re(det_phase), Im(det_phase)].
fn precompute_makhlin_args(
    prefix_op: &Mat4,
    basis_gate: &Mat4,
    target_makhlin: &[f64; 3],
) -> (Mat4, Mat4, [f64; 5]) {
    let magic_basis = magic_dag() * basis_gate;
    let prefix_magic = prefix_op * magic();
    let det_um = (magic_basis * prefix_magic).determinant();
    let det_norm = det_um.norm().max(1e-30);
    let det_phase = det_um / det_norm;
    (
        prefix_magic,
        magic_basis,
        [
            target_makhlin[0],
            target_makhlin[1],
            target_makhlin[2],
            det_phase.re,
            det_phase.im,
        ],
    )
}

/// Fused Makhlin residual: target − computed invariants.
///
/// Uses precomputed magic-basis transforms to skip 2 matmuls + 1 det.
fn makhlin_residual_fused(
    x: &[f64; 8],
    prefix_magic: &Mat4,
    magic_basis: &Mat4,
    target_packed: &[f64; 5],
) -> [f64; 3] {
    let det_phase = C64::new(target_packed[3], target_packed[4]);
    let (u0, u1) = params_to_unitaries(x);
    let k = kron_2x2(&u1, &u0);
    let um = magic_basis * k * prefix_magic;
    let m = um.transpose() * um;
    let t1 = m.trace();
    let t1s = t1 * t1;
    let tr_m2 = trace_of_square(&m);
    let g1 = t1s / (det_phase * 16.0);
    let g2 = (t1s - tr_m2) / (det_phase * 4.0);
    [
        target_packed[0] - g1.re,
        target_packed[1] - g1.im,
        target_packed[2] - g2.re,
    ]
}

/// Fused Makhlin residual AND analytic Jacobian in a single forward pass.
///
/// Shares the forward computation (params → kron → Um → M → traces) between
/// the residual and Jacobian, eliminating redundant 4×4 matmuls per call.
fn makhlin_residual_and_jacobian(
    x: &[f64; 8],
    prefix_magic: &Mat4,
    magic_basis: &Mat4,
    target_packed: &[f64; 5],
) -> ([f64; 3], [[f64; 8]; 3]) {
    let det_phase = C64::new(target_packed[3], target_packed[4]);

    // --- shared forward pass (used by both residual and Jacobian) ---
    let (u0, u1) = params_to_unitaries(x);
    let k = kron_2x2(&u1, &u0);
    let um = magic_basis * k * prefix_magic;
    let m = um.transpose() * um;
    let t1 = m.trace();

    // --- residual ---
    let t1s = t1 * t1;
    let tr_m2 = trace_of_square(&m);
    let g1 = t1s / (det_phase * 16.0);
    let g2 = (t1s - tr_m2) / (det_phase * 4.0);
    let r = [
        target_packed[0] - g1.re,
        target_packed[1] - g1.im,
        target_packed[2] - g2.re,
    ];

    // --- Jacobian (chain rule from shared intermediates) ---
    let mb_t = magic_basis.transpose();
    let pm_t = prefix_magic.transpose();
    let f1 = mb_t * um * pm_t;
    let um_m = um * m;
    let f2 = mb_t * um_m * pm_t;

    let g01 = contract_kron_u0(&f1, &u1);
    let g11 = contract_kron_u1(&f1, &u0);
    let g02 = contract_kron_u0(&f2, &u1);
    let g12 = contract_kron_u1(&f2, &u0);

    let norm0 = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3])
        .sqrt()
        .max(1e-12);
    let norm1 = (x[4] * x[4] + x[5] * x[5] + x[6] * x[6] + x[7] * x[7])
        .sqrt()
        .max(1e-12);
    let p0 = [x[0] / norm0, x[1] / norm0, x[2] / norm0, x[3] / norm0];
    let p1 = [x[4] / norm1, x[5] / norm1, x[6] / norm1, x[7] / norm1];

    let (e01, s01) = quat_sensitivity(&g01, &p0);
    let (e11, s11) = quat_sensitivity(&g11, &p1);
    let (e02, s02) = quat_sensitivity(&g02, &p0);
    let (e12, s12) = quat_sensitivity(&g12, &p1);

    let inv_16dp = C1 / (det_phase * 16.0);
    let inv_4dp = C1 / (det_phase * 4.0);
    let two_t1 = t1 * 2.0;

    let mut jac = [[0.0f64; 8]; 3];

    let inv_norm0 = 1.0 / norm0;
    for j in 0..4 {
        // complex×real: 2 FLOPs each, not complex×complex (6 FLOPs)
        let dt1_contrib = (e01[j] - s01 * p0[j]) * (2.0 * inv_norm0);
        let dtrm2_contrib = (e02[j] - s02 * p0[j]) * (4.0 * inv_norm0);
        let dg1 = two_t1 * dt1_contrib * inv_16dp;
        let dg2 = (two_t1 * dt1_contrib - dtrm2_contrib) * inv_4dp;
        jac[0][j] = -dg1.re;
        jac[1][j] = -dg1.im;
        jac[2][j] = -dg2.re;
    }

    let inv_norm1 = 1.0 / norm1;
    for j in 0..4 {
        let dt1_contrib = (e11[j] - s11 * p1[j]) * (2.0 * inv_norm1);
        let dtrm2_contrib = (e12[j] - s12 * p1[j]) * (4.0 * inv_norm1);
        let dg1 = two_t1 * dt1_contrib * inv_16dp;
        let dg2 = (two_t1 * dt1_contrib - dtrm2_contrib) * inv_4dp;
        jac[0][j + 4] = -dg1.re;
        jac[1][j + 4] = -dg1.im;
        jac[2][j + 4] = -dg2.re;
    }

    (r, jac)
}

// ---------------------------------------------------------------------------
// Analytic Jacobian of the fused Makhlin residual
// ---------------------------------------------------------------------------

/// Contract a 4×4 sensitivity matrix F with U_other to produce a 2×2 matrix.
///
/// For U0 derivatives (contract with U1):
///   G0[b,d] = Σ_{a,c} F[2a+b, 2c+d] · U1[a,c]
#[inline]
fn contract_kron_u0(f: &Mat4, u1: &Mat2) -> Mat2 {
    let mut g = Mat2::zeros();
    for b in 0..2 {
        for d in 0..2 {
            let mut s = C0;
            for a in 0..2 {
                for c in 0..2 {
                    s += f[(2 * a + b, 2 * c + d)] * u1[(a, c)];
                }
            }
            g[(b, d)] = s;
        }
    }
    g
}

/// For U1 derivatives (contract with U0):
///   G1[a,c] = Σ_{b,d} F[2a+b, 2c+d] · U0[b,d]
#[inline]
fn contract_kron_u1(f: &Mat4, u0: &Mat2) -> Mat2 {
    let mut g = Mat2::zeros();
    for a in 0..2 {
        for c in 0..2 {
            let mut s = C0;
            for b in 0..2 {
                for d in 0..2 {
                    s += f[(2 * a + b, 2 * c + d)] * u0[(b, d)];
                }
            }
            g[(a, c)] = s;
        }
    }
    g
}

/// From a 2×2 contracted sensitivity matrix G, compute the quaternion
/// derivative contributions [α, β, γ, δ] based on the SU(2) structure:
///   U = [[w+iz, x+iy], [-x+iy, w-iz]]
/// Returns (e[4], S) where e[j] = contribution for ∂/∂q_j before normalization,
/// and S = Σ p_j · e[j].
#[inline]
fn quat_sensitivity(g: &Mat2, p: &[f64; 4]) -> ([C64; 4], C64) {
    let alpha = g[(0, 0)] + g[(1, 1)]; // coefficient of dw
    let beta = g[(0, 1)] - g[(1, 0)]; // coefficient of dx
    let gamma = CI * (g[(0, 1)] + g[(1, 0)]); // coefficient of dy
    let delta = CI * (g[(0, 0)] - g[(1, 1)]); // coefficient of dz
    let e = [alpha, beta, gamma, delta];
    // p[j] is real - use complex×real (2 FLOPs) not complex×complex (6 FLOPs)
    let s = alpha * p[0] + beta * p[1] + gamma * p[2] + delta * p[3];
    (e, s)
}

// ---------------------------------------------------------------------------
// Linear algebra helpers
// ---------------------------------------------------------------------------

/// tr(M²) without forming the full product matrix.
///
///// tr(M²) = Σᵢⱼ M[i,j]·M[j,i]: 16 complex multiply-adds vs 64 for a matmul.
#[inline]
fn trace_of_square(m: &Mat4) -> C64 {
    let mut s = C0;
    for i in 0..4 {
        for j in 0..4 {
            s += m[(i, j)] * m[(j, i)];
        }
    }
    s
}

/// Compute S · M^T · S where S = σ_y⊗σ_y is a signed permutation.
///
/// S permutes rows/cols via π: (0↔3, 1↔2) with signs (-,+,+,-).
/// Result[i,j] = sign[i] · sign[j] · M[π(j), π(i)].
#[inline]
fn sysy_transpose_sysy(m: &Mat4) -> Mat4 {
    // sign[0]=-1, sign[1]=+1, sign[2]=+1, sign[3]=-1
    // π(0)=3, π(1)=2, π(2)=1, π(3)=0
    // Result[i,j] = sign[i]*sign[j] * M^T[π(i),π(j)] = sign[i]*sign[j] * M[π(j),π(i)]
    Mat4::new(
        // row 0 (sign=-1): M[π(j),π(0)] * sign[0]*sign[j]
        //   j=0: (-1)(-1)*M[3,3] = M[3,3]
        //   j=1: (-1)(+1)*M[2,3] = -M[2,3]
        //   j=2: (-1)(+1)*M[1,3] = -M[1,3]
        //   j=3: (-1)(-1)*M[0,3] = M[0,3]
        m[(3, 3)], -m[(2, 3)], -m[(1, 3)], m[(0, 3)],
        // row 1 (sign=+1):
        //   j=0: (+1)(-1)*M[3,2] = -M[3,2]
        //   j=1: (+1)(+1)*M[2,2] = M[2,2]
        //   j=2: (+1)(+1)*M[1,2] = M[1,2]
        //   j=3: (+1)(-1)*M[0,2] = -M[0,2]
        -m[(3, 2)], m[(2, 2)], m[(1, 2)], -m[(0, 2)],
        // row 2 (sign=+1):
        -m[(3, 1)], m[(2, 1)], m[(1, 1)], -m[(0, 1)],
        // row 3 (sign=-1):
        m[(3, 0)], -m[(2, 0)], -m[(1, 0)], m[(0, 0)],
    )
}

/// L∞ norm of a 3-vector.
#[inline]
fn linf_3(r: &[f64; 3]) -> f64 {
    r[0].abs().max(r[1].abs()).max(r[2].abs())
}

/// Gram matrix J·Jᵀ + λI for a 3×8 Jacobian.
#[inline]
fn gram_3x3(jac: &[[f64; 8]; 3], damping: f64) -> [[f64; 3]; 3] {
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

/// 3×3 linear solve via Cholesky decomposition.
///
/// The Gram matrix J·Jᵀ + λI is symmetric positive definite, so Cholesky
/// is both fast and numerically stable.
#[inline]
fn solve_3x3(a: &[[f64; 3]; 3], b: &[f64; 3]) -> Option<[f64; 3]> {
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
fn jt_times_v(jac: &[[f64; 8]; 3], v: &[f64; 3]) -> [f64; 8] {
    let mut out = [0.0f64; 8];
    for k in 0..8 {
        out[k] = jac[0][k] * v[0] + jac[1][k] * v[1] + jac[2][k] * v[2];
    }
    out
}
