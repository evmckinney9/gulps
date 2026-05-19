//! Makhlin and Weyl residuals with their analytic Jacobians.

use crate::gate_primitives::{
    kron_2x2, magic, magic_dag, params_to_unitaries, weyl_coordinates, weyl_res_both_branches,
};
use crate::linalg::{
    eigendecomp_4x4, mat4_vec, normalize_quat, sysy_transpose_sysy, trace_of_square,
};
use crate::{Mat2, Mat4, C0, C1, C64, CI};
use std::f64::consts::PI;

/// Weyl residual and its analytic Jacobian against `branch_target`.
pub(crate) fn weyl_residual_and_jacobian_analytic(
    x: &[f64; 8],
    prefix_op: &Mat4,
    basis_gate: &Mat4,
    basis_t: &Mat4,
    branch_target: &[f64; 3],
) -> ([f64; 3], [[f64; 8]; 3]) {
    let (u0, u1) = params_to_unitaries(x);
    let kk = kron_2x2(&u1, &u0);
    let u = basis_gate * kk * prefix_op;

    let det_u = u.determinant();
    let quarter_phase = -det_u.arg() * 0.25;
    let alpha = C64::new(quarter_phase.cos(), quarter_phase.sin());
    let v = u * alpha;

    let v_tilde = sysy_transpose_sysy(&v);
    let p = v * v_tilde;

    let (eigvals, raw_phases, vecs) = eigendecomp_4x4(&p);

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

    let c1_raw = rolled[0] + rolled[1];
    let c2_raw = rolled[0] + rolled[2];
    let c3_raw = rolled[1] + rolled[2];
    let folded = c3_raw < 0.0;
    let c = [
        if folded { 1.0 - c1_raw } else { c1_raw },
        c2_raw.max(0.0),
        if folded { -c3_raw } else { c3_raw.max(0.0) },
    ];

    let r = [
        c[0] - branch_target[0],
        c[1] - branch_target[1],
        c[2] - branch_target[2],
    ];

    let basis_h = basis_gate.adjoint();
    let v_h = v.adjoint();

    let (p0, norm0) = normalize_quat(&x[0..4]);
    let (p1, norm1) = normalize_quat(&x[4..8]);

    let mut jac = [[0.0f64; 8]; 3];

    for ei in 0..4 {
        let vi = &vecs[ei];

        let wi = mat4_vec(&v_tilde, vi);
        let f1i = mat4_vec(prefix_op, &wi);
        let bi = mat4_vec(&basis_h, vi);

        let qi = mat4_vec(&v_h, vi);

        let yi = [-vi[3], vi[2], vi[1], -vi[0]];
        let ri = [-qi[3], qi[2], qi[1], -qi[0]];
        let ri_conj = [ri[0].conj(), ri[1].conj(), ri[2].conj(), ri[3].conj()];
        let ai = mat4_vec(prefix_op, &ri_conj);
        let di = mat4_vec(basis_t, &yi); // basis_t is the transpose, not the adjoint

        let mut lambda_i = Mat4::zeros();
        for m in 0..4 {
            for nn in 0..4 {
                lambda_i[(m, nn)] = alpha * (bi[m].conj() * f1i[nn] + di[m] * ai[nn]);
            }
        }

        let g0 = contract_kron_u0(&lambda_i, &u1);
        let g1 = contract_kron_u1(&lambda_i, &u0);

        let (e0, s0_qs) = quat_sensitivity(&g0, &p0);
        let (e1, s1_qs) = quat_sensitivity(&g1, &p1);

        let inv_norm0 = 1.0 / norm0;
        let inv_norm1 = 1.0 / norm1;
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

        // ei is the unsorted eigenvalue; follow it through the sort and shift.
        let sorted_pos = perm.iter().position(|&p| p == ei).unwrap();
        let rolled_pos = (sorted_pos + 4 - shift) % 4;

        // rolled[3] feeds no Weyl coord, so its contribution is zero.
        let half = 0.5;
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

        let (dc1_f, dc3_f) = if folded {
            (-dc1_contrib, -dc3_contrib)
        } else {
            (dc1_contrib, dc3_contrib)
        };
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
fn params_to_weyl(params: &[f64; 8], prefix_op: &Mat4, basis_gate: &Mat4) -> [f64; 3] {
    let (u0, u1) = params_to_unitaries(params);
    let k = kron_2x2(&u1, &u0);
    let u = basis_gate * k * prefix_op;
    weyl_coordinates(&u)
}

/// Signed Weyl residual vector against a specific branch target.
pub(crate) fn weyl_residual_vec(
    params: &[f64; 8],
    prefix_op: &Mat4,
    basis_gate: &Mat4,
    target: &[f64; 3],
) -> [f64; 3] {
    let c = params_to_weyl(params, prefix_op, basis_gate);
    [c[0] - target[0], c[1] - target[1], c[2] - target[2]]
}

/// Weyl residual over both direct and ρ-reflected branches.
pub(crate) fn weyl_check(
    params: &[f64; 8],
    prefix_op: &Mat4,
    basis_gate: &Mat4,
    target_weyl: &[f64; 3],
) -> f64 {
    let c = params_to_weyl(params, prefix_op, basis_gate);
    weyl_res_both_branches(&c, target_weyl)
}

/// Magic-basis form of a segment: the precomputed prefix and basis,
/// the Makhlin target `[Re G1, Im G1, Re G2]`, and the unit-norm
/// det-phase pulled out of the product.
pub(crate) struct MakhlinForm {
    pub prefix_magic: Mat4,
    pub basis_magic: Mat4,
    pub target: [f64; 3],
    pub det_phase: C64,
}

pub(crate) fn precompute_makhlin(
    prefix_op: &Mat4,
    basis_gate: &Mat4,
    target_makhlin: &[f64; 3],
) -> MakhlinForm {
    let basis_magic = magic_dag() * basis_gate;
    let prefix_magic = prefix_op * magic();
    let det_um = (basis_magic * prefix_magic).determinant();
    let det_phase = det_um / det_um.norm().max(1e-30);
    MakhlinForm {
        prefix_magic,
        basis_magic,
        target: *target_makhlin,
        det_phase,
    }
}

/// Makhlin residual using the precomputed form.
pub(crate) fn makhlin_residual_fused(x: &[f64; 8], mf: &MakhlinForm) -> [f64; 3] {
    let (u0, u1) = params_to_unitaries(x);
    let k = kron_2x2(&u1, &u0);
    let um = mf.basis_magic * k * mf.prefix_magic;
    let m = um.transpose() * um;
    let t1 = m.trace();
    let t1s = t1 * t1;
    let tr_m2 = trace_of_square(&m);
    let g1 = t1s / (mf.det_phase * 16.0);
    let g2 = (t1s - tr_m2) / (mf.det_phase * 4.0);
    [mf.target[0] - g1.re, mf.target[1] - g1.im, mf.target[2] - g2.re]
}

/// Makhlin residual and its analytic Jacobian, sharing one forward pass.
pub(crate) fn makhlin_residual_and_jacobian(
    x: &[f64; 8],
    mf: &MakhlinForm,
) -> ([f64; 3], [[f64; 8]; 3]) {
    let (u0, u1) = params_to_unitaries(x);
    let k = kron_2x2(&u1, &u0);
    let um = mf.basis_magic * k * mf.prefix_magic;
    let m = um.transpose() * um;
    let t1 = m.trace();

    let t1s = t1 * t1;
    let tr_m2 = trace_of_square(&m);
    let g1 = t1s / (mf.det_phase * 16.0);
    let g2 = (t1s - tr_m2) / (mf.det_phase * 4.0);
    let r = [mf.target[0] - g1.re, mf.target[1] - g1.im, mf.target[2] - g2.re];

    let mb_t = mf.basis_magic.transpose();
    let pm_t = mf.prefix_magic.transpose();
    let f1 = mb_t * um * pm_t;
    let um_m = um * m;
    let f2 = mb_t * um_m * pm_t;

    let g01 = contract_kron_u0(&f1, &u1);
    let g11 = contract_kron_u1(&f1, &u0);
    let g02 = contract_kron_u0(&f2, &u1);
    let g12 = contract_kron_u1(&f2, &u0);

    let (p0, norm0) = normalize_quat(&x[0..4]);
    let (p1, norm1) = normalize_quat(&x[4..8]);

    let (e01, s01) = quat_sensitivity(&g01, &p0);
    let (e11, s11) = quat_sensitivity(&g11, &p1);
    let (e02, s02) = quat_sensitivity(&g02, &p0);
    let (e12, s12) = quat_sensitivity(&g12, &p1);

    let inv_16dp = C1 / (mf.det_phase * 16.0);
    let inv_4dp = C1 / (mf.det_phase * 4.0);
    let two_t1 = t1 * 2.0;

    let mut jac = [[0.0f64; 8]; 3];

    let inv_norm0 = 1.0 / norm0;
    for j in 0..4 {
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

/// Contract the 4×4 sensitivity with u1 to get the 2×2 u0-derivative block.
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

/// Contract the 4×4 sensitivity with u0 to get the 2×2 u1-derivative block.
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

/// Quaternion derivative of the SU(2) parameterization; returns the (dw, dx, dy, dz)
/// coefficients and their p-weighted sum.
#[inline]
fn quat_sensitivity(g: &Mat2, p: &[f64; 4]) -> ([C64; 4], C64) {
    let alpha = g[(0, 0)] + g[(1, 1)];
    let beta = g[(0, 1)] - g[(1, 0)];
    let gamma = CI * (g[(0, 1)] + g[(1, 0)]);
    let delta = CI * (g[(0, 0)] - g[(1, 1)]);
    let e = [alpha, beta, gamma, delta];
    let s = alpha * p[0] + beta * p[1] + gamma * p[2] + delta * p[3];
    (e, s)
}
