//! Two-stage segment solver: Makhlin-residual Gauss-Newton, then a Weyl
//! Levenberg-Marquardt polish.

use crate::gate_primitives::{
    kron_2x2, makhlin_invariants, params_to_unitaries, weyl_coordinates, weyl_res_both_branches,
};
use crate::jacobian::{
    makhlin_residual_and_jacobian, makhlin_residual_fused, precompute_makhlin, weyl_check,
    weyl_residual_and_jacobian_analytic, weyl_residual_vec, MakhlinForm,
};
use crate::linalg::{gram_3x3, jt_times_v, linf_3, solve_3x3};
use crate::{Mat2, Mat4};
use rand::prelude::*;
use rand::rngs::SmallRng;

const MAXITER: usize = 512;
const MAX_RESTARTS: usize = 512;
const POLISH_FLOOR: f64 = 1e-12;
const POLISH_BUDGET: usize = 128;
const LAM_K: f64 = 1e-4; // self-scaling Levenberg coefficient: damp = 1e-12 + LAM_K*|r|^2
const RNG_BASE: u64 = 0x9E37_79B9_7F4A_7C15;
/// One segment of the synthesis problem.
pub struct Segment<'a> {
    pub prefix: &'a Mat4,
    pub basis: &'a Mat4,
    pub target: &'a Mat4,
}

/// Solver state for a single segment.
struct Solver<'a> {
    seg: Segment<'a>,
    target_weyl: [f64; 3],
    mf: MakhlinForm,
    makhlin_tol: f64,
    weyl_tol: f64,
}

/// Solve one segment; returns `(u0, u1, weyl_res, makhlin_res)`.
pub fn solve(
    seg: Segment<'_>,
    makhlin_tol: f64,
    weyl_tol: f64,
    identity_warmstart: bool,
) -> (Mat2, Mat2, f64, f64) {
    let solver = Solver::new(seg, makhlin_tol, weyl_tol);
    let (mut params, makhlin_res) = solver.gn_restart_loop(identity_warmstart);
    let weyl_res = solver.weyl_polish(&mut params, POLISH_BUDGET);
    let (u0, u1) = params_to_unitaries(&params);
    (u0, u1, weyl_res, makhlin_res)
}

impl<'a> Solver<'a> {
    fn new(seg: Segment<'a>, makhlin_tol: f64, weyl_tol: f64) -> Self {
        let target_makhlin = makhlin_invariants(seg.target);
        let target_weyl = weyl_coordinates(seg.target);
        let mf = precompute_makhlin(seg.prefix, seg.basis, &target_makhlin);
        Self {
            seg,
            target_weyl,
            mf,
            makhlin_tol,
            weyl_tol,
        }
    }

    /// Best Makhlin solution across restarts; early-exits on Makhlin or Weyl convergence.
    fn gn_restart_loop(&self, identity_warmstart: bool) -> ([f64; 8], f64) {
        let mut best_params = [0.0f64; 8];
        let mut best_res = f64::INFINITY;

        for restart in 0..MAX_RESTARTS {
            let mut init = [0.0f64; 8];
            if identity_warmstart && restart == 0 {
                init[0] = 1.0;
                init[4] = 1.0;
            } else {
                let mut rng = SmallRng::seed_from_u64(RNG_BASE.wrapping_add(restart as u64));
                for v in &mut init {
                    *v = rng.random_range(-0.1..0.1);
                }
            }

            let (final_x, final_res) = self.gn_inner_loop(&init);

            if final_res < best_res && final_x.iter().all(|v| v.is_finite()) {
                best_params = final_x;
                best_res = final_res;

                if best_res <= self.makhlin_tol {
                    break;
                }
                let wres = weyl_check(
                    &best_params,
                    self.seg.prefix,
                    self.seg.basis,
                    &self.target_weyl,
                );
                if wres <= self.weyl_tol {
                    break;
                }
            }
        }

        (best_params, best_res)
    }

    /// One restart: Gauss-Newton until convergence or stagnation.
    fn gn_inner_loop(&self, init: &[f64; 8]) -> ([f64; 8], f64) {
        let mut x = *init;
        let mut best_x = *init;
        let mut best_norm = f64::INFINITY;
        let mut prev_norm = f64::INFINITY;

        for j in 0..MAXITER {
            let stagnated = (j >= 8 && prev_norm > 3e-1)
                || (j >= 16 && prev_norm > 5e-2)
                || (j >= 32 && prev_norm > 1e-3);
            if best_norm <= self.makhlin_tol || stagnated {
                break;
            }

            let (r, jac) = makhlin_residual_and_jacobian(&x, &self.mf);
            let curr_norm = linf_3(&r);

            if curr_norm < best_norm {
                best_x = x;
                best_norm = curr_norm;
            }

            let damp = 1e-12 + LAM_K * curr_norm * curr_norm;
            let gram = gram_3x3(&jac, damp);
            if let Some(sol) = solve_3x3(&gram, &[-r[0], -r[1], -r[2]]) {
                let mut delta = jt_times_v(&jac, &sol);
                let nrm = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
                if nrm > 0.6 {
                    let s = 0.6 / nrm;
                    for d in &mut delta {
                        *d *= s;
                    }
                }
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

        // score the last step (applied but not yet evaluated)
        let r_last = makhlin_residual_fused(&x, &self.mf);
        let last_norm = linf_3(&r_last);
        if last_norm < best_norm && x.iter().all(|v| v.is_finite()) {
            (x, last_norm)
        } else {
            (best_x, best_norm)
        }
    }

    /// Levenberg-Marquardt Weyl polish; stalls harmlessly at c1≈c2 (singular Jacobian).
    fn weyl_polish(&self, params: &mut [f64; 8], max_iters: usize) -> f64 {
        let (u0, u1) = params_to_unitaries(params);
        let k = kron_2x2(&u1, &u0);
        let u = self.seg.basis * k * self.seg.prefix;
        let c = weyl_coordinates(&u);
        let init_res = weyl_res_both_branches(&c, &self.target_weyl);
        if init_res <= POLISH_FLOOR {
            return init_res;
        }

        // pick the branch once and hold it for the whole polish
        let tw = self.target_weyl;
        let refl = [1.0 - tw[0], tw[1], -tw[2]];
        let direct = linf_3(&[c[0] - tw[0], c[1] - tw[1], c[2] - tw[2]]);
        let reflected = linf_3(&[c[0] - refl[0], c[1] - refl[1], c[2] - refl[2]]);
        let branch = if reflected < direct { refl } else { tw };

        let basis_t = self.seg.basis.transpose();

        let mut x = *params;
        let mut best_x = *params;
        let mut best_res = init_res;
        let mut damping = 1e-6;
        let mut stale = 0u32;

        for _ in 0..max_iters {
            if best_res <= POLISH_FLOOR || stale >= 4 {
                break;
            }

            let (r0, jac) = weyl_residual_and_jacobian_analytic(
                &x,
                self.seg.prefix,
                self.seg.basis,
                &basis_t,
                &branch,
            );
            let r0_norm = linf_3(&r0);

            let gram = gram_3x3(&jac, damping);
            if let Some(sol) = solve_3x3(&gram, &[-r0[0], -r0[1], -r0[2]]) {
                let delta = jt_times_v(&jac, &sol);
                let mut x_new = x;
                for i in 0..8 {
                    x_new[i] += delta[i];
                }
                if x_new.iter().all(|v| v.is_finite()) {
                    let r_new = weyl_residual_vec(&x_new, self.seg.prefix, self.seg.basis, &branch);
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
}
