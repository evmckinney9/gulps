"""Adaptive Makhlin/Weyl segment solver (Gauss-Newton + Levenberg-Marquardt).

Orchestration logic: restart loops, dispatch strategy, and the public
JaxLMSegmentSolver class.  All JAX math primitives live in jax_primitives.py.
"""

import time

import jax.numpy as jnp
import numpy as np
from jax import jacrev, jit, lax
from jax.random import PRNGKey, fold_in, uniform

from gulps.config import GulpsConfig
from gulps.synthesis.jax_primitives import (
    NUM_PARAMS,
    _get_jax_matrix,
    _kron_2x2,
    _makhlin_residual_fused,
    _params_to_unitaries,
    _precompute_makhlin_args,
    _weyl_residual,
    makhlin_invariants,
    weyl_coordinates,
)
from gulps.synthesis.segments_abc import SegmentSolution, SegmentSolver


# ---------------------------------------------------------------------------
# Restart loop factories
# ---------------------------------------------------------------------------
def _gn_step(residual_fn, x, prefix_op, basis_gate, target_inv, damping):
    """Single damped Gauss-Newton step -> (x_new, residual_norm)."""
    r = residual_fn(x, prefix_op, basis_gate, target_inv)
    J = jacrev(residual_fn, argnums=0)(x, prefix_op, basis_gate, target_inv)
    gram = J @ J.T + damping * jnp.eye(J.shape[0], dtype=jnp.float64)
    x_new = x + J.T @ jnp.linalg.solve(gram, -r)
    r_new = residual_fn(x_new, prefix_op, basis_gate, target_inv)
    return x_new, jnp.max(jnp.abs(r_new))


def _make_gn_restart_loop(
    residual_fn,
    *,
    maxiter: int,
    max_restarts: int,
    solver_tol: float,
    stagnation_iters: int = 32,
    stagnation_factor: float = 0.1,
):
    """Build a JIT'd random-restart GN solver with stagnation early-exit."""

    @jit
    def run_until_success(
        key,
        prefix_op,
        basis_gate,
        target_inv,
        tol,
        init_min=-jnp.pi / 2,
        init_max=jnp.pi / 2,
    ):
        def cond_fn(carry):
            i, _, _, best_res, done = carry
            return (i < max_restarts) & (~done)

        def body_fn(carry):
            i, key, best_params, best_res, done = carry
            init = uniform(
                fold_in(key, i), shape=(NUM_PARAMS,), minval=init_min, maxval=init_max
            )

            def gn_cond(inner):
                j, _, prev_norm, init_n = inner
                stagnated = (
                    (j >= stagnation_iters)
                    & (prev_norm > init_n * stagnation_factor)
                    & (prev_norm > 1e-3)
                )
                return (j < maxiter) & (prev_norm > solver_tol) & (~stagnated)

            def gn_body(inner):
                j, x, _, init_n = inner
                x_new, new_norm = _gn_step(
                    residual_fn, x, prefix_op, basis_gate, target_inv, 1e-14
                )
                finite = jnp.all(jnp.isfinite(x_new)) & jnp.isfinite(new_norm)
                return (
                    j + 1,
                    jnp.where(finite, x_new, x),
                    jnp.where(finite, new_norm, jnp.inf),
                    init_n,
                )

            r0 = residual_fn(init, prefix_op, basis_gate, target_inv)
            init_norm = jnp.max(jnp.abs(r0))
            _, final_x, final_res, _ = lax.while_loop(
                gn_cond, gn_body, (jnp.int32(0), init, init_norm, init_norm)
            )

            params_finite = jnp.all(jnp.isfinite(final_x))
            improved = (final_res < best_res) & params_finite
            safe_params = jnp.where(params_finite, final_x, init)

            return (
                i + 1,
                key,
                jnp.where(improved, safe_params, best_params),
                jnp.where(improved, final_res, best_res),
                done | (final_res <= tol),
            )

        _, _, best_params, best_res, _ = lax.while_loop(
            cond_fn,
            body_fn,
            (
                jnp.int32(0),
                key,
                jnp.zeros((NUM_PARAMS,), dtype=jnp.float64),
                jnp.array(jnp.inf, dtype=jnp.float64),
                jnp.array(False),
            ),
        )
        return best_params, best_res

    return run_until_success


def _make_lm_warmstart_loop(
    residual_fn,
    *,
    maxiter: int,
    max_restarts: int,
    solver_tol: float,
    perturb_scale: float,
):
    """Build a JIT'd warm-start LM solver with adaptive damping."""

    @jit
    def run_until_success(key, prefix_op, basis_gate, target_inv, tol, warm_start):
        def cond_fn(carry):
            i, _, _, best_res, done = carry
            return (i < max_restarts) & (~done)

        def body_fn(carry):
            i, key, best_params, best_res, done = carry
            perturb = uniform(
                fold_in(key, i),
                shape=(NUM_PARAMS,),
                minval=-perturb_scale,
                maxval=perturb_scale,
            )

            # Restart 0: just evaluate warm_start (no perturbation, no iterations).
            # Restart 1+: perturbed warm_start -> LM iterations.
            def iter_zero(_):
                r = residual_fn(warm_start, prefix_op, basis_gate, target_inv)
                return warm_start, jnp.max(jnp.abs(r))

            def iter_nonzero(_):
                init = warm_start + perturb

                def lm_cond(inner):
                    j, _, prev_norm, _ = inner
                    return (j < maxiter) & (prev_norm > solver_tol)

                def lm_body(inner):
                    j, x, prev_norm, lam = inner
                    x_new, new_norm = _gn_step(
                        residual_fn, x, prefix_op, basis_gate, target_inv, lam
                    )
                    improved = jnp.isfinite(new_norm) & (new_norm < prev_norm)
                    lam_new = jnp.clip(
                        jnp.where(improved, lam * 0.5, lam * 2.0), 1e-12, 1e6
                    )
                    return (
                        j + 1,
                        jnp.where(improved, x_new, x),
                        jnp.where(improved, new_norm, prev_norm),
                        lam_new,
                    )

                r0 = residual_fn(init, prefix_op, basis_gate, target_inv)
                init_norm = jnp.max(jnp.abs(r0))
                _, final_x, final_res, _ = lax.while_loop(
                    lm_cond, lm_body, (jnp.int32(0), init, init_norm, jnp.float64(1e-3))
                )
                return final_x, final_res

            params, res = lax.cond(i == 0, iter_zero, iter_nonzero, None)
            params_finite = jnp.all(jnp.isfinite(params))
            improved = (res < best_res) & params_finite & jnp.isfinite(res)

            return (
                i + 1,
                key,
                jnp.where(improved, params, best_params),
                jnp.where(improved, res, best_res),
                done | (res <= tol),
            )

        _, _, best_params, best_res, _ = lax.while_loop(
            cond_fn,
            body_fn,
            (
                jnp.int32(0),
                key,
                warm_start,
                jnp.array(jnp.inf, dtype=jnp.float64),
                jnp.array(False),
            ),
        )
        return best_params, best_res

    return run_until_success


def _warm_polish(solver, params, key, prefix_op, basis_gate, target_weyl, tol):
    """Branch-detect -> warm-start LM solve -> extract u0, u1."""
    # Pick closer of direct or rho-reflected Weyl target
    u0, u1 = _params_to_unitaries(params)
    U = basis_gate @ _kron_2x2(u1, u0) @ prefix_op
    c = weyl_coordinates(U)
    refl = target_weyl * jnp.array([-1, 1, -1]) + jnp.array([1, 0, 0])
    target = jnp.where(
        jnp.max(jnp.abs(c - refl)) < jnp.max(jnp.abs(c - target_weyl)),
        refl,
        target_weyl,
    )
    final, res = solver(key, prefix_op, basis_gate, target, tol, params)
    u0, u1 = _params_to_unitaries(final)
    return u0, u1, res


# ---------------------------------------------------------------------------
# Public solver class
# ---------------------------------------------------------------------------


class JaxLMSegmentSolver(SegmentSolver):
    """Adaptive Makhlin/Weyl segment solver.

    Single JIT dispatch:
      1. Makhlin GN probe (4 restarts) -- basin-finding.
      2. Converged -> branch detect -> Weyl LM polish.
      3. Near-miss -> quick Weyl LM (2 restarts).
      4. Both fail -> cold Weyl GN (full budget).

    Python-level fallback with separate JIT as safety net.
    """

    def __init__(self, config: GulpsConfig | None = None, rng_seed: int | None = None):
        self.config = config or GulpsConfig()
        self._key = PRNGKey(rng_seed or 0)

        # --- Build solver components ---
        solve_makhlin_probe = _make_gn_restart_loop(
            _makhlin_residual_fused,
            maxiter=self.config.makhlin_maxiter,
            max_restarts=self.config.makhlin_probe_restarts,
            solver_tol=self.config.segment_solver_tol,
        )
        solve_weyl_polish = _make_lm_warmstart_loop(
            _weyl_residual,
            maxiter=self.config.weyl_maxiter,
            max_restarts=self.config.weyl_restarts,
            solver_tol=self.config.segment_solver_tol,
            perturb_scale=self.config.weyl_perturb_scale,
        )
        solve_weyl_quick = _make_lm_warmstart_loop(
            _weyl_residual,
            maxiter=self.config.weyl_maxiter,
            max_restarts=2,
            solver_tol=self.config.segment_solver_tol,
            perturb_scale=self.config.weyl_perturb_scale,
        )
        solve_weyl_direct = _make_gn_restart_loop(
            _weyl_residual,
            maxiter=self.config.makhlin_maxiter,
            max_restarts=self.config.makhlin_restarts,
            solver_tol=self.config.segment_solver_tol,
        )
        solve_makhlin_full = _make_gn_restart_loop(
            _makhlin_residual_fused,
            maxiter=self.config.makhlin_maxiter,
            max_restarts=self.config.makhlin_restarts,
            solver_tol=self.config.segment_solver_tol,
        )

        # --- Primary solve: Makhlin probe -> graduated Weyl refinement ---
        @jit
        def _primary_solve(
            key, prefix_op, basis_gate, target_mat, makhlin_tol, weyl_tol
        ):
            target_makhlin = makhlin_invariants(target_mat)
            target_weyl = weyl_coordinates(target_mat)
            prefix_magic, magic_basis, target_packed = _precompute_makhlin_args(
                prefix_op, basis_gate, target_makhlin
            )
            makhlin_params, makhlin_res = solve_makhlin_probe(
                key,
                prefix_magic,
                magic_basis,
                target_packed,
                makhlin_tol,
            )

            def warm_path(_):
                return _warm_polish(
                    solve_weyl_polish,
                    makhlin_params,
                    key,
                    prefix_op,
                    basis_gate,
                    target_weyl,
                    weyl_tol,
                )

            def cold_path(_):
                u0_q, u1_q, res_q = _warm_polish(
                    solve_weyl_quick,
                    makhlin_params,
                    key,
                    prefix_op,
                    basis_gate,
                    target_weyl,
                    weyl_tol,
                )

                def quick_done(_):
                    return u0_q, u1_q, res_q

                def gn_fallback(_):
                    p, r = solve_weyl_direct(
                        key,
                        prefix_op,
                        basis_gate,
                        target_weyl,
                        weyl_tol,
                    )
                    u0, u1 = _params_to_unitaries(p)
                    return u0, u1, r

                return lax.cond(res_q <= weyl_tol, quick_done, gn_fallback, None)

            u0, u1, weyl_res = lax.cond(
                makhlin_res <= makhlin_tol, warm_path, cold_path, None
            )
            return u0, u1, weyl_res, makhlin_res

        self._primary_solve = _primary_solve

        # --- Fallback: full Makhlin (256) -> Weyl polish ---
        @jit
        def _fallback_solve(
            key, prefix_op, basis_gate, target_mat, makhlin_tol, weyl_tol
        ):
            target_makhlin = makhlin_invariants(target_mat)
            target_weyl = weyl_coordinates(target_mat)
            prefix_magic, magic_basis, target_packed = _precompute_makhlin_args(
                prefix_op, basis_gate, target_makhlin
            )
            makhlin_params, _ = solve_makhlin_full(
                key,
                prefix_magic,
                magic_basis,
                target_packed,
                makhlin_tol,
            )
            return _warm_polish(
                solve_weyl_polish,
                makhlin_params,
                key,
                prefix_op,
                basis_gate,
                target_weyl,
                weyl_tol,
            )

        self._fallback_solve = _fallback_solve

        # Eagerly compile both traces (identity converges in 1 restart).
        _dummy = jnp.eye(4, dtype=jnp.complex128)
        _primary_solve(self._key, _dummy, _dummy, _dummy, 1e-9, 1e-5)
        _fallback_solve(self._key, _dummy, _dummy, _dummy, 1e-9, 1e-5)

    def try_solve(
        self,
        prefix_inv,
        basis_inv,
        target_inv,
        *,
        step: int | None = 0,
        rng_seed: int | None = None,
    ) -> SegmentSolution:
        """Solve one segment via adaptive Makhlin/Weyl dispatch."""
        start_time = time.perf_counter()

        key = self._key
        if rng_seed is not None:
            key = fold_in(key, rng_seed)

        j_prefix = _get_jax_matrix(prefix_inv)
        j_gate = _get_jax_matrix(basis_inv)
        j_target = _get_jax_matrix(target_inv)

        u0, u1, weyl_res, makhlin_res = self._primary_solve(
            key,
            j_prefix,
            j_gate,
            j_target,
            self.config.makhlin_conv_tol,
            self.config.weyl_conv_tol,
        )
        weyl_res_py = float(weyl_res)

        # Safety-net fallback: full Makhlin + Weyl polish
        if weyl_res_py > self.config.weyl_conv_tol:
            u0, u1, weyl_res_fb = self._fallback_solve(
                key,
                j_prefix,
                j_gate,
                j_target,
                self.config.makhlin_conv_tol,
                self.config.weyl_conv_tol,
            )
            weyl_res_py = float(weyl_res_fb)

            if weyl_res_py > self.config.weyl_conv_tol:
                raise RuntimeError(
                    f"Optimization failed: Stage 1 residual {float(makhlin_res):.2e}, "
                    f"Stage 2 residual {weyl_res_py:.2e}"
                )

        return SegmentSolution(
            u0=np.asarray(u0),
            u1=np.asarray(u1),
            max_residual=weyl_res_py,
            success=True,
            metadata={
                "stage": "weyl",
                "elapsed": time.perf_counter() - start_time,
            },
        )
