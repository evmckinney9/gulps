"""Two-stage Makhlin/Weyl segment solver (Gauss-Newton + Levenberg-Marquardt).

Stage 1 (Makhlin GN) finds the basin; Stage 2 (Weyl LM) polishes to
high accuracy.  Restart loop factories and the public JaxLMSegmentSolver
class live here.  All JAX math primitives live in jax_primitives.py.

The Makhlin invariants symmetry is causes the Jacobian to become nearly
rank-deficient at the symmetry line c1 = c2.  GN convergence is steady
but slow there (~40-60 iters vs ~10 typical); the stagnation detector
uses a rate-based criterion (no halving in N iters) rather than an
absolute-improvement threshold to avoid killing these restarts early.
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
    stagnation_window: int = 32,
    progress_ratio: float = 0.5,
    restart_patience: int = 0,
):
    """Build a JIT'd random-restart GN solver with rate-based stagnation.

    A snapshot of the residual is taken whenever it improves by progress_ratio
    (halves by default).  A restart is abandoned if stagnation_window iterations
    pass without crossing the next threshold.  This lets slow-converging
    restarts at rank-deficient symmetry points (c1 ≈ c2) run to completion.

    If restart_patience > 0, the restart loop exits early when
    restart_patience consecutive restarts fail to improve the global best.
    """

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
            i, _, _, best_res, done, stale_count = carry
            exhausted = (restart_patience > 0) & (stale_count >= restart_patience)
            return (i < max_restarts) & (~done) & (~exhausted)

        def body_fn(carry):
            i, key, best_params, best_res, done, stale_count = carry
            init = uniform(
                fold_in(key, i), shape=(NUM_PARAMS,), minval=init_min, maxval=init_max
            )

            def gn_cond(inner):
                j, _, prev_norm, _init_n, _snap_norm, snap_j = inner
                stagnated = ((j - snap_j) >= stagnation_window) & (prev_norm > 1e-3)
                return (j < maxiter) & (prev_norm > solver_tol) & (~stagnated)

            def gn_body(inner):
                j, x, _, init_n, snap_norm, snap_j = inner
                x_new, new_norm = _gn_step(
                    residual_fn, x, prefix_op, basis_gate, target_inv, 1e-14
                )
                finite = jnp.all(jnp.isfinite(x_new)) & jnp.isfinite(new_norm)
                out_norm = jnp.where(finite, new_norm, jnp.inf)
                progressed = finite & (out_norm < snap_norm * progress_ratio)
                new_snap_norm = jnp.where(progressed, out_norm, snap_norm)
                new_snap_j = jnp.where(progressed, j + 1, snap_j)
                return (
                    j + 1,
                    jnp.where(finite, x_new, x),
                    out_norm,
                    init_n,
                    new_snap_norm,
                    new_snap_j,
                )

            r0 = residual_fn(init, prefix_op, basis_gate, target_inv)
            init_norm = jnp.max(jnp.abs(r0))
            _, final_x, final_res, _, _, _ = lax.while_loop(
                gn_cond,
                gn_body,
                (
                    jnp.int32(0),
                    init,
                    init_norm,
                    init_norm,
                    init_norm,
                    jnp.int32(0),
                ),
            )

            params_finite = jnp.all(jnp.isfinite(final_x))
            improved = (final_res < best_res) & params_finite
            safe_params = jnp.where(params_finite, final_x, init)
            new_stale = jnp.where(improved, jnp.int32(0), stale_count + 1)

            return (
                i + 1,
                key,
                jnp.where(improved, safe_params, best_params),
                jnp.where(improved, final_res, best_res),
                done | (final_res <= tol),
                new_stale,
            )

        _, _, best_params, best_res, _, _ = lax.while_loop(
            cond_fn,
            body_fn,
            (
                jnp.int32(0),
                key,
                jnp.zeros((NUM_PARAMS,), dtype=jnp.float64),
                jnp.array(jnp.inf, dtype=jnp.float64),
                jnp.array(False),
                jnp.int32(0),
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
    """Two-stage Makhlin/Weyl segment solver.

    Single JIT dispatch:
      1. Makhlin GN (random restarts, early-exit) -- basin-finding.
      2. Branch detect -> Weyl LM polish (warm-started from Makhlin result).

    The Weyl polish prevents residual accumulation across subsequent segments,
    even in cases where the Makhlin solution is already close.
    """

    def __init__(self, config: GulpsConfig | None = None, rng_seed: int | None = None):
        """Initialize the two-stage Makhlin→Weyl segment solver."""
        self.config = config or GulpsConfig()
        self._key = PRNGKey(rng_seed or 0)

        # --- Build solver components ---
        solve_makhlin = _make_gn_restart_loop(
            _makhlin_residual_fused,
            maxiter=self.config.makhlin_maxiter,
            max_restarts=self.config.makhlin_restarts,
            solver_tol=self.config.makhlin_solver_tol,
            restart_patience=self.config.makhlin_restart_patience,
        )
        solve_weyl = _make_lm_warmstart_loop(
            _weyl_residual,
            maxiter=self.config.weyl_maxiter,
            max_restarts=self.config.weyl_restarts,
            solver_tol=self.config.weyl_solver_tol,
            perturb_scale=self.config.weyl_perturb_scale,
        )

        # --- Makhlin -> warm Weyl polish ---
        @jit
        def _solve(key, prefix_op, basis_gate, target_mat, makhlin_tol, weyl_tol):
            target_makhlin = makhlin_invariants(target_mat)
            target_weyl = weyl_coordinates(target_mat)
            prefix_magic, magic_basis, target_packed = _precompute_makhlin_args(
                prefix_op, basis_gate, target_makhlin
            )
            makhlin_params, makhlin_res = solve_makhlin(
                key,
                prefix_magic,
                magic_basis,
                target_packed,
                makhlin_tol,
            )
            u0, u1, weyl_res = _warm_polish(
                solve_weyl,
                makhlin_params,
                key,
                prefix_op,
                basis_gate,
                target_weyl,
                weyl_tol,
            )
            return u0, u1, weyl_res, makhlin_res

        self._solve = _solve

        # Eagerly compile (identity converges in 1 restart).
        _dummy = jnp.eye(4, dtype=jnp.complex128)
        _solve(self._key, _dummy, _dummy, _dummy, 1e-9, 1e-5)

    def try_solve(
        self,
        prefix_inv,
        basis_inv,
        target_inv,
        *,
        step: int | None = 0,
        rng_seed: int | None = None,
    ) -> SegmentSolution:
        """Solve one segment via Makhlin -> warm Weyl polish."""
        start_time = time.perf_counter()

        key = self._key
        if rng_seed is not None:
            key = fold_in(key, rng_seed)

        j_prefix = _get_jax_matrix(prefix_inv)
        j_gate = _get_jax_matrix(basis_inv)
        j_target = _get_jax_matrix(target_inv)

        u0, u1, weyl_res, makhlin_res = self._solve(
            key,
            j_prefix,
            j_gate,
            j_target,
            self.config.makhlin_conv_tol,
            self.config.weyl_conv_tol,
        )
        weyl_res_py = float(weyl_res)

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
