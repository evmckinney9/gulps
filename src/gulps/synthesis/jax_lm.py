"""Makhlin GN segment solver with Weyl-coordinate early-exit.

Random-restart Gauss-Newton on Makhlin invariants, with early-exit: at degenerate Weyl faces the Makhlin floor (~2e-9) maps to
acceptable Weyl residual via quadratic sensitivity, so the restart loop
checks Weyl coordinates directly.  All JAX primitives in jax_primitives.py.
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
):
    """Build a JIT'd random-restart GN solver.

    Three exit conditions per restart: Makhlin convergence, stagnation
    (residual > 5e-2 after 16 iters or > 1e-3 after 32), or Weyl-coordinate
    accuracy.  The Weyl check handles degenerate faces where the Makhlin
    floor exceeds tol but the Weyl residual is already acceptable.
    """

    @jit
    def run_until_success(
        key,
        prefix_op,
        basis_gate,
        target_inv,
        tol,
        weyl_prefix,
        weyl_gate,
        weyl_target,
        weyl_tol,
    ):
        def _weyl_check(params):
            """Weyl-coordinate error for params, checking both branches."""
            u0, u1 = _params_to_unitaries(params)
            U = weyl_gate @ _kron_2x2(u1, u0) @ weyl_prefix
            c = weyl_coordinates(U)
            refl = weyl_target * jnp.array([-1.0, 1.0, -1.0]) + jnp.array(
                [1.0, 0.0, 0.0]
            )
            return jnp.minimum(
                jnp.max(jnp.abs(c - weyl_target)),
                jnp.max(jnp.abs(c - refl)),
            )

        def cond_fn(carry):
            i, _, _, best_res, done = carry
            return (i < max_restarts) & (~done)

        def body_fn(carry):
            i, key, best_params, best_res, done = carry
            init = uniform(
                fold_in(key, i), shape=(NUM_PARAMS,), minval=-0.1, maxval=0.1
            )

            def gn_cond(inner):
                j, _, prev_norm, _, _ = inner
                stagnated = (
                    ((j >= 8) & (prev_norm > 3e-1))
                    | ((j >= 16) & (prev_norm > 5e-2))
                    | ((j >= 32) & (prev_norm > 1e-3))
                )
                return (j < maxiter) & (prev_norm > tol) & (~stagnated)

            def gn_body(inner):
                j, x, _, best_x, best_norm = inner
                x_new, new_norm = _gn_step(
                    residual_fn, x, prefix_op, basis_gate, target_inv, 1e-14
                )
                finite = jnp.all(jnp.isfinite(x_new)) & jnp.isfinite(new_norm)
                safe_x = jnp.where(finite, x_new, x)
                safe_norm = jnp.where(finite, new_norm, jnp.inf)
                is_best = safe_norm < best_norm
                return (
                    j + 1,
                    safe_x,
                    safe_norm,
                    jnp.where(is_best, safe_x, best_x),
                    jnp.where(is_best, safe_norm, best_norm),
                )

            r0 = residual_fn(init, prefix_op, basis_gate, target_inv)
            init_norm = jnp.max(jnp.abs(r0))
            _, _, _, final_x, final_res = lax.while_loop(
                gn_cond,
                gn_body,
                (jnp.int32(0), init, init_norm, init, init_norm),
            )

            improved = (final_res < best_res) & jnp.all(jnp.isfinite(final_x))
            new_best = jnp.where(improved, final_x, best_params)
            new_res = jnp.where(improved, final_res, best_res)

            # Weyl-based early exit: at degenerate faces, the Makhlin
            # floor exceeds tol but maps to acceptable Weyl residual.
            return (
                i + 1,
                key,
                new_best,
                new_res,
                done | (new_res <= tol) | (_weyl_check(new_best) <= weyl_tol),
            )

        _, _, best_params, best_res, _ = lax.while_loop(
            cond_fn,
            body_fn,
            (
                jnp.int32(0),
                key,
                jnp.zeros(NUM_PARAMS, dtype=jnp.float64),
                jnp.array(jnp.inf, dtype=jnp.float64),
                jnp.array(False),
            ),
        )
        return best_params, best_res

    return run_until_success


# ---------------------------------------------------------------------------
# Public solver class
# ---------------------------------------------------------------------------


class JaxLMSegmentSolver(SegmentSolver):
    """Makhlin GN segment solver with Weyl-coordinate early-exit.

    Random-restart GN on Makhlin invariants. At degenerate Weyl faces,
    the Makhlin convergence floor maps to acceptable Weyl residual via
    quadratic sensitivity — the restart loop checks this directly.
    """

    def __init__(self, config: GulpsConfig | None = None, rng_seed: int | None = None):
        """Initialize solvers and cache from config."""
        self.config = config or GulpsConfig()
        self._key = PRNGKey(rng_seed or 0)

        # --- Build solver ---
        solve_makhlin = _make_gn_restart_loop(
            _makhlin_residual_fused,
            maxiter=self.config.makhlin_maxiter,
            max_restarts=self.config.makhlin_restarts,
        )

        @jit
        def _solve(key, prefix_op, basis_gate, target_mat, makhlin_tol, weyl_tol):
            target_makhlin = makhlin_invariants(target_mat)
            target_weyl = weyl_coordinates(target_mat)
            prefix_magic, magic_basis, target_packed = _precompute_makhlin_args(
                prefix_op, basis_gate, target_makhlin
            )
            params, makhlin_res = solve_makhlin(
                key,
                prefix_magic,
                magic_basis,
                target_packed,
                makhlin_tol,
                prefix_op,
                basis_gate,
                target_weyl,
                weyl_tol,
            )
            # Extract unitaries and compute Weyl residual (both branches).
            u0, u1 = _params_to_unitaries(params)
            U = basis_gate @ _kron_2x2(u1, u0) @ prefix_op
            c = weyl_coordinates(U)
            refl = target_weyl * jnp.array([-1.0, 1.0, -1.0]) + jnp.array(
                [1.0, 0.0, 0.0]
            )
            direct_res = jnp.max(jnp.abs(c - target_weyl))
            refl_res = jnp.max(jnp.abs(c - refl))
            weyl_res = jnp.minimum(direct_res, refl_res)

            # Borderline polish: at degenerate faces, quadratic
            # Makhlin→Weyl sensitivity puts the result near weyl_tol.
            # A few LM iterations in Weyl space (1:1 sensitivity)
            # close the gap without restarts or perturbation.
            weyl_target = jnp.where(refl_res < direct_res, refl, target_weyl)

            def _polish_cond(inner):
                j, _, res, _ = inner
                return (j < 32) & (res > weyl_tol)

            def _polish_body(inner):
                j, x, res, lam = inner
                x_new, new_res = _gn_step(
                    _weyl_residual, x, prefix_op, basis_gate, weyl_target, lam
                )
                improved = jnp.isfinite(new_res) & (new_res < res)
                return (
                    j + 1,
                    jnp.where(improved, x_new, x),
                    jnp.where(improved, new_res, res),
                    jnp.clip(jnp.where(improved, lam * 0.5, lam * 2.0), 1e-12, 1e6),
                )

            _, polished, polished_res, _ = lax.while_loop(
                _polish_cond,
                _polish_body,
                (jnp.int32(0), params, weyl_res, jnp.float64(1e-3)),
            )
            u0, u1 = _params_to_unitaries(polished)
            return u0, u1, polished_res, makhlin_res

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

        # Retry with different random keys if the Weyl polish
        # barely fails.  A different key changes which restart wins
        # and can land on a Weyl-friendlier basin.
        best_weyl = float("inf")
        best_result = None
        for attempt in range(3):
            attempt_key = key if attempt == 0 else fold_in(key, attempt + 999)
            u0, u1, weyl_res, makhlin_res = self._solve(
                attempt_key,
                j_prefix,
                j_gate,
                j_target,
                self.config.makhlin_conv_tol,
                self.config.weyl_conv_tol,
            )
            weyl_res_py = float(weyl_res)
            if weyl_res_py <= self.config.weyl_conv_tol:
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
            if weyl_res_py < best_weyl:
                best_weyl = weyl_res_py
                best_result = (u0, u1, weyl_res_py, float(makhlin_res))

        u0, u1, weyl_res_py, makhlin_res_py = best_result
        raise RuntimeError(
            f"Optimization failed: makhlin_res={makhlin_res_py:.2e}, "
            f"weyl_res={weyl_res_py:.2e}"
        )
