# Copyright 2025-2026 Lev S. Bishop, Evan McKinney
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Makhlin GN segment solver with Weyl-coordinate early-exit.

Random-restart Gauss-Newton on Makhlin invariants, with early-exit: at
degenerate Weyl faces the Makhlin floor (~2e-9) maps to acceptable Weyl
residual via quadratic sensitivity, so the restart loop checks Weyl
coordinates directly.  All JAX primitives in jax_primitives.py.
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

_REFL_SIGN = jnp.array([-1.0, 1.0, -1.0])
_REFL_OFFSET = jnp.array([1.0, 0.0, 0.0])


def _weyl_res_both_branches(c, target_weyl):
    """Min Weyl residual over direct and rho-reflected branches.

    Returns (min_res, refl_target, use_refl) where use_refl indicates
    the reflected branch is closer.
    """
    refl = target_weyl * _REFL_SIGN + _REFL_OFFSET
    direct = jnp.max(jnp.abs(c - target_weyl))
    reflected = jnp.max(jnp.abs(c - refl))
    return jnp.minimum(direct, reflected), refl, reflected < direct


# ---------------------------------------------------------------------------
# Restart loop factory
# ---------------------------------------------------------------------------


def _gn_step(residual_fn, x, prefix_op, basis_gate, target_inv, damping):
    """Single damped Gauss-Newton step -> (x_new, residual_norm)."""
    r = residual_fn(x, prefix_op, basis_gate, target_inv)
    J = jacrev(residual_fn, argnums=0)(x, prefix_op, basis_gate, target_inv)
    gram = J @ J.T + damping * jnp.eye(J.shape[0], dtype=jnp.float64)
    x_new = x + J.T @ jnp.linalg.solve(gram, -r)
    r_new = residual_fn(x_new, prefix_op, basis_gate, target_inv)
    return x_new, jnp.max(jnp.abs(r_new))


def _make_gn_restart_loop(residual_fn, *, maxiter, max_restarts):
    """Build a JIT'd random-restart GN solver.

    Three exit conditions per restart: Makhlin convergence, stagnation
    (absolute thresholds at iter 8/16/32), or Weyl-coordinate accuracy.
    """

    @jit
    def run(
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
            u0, u1 = _params_to_unitaries(params)
            U = weyl_gate @ _kron_2x2(u1, u0) @ weyl_prefix
            res, _, _ = _weyl_res_both_branches(weyl_coordinates(U), weyl_target)
            return res

        def cond_fn(carry):
            i, _, best_res, done = carry
            return (i < max_restarts) & (~done)

        def body_fn(carry):
            i, best_params, best_res, done = carry
            init = uniform(
                fold_in(key, i), shape=(NUM_PARAMS,), minval=-0.1, maxval=0.1
            )

            def gn_cond(inner):
                j, _, prev_norm, _, best_norm = inner
                stagnated = (
                    ((j >= 8) & (prev_norm > 3e-1))
                    | ((j >= 16) & (prev_norm > 5e-2))
                    | ((j >= 32) & (prev_norm > 1e-3))
                )
                return (j < maxiter) & (best_norm > tol) & (~stagnated)

            def gn_body(inner):
                j, x, _, best_x, best_norm = inner
                # r and jacrev share the forward pass via XLA CSE.
                r = residual_fn(x, prefix_op, basis_gate, target_inv)
                J = jacrev(residual_fn, argnums=0)(x, prefix_op, basis_gate, target_inv)
                curr_norm = jnp.max(jnp.abs(r))
                is_best = curr_norm < best_norm
                new_best_x = jnp.where(is_best, x, best_x)
                new_best_norm = jnp.where(is_best, curr_norm, best_norm)
                # GN step (no r_new evaluation - norm is taken at the
                # TOP of the next iteration via CSE with jacrev).
                gram = J @ J.T + 1e-14 * jnp.eye(3, dtype=jnp.float64)
                x_new = x + J.T @ jnp.linalg.solve(gram, -r)
                finite = jnp.all(jnp.isfinite(x_new))
                safe_x = jnp.where(finite, x_new, x)
                return (
                    j + 1,
                    safe_x,
                    curr_norm,
                    new_best_x,
                    new_best_norm,
                )

            _, last_x, _, best_x, best_norm = lax.while_loop(
                gn_cond,
                gn_body,
                (jnp.int32(0), init, jnp.inf, init, jnp.inf),
            )
            # Evaluate the final iterate (not checked inside the loop).
            r_last = residual_fn(last_x, prefix_op, basis_gate, target_inv)
            last_norm = jnp.max(jnp.abs(r_last))
            last_better = (last_norm < best_norm) & jnp.all(jnp.isfinite(last_x))
            final_x = jnp.where(last_better, last_x, best_x)
            final_res = jnp.where(last_better, last_norm, best_norm)

            improved = (final_res < best_res) & jnp.all(jnp.isfinite(final_x))
            new_best = jnp.where(improved, final_x, best_params)
            new_res = jnp.where(improved, final_res, best_res)
            return (
                i + 1,
                new_best,
                new_res,
                done | (new_res <= tol) | (_weyl_check(new_best) <= weyl_tol),
            )

        _, best_params, best_res, _ = lax.while_loop(
            cond_fn,
            body_fn,
            (
                jnp.int32(0),
                jnp.zeros(NUM_PARAMS, dtype=jnp.float64),
                jnp.array(jnp.inf, dtype=jnp.float64),
                jnp.array(False),
            ),
        )
        return best_params, best_res

    return run


# ---------------------------------------------------------------------------
# Compiled solver: built once at first use, reused for all ISAs.
# Gate sets and tolerances are runtime args; only maxiter/max_restarts
# are compile-time constants (hardcoded, not configurable).
# ---------------------------------------------------------------------------

_MAXITER = 512
_MAX_RESTARTS = 256


def _build_solve():
    """Build and JIT-compile the solver function, with eager warmup."""
    solve_makhlin = _make_gn_restart_loop(
        _makhlin_residual_fused,
        maxiter=_MAXITER,
        max_restarts=_MAX_RESTARTS,
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

        # Weyl residual (both branches) from Makhlin solution.
        u0, u1 = _params_to_unitaries(params)
        U = basis_gate @ _kron_2x2(u1, u0) @ prefix_op
        c = weyl_coordinates(U)
        weyl_res, refl, use_refl = _weyl_res_both_branches(c, target_weyl)

        # Weyl LM polish: close the Makhlin->Weyl gap at degenerate
        # faces where makhlin_res ~1e-9 maps to weyl_res ~1e-5.
        weyl_target = jnp.where(use_refl, refl, target_weyl)

        def _polish_cond(inner):
            j, _, res, _ = inner
            return (j < 128) & (res > weyl_tol)

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

    # Eagerly compile (identity converges in 1 restart).
    _dummy = jnp.eye(4, dtype=jnp.complex128)
    _solve(PRNGKey(0), _dummy, _dummy, _dummy, 1e-9, 1e-5)
    return _solve


_compiled_solve = _build_solve()


# ---------------------------------------------------------------------------
# Public solver class
# ---------------------------------------------------------------------------


class JaxLMSegmentSolver(SegmentSolver):
    """Makhlin GN segment solver with Weyl-coordinate early-exit."""

    def __init__(self, config: GulpsConfig | None = None, rng_seed: int | None = None):
        """Initialize the solver."""
        self.config = config or GulpsConfig()
        self._key = PRNGKey(rng_seed or 0)
        self._solve = _compiled_solve

    def try_solve(
        self, prefix_inv, basis_inv, target_inv, *, step=0, rng_seed=None
    ) -> SegmentSolution:
        """Solve one segment via Makhlin GN -> Weyl polish."""
        start = time.perf_counter()
        key = self._key
        if rng_seed is not None:
            key = fold_in(key, rng_seed)

        prefix_jax = _get_jax_matrix(prefix_inv)
        basis_jax = _get_jax_matrix(basis_inv)
        target_jax = _get_jax_matrix(target_inv)
        makhlin_tol = self.config.makhlin_conv_tol
        weyl_tol = self.config.weyl_conv_tol

        u0, u1, weyl_res, makhlin_res = self._solve(
            key,
            prefix_jax,
            basis_jax,
            target_jax,
            makhlin_tol,
            weyl_tol,
        )
        weyl_ok = float(weyl_res) <= weyl_tol
        makhlin_ok = float(makhlin_res) <= makhlin_tol

        # PRNG retry: at near-degenerate Weyl faces, the default key may
        # produce 256 random inits that all miss convergent basins.  A
        # diversified key explores a different set of basins.
        if not (weyl_ok or makhlin_ok):
            key2 = fold_in(self._key, step ^ 0xCAFE)
            u0, u1, weyl_res, makhlin_res = self._solve(
                key2,
                prefix_jax,
                basis_jax,
                target_jax,
                makhlin_tol,
                weyl_tol,
            )
            weyl_ok = float(weyl_res) <= weyl_tol
            makhlin_ok = float(makhlin_res) <= makhlin_tol

        if weyl_ok or makhlin_ok:
            return SegmentSolution(
                u0=np.asarray(u0),
                u1=np.asarray(u1),
                max_residual=float(weyl_res),
                success=True,
                metadata={"elapsed": time.perf_counter() - start},
            )

        raise RuntimeError(
            f"Optimization failed: makhlin_res={float(makhlin_res):.2e}, "
            f"weyl_res={float(weyl_res):.2e}"
        )
