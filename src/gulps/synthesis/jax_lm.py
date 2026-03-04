import time

import jax.numpy as jnp
import numpy as np
from jax import jacrev, jit, lax
from jax.random import PRNGKey, fold_in, uniform

from gulps.config import GulpsConfig
from gulps.core.jax_invariants import makhlin_invariants, weyl_coordinates
from gulps.synthesis.segments_abc import SegmentSolution, SegmentSolver

NUM_PARAMS = 8


@jit
def _params_to_unitaries(params: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract u0, u1 from 8D params as two normalized quaternions -> SU(2).

    params[:4]  = q0 = (w,x,y,z) for u0
    params[4:]  = q1 = (w,x,y,z) for u1
    """
    eps = 1e-12

    def quat_to_su2(q):
        # normalize onto S^3
        q = q / jnp.maximum(jnp.linalg.norm(q), eps)
        w, x, y, z = q
        a = w + 1j * z
        b = x + 1j * y
        return jnp.array([[a, b], [-jnp.conj(b), jnp.conj(a)]], dtype=jnp.complex128)

    u0 = quat_to_su2(params[:4])
    u1 = quat_to_su2(params[4:])
    return u0, u1


@jit
def _kron_2x2(u0: jnp.ndarray, u1: jnp.ndarray) -> jnp.ndarray:
    """Kronecker product of two 2x2 matrices."""
    return jnp.block(
        [
            [u0[0, 0] * u1, u0[0, 1] * u1],
            [u0[1, 0] * u1, u0[1, 1] * u1],
        ]
    )


@jit
def _construct_unitary(params, prefix_op, basis_gate):
    """Construct U = basis_gate @ (u1 ⊗ u0) @ prefix_op from rotation-vector params."""
    u0, u1 = _params_to_unitaries(params)
    return basis_gate @ _kron_2x2(u1, u0) @ prefix_op


def _make_residual_fn(invariant_fn):
    """Create residual function: target_inv - constructed_inv."""

    @jit
    def residual(x, prefix_op, basis_gate, target_inv):
        U = _construct_unitary(x, prefix_op, basis_gate)
        return target_inv - invariant_fn(U)

    return residual


# Pre-built residual functions
_makhlin_residual = _make_residual_fn(makhlin_invariants)
_weyl_residual = _make_residual_fn(weyl_coordinates)


def _get_jax_matrix(inv) -> jnp.ndarray:
    """Get JAX 4x4 unitary matrix from GateInvariants, with caching."""
    cached = getattr(inv, "_jax_matrix", None)
    if cached is not None:
        return cached
    # Use canonical_matrix directly (avoids Qiskit UnitaryGate round-trip)
    if inv._unitary is not None:
        mat_np = np.asarray(inv._unitary, dtype=np.complex128)
    else:
        mat_np = inv.canonical_matrix
    mat = jnp.asarray(mat_np, dtype=jnp.complex128)
    try:
        inv._jax_matrix = mat
    except AttributeError:
        pass  # frozen dataclass or similar
    return mat


@jit
def _compute_target_invariants(target_mat):
    """Fused target Makhlin + Weyl invariant computation (1 dispatch instead of 2)."""
    return makhlin_invariants(target_mat), weyl_coordinates(target_mat)


@jit
def _detect_branch_and_weyl_target(makhlin_params, prefix_op, basis_gate, target_weyl):
    """Fused branch detection (1 dispatch instead of ~5).

    Constructs the current unitary from Makhlin params, computes its Weyl coordinates,
    and determines whether the direct or rho-reflected Weyl target is closer.
    """
    U = _construct_unitary(makhlin_params, prefix_op, basis_gate)
    constructed_weyl = weyl_coordinates(U)
    reflected_weyl = target_weyl * jnp.array([-1.0, 1.0, -1.0]) + jnp.array(
        [1.0, 0.0, 0.0]
    )
    direct_res = jnp.max(jnp.abs(constructed_weyl - target_weyl))
    reflect_res = jnp.max(jnp.abs(constructed_weyl - reflected_weyl))
    use_reflected = reflect_res < direct_res
    return jnp.where(use_reflected, reflected_weyl, target_weyl)


def _gn_step(residual_fn, x, prefix_op, basis_gate, target_inv, damping):
    """Single damped Gauss-Newton step → (x_new, residual_norm)."""
    r = residual_fn(x, prefix_op, basis_gate, target_inv)
    J = jacrev(residual_fn, argnums=0)(x, prefix_op, basis_gate, target_inv)
    gram = J @ J.T + damping * jnp.eye(J.shape[0], dtype=jnp.float64)
    w = jnp.linalg.solve(gram, -r)
    x_new = x + J.T @ w
    r_new = residual_fn(x_new, prefix_op, basis_gate, target_inv)
    return x_new, jnp.max(jnp.abs(r_new))


def _warm_polish(solver, params, key, prefix_op, basis_gate, target_weyl, tol):
    """Branch-detect → warm-start LM solve → extract u0, u1.

    Common pattern used by primary and fallback solve paths.
    ``solver`` must accept (key, prefix_op, basis_gate, target, tol, warm_start).
    """
    target = _detect_branch_and_weyl_target(params, prefix_op, basis_gate, target_weyl)
    final, res = solver(key, prefix_op, basis_gate, target, tol, params)
    u0, u1 = _params_to_unitaries(final)
    return u0, u1, res


def _make_gn_restart_loop(
    residual_fn,
    *,
    maxiter: int,
    max_restarts: int,
    solver_tol: float,
    stagnation_iters: int = 32,
    stagnation_factor: float = 0.1,
):
    """Gauss-Newton restart loop with stagnation-based early exit.

    Each restart draws random initial params, runs GN with minimum-norm
    steps (3x3 Gram solve), and abandons stagnant restarts early.
    """

    @jit
    def run_until_success(
        key: PRNGKey,
        prefix_op: jnp.ndarray,
        basis_gate: jnp.ndarray,
        target_inv: jnp.ndarray,
        tol: float,
        init_min: float = -jnp.pi / 2,
        init_max: float = jnp.pi / 2,
    ):
        def cond_fn(carry):
            i, _, _, best_res, done = carry
            return (i < max_restarts) & (~done)

        def body_fn(carry):
            i, key, best_params, best_res, done = carry

            key_i = fold_in(key, i)
            init = uniform(key_i, shape=(NUM_PARAMS,), minval=init_min, maxval=init_max)

            # --- Inner GN iterations with stagnation detection ---
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

        init_state = (
            jnp.int32(0),
            key,
            jnp.zeros((NUM_PARAMS,), dtype=jnp.float64),
            jnp.array(jnp.inf, dtype=jnp.float64),
            jnp.array(False),
        )
        _, _, best_params, best_res, _ = lax.while_loop(cond_fn, body_fn, init_state)
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
    """Levenberg-Marquardt warm-start restart loop with adaptive damping.

    Restart 0 evaluates the warm start directly; subsequent restarts
    perturb it and run LM with adaptive damping via the 3x3 Gram solve.
    """

    @jit
    def run_until_success(
        key: PRNGKey,
        prefix_op: jnp.ndarray,
        basis_gate: jnp.ndarray,
        target_inv: jnp.ndarray,
        tol: float,
        warm_start: jnp.ndarray,
    ):
        def cond_fn(carry):
            i, _, _, best_res, done = carry
            return (i < max_restarts) & (~done)

        def body_fn(carry):
            i, key, best_params, best_res, done = carry

            key_i = fold_in(key, i)
            perturb = uniform(
                key_i, shape=(NUM_PARAMS,), minval=-perturb_scale, maxval=perturb_scale
            )

            # Iteration 0: just evaluate warm_start, no solver call
            # Iteration 1+: run LM with perturbed warm_start
            def iter_zero(_):
                res_vec = residual_fn(warm_start, prefix_op, basis_gate, target_inv)
                return warm_start, jnp.max(jnp.abs(res_vec))

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
            res_finite = jnp.isfinite(res)
            improved = (res < best_res) & params_finite & res_finite

            return (
                i + 1,
                key,
                jnp.where(improved, params, best_params),
                jnp.where(improved, res, best_res),
                done | (res <= tol),
            )

        init_state = (
            jnp.int32(0),
            key,
            warm_start,
            jnp.array(jnp.inf, dtype=jnp.float64),
            jnp.array(False),
        )
        _, _, best_params, best_res, _ = lax.while_loop(cond_fn, body_fn, init_state)
        return best_params, best_res

    return run_until_success


class JaxLMSegmentSolver(SegmentSolver):
    """Adaptive Makhlin/Weyl segment solver exploiting landscape complementarity.

    Single JIT dispatch:
      1. Makhlin GN probe (4 restarts) — basin-finding, not full convergence.
      2. Converged → branch detect → Weyl LM polish.
      3. Near-miss → quick Weyl LM from probe params (2 restarts).
      4. Both fail → cold Weyl GN (full budget).

    Python-level fallback with separate JIT as safety net.
    """

    _MAKHLIN_PROBE = 4  # Bimodal: converges in <4 or not at all

    def __init__(
        self,
        config: GulpsConfig | None = None,
        rng_seed: int | None = None,
    ):
        self.config = config or GulpsConfig()
        self._key = PRNGKey(rng_seed or 0)

        solve_makhlin_probe = _make_gn_restart_loop(
            _makhlin_residual,
            maxiter=self.config.makhlin_maxiter,
            max_restarts=self._MAKHLIN_PROBE,
            solver_tol=self.config.segment_solver_tol,
        )

        solve_weyl_polish = _make_lm_warmstart_loop(
            _weyl_residual,
            maxiter=self.config.weyl_maxiter,
            max_restarts=self.config.weyl_restarts,
            solver_tol=self.config.segment_solver_tol,
            perturb_scale=self.config.weyl_perturb_scale,
        )

        # Quick LM: 2-restart attempt to salvage near-miss Makhlin probes.
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

        @jit
        def _primary_solve(
            key, prefix_op, basis_gate, target_mat, makhlin_tol, weyl_tol
        ):
            """Makhlin probe → graduated Weyl refinement (single JIT)."""
            target_makhlin, target_weyl = _compute_target_invariants(target_mat)

            makhlin_params, makhlin_res = solve_makhlin_probe(
                key,
                prefix_op,
                basis_gate,
                target_makhlin,
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

        solve_makhlin_full = _make_gn_restart_loop(
            _makhlin_residual,
            maxiter=self.config.makhlin_maxiter,
            max_restarts=self.config.makhlin_restarts,
            solver_tol=self.config.segment_solver_tol,
        )

        @jit
        def _fallback_solve(
            key, prefix_op, basis_gate, target_mat, makhlin_tol, weyl_tol
        ):
            """Full Makhlin (256) → Weyl polish (separate JIT, safety net)."""
            target_makhlin, target_weyl = _compute_target_invariants(target_mat)
            makhlin_params, _ = solve_makhlin_full(
                key,
                prefix_op,
                basis_gate,
                target_makhlin,
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

        # Eagerly compile both traces so the first real invocation
        # doesn't pay ~1s of JIT overhead.  Running on identity is nearly
        # instantaneous (converges in 1 restart).
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
        """Solve one segment via adaptive Makhlin/Weyl dispatch.

        Returns a SegmentSolution (always succeeds or raises RuntimeError).
        Uses ``rng_seed`` to diversify restart sequences across calls.
        """
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
                makhlin_res_py = float(makhlin_res)
                raise RuntimeError(
                    f"Optimization failed: Stage 1 residual {makhlin_res_py:.2e}, "
                    f"Stage 2 residual {weyl_res_py:.2e}"
                )

        # Convert to numpy once (avoids repeated JAX device→host transfers
        # when u0/u1 are used in downstream numpy operations)
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
