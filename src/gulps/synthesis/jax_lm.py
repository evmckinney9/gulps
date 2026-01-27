import time

import jax.numpy as jnp
import numpy as np
from jax import jit, lax
from jax.random import PRNGKey, fold_in, uniform
from jaxopt import GaussNewton, LevenbergMarquardt
from jaxopt.linear_solve import solve_lu

from gulps.config import GulpsConfig
from gulps.core.jax_invariants import makhlin_invariants, weyl_coordinates
from gulps.synthesis.segments_abc import SegmentSolution, SegmentSolver

# Pauli matrices
X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
I2 = jnp.eye(2, dtype=jnp.complex128)


# @jit
# def _rv(v: jnp.ndarray) -> jnp.ndarray:
#     """Rotation vector to SU(2) unitary via Rodriguez formula."""
#     a = jnp.linalg.norm(v)
#     half = 0.5 * a
#     s = jnp.where(
#         jnp.abs(half) < 1e-8,
#         1.0 - (half * half) / 6.0 + (half**4) / 120.0,
#         jnp.sin(half) / half,
#     )
#     c = jnp.cos(half)
#     vx, vy, vz = v
#     H = vx * X + vy * Y + vz * Z
#     return c * I2 - 1j * (0.5 * s) * H

NUM_PARAMS = 8


# @jit
# def _params_to_unitaries(params: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
#     """Extract u0, u1 from 6D params."""

#     def _rv(v: jnp.ndarray) -> jnp.ndarray:
#         """Rotation vector to SU(2) via stabilized Rodriguez formula."""
#         theta_sq = jnp.dot(v, v)
#         theta = jnp.sqrt(theta_sq)
#         half = 0.5 * theta

#         # Taylor expansion for small angles: sin(x)/x ≈ 1 - x²/6 + x⁴/120
#         # cos(x) ≈ 1 - x²/2 + x⁴/24
#         sinc_half = jnp.where(
#             theta_sq < 1e-8,
#             0.5 - theta_sq / 48.0,  # (1 - (θ/2)²/6) / 2
#             jnp.sin(half) / theta,
#         )
#         cos_half = jnp.where(
#             theta_sq < 1e-8,
#             1.0 - theta_sq / 8.0,  # 1 - (θ/2)²/2
#             jnp.cos(half),
#         )

#         vx, vy, vz = v
#         H = vx * X + vy * Y + vz * Z
#         return cos_half * I2 - 1j * sinc_half * H

#     return _rv(params[:3]), _rv(params[3:])


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
    """Construct U = basis_gate @ (u1 ⊗ u0) @ prefix_op from 6D rotation-vector params."""
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


def _make_restart_loop(run_solver, max_restarts: int):
    """Factory that creates a jitted restart loop for a specific solver."""

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
            result = run_solver(
                init, prefix_op=prefix_op, basis_gate=basis_gate, target_inv=target_inv
            )

            res = jnp.max(jnp.abs(result.state.residual))
            improved = res < best_res

            return (
                i + 1,
                key,
                jnp.where(improved, result.params, best_params),
                jnp.where(improved, res, best_res),
                done | (res <= tol),
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


def _make_warmstart_loop(
    run_solver, residual_fn, max_restarts: int, perturb_scale: float
):
    """Factory that creates a jitted warm-start restart loop.

    Iteration 0 evaluates the warm start directly (no solver call).
    Subsequent iterations run the solver with perturbations.
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

            # Iteration 0: just evaluate warm_start, no solver
            # Iteration 1+: run solver with perturbed warm_start
            def iter_zero(_):
                res_vec = residual_fn(warm_start, prefix_op, basis_gate, target_inv)
                return warm_start, jnp.max(jnp.abs(res_vec))

            def iter_nonzero(_):
                init = warm_start + perturb
                result = run_solver(
                    init,
                    prefix_op=prefix_op,
                    basis_gate=basis_gate,
                    target_inv=target_inv,
                )
                return result.params, jnp.max(jnp.abs(result.state.residual))

            params, res = lax.cond(i == 0, iter_zero, iter_nonzero, None)
            improved = res < best_res

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
    """Two-stage Makhlin→Weyl segment solver."""

    def __init__(
        self,
        config: GulpsConfig | None = None,
        rng_seed: int | None = None,
    ):
        self.config = config or GulpsConfig()
        self._key = PRNGKey(rng_seed or 0)

        makhlin_solver = GaussNewton(
            residual_fun=_makhlin_residual,
            maxiter=self.config.makhlin_maxiter,
            tol=self.config.segment_solver_tol,
            implicit_diff=False,
            implicit_diff_solve=solve_lu,
            jit=True,
        )

        weyl_solver = LevenbergMarquardt(
            residual_fun=_weyl_residual,
            maxiter=self.config.weyl_maxiter,
            tol=self.config.segment_solver_tol,
            implicit_diff=False,
            jit=True,
            materialize_jac=True,
            solver=solve_lu,
        )

        self._solve_makhlin = _make_restart_loop(
            makhlin_solver.run,
            self.config.makhlin_restarts,
        )

        self._solve_weyl = _make_warmstart_loop(
            weyl_solver.run,
            _weyl_residual,
            self.config.weyl_restarts,
            self.config.weyl_perturb_scale,
        )

    def try_solve(
        self,
        prefix_inv,
        basis_inv,
        target_inv,
        *,
        step: int | None = 0,
        rng_seed: int | None = None,
    ) -> SegmentSolution:
        """Solve for local unitaries using two-stage optimization.

        Always returns a solution (generic numeric solver).
        """
        start_time = time.perf_counter()

        # Extract matrices from invariants
        prefix_op = np.array(prefix_inv.unitary, dtype=np.complex128)
        basis_gate = np.array(basis_inv.unitary, dtype=np.complex128)
        target = np.array(target_inv.unitary, dtype=np.complex128)

        j_prefix = jnp.asarray(prefix_op, dtype=jnp.complex128)
        j_gate = jnp.asarray(basis_gate, dtype=jnp.complex128)
        j_target = jnp.asarray(target, dtype=jnp.complex128)

        target_makhlin = makhlin_invariants(j_target)
        target_weyl = weyl_coordinates(j_target)

        # Stage 1: Makhlin exploration
        makhlin_params, makhlin_res = self._solve_makhlin(
            self._key,
            j_prefix,
            j_gate,
            target_makhlin,
            self.config.makhlin_conv_tol,
        )

        # Optional early check (skipped by default to avoid device sync)
        if (
            self.config.strict_convergence_checks
            and float(makhlin_res) > self.config.makhlin_conv_tol
        ):
            raise RuntimeError(
                f"Stage 1 (Makhlin) failed: best residual {float(makhlin_res):.2e}"
            )

        # Determine Weyl branch (direct vs reflected)
        U = _construct_unitary(makhlin_params, j_prefix, j_gate)
        constructed_weyl = weyl_coordinates(U)
        reflected_weyl = target_weyl * jnp.array([-1.0, 1.0, -1.0]) + jnp.array(
            [1.0, 0.0, 0.0]
        )

        direct_res = jnp.max(jnp.abs(constructed_weyl - target_weyl))
        reflect_res = jnp.max(jnp.abs(constructed_weyl - reflected_weyl))
        use_reflected = reflect_res < direct_res

        weyl_target = jnp.where(use_reflected, reflected_weyl, target_weyl)

        # Stage 2: Weyl polishing with warm start
        weyl_params, weyl_res = self._solve_weyl(
            self._key,
            j_prefix,
            j_gate,
            weyl_target,
            self.config.weyl_conv_tol,
            makhlin_params,
        )

        # Check residual (need this value for return anyway)
        weyl_res_py = float(weyl_res)
        if weyl_res_py > self.config.weyl_conv_tol:
            makhlin_res_py = float(makhlin_res)
            raise RuntimeError(
                f"Optimization failed: Stage 1 residual {makhlin_res_py:.2e}, "
                f"Stage 2 residual {weyl_res_py:.2e}"
            )

        u0, u1 = _params_to_unitaries(weyl_params)
        return SegmentSolution(
            u0=u0,
            u1=u1,
            max_residual=weyl_res_py,
            success=True,
            metadata={
                "stage": "weyl",
                "elapsed": time.perf_counter() - start_time,
            },
        )
