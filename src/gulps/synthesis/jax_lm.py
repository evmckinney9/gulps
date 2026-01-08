import time
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import config, device_get, jit
from jax.random import PRNGKey, split, uniform
from jaxopt import GaussNewton, LevenbergMarquardt
from jaxopt.linear_solve import solve_lu

from gulps.config import GulpsConfig
from gulps.core.jax_invariants import get_invariant_function
from gulps.synthesis.segments_abc import SegmentSolution, SegmentSolver

# Pauli matrices
X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
I2 = jnp.eye(2, dtype=jnp.complex128)


@jit
def _rv(v: jnp.ndarray) -> jnp.ndarray:
    """Rotation vector to SU(2) unitary via Rodriguez formula."""
    a = jnp.linalg.norm(v)
    half = 0.5 * a
    s = jnp.where(
        jnp.abs(half) < 1e-8,
        1.0 - (half * half) / 6.0 + (half**4) / 120.0,
        jnp.sin(half) / half,
    )
    c = jnp.cos(half)
    vx, vy, vz = v
    H = vx * X + vy * Y + vz * Z
    return c * I2 - 1j * (0.5 * s) * H


@jit
def kron2(A, B):
    """Kronecker product for 2x2 matrices."""
    return jnp.einsum("ab,cd->acbd", A, B).reshape(4, 4)


@jit
def _construct_unitary(params, prefix_op, basis_gate):
    """Construct U = G · (u1 ⊗ u0) · C from 6D rotation vector params."""
    return basis_gate @ kron2(_rv(params[3:6]), _rv(params[:3])) @ prefix_op


def _params_to_locals(params: jnp.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert 6D rotation vector params to two 2x2 unitaries."""
    u0 = _rv(params[3:6])
    u1 = _rv(params[:3])
    return device_get(u0), device_get(u1)


def _make_residual_fn(invariant_fn):
    """Create residual function: target_inv - constructed_inv."""

    @jit
    def residual(x, prefix_op, basis_gate, target_inv):
        U = _construct_unitary(x, prefix_op, basis_gate)
        return target_inv - invariant_fn(U)

    return residual


class JaxLMSegmentSolver(SegmentSolver):
    """Two-stage Makhlin→Weyl segment solver."""

    def __init__(self, config: GulpsConfig | None = None):
        self.config = config or GulpsConfig()

        self._makhlin_fn = get_invariant_function("makhlin")
        self._weyl_fn = get_invariant_function("weyl")

        self._makhlin_solver = GaussNewton(
            residual_fun=_make_residual_fn(self._makhlin_fn),
            maxiter=self.config.makhlin_maxiter,
            tol=self.config.segment_solver_tol,
            implicit_diff=False,
            implicit_diff_solve=solve_lu,
            jit=True,
        )

        self._weyl_solver = LevenbergMarquardt(
            residual_fun=_make_residual_fn(self._weyl_fn),
            maxiter=self.config.weyl_maxiter,
            tol=self.config.segment_solver_tol,
            implicit_diff=False,
            jit=True,
            materialize_jac=True,
            solver=solve_lu,
        )

    def solve_segment(
        self,
        prefix_op: np.ndarray,
        basis_gate: np.ndarray,
        target: np.ndarray,
        *,
        rng_seed: int | None = None,
    ) -> SegmentSolution:
        """Solve for local unitaries using two-stage optimization."""
        start_time = time.perf_counter()

        j_prefix = jnp.asarray(prefix_op, dtype=jnp.complex128)
        j_gate = jnp.asarray(basis_gate, dtype=jnp.complex128)
        j_target = jnp.asarray(target, dtype=jnp.complex128)

        target_makhlin = self._makhlin_fn(j_target)
        target_weyl = self._weyl_fn(j_target)

        makhlin_conv_tol = self.config.makhlin_conv_tol
        weyl_conv_tol = self.config.weyl_conv_tol

        key = PRNGKey(rng_seed or 0)
        keys = split(key, self.config.makhlin_restarts + self.config.weyl_restarts)

        # Stage 1: Makhlin exploration
        best_params = None
        best_res = float("inf")  # Keep as Python float
        total_nfev = 0

        for i in range(self.config.makhlin_restarts):
            init = uniform(keys[i], shape=(6,), minval=-jnp.pi / 2, maxval=jnp.pi / 2)
            result = self._makhlin_solver.run(
                init, prefix_op=j_prefix, basis_gate=j_gate, target_inv=target_makhlin
            )
            res_jax = jnp.max(jnp.abs(result.state.residual))
            total_nfev += result.state.iter_num

            # Single device transfer per iteration
            res = float(device_get(res_jax))

            if res < best_res:
                best_res = res
                best_params = result.params

            if res <= makhlin_conv_tol:
                break

        # Check convergence (best_res already on CPU)
        if best_res > makhlin_conv_tol:
            raise RuntimeError(
                f"Stage 1 (Makhlin) failed: best residual {best_res:.2e}"
            )

        # Determine Weyl branch
        constructed_weyl = self._weyl_fn(
            _construct_unitary(best_params, j_prefix, j_gate)
        )

        # Check both branches (single device transfer for comparison)
        direct_res = jnp.max(jnp.abs(constructed_weyl - target_weyl))
        reflected = jnp.array([1.0 - target_weyl[0], target_weyl[1], -target_weyl[2]])
        reflect_res = jnp.max(jnp.abs(constructed_weyl - reflected))

        # Transfer both to CPU, decide, then keep target on device
        direct_val, reflect_val = (
            float(device_get(direct_res)),
            float(device_get(reflect_res)),
        )
        use_reflected = reflect_val < direct_val
        weyl_target = reflected if use_reflected else target_weyl
        weyl_res_val = reflect_val if use_reflected else direct_val

        # Early exit if already converged
        if weyl_res_val <= weyl_conv_tol:
            u0, u1 = _params_to_locals(best_params)
            return SegmentSolution(
                u0=u0,
                u1=u1,
                max_residual=weyl_res_val,
                success=True,
                metadata={
                    "stage": "makhlin",
                    "nfev": int(device_get(total_nfev)),
                    "elapsed": time.perf_counter() - start_time,
                    "use_reflected": use_reflected,
                },
            )

        # Stage 2: Weyl polishing
        warm_start = best_params
        weyl_best_params = None
        weyl_best_res = float("inf")  # Keep as Python float

        for i in range(self.config.weyl_restarts):
            if i == 0:
                init = warm_start
            else:
                perturb = uniform(
                    keys[self.config.makhlin_restarts + i],
                    shape=(6,),
                    minval=-self.config.weyl_perturb_scale,
                    maxval=self.config.weyl_perturb_scale,
                )
                init = warm_start + perturb

            result = self._weyl_solver.run(
                init, prefix_op=j_prefix, basis_gate=j_gate, target_inv=weyl_target
            )
            res_jax = jnp.max(jnp.abs(result.state.residual))
            total_nfev += result.state.iter_num

            # Single device transfer per iteration
            res = float(device_get(res_jax))

            if res < weyl_best_res:
                weyl_best_res = res
                weyl_best_params = result.params

            if res <= weyl_conv_tol:
                break

        # Check convergence (weyl_best_res already on CPU)
        if weyl_best_res > weyl_conv_tol:
            raise RuntimeError(
                f"Stage 2 (Weyl) failed: best residual {weyl_best_res:.2e}"
            )

        u0, u1 = _params_to_locals(weyl_best_params)
        return SegmentSolution(
            u0=u0,
            u1=u1,
            max_residual=weyl_best_res,
            success=True,
            metadata={
                "stage": "weyl",
                "nfev": int(device_get(total_nfev)),
                "elapsed": time.perf_counter() - start_time,
                "use_reflected": use_reflected,
            },
        )
