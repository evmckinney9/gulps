import logging
import time
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import config, jit
from jax.random import PRNGKey, split, uniform
from jaxopt import LevenbergMarquardt
from jaxopt.linear_solve import solve_cg

from gulps.synthesis.segments_abc import SegmentSolution, SegmentSolver

logger = logging.getLogger(__name__)

config.update("jax_enable_x64", True)

# jax setup and definitions
# NOTE these are highly-tunable parameters
# I believe two_qubit_local_invariants only has 8 decimal places of precision
CONV_TOL = 5e-9
A_TOL = 1e-9

MAGIC = jnp.array(
    [[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]],
    dtype=jnp.cdouble,
) / jnp.sqrt(2)


@jit
def _two_qubit_local_invariants(U):
    # from qiskit.synthesis.two_qubit.local_invariance import two_qubit_local_invariants
    Um = MAGIC.conj().T.dot(U.dot(MAGIC))
    det_um = jnp.complex128(jnp.linalg.det(Um))
    M = jnp.dot(Um.T, Um)
    t1 = jnp.trace(M)
    t1s = t1 * t1
    t2 = jnp.trace(M @ M)
    g1 = t1s / (16.0 * det_um)
    g2 = (t1s - t2) / (4.0 * det_um)
    return jnp.array([g1.real, g1.imag, g2.real], dtype=jnp.double)
    # # Orientation term --------------------------------------------
    # t3 = jnp.trace(M @ M @ M)
    # Continuous orientation moment  (breaks the symmetry)
    # delta_im = jnp.imag(t1**3 - 3.0 * t1 * t2 + 2.0 * t3)
    # # Normalise to match g-scales (~1): divide by 16*|det_um|
    # g4 = 1e-6 * delta_im / (16.0 * jnp.abs(det_um))
    # return jnp.array([g1.real, g1.imag, g2.real, g4], dtype=jnp.double)


@jit
def _rv(v: jnp.ndarray) -> jnp.ndarray:
    """Rotation-vector (RV) single-qubit gate, safe for ‖v‖≈0 and tracer-friendly."""
    angle = jnp.linalg.norm(v)  # scalar tracer
    safe_angle = jnp.where(angle < 1e-16, 1.0, angle)  # never zero
    nx, ny, nz = v / safe_angle

    sin = jnp.sin(angle / 2)
    cos = jnp.cos(angle / 2)

    rot = jnp.array(
        [
            [cos - 1j * nz * sin, (-ny - 1j * nx) * sin],
            [(ny - 1j * nx) * sin, cos + 1j * nz * sin],
        ],
        dtype=jnp.cdouble,
    )

    # If the vector’s length is (numerically) zero, return the identity instead.
    return jnp.where(angle < 1e-12, jnp.eye(2, dtype=jnp.cdouble), rot)


def _params_to_locals(params: jnp.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Helper to turn RV parameters into two 2x2 unitaries."""
    u0 = _rv(params[:3])
    u1 = _rv(params[3:6])
    return np.array(u0), np.array(u1)


@jit
def _objective_function(
    x: jnp.ndarray,
    prefix_op: jnp.ndarray,
    basis_gate: jnp.ndarray,
    target_inv: jnp.ndarray,
):
    U = basis_gate @ jnp.kron(_rv(x[:3]), _rv(x[3:6])) @ prefix_op
    construct_inv = _two_qubit_local_invariants(U)
    # return (target_inv - construct_inv) ** 2
    return target_inv - construct_inv


@dataclass(frozen=True)
class JaxLMConfig:
    easy_restarts: int = 8
    hard_restarts: int = 16
    conv_tol: float = CONV_TOL
    # could add xtol/gtol/etc here later


@dataclass
class _AttemptResult:
    params: jnp.ndarray | None
    residual_norm: float
    converged: bool
    label: str
    attempt_index: int  # 1-based within that phase
    nfev: int


class JaxLMSegmentSolver(SegmentSolver):
    """JAX+LM-backed segment solver with a simple restart policy.

    Policy is controlled entirely by JaxLMConfig and the `phases` list in solve_segment.
    """

    def __init__(self, config: JaxLMConfig | None = None):
        self.config = config or JaxLMConfig()

        # These are instance attributes instead of module globals.
        self._easy_lm = LevenbergMarquardt(
            residual_fun=_objective_function,
            solver=solve_cg,
            xtol=1e-6,
            gtol=1e-6,
            damping_parameter=1e-6,
            maxiter=128,
            tol=A_TOL,
            implicit_diff=False,
            materialize_jac=True,
            jit=True,
        )
        self._hard_lm = LevenbergMarquardt(
            residual_fun=_objective_function,
            solver=solve_cg,
            maxiter=2048,
            tol=0.0,  # never gives up until maxiter
            implicit_diff=False,
            materialize_jac=True,
            jit=True,
        )

    def _run_one_attempt(
        self,
        lm: LevenbergMarquardt,
        label: str,
        attempt_index: int,
        key: PRNGKey,
        prefix: jnp.ndarray,
        gate: jnp.ndarray,
        target_inv: jnp.ndarray,
    ) -> _AttemptResult:
        """Run a single LM attempt with random init from `key`."""
        j_init = uniform(
            key,
            shape=(6,),
            minval=-2 * jnp.pi,
            maxval=2 * jnp.pi,
        )

        j_attempt = lm.run(
            j_init,
            prefix_op=prefix,
            basis_gate=gate,
            target_inv=target_inv,
        )
        # We need residual_array to test convergence; this blocks anyway.
        residual_array = j_attempt.state.residual.block_until_ready()
        residual_norm = float(j_attempt.state.value)
        nfev = int(j_attempt.state.iter_num)

        conv = bool(jnp.all(jnp.abs(residual_array) <= self.config.conv_tol))

        return _AttemptResult(
            params=j_attempt.params,
            residual_norm=residual_norm,
            converged=conv,
            label=label,
            attempt_index=attempt_index,
            nfev=nfev,
        )

    def solve_segment(
        self,
        prefix_op: np.ndarray,
        basis_gate: np.ndarray,
        target: np.ndarray,
        *,
        rng_seed: int | None = None,
    ) -> SegmentSolution:
        """Solve for RV parameters for a single segment.

        prefix_op, basis_gate, target are 4x4 unitary matrices.
        We convert `target` once to invariant coordinates and then optimize.
        """
        rng_seed_base = 0 if rng_seed is None else rng_seed

        j_prefix = jnp.array(prefix_op, dtype=jnp.complex128)
        j_gate = jnp.array(basis_gate, dtype=jnp.complex128)
        j_target_inv = _two_qubit_local_invariants(jnp.array(target, dtype=jnp.cdouble))

        # Policy is now expressed as data:
        phases = [
            (self._easy_lm, "easy", self.config.easy_restarts),
            (self._hard_lm, "hard", self.config.hard_restarts),
        ]

        best_result: _AttemptResult | None = None
        total_attempts = sum(restarts for _, _, restarts in phases)
        total_nfev = 0

        start_time = time.perf_counter()

        for lm, label, restarts in phases:
            for local_idx in range(restarts):
                # Derive a fresh subkey for each attempt
                subkey = PRNGKey(rng_seed_base + total_nfev + local_idx)
                attempt = self._run_one_attempt(
                    lm=lm,
                    label=label,
                    attempt_index=local_idx + 1,
                    key=subkey,
                    prefix=j_prefix,
                    gate=j_gate,
                    target_inv=j_target_inv,
                )
                total_nfev += attempt.nfev

                # Update global best (even if not converged)
                if (
                    best_result is None
                    or attempt.residual_norm < best_result.residual_norm
                ):
                    best_result = attempt

                # Early exit on first converged attempt
                if attempt.converged:
                    elapsed = time.perf_counter() - start_time
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"Segment solver SUCCESS [{attempt.label} attempt {attempt.attempt_index}] "
                            f"(residual={attempt.residual_norm:.2e}, "
                            f"nfev={total_nfev}) in {elapsed:.3f}s"
                        )
                    u0, u1 = _params_to_locals(attempt.params)
                    return SegmentSolution(
                        u0=u0,
                        u1=u1,
                        residual_norm=attempt.residual_norm,
                        success=True,
                        metadata={
                            "label": attempt.label,
                            "attempt": attempt.attempt_index,
                            "nfev": total_nfev,
                            "elapsed": elapsed,
                        },
                    )

        # No converged attempt → return best attempt marked as failure
        elapsed = time.perf_counter() - start_time
        assert best_result is not None  # there was at least one attempt

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Segment solver FAILED after {total_attempts} attempts "
                f"(best residual={best_result.residual_norm:.2e}, "
                f"nfev={total_nfev}) in {elapsed:.3f}s"
            )

        if best_result.params is not None:
            u0, u1 = _params_to_locals(best_result.params)
        else:
            u0 = u1 = None
        return SegmentSolution(
            u0=u0,
            u1=u1,
            residual_norm=best_result.residual_norm,
            success=False,
            metadata={
                "label": best_result.label,
                "attempt": best_result.attempt_index,
                "nfev": total_nfev,
                "elapsed": elapsed,
            },
        )
