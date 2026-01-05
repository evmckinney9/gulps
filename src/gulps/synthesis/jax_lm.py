import logging
import time
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import config, device_get, jit
from jax.random import PRNGKey, split, uniform
from jaxopt import GaussNewton, LevenbergMarquardt
from jaxopt.linear_solve import solve_lu

from gulps.synthesis.segments_abc import SegmentSolution, SegmentSolver

logger = logging.getLogger(__name__)

config.update("jax_enable_x64", True)

# jax setup and definitions
# NOTE these are highly-tunable parameters
# I believe two_qubit_local_invariants only has 8 decimal places of precision
CONV_TOL = 1e-9
A_TOL = 1e-10

MAGIC = jnp.array(
    [[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]],
    dtype=jnp.complex128,
) / jnp.sqrt(2)
MAGIC_DAG = MAGIC.conj().T  # precompute once


@jit
def _two_qubit_local_invariants(U):
    # from qiskit.synthesis.two_qubit.local_invariance import two_qubit_local_invariants
    Um = MAGIC_DAG @ (U @ MAGIC)
    det_um = jnp.linalg.det(Um)
    det_um = det_um / jnp.abs(det_um)
    det_um = 1.0  # jnp.linalg.det(Um) #XXX enforce this earlier?
    M = Um.T @ Um
    t1 = jnp.trace(M)
    t1s = t1 * t1
    t2 = jnp.trace(M @ M)
    g1 = t1s / (16.0 * det_um)
    g2 = (t1s - t2) / (4.0 * det_um)
    return jnp.array([jnp.real(g1), jnp.imag(g1), jnp.real(g2)], dtype=jnp.float64)


X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
I2 = jnp.eye(2, dtype=jnp.complex128)


@jit
def _rv(v: jnp.ndarray) -> jnp.ndarray:
    a = jnp.linalg.norm(v)
    half = 0.5 * a

    # stable sinc(half) = sin(half)/half
    s = jnp.where(
        jnp.abs(half) < 1e-6,
        1.0 - (half * half) / 6.0 + (half**4) / 120.0,
        jnp.sin(half) / half,
    )
    c = jnp.cos(half)

    vx, vy, vz = v
    H = vx * X + vy * Y + vz * Z
    return c * I2 - 1j * (0.5 * s) * H


def _params_to_locals(params: jnp.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Helper to turn RV parameters into two 2x2 unitaries."""
    u0 = _rv(params[3:6])
    u1 = _rv(params[:3])
    return device_get(u0), device_get(u1)


@jit
def kron2(A, B):
    # (2,2) ⊗ (2,2) -> (4,4)
    return jnp.einsum("ab,cd->acbd", A, B).reshape(4, 4)


@jit
def _objective_function(
    x: jnp.ndarray,
    prefix_op: jnp.ndarray,
    basis_gate: jnp.ndarray,
    target_inv: jnp.ndarray,
):
    U = basis_gate @ kron2(_rv(x[3:6]), _rv(x[:3])) @ prefix_op
    construct_inv = _two_qubit_local_invariants(U)
    return target_inv - construct_inv


@dataclass(frozen=True)
class JaxLMConfig:
    easy_restarts: int = 8
    hard_restarts: int = 4
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
        self._easy_lm = GaussNewton(
            residual_fun=_objective_function,
            maxiter=128,
            tol=A_TOL,
            implicit_diff=False,
            jit=True,
        )
        self._hard_lm = LevenbergMarquardt(
            residual_fun=_objective_function,
            solver=solve_lu,
            maxiter=1024,
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
            minval=-jnp.pi / 2,
            maxval=jnp.pi / 2,
        )

        j_attempt = lm.run(
            j_init,
            prefix_op=prefix,
            basis_gate=gate,
            target_inv=target_inv,
        )
        # # We need residual_array to test convergence; this blocks anyway.
        # residual_array = j_attempt.state.residual.block_until_ready()
        # residual_norm = float(j_attempt.state.value)
        # conv = bool(jnp.all(jnp.abs(residual_array) <= self.config.conv_tol))

        # No residual array readback; use objective value only.
        residual = j_attempt.state.residual  # shape (3,)
        conv = jnp.max(jnp.abs(residual)) <= self.config.conv_tol
        residual_norm = jnp.linalg.norm(residual)  # this is a real norm for logging
        nfev = int(j_attempt.state.iter_num)

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
        j_target_inv = _two_qubit_local_invariants(
            jnp.array(target, dtype=jnp.complex128)
        )
        phases = [
            (self._easy_lm, "easy", self.config.easy_restarts),
            (self._hard_lm, "hard", self.config.hard_restarts),
        ]

        best_result: _AttemptResult | None = None
        total_attempts = sum(restarts for _, _, restarts in phases)
        total_nfev = 0
        # Pre-split all attempt keys up front.
        base_key = PRNGKey(rng_seed_base)
        attempt_keys = split(base_key, total_attempts)
        key_cursor = 0  # walks through attempt_keys
        start_time = time.perf_counter()

        for lm, label, restarts in phases:
            for local_idx in range(restarts):
                # Derive a fresh subkey for each attempt
                subkey = attempt_keys[key_cursor]
                key_cursor += 1
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
                            f"(residual norm={attempt.residual_norm:.2e}, "
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
                f"(best residual norm={best_result.residual_norm:.2e}, "
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
