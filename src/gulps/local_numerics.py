import logging
import time

import jax.numpy as jnp
import numpy as np
from jax import config, jit
from jax.random import PRNGKey, uniform
from jaxopt import LevenbergMarquardt
from jaxopt.linear_solve import solve_cg
from qiskit import QuantumCircuit
from qiskit.circuit.library import RVGate, UnitaryGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import Operator

from gulps.utils.invariants import GateInvariants
from gulps.utils.recover_equiv import recover_local_equivalence

logger = logging.getLogger(__name__)

config.update("jax_enable_x64", True)

# jax setup and definitions
# NOTE these are highly-tunable parameters
# I believe two_qubit_local_invariants only has 8 decimal places of precision
CONV_TOL = 5e-9  # 1e-8
A_TOL = 1e-9

MAGIC = jnp.array(
    [[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]],
    dtype=jnp.cdouble,
) / jnp.sqrt(2)


@jit
def _two_qubit_local_invariants(U):
    from qiskit.synthesis.two_qubit.local_invariance import two_qubit_local_invariants

    Um = MAGIC.conj().T.dot(U.dot(MAGIC))
    det_um = jnp.complex128(jnp.linalg.det(Um))
    M = jnp.dot(Um.T, Um)
    t1 = jnp.trace(M)
    t1s = t1 * t1
    t2 = jnp.trace(M @ M)
    # t3 = jnp.trace(M @ M @ M)
    g1 = t1s / (16.0 * det_um)
    g2 = (t1s - t2) / (4.0 * det_um)
    return jnp.array([g1.real, g1.imag, g2.real], dtype=jnp.double)
    # # Orientation term --------------------------------------------
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


@jit
def _objective_function(
    x: jnp.ndarray,
    prefix_op: jnp.ndarray,
    basis_gate: jnp.ndarray,
    target_inv: jnp.ndarray,
):
    U = (
        basis_gate
        # @ jnp.kron(
        #     jnp.array([[0.0 + 0.0j, 0.0 - 1.0j], [0.0 + 1.0j, 0.0 + 0.0j]]),
        #     jnp.eye(2, dtype=jnp.cdouble),
        # )
        @ jnp.kron(_rv(x[:3]), _rv(x[3:6]))
        # @ jnp.kron(
        #     jnp.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]]),
        #     jnp.array([[0.0 + 0.0j, 0.0 - 1.0j], [0.0 + 1.0j, 0.0 + 0.0j]]),
        # )
        @ prefix_op
    )
    construct_inv = _two_qubit_local_invariants(U)
    # return (target_inv - construct_inv) ** 2
    return target_inv - construct_inv
    # return (target_inv - construct_inv) + 0.0 * target_inv


EASY_LM = LevenbergMarquardt(
    residual_fun=_objective_function,
    solver=solve_cg,
    xtol=1e-6,  # 1e-3
    gtol=1e-6,  # 1e-3
    damping_parameter=1e-6,  # 1e-4
    maxiter=128,
    tol=A_TOL,
    implicit_diff=False,
    materialize_jac=True,
    jit=True,
)

HARD_LM = LevenbergMarquardt(
    residual_fun=_objective_function,
    solver=solve_cg,
    implicit_diff=False,
    materialize_jac=True,
    jit=True,
    maxiter=2048,
    tol=0.0,  # never gives up until maxiter
)


class SegmentNumericSynthesizer:
    """Class for synthesizing segments of a two-qubit gate sequence using numeric methods.

    Requires having already determining sequence of intermedaite canonical invariants {C_i} and a basis gate sentence {G_i}.
    See section 3.B of https://arxiv.org/pdf/2505.00543
    """

    _segment_stats = []  # List of attempts

    @staticmethod
    def _segment_interior_solve(
        g_op,
        c_op,
        target,
        easy_restarts,
        hard_restarts,
        seed=0,
    ):
        start_time = time.time()
        j_target_inv = jnp.array(target, dtype=jnp.double)

        j_prefix = jnp.array(c_op, dtype=jnp.complex128)
        j_gate = jnp.array(g_op, dtype=jnp.complex128)

        total_nfev, success_nfev = 0, 0
        best_residual = float("inf")
        best_params = None
        success = False
        success_label = None
        success_attempt = None

        def run_attempts(j_lm, label: str, restarts: int, start_idx: int):
            nonlocal total_nfev, success_nfev, best_residual, best_params
            nonlocal success, success_label, success_attempt

            for i in range(restarts):
                attempt_idx = start_idx + i
                j_init = uniform(
                    PRNGKey(seed + attempt_idx),
                    shape=(6,),
                    minval=-2 * jnp.pi,
                    maxval=2 * jnp.pi,
                )

                j_attempt = j_lm.run(
                    j_init,
                    prefix_op=j_prefix,
                    basis_gate=j_gate,
                    target_inv=j_target_inv,
                )
                residual_array = j_attempt.state.residual.block_until_ready()
                residual_norm = j_attempt.state.value
                nfev = j_attempt.state.iter_num
                total_nfev += nfev

                if residual_norm < best_residual:
                    best_residual = residual_norm
                    best_params = j_attempt.params

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"[{label.upper()} {i + 1}/{restarts}] "
                        f"residual={residual_array} (‖residual‖={residual_norm:.2e}, nfev={nfev})"
                    )

                if jnp.all(jnp.abs(residual_array) <= CONV_TOL):
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"=> Success on [{label.upper()} {i + 1}] "
                            f"(componentwise |residual| ≤ {CONV_TOL:.1e})"
                        )
                    success_nfev += nfev
                    success = True
                    success_label = label
                    success_attempt = i + 1
                    return

        # Phase 1: Easy
        run_attempts(
            EASY_LM,
            "easy",
            restarts=easy_restarts,
            start_idx=0,
        )

        # Phase 2: Hard (only if easy failed)
        if not success:
            run_attempts(
                HARD_LM,
                "hard",
                restarts=hard_restarts,
                start_idx=easy_restarts,
            )

        # XXX, messy way to get an attribute from a class used statically
        SegmentNumericSynthesizer._segment_stats.append((success_attempt or -1))

        ##############
        if logger.isEnabledFor(logging.DEBUG):
            elapsed_time = time.time() - start_time
            if success:
                logger.debug(
                    f"✅ LM synthesis SUCCESS on {success_label.upper()} attempt {success_attempt} "
                    f"(residual={best_residual:.2e}, total_nfev={total_nfev}) in {elapsed_time:.3f}s"
                )
            else:
                logger.debug(
                    f"❌ LM synthesis FAILED after {easy_restarts + hard_restarts} attempts "
                    f"(total_nfev={total_nfev}, best residual={best_residual:.2e}) in {elapsed_time:.3f}s"
                )
        return np.array(best_params)

    # NOTE, in principle this could be computed in parallel
    @staticmethod
    def _synthesize_segments(
        gate_list: list[GateInvariants],
        invariant_list: list[GateInvariants],
        easy_attempts: int = 8,
        hard_attempts: int = 16,
    ) -> list[np.ndarray]:
        # gate list given as G1, G2, ..., Gn as Operator
        # invariant list given as C1, C2, ..., Cn as 3-tuple floats (monodromy)
        # assert C1 is G1 (and C0 is Identity)
        segment_sols = []

        # loop gi.v.ci-1
        for i in range(1, len(invariant_list)):
            g_op = gate_list[i].unitary
            c_op = (
                gate_list[0].unitary
                if i == 1
                else invariant_list[i - 1].canonical_matrix
            )
            # c_op = invariant_list[i - 1].canonical_matrix

            # XXX FIXME temporary, use function instaed of makhlin while I am
            # messing around with the definition of the invariants
            # target = invariant_list[i].makhlin
            target = _two_qubit_local_invariants(invariant_list[i].canonical_matrix)

            # solve for segment
            SegmentNumericSynthesizer._segment_stats = []
            segment_sol = SegmentNumericSynthesizer._segment_interior_solve(
                g_op,
                c_op,
                target,
                easy_restarts=easy_attempts,
                hard_restarts=hard_attempts,
            )
            segment_sols.append(segment_sol)

            # ######################################
            # # # XXX DEBUG only (slow)
            if (
                logger.isEnabledFor(logging.DEBUG)
                # and SegmentNumericSynthesizer._segment_stats[-1] == -1  # failed
            ):
                from qiskit._accelerate.two_qubit_decompose import weyl_coordinates

                # piece back to see what the segment looks like
                U = (
                    np.array(g_op)
                    @ np.kron(_rv(segment_sol[:3]), _rv(segment_sol[3:6]))
                    @ np.array(c_op)
                )
                U_inv = GateInvariants.from_unitary(U)
                logger.debug(f"constructed makh: {U_inv.makhlin}")
                logger.debug(f"target makh: {invariant_list[i].makhlin}")
                logger.debug(f"rho(t) makh: {invariant_list[i].rho_reflect.makhlin}")
                logger.debug(f"constructed weyl: {U_inv.weyl}")
                logger.debug(f"target weyl: {invariant_list[i].weyl}")
                logger.debug(f"rho(t) weyl: {invariant_list[i].rho_reflect.weyl}")
                logger.debug(f"constructed logspec: {U_inv.logspec}")
                logger.debug(f"target logspec: {invariant_list[i].logspec}")
                logger.debug(f"rho(t) logspec: {invariant_list[i].rho_reflect.logspec}")
            # # ######################################

        return segment_sols

    @staticmethod
    def _stitch_segments(
        gate_list: list[GateInvariants],
        invariant_list: list[GateInvariants],
        segment_sols: list[np.ndarray],  # [(v1…v6), …]      (inner-RV parameters)
        target: GateInvariants = None,  # final target unitary
        return_dag: bool = False,
    ) -> QuantumCircuit | DAGCircuit:
        """Piece together while recovering unitary equivalence.

        After `_synthesize_segments` gives the six angles for every interior RV pair,
        build a single circuit *in a DAG* such that after each segment the entire
        prefix equals CAN(C_i).  The outer locals are injected with
        `apply_operation_front` / `apply_operation_back`.
        """
        if not (len(gate_list) == len(invariant_list) == len(segment_sols) + 1):
            raise ValueError("len(gates) must equal len(invariants) = len(sols)+1")

        # blank two-qubit circuit → DAG
        base_qc = QuantumCircuit(2, global_phase=0)
        dag = circuit_to_dag(base_qc)
        qreg = base_qc.qregs[0]  # convenient handle to the two qubits

        # NOTE endianess  qreg[0:2] -> qreg[::-1]

        # first basis gate
        dag.apply_operation_back(gate_list[0].unitary, qreg[0:2])

        ## NOTE we skip local equiv recovery on first segment
        ## because c_op[0] is already gate_list[0].unitary
        # can_op = invariant_list[0].canonical_matrix
        # current_op = Operator(dag_to_circuit(dag)).to_matrix()
        # k1, k2, k3, k4, gphase = recover_local_equivalence(can_op, current_op)
        # dag.global_phase += gphase
        # # prepend k1 tensor k2
        # dag.apply_operation_front(UnitaryGate(k1), [qreg[0]])
        # dag.apply_operation_front(UnitaryGate(k2), [qreg[1]])
        # # append  k3 tensor k4
        # dag.apply_operation_back(UnitaryGate(k3), [qreg[0]])
        # dag.apply_operation_back(UnitaryGate(k4), [qreg[1]])

        # iterate over every remaining segment
        for idx, params in enumerate(segment_sols, start=1):
            # (a) inner local RVs
            rv0 = RVGate(*params[:3])
            rv1 = RVGate(*params[3:6])

            dag.apply_operation_back(rv0, [qreg[1]])
            dag.apply_operation_back(rv1, [qreg[0]])

            # (b) corresponding two-qubit basis gate Gi
            dag.apply_operation_back(gate_list[idx].unitary, qreg[0:2])

            # (c) recover local equivalence → CAN(C_i)
            # XXX TODO, unnecessary dag conversions?
            # can_op = invariant_list[idx].canonical_matrix
            # TODO, move this inside recover_local_equivalence?
            current_op = Operator(dag_to_circuit(dag)).to_matrix()
            current_inv = GateInvariants.from_unitary(current_op)
            can_inv = invariant_list[idx]
            can_op = can_inv.canonical_matrix
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"starting from weyl: {current_inv.weyl}")
                logger.debug(f"recovering to weyl {can_inv.weyl} ")
            # if np.allclose(can_inv.monodromy, current_inv.monodromy, rtol=1e-9):
            #     can_op = can_inv.canonical_matrix
            #     logger.debug(f"recovering to weyl {can_inv.weyl}")
            # elif np.allclose(
            #     can_inv.rho_reflect.monodromy, current_inv.monodromy, rtol=1e-9
            # ):
            #     can_op = can_inv.rho_reflect.canonical_matrix
            #     logger.debug(f"recovering to rho weyl {can_inv.rho_reflect.weyl}")
            # else:
            #     raise ValueError(
            #         f"Current invariant {current_inv} does not match "
            #         f"target invariant {can_inv} or its rho-reflect."
            #     )

            # on last iteration, recover to decomposition target instead of CAN
            if idx == len(segment_sols) and target is not None:
                k1, k2, k3, k4, gphase = recover_local_equivalence(
                    target.unitary, current_op
                )

            else:
                k1, k2, k3, k4, gphase = recover_local_equivalence(can_op, current_op)

            if logger.isEnabledFor(logging.DEBUG):
                after_recovery_op = Operator(dag_to_circuit(dag)).to_matrix()
                after_recovery_inv = GateInvariants.from_unitary(after_recovery_op)
                logger.debug(f"finished at weyl {after_recovery_inv.weyl} ")

            dag.global_phase += gphase
            # prepend k1 tensor k2
            dag.apply_operation_front(UnitaryGate(k1), [qreg[0]])
            dag.apply_operation_front(UnitaryGate(k2), [qreg[1]])
            # append  k3 tensor k4
            dag.apply_operation_back(UnitaryGate(k3), [qreg[0]])
            dag.apply_operation_back(UnitaryGate(k4), [qreg[1]])
            # after this, the full prefix equals CAN(C_i) – ready for next loop

        if return_dag:
            return dag
        return dag_to_circuit(dag)
