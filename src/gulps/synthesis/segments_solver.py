import logging
from typing import List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RVGate, UnitaryGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import Operator

from gulps.core.invariants import GateInvariants
from gulps.synthesis.recover_equiv import recover_local_equivalence
from gulps.synthesis.segments_abc import SegmentSolution, SegmentSolver

logger = logging.getLogger(__name__)


class SegmentSynthesizer:
    """Orchestrate segment-wise numeric synthesis and circuit stitching.

    This class:
      1. Breaks the invariant sequence {C_i} and gate sequence {G_i} into segments.
      2. Calls a pluggable SegmentSolver per segment.
      3. Stitches the resulting locals into a Qiskit circuit.

    Requires having already determining sequence of intermediate canonical invariants {C_i} and a basis gate sentence {G_i}.
    See section 3.B of https://arxiv.org/pdf/2505.00543 for details.
    """

    def __init__(self, solver: SegmentSolver):
        self._solver = solver
        self._segment_stats: list[SegmentSolution] = []

    def synthesize_segments(
        self,
        gate_list: List[GateInvariants],
        invariant_list: List[GateInvariants],
    ) -> list[np.ndarray]:
        """Solve for local parameters of each interior segment.

        Returns a list of parameter vectors (v1…v6) for each segment.
        """
        if len(gate_list) != len(invariant_list):
            raise ValueError("Gate list and invariant list must have the same length.")

        if len(gate_list) < 2:
            raise ValueError("At least two gates are required for segment synthesis.")

        segment_sols: list[SegmentSolution] = []
        self._segment_stats.clear()

        for i in range(1, len(invariant_list)):
            g_op = gate_list[i].unitary
            c_op = (
                gate_list[0].unitary
                if i == 1
                else invariant_list[i - 1].canonical_matrix
            )
            target = invariant_list[i].canonical_matrix

            sol = self._solver.solve_segment(
                prefix_op=c_op,
                basis_gate=g_op,
                target=target,
                rng_seed=i,
            )
            self._segment_stats.append(sol)

            if not sol.success:
                raise RuntimeError(
                    f"Segment {i} synthesis failed (residual norm={sol.residual_norm:.2e})."
                )

            segment_sols.append(sol)

            if logger.isEnabledFor(logging.DEBUG):
                self._debug_segment(i, g_op, c_op, sol, invariant_list[i])

        return segment_sols

    def stitch_segments(
        self,
        gate_list: list[GateInvariants],
        invariant_list: list[GateInvariants],
        segment_sols: list[SegmentSolution],
        target: GateInvariants | None = None,
        return_dag: bool = False,
    ) -> QuantumCircuit | DAGCircuit:
        """Piece together the circuit while recovering unitary equivalence."""
        if not (len(gate_list) == len(invariant_list) == len(segment_sols) + 1):
            raise ValueError("len(gates) must equal len(invariants) = len(sols)+1")

        base_qc = QuantumCircuit(2, global_phase=0)
        dag = circuit_to_dag(base_qc)
        qreg = base_qc.qregs[0]

        dag.apply_operation_back(gate_list[0].unitary, qreg[0:2])

        for idx, sol in enumerate(segment_sols, start=1):
            if sol.u0 is None or sol.u1 is None:
                raise RuntimeError(f"Missing locals for segment {idx}")

            # NOTE, endianess is flipping here
            dag.apply_operation_back(UnitaryGate(sol.u0), [qreg[1]])
            dag.apply_operation_back(UnitaryGate(sol.u1), [qreg[0]])

            dag.apply_operation_back(gate_list[idx].unitary, qreg[0:2])

            current_op = Operator(dag_to_circuit(dag)).to_matrix()
            current_inv = GateInvariants.from_unitary(current_op)
            can_inv = invariant_list[idx]
            can_op = can_inv.canonical_matrix

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"[segment {idx}] starting from weyl {current_inv.weyl}, "
                    f"recovering to {can_inv.weyl}"
                )

            if idx == len(segment_sols) and target is not None:
                k1, k2, k3, k4, gphase = recover_local_equivalence(
                    target.unitary, current_op
                )
            else:
                k1, k2, k3, k4, gphase = recover_local_equivalence(can_op, current_op)

            dag.global_phase += gphase
            dag.apply_operation_front(UnitaryGate(k1), [qreg[0]])
            dag.apply_operation_front(UnitaryGate(k2), [qreg[1]])
            dag.apply_operation_back(UnitaryGate(k3), [qreg[0]])
            dag.apply_operation_back(UnitaryGate(k4), [qreg[1]])

        if return_dag:
            return dag
        return dag_to_circuit(dag)

    def _debug_segment(
        self,
        i: int,
        g_op: np.ndarray,
        c_op: np.ndarray,
        sol: SegmentSolution,
        target_inv: GateInvariants,
    ) -> None:
        """Optional debugging reconstruction to log invariants."""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        U = np.array(g_op) @ np.kron(sol.u0, sol.u1) @ np.array(c_op)
        U_inv = GateInvariants.from_unitary(U)
        logger.debug(f"[segment {i}] constructed makh: {U_inv.makhlin}")
        logger.debug(f"[segment {i}] target makh: {target_inv.makhlin}")
        logger.debug(f"[segment {i}] constructed weyl: {U_inv.weyl}")
        logger.debug(f"[segment {i}] target weyl: {target_inv.weyl}")
        logger.debug(
            f"[segment {i}] construct rho(makhlin): {U_inv.rho_reflect.makhlin}"
        )
        logger.debug(
            f"[segment {i}] target rho(makhlin): {target_inv.rho_reflect.makhlin}"
        )
