"""Segment-wise synthesis orchestrator with circuit stitching."""

import logging
from typing import List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit

from gulps.config import GulpsConfig
from gulps.core.invariants import GateInvariants
from gulps.synthesis.jax_lm import JaxLMSegmentSolver
from gulps.synthesis.recover_equiv import recover_local_equivalence
from gulps.synthesis.segments_cache import SegmentCache

logger = logging.getLogger(__name__)


class SegmentSynthesizer:
    """Orchestrate segment-wise synthesis and circuit stitching.

    Solves all segments independently using canonical placeholders, then stitches.
    Each solve assumes an idealized prefix that doesn't exactly match the accumulated
    circuit, so small errors compound. All drift is absorbed by the final recovery step.
    """

    def __init__(self, config: GulpsConfig | None = None):
        """Initialize solvers and cache from config."""
        self.config = config or GulpsConfig()
        self._cache = SegmentCache(max_entries_per_step=self.config.segment_cache_size)
        self._jax_lm_solver = JaxLMSegmentSolver(config=self.config)

    def synthesize_segments(
        self,
        gate_list: List[GateInvariants],
        invariant_list: List[GateInvariants],
        target: GateInvariants,
        *,
        return_dag: bool = False,
    ) -> QuantumCircuit | DAGCircuit:
        """Solve for local parameters of each interior segment.

        Args:
            gate_list: Sequence of basis gates {G_i} with precomputed invariants.
            invariant_list: Sequence of target invariants {C_i} for each segment.
            target: Exact target unitary for final recovery.
            return_dag: If True, return DAGCircuit instead of QuantumCircuit.

        Returns:
            Synthesized circuit implementing the target unitary.

        Raises:
            ValueError: If gate_list and invariant_list have mismatched lengths,
                or fewer than 2 gates provided.
            RuntimeError: If any segment synthesis fails.
        """
        if len(gate_list) != len(invariant_list):
            raise ValueError("Gate list and invariant list must have the same length.")

        if len(gate_list) < 1:
            raise ValueError("At least one gate is required for segment synthesis.")

        dag = circuit_to_dag(QuantumCircuit(2, global_phase=0))
        qreg = dag.qregs["q"]

        P = self._synthesize_batch(dag, qreg, gate_list, invariant_list)

        # Final recovery to true target
        k1, k2, k3, k4, gphase = recover_local_equivalence(
            target.unitary, P, config=self.config
        )
        dag.global_phase += gphase
        dag.apply_operation_front(UnitaryGate(k1, check_input=False), [qreg[0]])
        dag.apply_operation_front(UnitaryGate(k2, check_input=False), [qreg[1]])
        dag.apply_operation_back(UnitaryGate(k3, check_input=False), [qreg[0]])
        dag.apply_operation_back(UnitaryGate(k4, check_input=False), [qreg[1]])

        return dag if return_dag else dag_to_circuit(dag)

    def _synthesize_batch(self, dag, qreg, gate_list, invariant_list):
        """Solve segments and stitch with intermediate recovery.

        Each segment is solved independently using the canonical matrix of the
        previous segment as the prefix. During stitching, intermediate recovery
        (k3, k4 only) snaps the accumulated unitary back toward canonical.

        Uses per-step caching to avoid recomputing identical segments across
        decompositions.
        """
        # Initialize with first gate
        dag.apply_operation_back(gate_list[0].unitary, qreg[:])
        P = gate_list[0].unitary.to_matrix()
        cache = self._cache
        jax_lm = self._jax_lm_solver
        num_segments = len(invariant_list)

        # Solve and stitch each segment
        # Segment i (1-indexed): append g_i to c_{i-1} to reach c_i
        use_cache = True
        for i in range(1, num_segments):
            step = i - 1
            prefix_inv = gate_list[0] if i == 1 else invariant_list[i - 1]
            basis_inv = gate_list[i]
            target_inv = invariant_list[i]

            # Cache lookup: once it misses, downstream prefixes diverge
            # (LP intermediates are target-dependent) so skip future lookups.
            seg_sol = None
            if use_cache:
                seg_sol = cache.try_solve(prefix_inv, basis_inv, target_inv, step=step)
                if seg_sol is None:
                    use_cache = False

            if seg_sol is None:
                seg_sol = jax_lm.try_solve(prefix_inv, basis_inv, target_inv, step=step)
                if seg_sol is not None and seg_sol.success:
                    cache.put(step, prefix_inv, basis_inv, target_inv, seg_sol)

            if seg_sol is None or not seg_sol.success:
                residual = seg_sol.max_residual if seg_sol else float("inf")
                raise RuntimeError(
                    f"Segment {i} synthesis failed (residual norm={residual:.2e})."
                )

            # Stitch solution into circuit
            u0, u1 = seg_sol.u0, seg_sol.u1
            dag.apply_operation_back(UnitaryGate(u0, check_input=False), [qreg[0]])
            dag.apply_operation_back(UnitaryGate(u1, check_input=False), [qreg[1]])
            dag.apply_operation_back(basis_inv.unitary, qreg[:])

            # Update accumulated unitary
            u1u0 = np.empty((4, 4), dtype=np.complex128)
            u1u0[0:2, 0:2] = u1[0, 0] * u0
            u1u0[0:2, 2:4] = u1[0, 1] * u0
            u1u0[2:4, 0:2] = u1[1, 0] * u0
            u1u0[2:4, 2:4] = u1[1, 1] * u0
            P = np.asarray(basis_inv.unitary, dtype=np.complex128) @ u1u0 @ P

            # Intermediate recovery (skip for final segment)
            if i < num_segments - 1:
                Ci = np.asarray(target_inv.canonical_matrix, dtype=np.complex128)
                _, _, k3, k4, gphase = recover_local_equivalence(
                    Ci, P, config=self.config
                )
                dag.global_phase += gphase
                dag.apply_operation_back(UnitaryGate(k3, check_input=False), [qreg[0]])
                dag.apply_operation_back(UnitaryGate(k4, check_input=False), [qreg[1]])
                k4k3 = np.empty((4, 4), dtype=np.complex128)
                k4k3[0:2, 0:2] = k4[0, 0] * k3
                k4k3[0:2, 2:4] = k4[0, 1] * k3
                k4k3[2:4, 0:2] = k4[1, 0] * k3
                k4k3[2:4, 2:4] = k4[1, 1] * k3
                P = k4k3 @ P

        return P
