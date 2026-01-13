import logging
from typing import List, Literal

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit

from gulps.config import GulpsConfig
from gulps.core.invariants import GateInvariants
from gulps.synthesis.recover_equiv import recover_local_equivalence
from gulps.synthesis.segments_abc import SegmentSolution, SegmentSolver
from gulps.synthesis.segments_cache import SegmentCache

logger = logging.getLogger(__name__)


class SegmentSynthesizer:
    """Orchestrate segment-wise numeric synthesis and circuit stitching.

    Two methods are available:

    - "batch" (default): Solves all segments independently using canonical
      placeholders, then stitches. Each solve assumes an idealized prefix that
      doesn't exactly match the accumulated circuit, so small errors compound.
      All drift is absorbed by the final recovery step.

    - "sequential": Interleaves solving and stitching, using the actual
      accumulated unitary as each prefix. Theoretically cleaner since each
      solve accounts for the true current state, effectively self-correcting
      drift. However, requires a solver robust to non-canonical inputs.
      Currently experimental due to solver convergence issues.

      NOTE: I believe the main source of drift is something analgous to undershooting (as
      opposed to overshooting). overshooting is correctable drift but undershooting is
      not. if we undershoot then the prefix won't be in the solution space for the next
      segment. a reason for undershooting, in monodromy polytope land, a decomposition
      likely saturates the intermediates to be exactly on the polytope facet. but we
      aren't getting exactly there, which means our segment is shorter than it needs to
      be to reach the next one.... I'm not sure what would be a good fix here since the
      facet points might just be intrinsically difficult solutions to find.

      Showing that facets are hard is a good motivating example for why the numeric
      solver is so unsatisfactory.
    """

    def __init__(self, solver: SegmentSolver, config: GulpsConfig | None = None):
        self._solver = solver
        self.config = config or GulpsConfig()
        self._cache = SegmentCache(max_entries_per_step=self.config.segment_cache_size)

    def synthesize_segments(
        self,
        gate_list: List[GateInvariants],
        invariant_list: List[GateInvariants],
        target: GateInvariants,
        *,
        method: Literal["sequential", "batch"] = "batch",
        return_dag: bool = False,
    ) -> QuantumCircuit | DAGCircuit:
        """Solve for local parameters of each interior segment.

        Args:
            gate_list: Sequence of basis gates {G_i} with precomputed invariants.
            invariant_list: Sequence of target invariants {C_i} for each segment.
            target: Exact target unitary for final recovery.
            method: Synthesis strategy ("batch" or "sequential"). See class docstring.
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

        # Initialize with first gate
        dag.apply_operation_back(gate_list[0].unitary, qreg[:])

        if method == "sequential":
            raise NotImplementedError(
                "Sequential segment synthesis is currently disabled."
            )
            # P = self._synthesize_sequential(dag, qreg, gate_list, invariant_list)
        else:
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
        """Solve all segments with canonical placeholders, then stitch with recovery.

        Each segment is solved independently using the canonical matrix of the
        previous segment as the prefix. This keeps the solver's inputs clean and
        well-conditioned. During stitching, intermediate recovery (k3, k4 only)
        snaps the accumulated unitary back toward canonical.

        Uses per-step caching (k=1) to avoid recomputing identical segments across
        decompositions. Cache matches on (target, prefix, basis) invariants with
        rho-reflection support.
        """
        segment_sols = []

        # Phase 1: Solve all segments with cache lookup
        # Segment i (1-indexed): append g_i to c_{i-1} to reach c_i
        # gate_list[i] = g_{i+1}, invariant_list[i] = c_{i+1}
        for i in range(1, len(invariant_list)):
            Gi = np.array(gate_list[i].unitary, dtype=np.complex128)  # g_{i+1}
            Ci = np.array(
                invariant_list[i].canonical_matrix, dtype=np.complex128
            )  # c_{i+1}

            # c_i = prefix for this segment (c_1 = g_1 for first segment)
            if i == 1:
                Cim1 = np.array(gate_list[0].unitary, dtype=np.complex128)  # c_1 = g_1
                prefix_inv = gate_list[0]
            else:
                Cim1 = np.array(
                    invariant_list[i - 1].canonical_matrix, dtype=np.complex128
                )  # c_i
                prefix_inv = invariant_list[i - 1]

            step = i - 1  # 0-indexed step number
            basis_inv = gate_list[i]
            target_inv = invariant_list[i]

            # Try cache lookup first
            seg_sol = self._cache.get(step, prefix_inv, basis_inv, target_inv)

            if seg_sol is None:
                # Cache miss - solve and store
                seg_sol = self._solver.solve_segment(
                    prefix_op=Cim1, basis_gate=Gi, target=Ci
                )
                if seg_sol.success:
                    self._cache.put(step, prefix_inv, basis_inv, target_inv, seg_sol)

            # TODO FIXME, raise a failure here or inside the solver?
            if not seg_sol.success:
                raise RuntimeError(
                    f"Segment {i} synthesis failed (residual norm={seg_sol.max_residual:.2e})."
                )
            segment_sols.append((Gi, Ci, seg_sol))

        # Phase 2: Stitch solutions together with intermediate recovery
        P = gate_list[0].unitary.to_matrix()
        for idx, (Gi, Ci, seg_sol) in enumerate(segment_sols):
            dag.apply_operation_back(
                UnitaryGate(seg_sol.u0, check_input=False), [qreg[0]]
            )
            dag.apply_operation_back(
                UnitaryGate(seg_sol.u1, check_input=False), [qreg[1]]
            )
            dag.apply_operation_back(gate_list[idx + 1].unitary, qreg[:])
            # track P manually instead of recomputing from DAG
            P = Gi @ np.kron(seg_sol.u1, seg_sol.u0) @ P

            is_final = idx == len(segment_sols) - 1
            if is_final:
                break
            else:
                # Intermediate recovery: only back gates (k3, k4)
                # Front gates (k1, k2) are deferred to final recovery
                _, _, k3, k4, gphase = recover_local_equivalence(
                    Ci, P, config=self.config
                )
                dag.global_phase += gphase
                dag.apply_operation_back(UnitaryGate(k3, check_input=False), [qreg[0]])
                dag.apply_operation_back(UnitaryGate(k4, check_input=False), [qreg[1]])
                # Update accumulated unitary (and global phase)
                P = np.kron(k4, k3) @ P
        return P

    # def _synthesize_sequential(self, dag, qreg, gate_list, invariant_list):
    #     """Interleave solving and stitching using accumulated prefix.

    #     WARNING: This method is experimental.

    #     Each segment is solved using the actual accumulated unitary as the prefix,
    #     avoiding intermediate recovery calls. However, numerical error from imperfect
    #     solver solutions compounds across segments, causing the prefix to drift from
    #     ideal. This makes later segments harder to solve, resulting in ~4x higher
    #     failure rates compared to batch mode in testing.

    #     However, (a) this method avoids intermediate recovery calls, which may be
    #     desirable in some contexts, and (b) more robust to drift, provided stil in solution space
    #     (i.e., the solver can still converge and removes drift from the prefix).
    #     """
    #     P = Operator(dag_to_circuit(dag)).data

    #     for i in range(1, len(invariant_list)):
    #         Gi = np.array(gate_list[i].unitary, dtype=np.complex128)
    #         Ci = np.array(invariant_list[i].canonical_matrix, dtype=np.complex128)

    #         seg_sol = self._solver.solve_segment(prefix_op=P, basis_gate=Gi, target=Ci)

    #         if not seg_sol.success:
    #             raise RuntimeError(
    #                 f"Segment {i} synthesis failed (residual norm={seg_sol.max_residual:.2e})."
    #             )

    #         dag.apply_operation_back(UnitaryGate(seg_sol.u0), [qreg[0]])
    #         dag.apply_operation_back(UnitaryGate(seg_sol.u1), [qreg[1]])
    #         dag.apply_operation_back(UnitaryGate(Gi), qreg[:])

    #         # Update accumulated unitary from DAG (avoids additional drift from
    #         # manual matrix multiplication)
    #         P = Operator(dag_to_circuit(dag)).data
    #         k1, k2, k3, k4, gphase = recover_local_equivalence(
    #             Ci, P, config=self.config
    #         )
    #         dag.global_phase += gphase
    #         dag.apply_operation_back(UnitaryGate(k3), [qreg[0]])
    #         dag.apply_operation_back(UnitaryGate(k4), [qreg[1]])
    #         P = Operator(dag_to_circuit(dag)).data

    #     return P
