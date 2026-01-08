import logging
from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import Operator

from gulps.config import GulpsConfig
from gulps.core.invariants import GateInvariants
from gulps.synthesis.jax_lm import JaxLMSegmentSolver
from gulps.synthesis.recover_equiv import recover_local_equivalence
from gulps.synthesis.segments_abc import SegmentSolution, SegmentSolver

logger = logging.getLogger(__name__)


@dataclass
class SegmentCacheEntry:
    """Cache entry for a single segment solution.

    Stores the inputs as GateInvariants, the solution, and hit count for LFU eviction.
    Matching uses GateInvariants.__eq__ (monodromy-based) with rho-reflection support.
    """

    prefix_inv: GateInvariants
    basis_inv: GateInvariants
    target_inv: GateInvariants
    solution: SegmentSolution
    hit_count: int = 0

    def matches(
        self, prefix_inv: GateInvariants, basis_inv: GateInvariants, target_inv: GateInvariants
    ) -> bool:
        """Check if the cache entry matches the given inputs.

        Checks in order: target (with rho), prefix, basis - for fastest fail-early.

        Args:
            prefix_inv: The prefix invariants to match.
            basis_inv: The basis gate invariants to match.
            target_inv: The target invariants to match (checked against both
                        original and rho-reflected).

        Returns:
            True if all inputs match.
        """
        # Check target first (most likely to differ) - allow rho-reflection match
        if self.target_inv != target_inv and self.target_inv != target_inv.rho_reflect:
            return False
        if self.prefix_inv != prefix_inv:
            return False
        if self.basis_inv != basis_inv:
            return False
        return True


class SegmentCache:
    """Per-step cache for segment solutions with LFU eviction.

    Maintains up to k cache entries per discrete step index, where k is
    configured by segment_cache_size. When a step's cache is full, the
    least frequently used entry is evicted.

    Early steps in decomposition often compute similar solutions across
    different targets, making this cache effective for iterative workflows.
    """

    def __init__(self, max_entries_per_step: int = 2):
        self._entries: dict[int, list[SegmentCacheEntry]] = {}
        self._max_entries = max_entries_per_step
        self.hits = 0
        self.misses = 0

    def get(
        self,
        step: int,
        prefix_inv: GateInvariants,
        basis_inv: GateInvariants,
        target_inv: GateInvariants,
    ) -> Optional[SegmentSolution]:
        """Look up a cached solution for the given step and inputs.

        Args:
            step: The discrete step index (0-indexed segment number).
            prefix_inv: The prefix invariants for this segment.
            basis_inv: The basis gate invariants for this segment.
            target_inv: The target invariants.

        Returns:
            Cached SegmentSolution if found and matching, None otherwise.
        """
        entries = self._entries.get(step, [])
        for entry in entries:
            if entry.matches(prefix_inv, basis_inv, target_inv):
                entry.hit_count += 1
                self.hits += 1
                return entry.solution
        self.misses += 1
        return None

    def put(
        self,
        step: int,
        prefix_inv: GateInvariants,
        basis_inv: GateInvariants,
        target_inv: GateInvariants,
        solution: SegmentSolution,
    ) -> None:
        """Store a solution in the cache for the given step.

        Uses LFU eviction: if the step has max_entries, evicts the entry
        with the lowest hit_count.

        Args:
            step: The discrete step index.
            prefix_inv: The prefix invariants used.
            basis_inv: The basis gate invariants used.
            target_inv: The target invariants.
            solution: The computed solution to cache.
        """
        if step not in self._entries:
            self._entries[step] = []

        entries = self._entries[step]

        # Check if already cached (shouldn't happen if called after get miss, but be safe)
        for entry in entries:
            if entry.matches(prefix_inv, basis_inv, target_inv):
                return

        new_entry = SegmentCacheEntry(
            prefix_inv=prefix_inv,
            basis_inv=basis_inv,
            target_inv=target_inv,
            solution=solution,
            hit_count=0,
        )

        if len(entries) < self._max_entries:
            entries.append(new_entry)
        else:
            # LFU eviction: find and replace entry with lowest hit_count
            min_idx = min(range(len(entries)), key=lambda i: entries[i].hit_count)
            entries[min_idx] = new_entry

    def clear(self) -> None:
        """Clear all cached entries and reset statistics."""
        self._entries.clear()
        self.hits = 0
        self.misses = 0

    @property
    def stats(self) -> dict:
        """Return cache hit/miss statistics."""
        total = self.hits + self.misses
        total_entries = sum(len(e) for e in self._entries.values())
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0.0,
            "entries": total_entries,
            "steps": len(self._entries),
        }


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

        if len(gate_list) < 2:
            raise ValueError("At least two gates are required for segment synthesis.")

        dag = circuit_to_dag(QuantumCircuit(2, global_phase=0))
        qreg = dag.qregs["q"]

        # Initialize with first gate
        dag.apply_operation_back(UnitaryGate(gate_list[0].unitary), qreg[:])

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
        dag.apply_operation_front(UnitaryGate(k1), [qreg[0]])
        dag.apply_operation_front(UnitaryGate(k2), [qreg[1]])
        dag.apply_operation_back(UnitaryGate(k3), [qreg[0]])
        dag.apply_operation_back(UnitaryGate(k4), [qreg[1]])

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
        for i in range(1, len(invariant_list)):
            Gi = np.array(gate_list[i].unitary, dtype=np.complex128)
            Ci = np.array(invariant_list[i].canonical_matrix, dtype=np.complex128)

            # Use canonical of previous segment (or first gate's unitary for i=1)
            if i == 1:
                Cim1 = np.array(gate_list[0].unitary, dtype=np.complex128)
                prefix_inv = gate_list[0]
            else:
                Cim1 = np.array(
                    invariant_list[i - 1].canonical_matrix, dtype=np.complex128
                )
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

            if not seg_sol.success:
                raise RuntimeError(
                    f"Segment {i} synthesis failed (residual norm={seg_sol.max_residual:.2e})."
                )
            segment_sols.append((Gi, Ci, seg_sol))

        # Phase 2: Stitch solutions together with intermediate recovery
        P = gate_list[0].unitary.to_matrix()
        for idx, (Gi, Ci, seg_sol) in enumerate(segment_sols):
            dag.apply_operation_back(UnitaryGate(seg_sol.u0), [qreg[0]])
            dag.apply_operation_back(UnitaryGate(seg_sol.u1), [qreg[1]])
            dag.apply_operation_back(UnitaryGate(Gi), qreg[:])
            # track P manually instead of recomputing from DAG
            # P = Operator(dag_to_circuit(dag)).data
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
                dag.apply_operation_back(UnitaryGate(k3), [qreg[0]])
                dag.apply_operation_back(UnitaryGate(k4), [qreg[1]])
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
