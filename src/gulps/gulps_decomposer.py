"""Gulps Decomposer module for two-qubit unitary synthesis."""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import UnitaryGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit

from gulps import GateInvariants
from gulps._internal.logging_config import logger
from gulps.core.isa import ISAInvariants
from gulps.linear_program.scipy_lp import MinimalOrderedISAConstraints
from gulps.synthesis.jax_lm import JaxLMSegmentSolver
from gulps.synthesis.recover_equiv import recover_local_equivalence
from gulps.synthesis.segments_abc import SegmentSolver
from gulps.synthesis.segments_solver import SegmentSynthesizer

logger = logging.getLogger(__name__)


@dataclass
class ToleranceConfig:
    """Tolerance settings for GULPS decomposition pipeline.

    Attributes:
        lp_feasibility_tol: Linear program primal/dual feasibility tolerance.
            Used in scipy linprog solver. Default: 1e-10
        makhlin_conv_tol: Stage 1 (Makhlin) convergence tolerance.
            Maximum residual in Makhlin invariant space. Default: 1e-7
        weyl_conv_tol: Stage 2 (Weyl) convergence tolerance.
            Maximum residual in Weyl coordinate space. Default: 1e-12
        segment_solver_tol: Linear solver tolerance for Newton steps.
            Controls numerical precision of internal linear algebra. Default: 1e-10
        equiv_recovery_tol: Local equivalence matching tolerance.
            Used when comparing Weyl coordinates in recovery. Default: 1e-5
    """

    lp_feasibility_tol: float = 1e-10
    makhlin_conv_tol: float = 1e-9
    weyl_conv_tol: float = 5e-5
    segment_solver_tol: float = 1e-11
    equiv_recovery_tol: float = 1e-5


class GulpsDecomposer:
    """Decompose two-qubit unitaries optimally into heterogeneous instruction sets.

    GULPS (Global Unitary Linear Programming Synthesis) combines linear programming
    over monodromy polytopes with numerical segment synthesis to compile arbitrary
    two-qubit unitaries into optimal gate sequences from non-standard ISAs.

    The decomposition process:
        1. Find the cheapest valid gate sentence using polytope lookup or enumeration
        2. Solve a linear program to determine intermediate invariant targets
        3. Synthesize each segment using numerical optimization (Levenberg-Marquardt)
        4. Stitch segments together and recover full unitary equivalence

    Attributes:
        isa: ISAInvariants object containing gate set, costs, and polytope coverage.
        last_timing: Dict with timing breakdown of last decomposition (lp_sentence, segments).
            Units are seconds. Only populated after a successful decomposition.
    """

    def __init__(
        self,
        gate_set: Optional[List[Gate]] = None,
        costs: Optional[List[float]] = None,
        names: Optional[List[str]] = None,
        precompute_polytopes: bool = True,
        isa: Optional[ISAInvariants] = None,
        segment_solver: Optional[SegmentSolver] = None,
        tolerance_config: Optional[ToleranceConfig] = None,
    ):
        """Initialize the GulpsDecomposer.

        Args:
            gate_set: List of two-qubit Gate objects comprising the ISA.
                Required if isa not provided. All gates must be two-qubit.
            costs: List of costs corresponding to gate_set. Required if isa not provided.
                Costs are typically normalized gate durations or error rates.
            names: Optional list of names for the gates in gate_set.
                Used for debugging logs. Defaults to generic labels if None.
            precompute_polytopes: Whether to precompute monodromy polytope coverage.
                When True, enables O(1) sentence lookup. When False, enumerates sentences
                on-demand. Recommended True for repeated decompositions.
            isa: Optional pre-built ISAInvariants instance. If provided, gate_set and
                costs are ignored. Use this to share ISA configuration across decomposers.
            segment_solver: Optional SegmentSolver for numerical synthesis.
                Defaults to JaxLMSegmentSolver() if None.
            tolerance_config: Optional ToleranceConfig for pipeline tolerances.
                If None, uses default tolerances. See ToleranceConfig for details.

        Raises:
            ValueError: If neither isa nor (gate_set, costs) are provided.
            ValueError: If gate_set is empty.
        """
        # gate_set/costs can only be None if isa is provided
        if not isa and (gate_set is None or costs is None):
            raise ValueError("Either isa or (gate_set, costs) must be provided.")

        if isa:
            self.isa = isa
        else:
            if not gate_set:
                raise ValueError("gate_set can't be empty.")

            self.isa = ISAInvariants(
                gate_set=gate_set,
                costs=costs,
                names=names,
                precompute_polytopes=precompute_polytopes,
            )

        self.tolerance_config = tolerance_config or ToleranceConfig()

        if segment_solver is None:
            from gulps.synthesis.jax_lm import JaxLMConfig

            segment_solver = JaxLMSegmentSolver(
                config=JaxLMConfig(
                    makhlin_conv_tol=self.tolerance_config.makhlin_conv_tol,
                    weyl_conv_tol=self.tolerance_config.weyl_conv_tol,
                    solver_tol=self.tolerance_config.segment_solver_tol,
                )
            )
        self._local_synthesis = SegmentSynthesizer(
            solver=segment_solver,
            equiv_recovery_tol=self.tolerance_config.equiv_recovery_tol,
        )

    def _eval_edge_case(
        self, target: GateInvariants, return_dag: bool
    ) -> Optional[Union[QuantumCircuit, DAGCircuit]]:
        """Check if target is locally equivalent to identity or any basis gate.

        This fast path handles edge cases where the target can be synthesized exactly
        using only single-qubit gates or a single basis gate plus single-qubit corrections.
        Checks both the target and its rho-reflection.

        Args:
            target: GateInvariants representation of the target unitary.
            return_dag: If True, return DAGCircuit instead of QuantumCircuit.

        Returns:
            Synthesized circuit if an edge case is detected, None otherwise.
            The circuit has structure: k1⊗k2 · G · k3⊗k4 where G is identity or a basis gate.

        Note:
            Works even when polytope precomputation is disabled, providing a reliable
            fallback for trivial cases.
        """
        # Check both the target and its rho-reflection against the ISA
        target_variants = [target, target.rho_reflect]
        for variant in target_variants:
            if variant == self.isa.identity_inv:  # use GateInvariants __eq__
                # NOTE: Target is locally equivalent to identity - return 1Q corrections only
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Target is identity, returning empty circuit")
                k1, k2, k3, k4, gphase = recover_local_equivalence(
                    target.unitary,
                    self.isa.identity_inv.unitary,
                    tol=self.tolerance_config.equiv_recovery_tol,
                )
                qc = QuantumCircuit(2, global_phase=gphase)
                qc.append(UnitaryGate(k1), [0])
                qc.append(UnitaryGate(k2), [1])
                qc.append(UnitaryGate(k3), [0])
                qc.append(UnitaryGate(k4), [1])
                return circuit_to_dag(qc) if return_dag else qc

        for basis_gate in self.isa.gate_set:
            for variant in target_variants:
                if variant == basis_gate:  # use GateInvariants __eq__
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Target is local to a gate in the ISA")
                    k1, k2, k3, k4, gphase = recover_local_equivalence(
                        target.unitary,
                        basis_gate.unitary,
                        tol=self.tolerance_config.equiv_recovery_tol,
                    )
                    qc = QuantumCircuit(2, global_phase=gphase)
                    qc.append(UnitaryGate(k1), [0])
                    qc.append(UnitaryGate(k2), [1])
                    qc.append(basis_gate.unitary, [0, 1])
                    qc.append(UnitaryGate(k3), [0])
                    qc.append(UnitaryGate(k4), [1])
                    return circuit_to_dag(qc) if return_dag else qc

        return None

    def _try_lp(
        self,
        sentence: List[GateInvariants],
        target: GateInvariants,
        rho_bool: bool = False,
        log_output: bool = False,
    ) -> Tuple[
        Optional[List[GateInvariants]], Optional[List[GateInvariants]], Optional[bool]
    ]:
        """Solve linear program to find intermediate invariants for a gate sentence.

        Attempts to solve the LP with the given rho-reflection orientation. If that fails,
        automatically retries with the opposite orientation.

        Args:
            sentence: Ordered list of basis gates forming a candidate decomposition.
            target: Target gate invariants to reach.
            rho_bool: Initial rho-reflection orientation. If True, uses target.rho_reflect.
            log_output: If True, enable verbose LP solver output.

        Returns:
            Tuple of (sentence_out, intermediates, final_rho_bool):
                - sentence_out: The input sentence if successful, None if LP infeasible
                - intermediates: List of intermediate GateInvariants if successful, None otherwise
                - final_rho_bool: The rho orientation that succeeded, None if both failed

        Note:
            The LP determines a path through monodromy space: I → C₁ → C₂ → ... → target,
            where each Cᵢ represents the cumulative action after gate i.
        """
        constraints = MinimalOrderedISAConstraints(
            sentence, epsilon_lp=self.tolerance_config.lp_feasibility_tol
        )
        constraints.set_target(target, rho_bool=rho_bool)
        sentence_out, intermediates = constraints.solve(log_output=log_output)
        if sentence_out is not None:
            return sentence_out, intermediates, rho_bool

        # if LP fails, try opposite rho-reflection
        # constraints = MinimalOrderedISAConstraints(sentence)
        constraints.set_target(target, rho_bool=not rho_bool)
        sentence_out, intermediates = constraints.solve(log_output=log_output)
        if sentence_out is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("LP succeeded on opposite rho_reflect")
            return sentence_out, intermediates, not rho_bool

        return None, None, None

    def _best_decomposition(
        self, target_inv: GateInvariants, log_output: bool = False
    ) -> Tuple[List[GateInvariants], List[GateInvariants]]:
        """Find the optimal gate sentence and intermediate invariants for the target.

        Uses either polytope lookup (if precomputed) or sentence enumeration to find
        the cheapest valid decomposition. The LP determines intermediate targets.

        Args:
            target_inv: Target gate invariants to decompose.
            log_output: If True, enable verbose LP solver output.

        Returns:
            Tuple of (sentence_out, intermediates):
                - sentence_out: Ordered list of basis gates forming the optimal sentence
                - intermediates: Corresponding intermediate invariants for each gate

        Raises:
            RuntimeError: If no valid sentence found (polytope mode) or LP fails for all
                enumerated sentences.

        Note:
            In polytope mode, the lookup is against the alcove-normalized target.
            The LP then solves against the true target to handle orientations correctly.
        """
        rho_bool = False  # assume this is False by default
        alcove_target = GateInvariants.from_unitary(
            target_inv.unitary, enforce_alcove=True
        )

        if self.isa._precompute_polytopes:
            sentence, rho_bool = self.isa.polytope_lookup(alcove_target)

            if sentence is None:
                raise RuntimeError(
                    f"No precomputed ISA sentence found for target with monodromy {alcove_target.monodromy}. "
                    f"The target may lie outside all polytope coverage. "
                    f"Try disabling precompute_polytopes or expanding the gate set."
                )
            sentence_out, intermediates, _ = self._try_lp(
                sentence, target_inv, rho_bool=rho_bool, log_output=log_output
            )
        else:
            for sentence in self.isa.enumerate():
                # heuristic filter to skip a full LP for obvious non-starters
                if sum(gate.strength for gate in sentence) < target_inv.strength:
                    continue

                sentence_out, intermediates, _ = self._try_lp(
                    sentence, alcove_target, log_output=log_output
                )
                if sentence_out is not None:
                    break

        if sentence_out is None:
            raise RuntimeError(
                f"No valid ISA sentence found for target with monodromy {alcove_target.monodromy}. "
                f"All enumerated sentences failed the LP feasibility check. "
                f"This may indicate insufficient gate set strength or numerical issues."
            )

        return sentence_out, intermediates

    def _run(
        self,
        target: Union[np.ndarray, Gate],
        return_dag: bool = False,
        log_output: bool = False,
    ) -> Union[QuantumCircuit, DAGCircuit]:
        """Core decomposition routine.

        Args:
            target: Two-qubit unitary as 4x4 numpy array or Qiskit Gate.
            return_dag: If True, return DAGCircuit instead of QuantumCircuit.
            log_output: If True, enable verbose LP solver output.

        Returns:
            Quantum circuit implementing the target unitary using the configured ISA.

        Raises:
            RuntimeError: If no valid sentence found or segment synthesis fails.
            ValueError: If gate_list and invariant_list have mismatched lengths or < 2 gates.

        Note:
            Timing information for the last decomposition is stored in self.last_timing
            with keys 'lp_sentence' and 'segments' (in seconds).
        """
        true_target = GateInvariants.from_unitary(target)

        edge_output = self._eval_edge_case(true_target, return_dag)
        if edge_output is not None:
            return edge_output

        # --- A) LP ---
        t0 = time.perf_counter()  # TIMING
        sentence_out, intermediates = self._best_decomposition(
            true_target, log_output=log_output
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Sentence: {[g.name for g in sentence_out]}")
            logger.debug(f"Intermediates: {[g.logspec for g in intermediates]}")

        # --- B1) Segment synthesis ---
        t1 = time.perf_counter()  # TIMING
        if len(sentence_out) != len(intermediates):
            raise ValueError("Gate list and invariant list must have the same length.")
        if len(sentence_out) < 2:
            raise ValueError("At least two gates are required for segment synthesis.")

        stitched_circuit = self._local_synthesis.synthesize_segments(
            gate_list=sentence_out,
            invariant_list=intermediates,
            target=true_target,
            return_dag=return_dag,
        )
        t2 = time.perf_counter()  # TIMING

        # Optional: Store or return timing info for analysis
        self.last_timing = {
            "lp_sentence": t1 - t0,
            "segments": t2 - t1,
        }

        return stitched_circuit

    def __call__(
        self,
        target: Union[np.ndarray, Gate],
        return_dag: bool = False,
        log_output: bool = False,
    ) -> Union[QuantumCircuit, DAGCircuit]:
        """Decompose a two-qubit unitary into the configured instruction set.

        Args:
            target: Two-qubit unitary as 4x4 numpy array or Qiskit Gate.
            return_dag: If True, return DAGCircuit instead of QuantumCircuit.
                Useful for transpiler integration.
            log_output: If True, enable verbose LP solver output for debugging.

        Returns:
            Quantum circuit implementing the target unitary using the configured ISA.
            Single-qubit gates are returned as generic UnitaryGate objects.

        Raises:
            RuntimeError: If no valid sentence found or segment synthesis fails.
            ValueError: If target is not a valid two-qubit unitary.

        Examples:
            >>> from qiskit.circuit.library import iSwapGate
            >>> from qiskit.quantum_info import random_unitary
            >>> gate_set = [iSwapGate().power(1/2), iSwapGate().power(1/3)]
            >>> costs = [0.5, 0.33]
            >>> decomposer = GulpsDecomposer(gate_set=gate_set, costs=costs)
            >>> u = random_unitary(4, seed=42)
            >>> circuit = decomposer(u)
            >>> print(f"Used {len([op for op in circuit.data if op[0].num_qubits == 2])} 2Q gates")
        """
        return self._run(
            target=target,
            return_dag=return_dag,
            log_output=log_output,
        )
