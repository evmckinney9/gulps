"""Gulps Decomposer module for two-qubit unitary synthesis."""

import logging
import time
from typing import List, Optional, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import UnitaryGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit

from gulps import GateInvariants
from gulps._internal.logging_config import logger
from gulps.config import GulpsConfig
from gulps.core.isa import ContinuousISA, DiscreteISA, ISAInvariants
from gulps.linear_program.lp_abc import ConstraintSolution
from gulps.linear_program.scipy_lp import MinimalOrderedISAConstraints
from gulps.synthesis.recover_equiv import recover_local_equivalence
from gulps.synthesis.segments_solver import SegmentSynthesizer

logger = logging.getLogger(__name__)


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
        config_options: Optional[GulpsConfig] = None,
    ):
        """Initialize the GulpsDecomposer.

        Args:
            gate_set: List of two-qubit Gate objects comprising the ISA.
                Required if isa not provided. All gates must be two-qubit.
                Creates a DiscreteISA internally. For ContinuousISA, use isa parameter.
            costs: List of costs corresponding to gate_set. Required if isa not provided.
                Costs are typically normalized gate durations or error rates.
            names: Optional list of names for the gates in gate_set.
                Used for debugging logs. Defaults to generic labels if None.
            precompute_polytopes: Whether to precompute monodromy polytope coverage.
                When True, enables O(1) sentence lookup. When False, enumerates sentences
                on-demand. Recommended True for repeated decompositions. Only applies
                when constructing DiscreteISA via gate_set/costs.
            isa: Optional pre-built ISA instance (DiscreteISA or ContinuousISA).
                If provided, gate_set, costs, names, and precompute_polytopes are ignored.
            config_options: Optional GulpsConfig for all pipeline settings.
                If None, uses default values. See GulpsConfig for all tunable parameters.

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

            self.isa = DiscreteISA(
                gate_set=gate_set,
                costs=costs,
                names=names,
                precompute_polytopes=precompute_polytopes,
            )

        self._is_continuous = isinstance(self.isa, ContinuousISA)
        self.config = config_options or GulpsConfig()

        self._local_synthesis = SegmentSynthesizer(config=self.config)

    def _eval_edge_case(
        self, target: GateInvariants, return_dag: bool
    ) -> Optional[Union[QuantumCircuit, DAGCircuit]]:
        """Check if target is locally equivalent to identity.

        This fast path handles the edge case where the target can be synthesized
        using only single-qubit gates (no two-qubit gate needed).

        Args:
            target: GateInvariants representation of the target unitary.
            return_dag: If True, return DAGCircuit instead of QuantumCircuit.

        Returns:
            Synthesized circuit if target is identity, None otherwise.
        """
        # Check if target is locally equivalent to identity
        if target.is_identity or target.rho_reflect.is_identity:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Target is identity, returning empty circuit")
            k1, k2, k3, k4, gphase = recover_local_equivalence(
                target.unitary,
                np.eye(4),
                config=self.config,
            )
            dag = circuit_to_dag(QuantumCircuit(2, global_phase=gphase))
            qreg = dag.qregs["q"]
            dag.apply_operation_back(UnitaryGate(k1, check_input=False), [qreg[0]])
            dag.apply_operation_back(UnitaryGate(k2, check_input=False), [qreg[1]])
            dag.apply_operation_back(UnitaryGate(k3, check_input=False), [qreg[0]])
            dag.apply_operation_back(UnitaryGate(k4, check_input=False), [qreg[1]])
            return dag if return_dag else dag_to_circuit(dag)

        return None

    def _try_discrete_lp(
        self, target: GateInvariants, log_output: bool = False
    ) -> ConstraintSolution:
        """Find optimal decomposition using discrete ISA.

        Uses polytope lookup if precomputed, otherwise enumerates sentences.

        Args:
            target: Alcove-normalized target gate invariants.
            log_output: If True, enable verbose LP solver output.

        Returns:
            ConstraintSolution with success=True if a valid decomposition is found.
        """
        if self.isa._precompute_polytopes:
            sentence = self.isa.polytope_lookup(target)
            if sentence is None:
                raise RuntimeError(
                    f"No precomputed ISA sentence found for target with monodromy {target.monodromy}. "
                    f"The target may lie outside all polytope coverage. "
                    f"Try disabling precompute_polytopes or expanding the gate set."
                )
            # TODO: optimize by caching constraint objects for previously seen sentences
            constraints = MinimalOrderedISAConstraints(sentence, config=self.config)
            return constraints.solve(target, log_output=log_output)

        # Priority queue enumeration
        for sentence in self.isa.enumerate():
            # Heuristic filter to skip obvious non-starters
            if (
                sum(gate.strength for gate in sentence)
                < target.strength - self.config.lp_feasibility_tol
            ):
                continue
            # TODO: optimize by caching constraint objects for previously seen sentences
            constraints = MinimalOrderedISAConstraints(sentence, config=self.config)
            result = constraints.solve(target, log_output=log_output)
            if result.success:
                return result

        return ConstraintSolution(success=False)

    def _try_continuous_lp(
        self, target: GateInvariants, log_output: bool = False
    ) -> ConstraintSolution:
        """Find optimal decomposition using continuous ISA.

        Single LP/MILP solve with continuous gate parameters.
        For single-family ISA: uses LP (ContinuousISAConstraints)
        For multi-family ISA: uses MILP (HeterogeneousContinuousISAConstraints)

        Args:
            target: Alcove-normalized target gate invariants.
            log_output: If True, enable verbose LP solver output.

        Returns:
            ConstraintSolution with success=True if a valid decomposition is found.
        """
        # Import here to avoid requiring CPLEX for discrete-only usage
        if self.isa.is_single_family:
            from gulps.linear_program.cplex_lp import ContinuousISAConstraints

            # TODO: optimize by caching constraint object (construct once in __init__ or ISA)
            constraints = ContinuousISAConstraints(
                base=self.isa.gate_set[0],
                max_sequence_length=self.isa.max_depth,
                k_lb=self.isa.k_lb,
                single_qubit_cost=self.isa.single_qubit_cost,
                config=self.config,
            )
        else:
            from gulps.linear_program.cplex_lp import (
                HeterogeneousContinuousISAConstraints,
            )

            # Heterogeneous continuous ISA with multiple gate families
            # single_qubit_cost is pulled from isa.single_qubit_cost inside the constructor
            constraints = HeterogeneousContinuousISAConstraints(
                isa=self.isa,
                max_sequence_length=self.isa.max_depth,
                config=self.config,
            )

        return constraints.solve(target, log_output=log_output)

    def _best_decomposition(
        self, target_inv: GateInvariants, log_output: bool = False
    ) -> ConstraintSolution:
        """Find the optimal gate sentence and intermediate invariants for the target.

        Dispatches to discrete or continuous LP solver based on ISA type.

        Args:
            target_inv: Target gate invariants to decompose.
            log_output: If True, enable verbose LP solver output.

        Returns:
            ConstraintSolution with sentence and intermediates.

        Raises:
            RuntimeError: If no valid decomposition found.
        """
        alcove_target = GateInvariants.from_unitary(
            target_inv.unitary, enforce_alcove=True
        )

        if self._is_continuous:
            result = self._try_continuous_lp(alcove_target, log_output=log_output)
        else:
            result = self._try_discrete_lp(alcove_target, log_output=log_output)

        if not result.success:
            raise RuntimeError(
                f"No valid ISA sentence found for target with monodromy {alcove_target.monodromy}. "
                f"All candidates failed the LP feasibility check. "
                f"This may indicate insufficient gate set strength or numerical issues."
            )

        return result

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
        result = self._best_decomposition(true_target, log_output=log_output)
        sentence = result.sentence
        intermediates = result.intermediates

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Sentence: {[g.name for g in sentence]}")
            logger.debug(f"Intermediates: {[g.logspec for g in intermediates]}")

        # --- B1) Segment synthesis ---
        t1 = time.perf_counter()  # TIMING
        stitched_circuit = self._local_synthesis.synthesize_segments(
            gate_list=sentence,
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
