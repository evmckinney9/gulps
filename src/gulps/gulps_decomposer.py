# Copyright 2025-2026 Lev S. Bishop, Evan McKinney
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gulps Decomposer module for two-qubit unitary synthesis."""

from __future__ import annotations

import logging
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGCircuit

from gulps import GateInvariants
from gulps.config import GulpsConfig
from gulps.core.isa import ContinuousISA, DiscreteISA, ISAInvariants
from gulps.core.segments import SegmentSynthesizer
from gulps.linear_program.lp_dispatch import best_decomposition_batch
from gulps.linear_program.lp_solver import LPSolverCache

logger = logging.getLogger(__name__)


class GulpsDecomposer:
    """Decompose two-qubit unitaries optimally into heterogeneous instruction sets.

    GULPS (Global Unitary Linear Programming Synthesis) combines linear programming
    over monodromy polytopes with numerical segment synthesis to compile arbitrary
    two-qubit unitaries into optimal gate sequences from non-standard ISAs.

    The decomposition process:
        1. Find the cheapest valid gate sentence using polytope lookup or enumeration
        2. Solve a linear program to determine intermediate invariant targets
        3. Synthesize each segment using numerical optimization (Gauss-Newton)
        4. Stitch segments together and recover full unitary equivalence

    Attributes:
        isa: ISAInvariants object containing gate set, costs, and polytope coverage.
        last_timing: Dict with timing breakdown of last decomposition (lp_sentence, segments).
            Units are seconds. Only populated after a successful decomposition.

    References:
        Peterson, Crooks, Smith, "Fixed-Depth Two-Qubit Circuits and the Monodromy
        Polytope", Quantum 4, 247 (2020). https://doi.org/10.22331/q-2020-03-26-247

        Zhang et al., "Geometric theory of nonlocal two-qubit operations",
        Phys. Rev. A 67, 042313 (2003). https://doi.org/10.1103/PhysRevA.67.042313

        Zhang et al., "Exact Two-Qubit Universal Quantum Circuit",
        Phys. Rev. Lett. 91, 027903 (2003). https://doi.org/10.1103/PhysRevLett.91.027903

        Makhlin, "Nonlocal properties of two-qubit gates and mixed states",
        Quantum Inf. Process. 1, 243-252 (2002).
        https://doi.org/10.1023/A:1022144002391

        Tucci, "An Introduction to Cartan's KAK Decomposition for QC Programmers",
        arXiv:quant-ph/0507171 (2005). https://arxiv.org/abs/quant-ph/0507171
    """

    def __init__(
        self,
        gate_set: list[Gate] | None = None,
        costs: list[float] | None = None,
        names: list[str] | None = None,
        precompute_polytopes: bool = False,
        max_sequence_length: int = 8,
        isa: ISAInvariants | None = None,
        config_options: GulpsConfig | None = None,
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
            max_sequence_length: Maximum gate sentence length for enumeration.
                Only applies when constructing DiscreteISA via gate_set/costs.
                Ignored when isa is provided (ISA carries its own max_sequence_length).
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
                max_sequence_length=max_sequence_length,
            )

        self._is_continuous = isinstance(self.isa, ContinuousISA)
        self.config = config_options or GulpsConfig()

        self._lp_cache = LPSolverCache()
        # MOC cache: ``MinimalOrderedISAConstraints`` keyed by sentence tuple.
        self._moc_cache: dict[tuple, object] = {}
        self._local_synthesis = SegmentSynthesizer(config=self.config)
        self._continuous_lp = None  # lazily built on first continuous solve

    def _best_decomposition(
        self,
        alcove_target: GateInvariants,
        log_output: bool = False,
    ):
        """Single-target LP via :func:`best_decomposition_batch` (sentence + intermediates only)."""
        return best_decomposition_batch(self, [alcove_target], log_output=log_output)[
            alcove_target
        ]

    def _run(
        self,
        target: np.ndarray | Gate,
        return_dag: bool = False,
        log_output: bool = False,
    ) -> QuantumCircuit | DAGCircuit:
        """LP + segment synthesis on a single target.

        Populates ``self.last_timing`` with keys ``init``, ``lp_sentence``,
        ``numerics``, ``stitch``, ``total`` (seconds).
        """
        t_start = time.perf_counter()
        alcove_target = GateInvariants.from_unitary(target)

        t0 = time.perf_counter()
        result = self._best_decomposition(alcove_target, log_output=log_output)
        sentence = result.sentence
        intermediates = result.intermediates

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Sentence: {[g.name for g in sentence]}")
            logger.debug(f"Intermediates: {[g.logspec for g in intermediates]}")

        t1 = time.perf_counter()
        stitched_circuit = self._local_synthesis.synthesize_segments(
            gate_list=sentence,
            invariant_list=intermediates,
            target=alcove_target,
            return_dag=return_dag,
        )
        t2 = time.perf_counter()

        phase = self._local_synthesis.last_phase_timing
        self.last_timing = {
            "init": t0 - t_start,
            "lp_sentence": t1 - t0,
            "numerics": phase["numerics"],
            "stitch": phase["stitch"],
            "total": t2 - t_start,
        }

        flag = self.config.flag_duration
        if flag > 0 and self.last_timing["total"] > flag:
            logger.warning(
                "Decomposition took %.4fs (threshold: %.4fs). "
                "Suppress with GulpsConfig(flag_duration=0).",
                self.last_timing["total"],
                flag,
            )

        return stitched_circuit

    def __call__(
        self,
        target: np.ndarray | Gate,
        return_dag: bool = False,
        log_output: bool = False,
    ) -> QuantumCircuit | DAGCircuit:
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
            LPInfeasibleError: If no valid sentence found for the target.
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
