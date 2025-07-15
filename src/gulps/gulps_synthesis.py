from itertools import product
from typing import List, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGCircuit

from gulps.utils.invariants import GateInvariants
from gulps.utils.isa import ISAInvariants

from .linear_program import MinimalOrderedISAConstraints
from .local_numerics import SegmentNumericSynthesizer


class GulpsDecomposer:
    """Decompose a two-qubit unitary using a monodromy LP and numeric segment synthesis.
    Gate sentences are drawn from a fixed gate set and enumerated up to a specified depth.
    """

    def __init__(
        self,
        gate_set: List[GateInvariants],
        costs: List[float],
        precompute_polytopes: bool = False,
    ):
        if not gate_set:
            raise ValueError("gate_set must contain at least one GateInvariants.")

        self.isa = ISAInvariants(
            gate_set=gate_set,
            costs=costs,
            precompute_polytopes=precompute_polytopes,
        )
        self._numerics = SegmentNumericSynthesizer()
        self._constraint_cache = {}

    def _eval_edge_case(self, target: GateInvariants):
        """Handle edge cases where the target is a simple gate."""
        if target.monodromy == (0, 0, 0):
            raise NotImplementedError()
            return QuantumCircuit(2)
        if np.any(
            [np.isclose(target.monodromy, gate.monodromy) for gate in self.gate_set]
        ):
            raise NotImplementedError()
            return QuantumCircuit(2)

    def _try_lp(
        self,
        sentence: List[GateInvariants],
        target: GateInvariants,
        log_output: bool = False,
    ) -> tuple[Union[List[GateInvariants], None], Union[List[GateInvariants], None]]:
        """Try solving LP for the sentence with rho-reflection fallback."""
        constraints = MinimalOrderedISAConstraints(sentence)
        constraints.set_target(target)
        sentence_out, intermediates = constraints.solve(log_output=log_output)
        if sentence_out is not None:
            return sentence_out, intermediates

        # if LP fails, try rho-reflection
        constraints.set_target(target.rho_reflect())
        sentence_out, intermediates = constraints.solve(log_output=log_output)
        return sentence_out, intermediates

    def __call__(
        self,
        target: Union[np.ndarray, Gate, GateInvariants],
        return_dag: bool = False,
        log_output: bool = False,
    ) -> QuantumCircuit | DAGCircuit:
        target_inv = (
            target
            if isinstance(target, GateInvariants)
            else GateInvariants.from_unitary(target)
        )

        sentence_out = None
        intermediates = None

        if self.isa._precompute_polytopes:
            sentence = self.isa.polytope_lookup(target_inv)
            if sentence is None:
                raise RuntimeError("No precomputed ISA sentence found for target.")
            sentence_out, intermediates = self._try_lp(
                sentence, target_inv, log_output=log_output
            )
            if sentence_out is None:
                raise RuntimeError("LP failed for precomputed ISA sentence.")
        else:
            for sentence in self.isa.enumerate():
                if sum(gate.strength for gate in sentence) < target_inv.strength:
                    continue
                sentence_out, intermediates = self._try_lp(
                    sentence, target_inv, log_output=log_output
                )
                if sentence_out is not None:
                    break
            else:
                raise RuntimeError("No valid ISA sentence found via LP enumeration.")

        return self._numerics(
            sentence_out,
            intermediates[1:],  # skip identity
            target_inv,
            return_dag=return_dag,
        )
