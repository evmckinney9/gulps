from itertools import product
from typing import List, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGCircuit

from gulps.invariants import GateInvariants

from .linear_program import MinimalOrderedISAConstraints
from .local_numerics import SegmentNumericSynthesizer


class GulpsDecomposer:
    """Decompose a two-qubit unitary using a monodromy LP and numeric segment synthesis.
    Gate sentences are drawn from a fixed gate set and enumerated up to a specified depth.
    """

    def __init__(self, gate_set: List[GateInvariants], max_depth: int = 3):
        if not gate_set:
            raise ValueError("gate_set must contain at least one GateInvariants.")
        if max_depth < 2:
            raise ValueError("max_depth must be at least 2 (for a meaningful LP).")

        self.gate_set = gate_set
        self.max_depth = max_depth

    def enumerate_sentences(self) -> List[List[GateInvariants]]:
        """Generate all ordered gate sequences up to max_depth."""
        for length in range(2, self.max_depth + 1):
            for sequence in product(self.gate_set, repeat=length):
                yield list(sequence)

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

        if sentence_out is None:
            constraints.set_target(target.rho_reflect())
            sentence_out, intermediates = constraints.solve(log_output=log_output)

        return sentence_out, intermediates

    def __call__(
        self,
        target: Union[np.ndarray, Gate, GateInvariants],
        return_dag: bool = False,
        log_output: bool = False,
    ) -> QuantumCircuit | DAGCircuit:
        """Decompose the given target into a QuantumCircuit using LP + numeric stitching."""
        if isinstance(target, Gate):
            target_unitary = target.to_matrix()
        elif isinstance(target, GateInvariants):
            target_unitary = target.unitary
        elif isinstance(target, np.ndarray):
            target_unitary = target
        else:
            raise TypeError("Target must be a Gate, np.ndarray, or GateInvariants")

        target_inv = (
            target
            if isinstance(target, GateInvariants)
            else GateInvariants.from_unitary(target_unitary)
        )

        for sentence in self.enumerate_sentences():
            sentence_out, intermediates = self._try_lp(
                sentence, target_inv, log_output=log_output
            )
            if sentence_out is not None:
                return SegmentNumericSynthesizer()(
                    sentence_out,
                    intermediates,
                    target_unitary,
                    return_dag=return_dag,
                )

        raise RuntimeError(
            "No valid decomposition found via LP (including rho-reflection)."
        )
