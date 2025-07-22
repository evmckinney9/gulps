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
            raise NotImplementedError("trivial edge case")
            return QuantumCircuit(2)
        if np.any(
            [np.isclose(target.monodromy, gate.monodromy) for gate in self.isa.gate_set]
        ):
            raise NotImplementedError("trivial edge case")
            return QuantumCircuit(2)

    def _try_lp(
        self,
        sentence: List[GateInvariants],
        target: GateInvariants,
        rho_reflect: bool = False,
        log_output: bool = False,
    ) -> tuple[Union[List[GateInvariants], None], Union[List[GateInvariants], None]]:
        """Try solving LP for the sentence with rho-reflection fallback."""
        constraints = MinimalOrderedISAConstraints(sentence)
        constraints.set_target(target, rho_reflect=rho_reflect)
        sentence_out, intermediates = constraints.solve(log_output=log_output)
        if sentence_out is not None:
            return sentence_out, intermediates

        # if LP fails, try opposite rho-reflection
        # constraints = MinimalOrderedISAConstraints(sentence)
        print("lp falls back to opposite rho_reflect")
        constraints.set_target(target, rho_reflect=not rho_reflect)
        sentence_out, intermediates = constraints.solve(log_output=log_output)
        return sentence_out, intermediates

    def _best_decomposition(
        self, target_inv: GateInvariants, log_output: bool = False
    ) -> tuple[List[GateInvariants], List[GateInvariants]]:
        if self.isa._precompute_polytopes:
            sentence, rho_bool = self.isa.polytope_lookup(target_inv)

            if sentence is None:
                raise RuntimeError("No precomputed ISA sentence found for target.")
            sentence_out, intermediates = self._try_lp(
                sentence, target_inv, rho_reflect=rho_bool, log_output=log_output
            )
            if sentence_out is None:
                raise RuntimeError("LP failed for precomputed ISA sentence.")
        else:
            for sentence in self.isa.enumerate():
                # heuristic filter to skip a full LP for obvious non-starters
                if sum(gate.strength for gate in sentence) < target_inv.strength:
                    continue

                sentence_out, intermediates = self._try_lp(
                    sentence, target_inv, log_output=log_output
                )
                if sentence_out is not None:
                    break
            else:
                raise RuntimeError("No valid ISA sentence found via LP enumeration.")

        # FIXME
        useful_intermediates = intermediates[1:]  # Skip identity, and target
        # useful_intermediates += (target_inv,)  # Append target as last intermediate
        return sentence_out, useful_intermediates

    def _run(
        self,
        target: Union[np.ndarray, Gate, GateInvariants],
        return_dag: bool = False,
        log_output: bool = False,
    ) -> QuantumCircuit | DAGCircuit:
        # Convert target to GateInvariants if necessary
        target_inv = (
            target
            if isinstance(target, GateInvariants)
            else GateInvariants.from_unitary(target)
        )

        # TODO
        self._eval_edge_case(target_inv)

        # Find the best decomposition using LP
        sentence_out, useful_intermediates = self._best_decomposition(
            target_inv, log_output=log_output
        )
        # Convert the sentence to a circuit or DAG
        return self._numerics.run(
            sentence_out,
            useful_intermediates,
            target_inv,
            return_dag=return_dag,
        )

    def __call__(
        self,
        target: Union[np.ndarray, Gate, GateInvariants],
        return_dag: bool = False,
        log_output: bool = False,
    ) -> QuantumCircuit | DAGCircuit:
        return self._run(
            target=target,
            return_dag=return_dag,
            log_output=log_output,
        )
