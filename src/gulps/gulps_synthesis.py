import logging
from itertools import product
from typing import List, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGCircuit

from gulps.utils.invariants import GateInvariants
from gulps.utils.isa import ISAInvariants
from gulps.utils.logging_config import logger

from .linear_program import MinimalOrderedISAConstraints
from .local_numerics import SegmentNumericSynthesizer

logger = logging.getLogger(__name__)


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
        rho_bool: bool = False,
        log_output: bool = False,
    ) -> tuple[Union[List[GateInvariants], None], Union[List[GateInvariants], None]]:
        """Try solving LP for the sentence with rho-reflection fallback."""
        constraints = MinimalOrderedISAConstraints(sentence)
        constraints.set_target(target, rho_bool=rho_bool)
        sentence_out, intermediates = constraints.solve(log_output=log_output)
        if sentence_out is not None:
            return sentence_out, intermediates, rho_bool

        # if LP fails, try opposite rho-reflection
        # constraints = MinimalOrderedISAConstraints(sentence)
        constraints.set_target(target, rho_bool=not rho_bool)
        sentence_out, intermediates = constraints.solve(log_output=log_output)
        if sentence_out is not None:
            # if LP succeeds with rho-reflection, return the reflected trajectory
            logger.debug("lp falls back to opposite rho_reflect")
            return sentence_out, intermediates, not rho_bool

        return None, None, None

    def _best_decomposition(
        self, target_inv: GateInvariants, log_output: bool = False
    ) -> tuple[List[GateInvariants], List[GateInvariants]]:
        if self.isa._precompute_polytopes:
            sentence, rho_bool = self.isa.polytope_lookup(target_inv)

            if sentence is None:
                raise RuntimeError("No precomputed ISA sentence found for target.")
            sentence_out, intermediates, lp_rho = self._try_lp(
                sentence, target_inv, rho_bool=rho_bool, log_output=log_output
            )
            if sentence_out is None:
                raise RuntimeError("LP failed for precomputed ISA sentence.")
        else:
            for sentence in self.isa.enumerate():
                # heuristic filter to skip a full LP for obvious non-starters
                if sum(gate.strength for gate in sentence) < target_inv.strength:
                    continue

                sentence_out, intermediates, lp_rho = self._try_lp(
                    sentence, target_inv, log_output=log_output
                )
                if sentence_out is not None:
                    break
            else:
                raise RuntimeError("No valid ISA sentence found via LP enumeration.")

        useful_intermediates = intermediates[1:]  # Skip identity
        return sentence_out, useful_intermediates

    def _run(
        self,
        target: Union[np.ndarray, Gate],
        return_dag: bool = False,
        log_output: bool = False,
    ) -> QuantumCircuit | DAGCircuit:
        # NOTE, enforce alcove means target will always be valid by qlr
        target_inv = GateInvariants.from_unitary(target, enforce_alcove=True)
        true_target = GateInvariants.from_unitary(target)

        # TODO
        self._eval_edge_case(target_inv)

        # Find the best decomposition using LP
        sentence_out, intermediates = self._best_decomposition(
            true_target, log_output=log_output
        )

        if intermediates[-1] is not true_target:
            logger.debug("Final intermediate does not match target.")
            intermediates = [x.rho_reflect for x in intermediates]

        # Convert the sentence to a circuit or DAG
        return self._numerics.run(
            sentence_out,
            intermediates,
            true_target,
            return_dag=return_dag,
        )

    def __call__(
        self,
        target: Union[np.ndarray, Gate],
        return_dag: bool = False,
        log_output: bool = False,
    ) -> QuantumCircuit | DAGCircuit:
        return self._run(
            target=target,
            return_dag=return_dag,
            log_output=log_output,
        )
