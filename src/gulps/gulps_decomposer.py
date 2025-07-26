import logging
from itertools import product
from typing import List, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import UnitaryGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import Operator

from gulps.utils.invariants import GateInvariants, recover_local_equivalence
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
        gate_set: List[Gate],
        costs: List[float],
        names: List[str] | None = None,
        precompute_polytopes: bool = True,
    ):
        if not gate_set:
            raise ValueError("gate_set can't be empty.")

        self.isa = ISAInvariants(
            gate_set=gate_set,
            costs=costs,
            names=names,
            precompute_polytopes=precompute_polytopes,
        )
        self._numerics = SegmentNumericSynthesizer()
        self._constraint_cache = {}

    def _eval_edge_case(
        self, target: GateInvariants, return_dag: bool
    ) -> QuantumCircuit | None:
        """Return an exact synthesis if the target is locally equivalent to a basis gate.

        This handles edge cases where the target is exactly a gate in the ISA (up to local unitaries),
        including rho-reflected versions. Works even if polytope precomputation is disabled.
        """
        rtol = 1e-14
        atol = 1e-15

        # Check both the target and its rho-reflection against the ISA
        target_variants = [target.monodromy, target.rho_reflect.monodromy]
        for variant in target_variants:
            if variant == self.isa.identity:  # use GateInvariants __eq__
                # TODO FIXME Hand this to a 1Q decomposer instead?
                logger.debug("Target is identity, returning empty circuit")
                k1, k2, k3, k4, gphase = recover_local_equivalence(
                    target.unitary, self.isa.identity_inv.unitary
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
                    logger.debug("Target is local to a gate in the ISA")
                    k1, k2, k3, k4, gphase = recover_local_equivalence(
                        target.unitary, basis_gate.unitary
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
            if sentence_out is not None:
                return sentence_out, intermediates
        else:
            rho_bool = False  # FIXME!
            for sentence in self.isa.enumerate():
                # heuristic filter to skip a full LP for obvious non-starters
                if sum(gate.strength for gate in sentence) < target_inv.strength:
                    continue

                sentence_out, intermediates, lp_rho = self._try_lp(
                    sentence, target_inv, log_output=log_output
                )
                if sentence_out is not None:
                    return sentence_out, intermediates
        raise RuntimeError("No valid ISA sentence found!.")

    def _run(
        self,
        target: Union[np.ndarray, Gate],
        return_dag: bool = False,
        log_output: bool = False,
    ) -> QuantumCircuit | DAGCircuit:
        # FIXME there is better way to handle this
        # NOTE, enforce alcove means target will always be valid by qlr
        true_target = GateInvariants.from_unitary(target)
        target_inv = GateInvariants.from_unitary(target, enforce_alcove=True)

        # if these have same monodromy, already in alcove_c2
        target_in_ac2 = target_inv == true_target  # use the GateInvariants __eq__

        # NOTE edge case handles target is identity or local to a gate in this isa
        # this is because LP assumes sentences of at least two gates
        edge_output = self._eval_edge_case(true_target, return_dag)
        if edge_output is not None:
            return edge_output

        # Find the best decomposition using LP
        sentence_out, intermediates = self._best_decomposition(
            target_inv, log_output=log_output
        )

        if not target_in_ac2 and intermediates[-1] is not true_target:
            logger.debug("Trying reflection of intermediates")
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
