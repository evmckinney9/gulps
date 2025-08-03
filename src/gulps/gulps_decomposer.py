import logging
import time
from itertools import product
from typing import List, Union

import numpy as np
from monodromy.coordinates import normalize_logspec_AC2
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import UnitaryGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import Operator

from gulps.utils.invariants import GateInvariants
from gulps.utils.isa import ISAInvariants
from gulps.utils.logging_config import logger
from gulps.utils.recover_equiv import recover_local_equivalence

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
        isa: ISAInvariants | None = None,
    ):
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
        self._numerics = SegmentNumericSynthesizer()

    def _eval_edge_case(
        self, target: GateInvariants, return_dag: bool
    ) -> QuantumCircuit | None:
        """Return an exact synthesis if the target is locally equivalent to a basis gate.

        This handles edge cases where the target is exactly a gate in the ISA (up to local unitaries),
        including rho-reflected versions. Works even if polytope precomputation is disabled.
        """
        # Check both the target and its rho-reflection against the ISA
        target_variants = [target, target.rho_reflect]
        for variant in target_variants:
            if variant == self.isa.identity_inv:  # use GateInvariants __eq__
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

    # TODO FIXME, handle true vs alcove target differently
    def _best_decomposition(
        self, target_inv: GateInvariants, log_output: bool = False
    ) -> tuple[List[GateInvariants], List[GateInvariants]]:
        # assume this is False by default (which means )
        rho_bool = False
        alcove_target = GateInvariants.from_unitary(
            target_inv.unitary, enforce_alcove=True
        )
        target_in_ac2 = alcove_target == target_inv

        if self.isa._precompute_polytopes:
            sentence, rho_bool = self.isa.polytope_lookup(alcove_target)

            if sentence is None:
                raise RuntimeError("No precomputed ISA sentence found for target.")
            sentence_out, intermediates, lp_rho = self._try_lp(
                sentence, alcove_target, rho_bool=rho_bool, log_output=log_output
            )
        else:
            for sentence in self.isa.enumerate():
                # heuristic filter to skip a full LP for obvious non-starters
                if sum(gate.strength for gate in sentence) < target_inv.strength:
                    continue

                sentence_out, intermediates, lp_rho = self._try_lp(
                    sentence, alcove_target, log_output=log_output
                )
                if sentence_out is not None:
                    break

        if sentence_out is None:
            raise RuntimeError("No valid ISA sentence found!.")

        # TODO
        # FIXME, the condition seems to be optimized ISA dependent(?)
        # # FIXME, rho_bool should be used to determine if the LP required a reflection
        if not target_in_ac2:  # and intermediates[-1] != target_inv:
            # if intermediates[-1] != target_inv:
            logger.debug("Trying reflection of intermediates")
            intermediates = [x.rho_reflect for x in intermediates]
        # logger.debug("trying norm logspec ac2")
        # intermediates = [
        #     GateInvariants(normalize_logspec_AC2(g.logspec)) for g in intermediates
        # ]

        return sentence_out, intermediates

    def _run(
        self,
        target: Union[np.ndarray, Gate],
        return_dag: bool = False,
        log_output: bool = False,
        easy_attempts: int = 8,
        hard_attempts: int = 16,
    ) -> QuantumCircuit | DAGCircuit:
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

        segment_sols = self._numerics._synthesize_segments(
            sentence_out,
            intermediates,
            easy_attempts=easy_attempts,
            hard_attempts=hard_attempts,
        )
        t2 = time.perf_counter()  # TIMING

        stitched_circuit = self._numerics._stitch_segments(
            sentence_out,
            intermediates,
            segment_sols,
            true_target,
            return_dag=return_dag,
        )
        t3 = time.perf_counter()  # TIMING

        # Optional: Store or return timing info for analysis
        self.last_timing = {
            "lp": t1 - t0,
            "numeric": t2 - t1,
            "stitch": t3 - t2,
        }

        return stitched_circuit

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
