"""Solver and result classes for monodromy linear programming."""

from typing import List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate

from hetero_isas.monodromy_lp.isa import ISAHandler, MonodromyLPGate, MonodromyLPISA
from hetero_isas.monodromy_lp.lp_constraints.lp_constraints import ConstraintsFactory
from hetero_isas.monodromy_lp.mono_lp_result import MonodromyLPDecomposerResult
from hetero_isas.monodromy_lp.singleq_numerics import MonodromyLPNumericalDecomposer

GATE_INVARIANTS = 3  # define this constant to avoid magic values
len_gi = GATE_INVARIANTS


class MonodromyLPDecomposer:
    """Performs monodromy linear programming decomposition of quantum gates.

    Finds the cheapest sequence of basis gates which supports the best approximation to the target gate,
    iteratively attempting to solve the LP problem for increasingly complex gate sequences.

    Attributes:
        isa (MonodromyLPISA): The instruction set architecture to use.
        constraints_factory (ConstraintsFactory): Factory for creating constraint objects.
        basic_pruning (bool): Whether to use basic pruning of gate sequences.

    Methods:
        __call__: Perform the decomposition for a given target gate.
    """

    class InfeasibleDecomposition(Exception):
        """Exception raised when no feasible decomposition is found."""

        pass

    def __init__(
        self,
        isa_handler: ISAHandler,
        use_ordered_sequences: bool = True,
        pad_undetermined: bool = True,
        include_rho_target=True,
        basic_pruning=True,
    ):
        """Initialize a new decomposer."""
        self.isa_handler = isa_handler
        self.isa = self.isa_handler.isa
        self._use_ordered_sequences = use_ordered_sequences
        self._pad_undetermined = pad_undetermined
        self.constraints_factory = ConstraintsFactory(
            self.isa,
            use_ordered_sequences=use_ordered_sequences,
            include_rho_target=include_rho_target,
        )
        self._include_rho_target = include_rho_target
        self.basic_pruning = (
            self.isa.defined and basic_pruning and use_ordered_sequences
        )
        self.numeric_decomposer = MonodromyLPNumericalDecomposer()
        # HACK, need a better way to cache constraints?
        # as well as differentiate between build once vs every iter
        if not self._use_ordered_sequences:
            gate_sequence = next(
                self.isa.enumerate(self._use_ordered_sequences, self._pad_undetermined)
            )
            constraints = self.constraints_factory.build(gate_sequence)
            self._constraints_cache = {0: constraints}

    def _eval_base_case(self, target_gate: MonodromyLPGate):
        """Handle n=0 and n=1 base cases."""
        if target_gate.definition == (0, 0, 0):
            raise NotImplementedError(
                "Local Equiv Base Case: parse into empty constraints?"
            )
        if any(gate.local_equiv(target_gate) for gate in self.isa.gate_set):
            # TODO XXX careful for edge case where we have a gate in the ISA
            # but its possible a gate in the ISA is more expensive than some other decomp
            # for example, if cost CX >> iswap, then making CX out of iswap is better than natively on hardware
            raise NotImplementedError(
                "Local Equiv Base Case: parse into empty constraints?"
            )
        return None

    def _strength_prune(
        self, isa_sequence: List[MonodromyLPGate], target_gate: MonodromyLPGate
    ) -> bool:
        """Prune gate sequences that violate the strength inequality.

        See (Theorem 4.1 Peterson2022Optimal).
        Pruning is a special case that only applied if defined ISA
        XXX proved for XX gates, I'm assuming is true in general...
        """
        total_strength = sum(g.strength for g in isa_sequence)
        return total_strength < target_gate.strength

    def _best_decomposition(
        self,
        target_unitary,
    ) -> MonodromyLPDecomposerResult:
        """Finds the cheapest sequence of isa gates for decomp of target."""
        if not isinstance(target_unitary, MonodromyLPGate):
            target_gate = MonodromyLPGate.from_unitary(
                target_unitary, alcove_norm=False
            )
        else:
            target_gate = target_unitary

        # mono_lp_result is where we will store final result
        mono_lp_result = MonodromyLPDecomposerResult(self.isa_handler, target_unitary)

        optimize_result = self._eval_base_case(target_gate)
        if optimize_result is not None:
            return mono_lp_result._setter_trivial_edge_case()

        lp_calls = 0

        # FIXME HACKY way to skip enumerate if using precomputed coverages
        if self.isa_handler.coverage_set is not None:
            rho_target = target_gate.rho_reflect()
            gate_sequence, cost = self.isa_handler.coverage_lookup_decomposition(
                target_gate
            )
            rho_gate_sequence, rho_cost = (
                self.isa_handler.coverage_lookup_decomposition(rho_target)
            )
            if not cost and not rho_cost:
                raise self.InfeasibleDecomposition(
                    "unexpected, no polytopes contained the target."
                )
            elif not cost or (rho_cost and cost > rho_cost):
                gate_sequence = rho_gate_sequence
                # target_gate = target_gate.rho_reflect()

            constraints = self.constraints_factory.build(gate_sequence)
            lp_result = constraints.attempt_solve(target_gate)
            lp_calls += 1
            if not lp_result:
                lp_result = constraints.attempt_solve(rho_target)
                lp_calls += 1
            if lp_result:  # None on fail
                return mono_lp_result._setter_mono_lp(*lp_result, lp_calls)
            raise self.InfeasibleDecomposition(
                "unexpected, LP failed depsite in polytope"
            )

        for idx, gate_sequence in enumerate(
            self.isa.enumerate(self._use_ordered_sequences, self._pad_undetermined)
        ):
            if self.basic_pruning and self._strength_prune(gate_sequence, target_gate):
                continue

            if self._use_ordered_sequences:
                constraints = self.constraints_factory.build(gate_sequence)
            else:
                constraints = self._constraints_cache[0]

            # if idx in self._constraints_cache.keys():
            #     constraints = self._constraints_cache[idx]
            # else:
            #     constraints = self.constraints_factory.build(gate_sequence)
            #     self._constraints_cache[idx] = constraints

            lp_result = constraints.attempt_solve(target_gate)
            lp_calls += 1
            if lp_result:  # None on fail
                return mono_lp_result._setter_mono_lp(*lp_result, lp_calls)

            # TODO deprecate this
            # the LPConstraints handle rho-reflection internally
            # # try again using rho-relfected target gate
            # if not (
            #     target_gate.rho_reflect_invariant
            #     and all(gate.rho_reflect_invariant for gate in gate_sequence)
            # ):
            #     rho_constraints = constraints.rho_rotate()  # FIXME
            #     optimize_result = rho_constraints.attempt_lp_solve()
            #     lp_calls += 1
            #     if optimize_result.success:
            #         return mono_lp_result._setter_mono_lp(
            #             optimize_result, constraints, lp_calls
            #         )

        raise self.InfeasibleDecomposition(
            "Exhausted valid sequences without solution."
        )

    def __call__(
        self, target: Gate | np.ndarray, use_dag=False, debugging=False
    ) -> QuantumCircuit:
        """Perform the decomposition for a given target gate.

        Args:
            target (Gate): The target gate to decompose.

        Raises:
            InfeasibleDecomposition: If no feasible decomposition is found.
        """
        if isinstance(target, Gate):
            target = np.array(target.to_matrix(), dtype=np.cdouble)

        mono_lp_result = self._best_decomposition(target)
        return self.numeric_decomposer.run(
            target, mono_lp_result, debugging=debugging, use_dag=use_dag
        )
