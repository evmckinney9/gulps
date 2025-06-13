"""Gate and ISA definitions for monodromy linear programming."""

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib import cm
from monodromy.coverage import CircuitPolytope, gates_to_coverage
from monodromy.polytopes import ConvexPolytope

from hetero_isas.monodromy_lp.gate import MonodromyLPGate

GATE_INVARIANTS = 3  # define this constant to avoid magic values


class ISAHandler:
    """Temporary class to handle some injective gate definition mapping.

    # FIXME maybe this is better to have elsewhere?
    # but for now this is required for reconstructing into circuit ansatz?

    TODO this constructor builds from unitaries, but what about cont. family gates?
    FIXME is there a sort by cost assumption? (when building constraints)

    I need to be able to reversibly map invariants to unitaries. I don't know how this
    might be integrated into larger qiskit.Target/transpiler flow so I am leaving barebones for now.

    Importantly, how do we define this ISA for when gate is a family???
    E.g. how do we specifiy continuous-CXGate?
    Not that hard - but need to make some implementation decisions I want to wait on.
    """

    def __init__(self, gates, costs, names, compute_coverage_set=False):
        self._mono_g_list = []
        for gate, cost, name in zip(gates, costs, names):
            g = MonodromyLPGate.from_unitary(gate, cost=cost, name=name)
            self._mono_g_list.append(g)

        self.isa = MonodromyLPISA(*self._mono_g_list)
        self.g_to_u = {g: gate for g, gate in zip(self._mono_g_list, gates)}
        self._gates = gates
        self._costs = costs
        self._names = names
        self.coverage_set = None
        if compute_coverage_set:
            self._compute_coverage_set()
            self._preprocess_coverage_set()

    def get_unitary(self, mono_g: MonodromyLPGate):
        return self.g_to_u[mono_g]

    def coverage_lookup_decomposition(
        self, target_gate: MonodromyLPGate
    ) -> Tuple[List[MonodromyLPGate], float]:
        if not self.coverage_set:
            raise ValueError("Polytope volumes weren't precomputed")
        valid_subpolytopes = self._evaluate_constraints(target_gate)
        if len(valid_subpolytopes) == 0:
            return None, None
        # NOTE I'm assuming coverage_set is already sorted by cost
        # but monodromy doesn't have 2Q gates in our preferred lexicographic ordering
        return (
            sorted(
                self.coverage_set[min(valid_subpolytopes)[0]].instructions, reverse=True
            ),
            self.coverage_set[min(valid_subpolytopes)[0]].cost,
        )

    def _compute_coverage_set(self):
        print("Building coverage,... this operation is slow.")
        self.coverage_set = gates_to_coverage(
            *self._gates,
            costs=self._costs,
            names=self._names,
            single_qubit_cost=1e-3,
            instructions=self._mono_g_list,
        )

    def _preprocess_coverage_set(self):
        unique_inequalities = []
        unique_equalities = []
        subpolytope_to_ineq = {}
        subpolytope_to_eq = {}
        # idx_to_polytope = {}
        for cp_idx, circuit_polytope in enumerate(self.coverage_set):
            for sub_idx, subpolytope in enumerate(circuit_polytope.convex_subpolytopes):
                subpolytope_to_ineq[(cp_idx, sub_idx)] = []
                subpolytope_to_eq[(cp_idx, sub_idx)] = []

                for ineq in subpolytope.inequalities:
                    ineq_tuple = tuple(ineq)
                    if ineq_tuple not in unique_inequalities:
                        unique_inequalities.append(ineq_tuple)
                        unique_idx = len(unique_inequalities) - 1
                    else:
                        unique_idx = unique_inequalities.index(ineq_tuple)
                    subpolytope_to_ineq[(cp_idx, sub_idx)].append(unique_idx)

                for eq in subpolytope.equalities:
                    eq_tuple = tuple(eq)
                    if eq_tuple not in unique_equalities:
                        unique_equalities.append(eq_tuple)
                        unique_idx = len(unique_equalities) - 1
                    else:
                        unique_idx = unique_equalities.index(eq_tuple)
                    subpolytope_to_eq[(cp_idx, sub_idx)].append(unique_idx)
        ineq_matrix = np.array(list(unique_inequalities))
        eq_matrix = np.array(list(unique_equalities))

        self._ineq_matrix = ineq_matrix
        self._eq_matrix = eq_matrix
        self._subpolytope_to_ineq = subpolytope_to_ineq
        self._subpolytope_to_eq = subpolytope_to_eq
        return None

    def _evaluate_constraints(
        self,
        target_gate: MonodromyLPGate,
        epsilon: float = 1e-8,  # Fraction(1, 1_000_000_000)
    ) -> List[Tuple[int, int]]:
        xv = np.array(target_gate.definition).reshape((GATE_INVARIANTS, 1))
        ineq_results = np.full(len(self._ineq_matrix), True)
        eq_results = np.full(len(self._eq_matrix), True)
        if self._ineq_matrix.size > 0:
            ineq_results = (
                -epsilon
                <= (self._ineq_matrix[:, 1:] @ xv).squeeze() + self._ineq_matrix[:, 0]
            )
        if self._eq_matrix.size > 0:
            eq_results = (
                np.abs((self._eq_matrix[:, 1:] @ xv).squeeze() + self._eq_matrix[:, 0])
                <= epsilon
            )
        valid_subpolytopes = []
        for (cp_idx, sub_idx), ineq_indices in self._subpolytope_to_ineq.items():
            eq_indices = self._subpolytope_to_eq[(cp_idx, sub_idx)]
            if np.all(ineq_results[ineq_indices]) and np.all(eq_results[eq_indices]):
                valid_subpolytopes.append((cp_idx, sub_idx))

        return valid_subpolytopes


class MonodromyLPISA:
    """Represents an ISA for monodromy linear programming.

    This class encapsulates a set of gates and provides methods to iterate over
    possible gate sequences, either in a predefined order or sorted by cost.

    Attributes:
        MAX_N (int): The maximum number of gates in a sequence.
        gate_set (Tuple[MonodromyLPGate, ...]): The set of available gates.
        defined (bool): Whether all gates in the set are fully defined.
        isa_ordered (bool): Whether to use the predefined gate order.
        sequence_length (int): The fixed length of gate sequences (if specified).

    Methods:
        reset: Reset the iterator.
        color_map: Generate a color map for visualization.
        __next__: Generate the next gate sequence.
    """

    def __init__(
        self,
        *gate_set: MonodromyLPGate,
        fixed_sequence_length: Optional[int] = None,
    ):
        """Initialize MonodromyLPISA.

        optional parameters useful for debugging.
        - sequence_length will guarantee has this number of gates. can be used
        together to specify a singular gate sequence to be tested. will override sorting by increasing costs, will gate sequence
        exactly in the ordered defined by gate_set (can include duplicates).
        """
        self.MAX_N = 8
        self.gate_set = list(gate_set)
        # XXX for simplicitly must be all or nothing.
        # partial constant subs can come later
        self.defined = all(gate.defined for gate in gate_set)
        if not self.defined:
            # XXX this simplification is because
            # otherwise idk how to order the undefined gate by cost
            assert len(self.gate_set) == 1

        self._fixed_sequence_length = fixed_sequence_length or 0
        assert (
            len(self.gate_set) >= self._fixed_sequence_length
        ), "Sequence length cannot exceed gate set size"
        assert self.MAX_N >= 2, "MAX_N must be at least 2"

    @property
    def color_map(self):
        colors = cm.tab10(np.linspace(0, 1.0, len(self.gate_set)))
        return {gate: color for gate, color in zip(self.gate_set, colors)}

    def enumerate(self, use_ordered_sequences, pad_undetermined):
        """Returns a generator."""
        self.index = 0
        self.priority_queue = [(0, [])]
        if use_ordered_sequences and self._fixed_sequence_length:
            return self._fixed_length_sequence()
        elif use_ordered_sequences:
            return self._priority_queue_sequence()
        elif not pad_undetermined:
            return self._unordered_sequence()
        else:
            return self._unordered_padded_sequence()

    def _fixed_length_sequence(self):
        if self.index != 0:
            return
        self.index += 1
        yield list(self.gate_set[: self._fixed_sequence_length])

    def _priority_queue_sequence(self):
        # otherwise, sort by cheapest sequences first.
        cost_1q = 1e-3  # TODO adjust offset cost for 1Q gate layers

        while True:  # keep popping until find something valid
            if not self.priority_queue:
                return

            candidate_sum_cost, candidate_sequence = heapq.heappop(self.priority_queue)

            if len(candidate_sequence) >= self.MAX_N:  # if exceeds MAX_N don't
                self.index += 1
                yield candidate_sequence

            # extend into next heap set, preserve lexicographic ordering
            for basis_gate in self.gate_set:
                # sequence must be in decreasing order
                if candidate_sequence and basis_gate.cost > candidate_sequence[-1].cost:
                    continue

                next_total_cost = candidate_sum_cost + basis_gate.cost + cost_1q
                next_sequence = candidate_sequence + [basis_gate]
                heapq.heappush(self.priority_queue, (next_total_cost, next_sequence))

            # recall our LP solver is only well-defined for atleast two gates
            if len(candidate_sequence) >= 2:
                self.index += 1
                yield candidate_sequence

    def _unordered_sequence(self):
        yield self.gate_set

    def _unordered_padded_sequence(self):
        identity = MonodromyLPGate.from_unitary(np.eye(4), cost=0.0, name="id")
        yield list(reversed(self.gate_set + [identity]))
