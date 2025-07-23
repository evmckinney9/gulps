import heapq
import itertools
import logging
from typing import Generator, List

import numpy as np
from monodromy.coverage import gates_to_coverage
from monodromy.polytopes import ConvexPolytope
from qiskit.circuit import Gate

from gulps.utils.invariants import GateInvariants

logger = logging.getLogger(__name__)

COST_1Q = 1e-4  # adjust offset cost for 1Q gate layers


class ISAInvariants:
    """Base class for ISA invariants."""

    def __init__(
        self,
        gate_set: List[GateInvariants] | List[np.ndarray],
        costs: List[float],
        precompute_polytopes: bool = False,
    ):
        if not gate_set:
            raise ValueError("gate_set can't be empty.")
        if len(gate_set) != len(costs):
            raise ValueError("gate_set and costs must have the same length.")
        if isinstance(gate_set[0], GateInvariants):  # XXX assuming uniform dtypes
            self.gate_set = gate_set
        else:
            self.gate_set = [GateInvariants.from_unitary(g) for g in gate_set]
        if len(self.gate_set) != len(set(self.gate_set)):
            raise ValueError("gate_set must contain unique GateInvariants.")

        self.cost_dict = {g: c for g, c in zip(self.gate_set, costs)}
        self._precompute_polytopes = precompute_polytopes
        if precompute_polytopes:
            self._build_coverage_set()

    def enumerate(
        self, max_depth: int = 10
    ) -> Generator[List[GateInvariants], None, None]:
        """Generate all ordered gate sequences up to max_depth."""
        counter = itertools.count()  # acts as cost tie-breaker
        priority_queue = [(0, next(counter), [])]  # (cost, unique_index, sequence)

        while priority_queue:
            cost, _, sequence = heapq.heappop(priority_queue)

            if len(sequence) == max_depth:
                break

            for gate in self.gate_set:
                # enforce monotonic sequence cost order
                if sequence and self.cost_dict[gate] < self.cost_dict[sequence[-1]]:
                    continue
                new_sequence = sequence + [gate]
                new_cost = cost + self.cost_dict[gate] + COST_1Q
                heapq.heappush(priority_queue, (new_cost, next(counter), new_sequence))

            if len(sequence) >= 2:
                yield sequence

    def _build_coverage_set(self):
        self.coverage_set = gates_to_coverage(
            *[g.unitary for g in self.gate_set],
            costs=[self.cost_dict[g] for g in self.gate_set],
            names=[f"{g.name}_{i}" for i, g in enumerate(self.gate_set)],
            single_qubit_cost=COST_1Q,
            instructions=self.gate_set,
        )

    def polytope_lookup(
        self, target: GateInvariants
    ) -> tuple[List[GateInvariants], bool]:
        """Return a gate sentence that spans the target via convex polytope lookup."""
        if not hasattr(self, "coverage_set"):
            raise ValueError("Polytope coverage set not precomputed.")

        # # less optimal but for debugging lets check both
        # for convex_polytope in self.coverage_set:
        #     rho_bool = [False, False]
        #     if convex_polytope.has_element(target.monodromy):
        #         rho_bool[0] = True
        #     if convex_polytope.has_element(target.rho_reflect.monodromy):
        #         rho_bool[1] = True
        #     if any(rho_bool):
        #         print(rho_bool)
        #         return convex_polytope.instructions, rho_bool[1]
        # return None, None

        for convex_polytope in self.coverage_set:
            if convex_polytope.has_element(target.monodromy):
                return convex_polytope.instructions, False
            elif convex_polytope.has_element(target.rho_reflect.monodromy):
                logger.debug(
                    "lookup falls back to rho-reflect, check did you not enforce alcove_c2 on the target gate?"
                )
                return convex_polytope.instructions, True
        return None, None

        # XXXX does this do something different than the original?
        # xv = np.array(target.monodromy).reshape((3, 1))  # monodromy vector
        # ineq_results = np.full(len(self._ineq_matrix), True)
        # eq_results = np.full(len(self._eq_matrix), True)

        # if self._ineq_matrix.size > 0:
        #     ineq_results = (
        #         self._ineq_matrix[:, 1:] @ xv
        #     ).squeeze() + self._ineq_matrix[:, 0] >= -1e-8

        # if self._eq_matrix.size > 0:
        #     eq_results = (
        #         np.abs((self._eq_matrix[:, 1:] @ xv).squeeze() + self._eq_matrix[:, 0])
        #         <= 1e-8
        #     )

        # valid_subpolytopes = []
        # for (cp_idx, sub_idx), ineq_indices in self._subpolytope_to_ineq.items():
        #     eq_indices = self._subpolytope_to_eq[(cp_idx, sub_idx)]
        #     if np.all(ineq_results[ineq_indices]) and np.all(eq_results[eq_indices]):
        #         valid_subpolytopes.append((cp_idx, sub_idx))
        # valid_subpolytopes = []
        # for cp_idx, circuit_polytope in enumerate(self.coverage_set):
        #     for sub_idx, subpolytope in enumerate(circuit_polytope.convex_subpolytopes):
        #         if subpolytope.contains(target.monodromy):
        #             valid_subpolytopes.append((cp_idx, sub_idx))
        # if not valid_subpolytopes:
        #     return None

        # best_cp_idx = min(cp_idx for cp_idx, _ in valid_subpolytopes)
        # return self.coverage_set[best_cp_idx].instructions

    # def _preprocess_coverage_set(self):
    #     unique_inequalities = []
    #     unique_equalities = []
    #     subpolytope_to_ineq = {}
    #     subpolytope_to_eq = {}

    #     for cp_idx, circuit_polytope in enumerate(self.coverage_set):
    #         for sub_idx, subpolytope in enumerate(circuit_polytope.convex_subpolytopes):
    #             subpolytope_to_ineq[(cp_idx, sub_idx)] = []
    #             subpolytope_to_eq[(cp_idx, sub_idx)] = []

    #             for ineq in subpolytope.inequalities:
    #                 ineq_tuple = tuple(ineq)
    #                 if ineq_tuple not in unique_inequalities:
    #                     unique_inequalities.append(ineq_tuple)
    #                 idx = unique_inequalities.index(ineq_tuple)
    #                 subpolytope_to_ineq[(cp_idx, sub_idx)].append(idx)

    #             for eq in subpolytope.equalities:
    #                 eq_tuple = tuple(eq)
    #                 if eq_tuple not in unique_equalities:
    #                     unique_equalities.append(eq_tuple)
    #                 idx = unique_equalities.index(eq_tuple)
    #                 subpolytope_to_eq[(cp_idx, sub_idx)].append(idx)

    #     self._ineq_matrix = np.array(unique_inequalities)
    #     self._eq_matrix = np.array(unique_equalities)
    #     self._subpolytope_to_ineq = subpolytope_to_ineq
    #     self._subpolytope_to_eq = subpolytope_to_eq
