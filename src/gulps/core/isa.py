import heapq
import itertools
import logging
from typing import Generator, List

import numpy as np
from qiskit.circuit import Gate

from gulps import GateInvariants
from gulps.core.coverage import isa_to_coverage

logger = logging.getLogger(__name__)


class ISAInvariants:
    """Base class for ISA invariants."""

    identity_inv = GateInvariants(logspec=(0.0, 0.0, 0.0, 0.0))
    # NOTE this is useful for tiebreakers between fractional parts
    # example 2 iswaps versus 4 sqrtiswaps both cost 4
    # with adjustment it is 2+2eps versus 2+4eps
    DEFAULT_COST_1Q = 1e-5  # adjust offset cost for 1Q gate layers

    def __init__(
        self,
        gate_set: List[Gate] | List[np.ndarray],
        costs: List[float],
        names: List[str] | None = None,
        precompute_polytopes: bool = False,
        single_qubit_cost: float = DEFAULT_COST_1Q,
    ):
        if not gate_set:
            raise ValueError("gate_set can't be empty.")
        if len(gate_set) != len(costs):
            raise ValueError("gate_set and costs must have the same length.")
        if names is None:
            names = [None] * len(gate_set)
        self.gate_set = [
            GateInvariants.from_unitary(g, name=n) for g, n in zip(gate_set, names)
        ]
        if len(self.gate_set) != len(set(self.gate_set)):
            raise ValueError("gate_set must contain unique GateInvariants.")

        self.cost_dict = {g: c for g, c in zip(self.gate_set, costs)}
        self.single_qubit_cost = single_qubit_cost
        self._precompute_polytopes = precompute_polytopes
        if precompute_polytopes:
            self.coverage_set = isa_to_coverage(self)

    def enumerate(
        self, max_depth: int = 32
    ) -> Generator[List[GateInvariants], None, None]:
        """Generate all ordered gate sequences up to max_depth."""
        counter = itertools.count()  # acts as cost tie-breaker
        priority_queue = [(0, next(counter), [])]  # (cost, unique_index, sequence)

        while priority_queue:
            cost, _, sequence = heapq.heappop(priority_queue)

            if len(sequence) == max_depth:
                continue

            for gate in self.gate_set:
                # skip if trying to reuse a zero-cost gate already in the sequence
                # used for example with SWAP-mirrors, cost 0.0 basis gates
                if self.cost_dict[gate] == 0.0 and gate in sequence:
                    continue
                # enforce monotonic sequence cost order
                if sequence and self.cost_dict[gate] < self.cost_dict[sequence[-1]]:
                    continue
                new_sequence = sequence + [gate]
                new_cost = cost + self.cost_dict[gate] + self.single_qubit_cost
                heapq.heappush(priority_queue, (new_cost, next(counter), new_sequence))

            if len(sequence) >= 2:
                yield sequence

    def polytope_lookup(
        self, target: GateInvariants
    ) -> tuple[List[GateInvariants], bool]:
        """Return a gate sentence that spans the target via convex polytope lookup."""
        if not hasattr(self, "coverage_set"):
            raise ValueError("Polytope coverage set not precomputed.")

        for convex_polytope in self.coverage_set:
            if convex_polytope.has_element(target.monodromy):
                return convex_polytope.instructions, False
            elif convex_polytope.has_element(target.rho_reflect.monodromy):
                logger.debug(
                    "lookup falls back to rho-reflect, check did you not enforce alcove_c2 on the target gate?"
                )
                return convex_polytope.instructions, True
        return None, None
        return None, None
