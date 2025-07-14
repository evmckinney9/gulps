import heapq
import itertools
from typing import Generator, List

import numpy as np
from qiskit.circuit import Gate

from gulps.utils.invariants import GateInvariants

COST_1Q = 1e-2  # adjust offset cost for 1Q gate layers


class ISAInvariants:
    """Base class for ISA invariants."""

    def __init__(
        self,
        gate_set: List[GateInvariants] | List[np.ndarray],
        costs: List[float],
        precompute_polytopes: bool = False,
    ):
        if not gate_set:
            raise ValueError("gate_set must contain at least one GateInvariants.")
        if len(gate_set) != len(costs):
            raise ValueError("gate_set and costs must have the same length.")
        if isinstance(gate_set[0], GateInvariants):
            self.gate_set = gate_set
        else:
            self.gate_set = [GateInvariants.from_unitary(g) for g in gate_set]
        if len(self.gate_set) != len(set(self.gate_set)):
            raise ValueError("gate_set must contain unique GateInvariants.")

        self.cost_dict = {g: c for g, c in zip(self.gate_set, costs)}
        self._precompute_polytopes = precompute_polytopes

    def enumerate(
        self, max_depth: int = 10
    ) -> Generator[List[GateInvariants], None, None]:
        """Generate all ordered gate sequences up to max_depth."""
        counter = itertools.count()  # acts as cost tie-breaker
        priority_queue = [(0, next(counter), [])]  # (cost, unique_index, sequence)

        while priority_queue:
            cost, _, sequence = heapq.heappop(priority_queue)

            if len(sequence) == max_depth:
                yield sequence

            for gate in self.gate_set:
                # enforce monotonic sequence cost order
                if sequence and self.cost_dict[gate] < self.cost_dict[sequence[-1]]:
                    continue
                new_sequence = sequence + [gate]
                new_cost = cost + self.cost_dict[gate] + COST_1Q
                heapq.heappush(priority_queue, (new_cost, next(counter), new_sequence))

            if len(sequence) >= 2:
                yield sequence
