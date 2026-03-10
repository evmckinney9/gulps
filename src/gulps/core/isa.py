"""ISA data structures: ContinuousISA, DiscreteISA, and their generators."""

import heapq
import itertools
import logging
from abc import ABC
from collections.abc import Generator
from dataclasses import dataclass, field

import numpy as np
from qiskit.circuit import Gate

from gulps import GateInvariants

logger = logging.getLogger(__name__)


class ISAInvariants(ABC):
    """Abstract base class for instruction set architectures.

    Attributes:
        gate_set: List of basis gate invariants.
        cost_dict: Mapping from gate invariants to their costs.
        single_qubit_cost: Cost offset per gate to prioritize shorter depth sequences.
    """

    gate_set: list[GateInvariants]
    cost_dict: dict[GateInvariants, float]
    single_qubit_cost: float
    # NOTE this is useful for tiebreakers between fractional parts
    # example 2 iswaps versus 4 sqrtiswaps both cost 4
    # with adjustment it is 2+2eps versus 2+4eps
    MIN_COST_1Q = 1e-8


@dataclass
class ContinuousISA(ISAInvariants):
    """ISA with continuous gate families G(k) = k * base, k ∈ [k_lb, 1].

    For single-family ISA, gate_set contains one base gate.
    For heterogeneous ISA (future), gate_set contains multiple base gates.

    Attributes:
        gate_set: List of base gate invariants (one per family).
        cost_dict: Mapping from base gates to cost rates (cost = k * rate).
        k_lb: Minimum nonzero k value (gates with k < k_lb are pruned).
        single_qubit_cost: Cost offset per gate to prioritize shorter depth sequences.
    """

    gate_set: list[GateInvariants]
    cost_dict: dict[GateInvariants, float] = field(default_factory=dict)
    k_lb: float = 0.1
    single_qubit_cost: float = ISAInvariants.MIN_COST_1Q

    @property
    def is_single_family(self) -> bool:
        """Return True when the ISA uses exactly one base gate family."""
        return len(self.gate_set) == 1

    @classmethod
    def from_base_gate(
        cls,
        base_gate: Gate | np.ndarray,
        name: str | None = None,
        single_qubit_cost: float = ISAInvariants.MIN_COST_1Q,
    ) -> "ContinuousISA":
        """Create ContinuousISA from a single base gate.

        Args:
            base_gate: Base gate unitary or Qiskit Gate.
            name: Optional name for the base gate.
            single_qubit_cost: Cost offset per gate to prioritize shorter circuits.

        Returns:
            ContinuousISA instance.
        """
        base_inv = GateInvariants.from_unitary(base_gate, name=name)
        cost_dict = {base_inv: 1.0}  # cost rate per unit k
        return cls(
            gate_set=[base_inv],
            cost_dict=cost_dict,
            single_qubit_cost=single_qubit_cost,
        )


class DiscreteISA(ISAInvariants):
    """Discrete ISA with fixed gate set and cost-ordered enumeration.

    Attributes:
        gate_set: List of discrete gate invariants.
        cost_dict: Mapping from gate invariants to their costs.
    """

    def __init__(
        self,
        gate_set: list[Gate] | list[np.ndarray],
        costs: list[float],
        names: list[str] | None = None,
        precompute_polytopes: bool = False,
        single_qubit_cost: float = ISAInvariants.MIN_COST_1Q,
    ):
        """Initialize DiscreteISA from gates, costs, and optional names."""
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
        if self.single_qubit_cost <= 0.0:
            logger.warning(
                "Setting single_qubit_cost to zero may lead to unexpected behavior. "
                "This offset is used to prioritize otherwise cost-equivalent gate sequences with fewer total segments. "
                "For example, to prioritize 2 iswaps over 4 sqrtiswaps."
            )
            self.single_qubit_cost = self.MIN_COST_1Q
        self._precompute_polytopes = precompute_polytopes
        if precompute_polytopes:
            from gulps.core.coverage import isa_to_coverage

            self.coverage_set = isa_to_coverage(self)

    def enumerate(self, max_depth: int) -> Generator[list[GateInvariants], None, None]:
        """Generate all ordered gate sequences up to max_depth (inclusive)."""
        counter = itertools.count()  # acts as cost tie-breaker
        # (cost, unique_index, sequence)
        priority_queue = [(self.single_qubit_cost, next(counter), [])]

        while priority_queue:
            cost, _, sequence = heapq.heappop(priority_queue)

            if len(sequence) < max_depth:
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
                    heapq.heappush(
                        priority_queue, (new_cost, next(counter), new_sequence)
                    )

            if len(sequence) >= 1:
                yield sequence

    def polytope_lookup(self, target: GateInvariants) -> list[GateInvariants] | None:
        """Return a gate sentence that spans the target via convex polytope lookup.

        Args:
            target: Alcove-normalized target gate invariants. The caller is responsible
                for ensuring the target is in the alcove (enforce_alcove=True), which
                eliminates the need to check rho-reflected variants here.

        Returns:
            Sorted list of gate invariants forming a valid sentence, or None if
            no polytope contains the target.
        """
        if not hasattr(self, "coverage_set"):
            raise ValueError("Polytope coverage set not precomputed.")

        for convex_polytope in self.coverage_set:
            if convex_polytope.has_element(target.monodromy):
                # Sort instructions by cost for consistency with enumerate()
                return sorted(
                    convex_polytope.instructions, key=lambda g: self.cost_dict[g]
                )
        return None
