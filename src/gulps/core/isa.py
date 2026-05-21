# Copyright 2025-2026 Lev S. Bishop, Evan McKinney
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ISA data structures: ContinuousISA, DiscreteISA, and their generators."""

import heapq
import warnings
from abc import ABC
from dataclasses import dataclass, field

import numpy as np
from qiskit.circuit import Gate

from gulps import GateInvariants


class ISAInvariants(ABC):
    """Abstract base class for instruction set architectures.

    Attributes:
        gate_set: List of basis gate invariants.
        cost_dict: Mapping from gate invariants to their costs.
        single_qubit_cost: Cost offset per gate to prioritize shorter depth sequences.
        max_sequence_length: Maximum gate sentence length for full Weyl chamber coverage.
    """

    gate_set: list[GateInvariants]
    cost_dict: dict[GateInvariants, float]
    single_qubit_cost: float
    max_sequence_length: int
    # NOTE this is useful for tiebreakers between fractional parts
    # example 2 iswaps versus 4 sqrtiswaps both cost 4
    # with adjustment it is 2+2eps versus 2+4eps
    MIN_COST_1Q = 1e-8

    def sentence_cost(self, gates: list[GateInvariants]) -> float:
        """Cost of an n-gate sentence: ``sum(gate_costs) + (n+1) * single_qubit_cost``.

        The ``(n+1)`` counts single-qubit layers as a function of 2q-depth:
        depth 0 has one 1q layer (bare local decomposition), and each added
        2q gate brings one more, so depths 0, 1, 2, … carry 1, 2, 3, … 1q
        layers.
        """
        return (
            sum(self.cost_dict[g] for g in gates)
            + (len(gates) + 1) * self.single_qubit_cost
        )


class ContinuousISA(ISAInvariants):
    """ISA with continuous gate families G(k) = k * base, k in [k_lb, 1].

    For single-family ISA, gate_set contains one base gate.
    For heterogeneous ISA (future), gate_set contains multiple base gates.

    Attributes:
        gate_set: List of base gate invariants (one per family).
        cost_dict: Mapping from base gates to cost rates (cost = k * rate).
        k_lb: Minimum nonzero k value (gates with k < k_lb are pruned).
        single_qubit_cost: Cost offset per gate to prioritize shorter depth sequences.
        max_sequence_length: Maximum gate sentence length for the LP.
    """

    def __init__(
        self,
        gate_set: list[GateInvariants],
        cost_dict: dict[GateInvariants, float] | None = None,
        k_lb: float = 0.1,
        single_qubit_cost: float = ISAInvariants.MIN_COST_1Q,
        max_sequence_length: int = 3,
    ):
        """Validate gate_set and cost_dict, store fields."""
        if not gate_set:
            raise ValueError("gate_set can't be empty.")
        cost_dict = cost_dict or {}
        missing = [g for g in gate_set if g not in cost_dict]
        if missing:
            raise ValueError(f"cost_dict missing entries for: {missing}")
        self.gate_set = gate_set
        self.cost_dict = cost_dict
        self.k_lb = k_lb
        self.single_qubit_cost = single_qubit_cost
        self.max_sequence_length = max_sequence_length

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
        max_sequence_length: int = 3,
    ) -> "ContinuousISA":
        """Create ContinuousISA from a single base gate."""
        base_inv = GateInvariants.from_unitary(base_gate, name=name)
        return cls(
            gate_set=[base_inv],
            cost_dict={base_inv: 1.0},  # cost rate per unit k
            single_qubit_cost=single_qubit_cost,
            max_sequence_length=max_sequence_length,
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
        max_sequence_length: int = 8,
    ):
        """Initialize DiscreteISA from gates, costs, and optional names."""
        if not gate_set:
            raise ValueError("gate_set can't be empty.")
        if len(gate_set) != len(costs):
            raise ValueError("gate_set and costs must have the same length.")
        if not all(np.isfinite(c) and c >= 0 for c in costs):
            raise ValueError(f"costs must be finite and non-negative, got {costs}")
        if not np.isfinite(single_qubit_cost):
            raise ValueError(
                f"single_qubit_cost must be finite, got {single_qubit_cost}"
            )
        if not isinstance(max_sequence_length, int) or max_sequence_length < 1:
            raise ValueError(
                f"max_sequence_length must be a positive integer, "
                f"got {max_sequence_length}"
            )
        if names is None:
            names = [None] * len(gate_set)
        self.gate_set = [
            GateInvariants.from_unitary(g, name=n) for g, n in zip(gate_set, names)
        ]
        if len(self.gate_set) != len(set(self.gate_set)):
            raise ValueError("gate_set must contain unique GateInvariants.")

        self.cost_dict = {g: c for g, c in zip(self.gate_set, costs)}
        self.single_qubit_cost = max(single_qubit_cost, self.MIN_COST_1Q)
        self.max_sequence_length = max_sequence_length

        self.coverage_set: list = []
        self._polytope_by_sentence: dict[tuple, object] = {}
        self._precompute_polytopes = precompute_polytopes
        if precompute_polytopes:
            self._build_coverage()

        self._sentence_cache: list[tuple] = []
        self._sentence_iter = self._enumerate_pq()

    def _build_coverage(self) -> None:
        """Populate ``coverage_set`` and ``_polytope_by_sentence`` via monodromy.

        Falls back to enumeration (``_precompute_polytopes = False``) when the
        optional ``monodromy`` dependency is missing. ``isa_to_coverage`` sorts
        each polytope's ``instructions`` by ``cost_dict.__getitem__``, matching
        the non-decreasing-cost canonicalization the PQ enforces, so tuple
        keys agree between the dict and ``candidate_sentences``.
        """
        try:
            from gulps.core.coverage import isa_to_coverage
        except ImportError:
            warnings.warn(
                "Coverage precomputation requires the 'monodromy' package, which "
                "is not installed.  Falling back to enumeration-based decomposition "
                "(precompute_polytopes=False).  "
                "Install with: pip install -r requirements-monodromy.txt",
                UserWarning,
                stacklevel=3,
            )
            self._precompute_polytopes = False
            return
        self.coverage_set = isa_to_coverage(self)
        self._polytope_by_sentence = {
            tuple(p.instructions): p for p in self.coverage_set if p.instructions
        }

    def _enumerate_pq(self):
        """Walk the cost-monotone PQ once; yield ``(sentence, polytope, strength)``.

        Bounded by ``max_sequence_length``; prunes zero-cost gate reuse and
        enforces non-decreasing cost within a sentence to canonicalize
        permutations. ``strength`` is the sum of gate strengths along the
        sentence.  One-shot generator: consumed lazily by :meth:`candidate_sentences`, which memoizes yields into ``_sentence_cache`` for replay across calls.
        """
        cost_dict = self.cost_dict
        gate_set = self.gate_set
        max_seq = self.max_sequence_length
        s_q_c = self.single_qubit_cost
        polytope_for = self._polytope_by_sentence
        have_polytopes = bool(polytope_for)

        pq: list[tuple[float, tuple]] = [(self.sentence_cost(()), ())]
        while pq:
            parent_cost, sequence = heapq.heappop(pq)
            if len(sequence) < max_seq:
                last_cost = cost_dict[sequence[-1]] if sequence else None
                for gate in gate_set:
                    gate_cost = cost_dict[gate]
                    if gate_cost == 0.0 and gate in sequence:
                        continue
                    if last_cost is not None and gate_cost < last_cost:
                        continue
                    new_seq = sequence + (gate,)
                    heapq.heappush(pq, (parent_cost + gate_cost + s_q_c, new_seq))
            if not sequence:
                continue
            polytope = polytope_for.get(sequence)
            if have_polytopes and polytope is None:
                continue  # pruned from coverage_set: no target can match
            strength = sum(g.strength for g in sequence)
            yield sequence, polytope, strength

    def candidate_sentences(self):
        """Yield ``(sentence, polytope, strength)`` in cost-monotone order, memoized.

        First replays ``_sentence_cache`` (free for already-walked sentences),
        then lazily advances the underlying PQ generator, appending each new
        entry to the cache as it goes. Subsequent calls hit the cache for
        sentences past callers have already seen.
        """
        yield from self._sentence_cache
        for entry in self._sentence_iter:
            self._sentence_cache.append(entry)
            yield entry
