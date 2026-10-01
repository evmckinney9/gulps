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

"""Cost of an ISA over Haar-random targets, by integration or sampling.

``coverage_report`` visits the instruction set's gate sentences in cost order
until their computed Haar mass is within tolerance of one, weighting each
sentence by the mass it reaches for the first time. Closed-form integrals
are evaluated in floating point; stopping does not certify every target's
reachability. To evaluate a finite sample instead, ``empirical_cost`` selects
a sentence for each supplied target.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import count
from typing import TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ControlFlowOp, Gate
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    ConsolidateBlocks,
    HighLevelSynthesis,
    Unroll3qOrMore,
)

from gulps._accelerate import coverage_candidate
from gulps.analysis.region import ReachableRegion, RegionUnion
from gulps.decomposition import GulpsDecomposer
from gulps.invariants import LocalEquivalenceClass

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

# Stop when the computed union mass is this close to 1, not a set-coverage proof.
_FULL_COVERAGE_TOL = 1e-9

# Targets accepted by ``GulpsDecomposer.select``.
Targets = Sequence[Gate | Operator | np.ndarray | LocalEquivalenceClass]


@dataclass(frozen=True)
class SentenceCoverage:
    """A gate sentence, its reachable region, and its newly covered Haar mass.

    Args:
        cost: The cost of the sentence.
        gates: The sentence's gates, in order.
        region: One orientation of the sentence's reach; ``region.rho`` is the
            other.
        fresh_mass: The Haar fraction this sentence is the first to reach, or
            ``None`` when earlier sentences already cover its region.
    """

    cost: float
    gates: tuple[Gate, ...]
    region: ReachableRegion
    fresh_mass: float | None

    @property
    def names(self) -> tuple[str, ...]:
        """The gates' names."""
        return tuple(g.name for g in self.gates)


@dataclass(frozen=True)
class CoverageReport:
    """Haar coverage and cost of a fixed ISA, integrated in floating point.

    Args:
        rows: Every sentence the search emitted through the last entry, in
            search order, including those whose region earlier sentences
            already cover.
        decomposer: The instruction set the search ran on.
    """

    rows: tuple[SentenceCoverage, ...]
    decomposer: GulpsDecomposer = field(repr=False, compare=False)

    @property
    def entries(self) -> tuple[SentenceCoverage, ...]:
        """The cost-ordered rows that reach classes no earlier row reaches."""
        return tuple(r for r in self.rows if r.fresh_mass is not None)

    @property
    def expected_cost(self) -> float:
        """The Haar-weighted cost summed over the reported regions."""
        return sum(e.cost * e.fresh_mass for e in self.entries)

    @property
    def total_coverage(self) -> float:
        """The covered Haar fraction, within numerical tolerance of 1."""
        return sum(e.fresh_mass for e in self.entries)

    def __repr__(self) -> str:
        return (
            f"CoverageReport(expected_cost={self.expected_cost:.4f}, "
            f"coverage={self.total_coverage:.4f}, entries={len(self.entries)})"
        )

    def plot(self) -> matplotlib.figure.Figure | None:
        """Draw each entry's reachable region in its own Weyl-chamber subplot."""
        from gulps.analysis.viz.polytope_viz import plot_coverage_set

        return plot_coverage_set(self.entries)

    def plot_tree(self, names: Sequence[str] | None = None) -> matplotlib.figure.Figure:
        """Draw the search rows and the candidates pruned among them as a tree.

        Args:
            names: A display name for each gate of the decomposer, in order.
                Defaults to the gates' names.
        """
        from gulps.analysis.viz.polytope_viz import plot_search_tree

        return plot_search_tree(self, names)

    @property
    def cost_cdf(self) -> list[tuple[float, float]]:
        """``(cost, cumulative Haar fraction)`` at each cost, ascending."""
        mass: dict[float, float] = {}
        for e in self.entries:
            if e.fresh_mass > 0.0:
                mass[e.cost] = mass.get(e.cost, 0.0) + e.fresh_mass
        out, cum = [], 0.0
        for c in sorted(mass):
            cum += mass[c]
            out.append((c, cum))
        return out

    def percentile(self, q: float) -> float:
        """The cost at which the cumulative Haar fraction first reaches ``q``.

        ``percentile(0.5)`` estimates the median cost. ``percentile(1.0)``
        returns the endpoint of the reported distribution, not a guaranteed
        worst-case cost: small uncovered regions can remain when integration
        stops. Returns infinity if ``q`` exceeds the reported coverage by
        more than its numerical tolerance.

        Args:
            q: A cumulative probability from 0 through 1.
        """
        return next(
            (c for c, cum in self.cost_cdf if cum >= q - _FULL_COVERAGE_TOL), math.inf
        )

    def plot_reach(
        self, ax: matplotlib.axes.Axes | None = None
    ) -> matplotlib.axes.Axes:
        """Plot cumulative Haar coverage against cost on ``ax``, or on new axes."""
        from gulps.analysis.viz.report_viz import plot_reach_curve

        return plot_reach_curve(self, ax=ax)


def coverage_report(decomposer: GulpsDecomposer) -> CoverageReport:
    """Integrate Haar coverage and average cost of a fixed ISA.

    Uses closed-form integrals evaluated in floating point and stops when the
    computed covered mass is within a small tolerance of one. This does not
    certify coverage of every target. The expected cost omits any remaining
    tail; the mass tolerance alone does not bound that tail's cost.

    Args:
        decomposer: A fixed instruction set.

    Returns:
        Cost-ordered reachable regions and the Haar-average cost.
        Entries with zero Haar mass can overlap.

    Raises:
        SearchDepthError: If the search is exhausted at ``decomposer.max_depth``
            before the computed covered mass meets the stopping tolerance.
    """
    union = RegionUnion()
    rows = []
    for row in count():
        cost, gates, bounds = coverage_candidate(decomposer, row)
        region = ReachableRegion(tuple(bounds))
        rows.append(SentenceCoverage(cost, gates, region, union.add(region)))
        if union.mass >= 1.0 - _FULL_COVERAGE_TOL:
            break
    return CoverageReport(rows=tuple(rows), decomposer=decomposer)


@dataclass(frozen=True)
class SampledCost:
    """Costs of the selected sentences over a finite target sample.

    Args:
        costs: The selected cost for each target.
    """

    costs: tuple[float, ...]

    def __repr__(self) -> str:
        return (
            f"SampledCost(total_cost={self.total_cost:.4f}, "
            f"expected_cost={self.expected_cost:.4f}, n={len(self.costs)})"
        )

    @property
    def total_cost(self) -> float:
        """The cost summed over the sample."""
        return sum(self.costs, 0.0)

    @property
    def expected_cost(self) -> float:
        """The mean cost per target."""
        return self.total_cost / len(self.costs) if self.costs else 0.0

    def percentile(self, q: float) -> float:
        """The smallest sampled cost with at least a ``q`` fraction of samples at or below it.

        Returns 0.0 for an empty sample.
        """
        if not self.costs:
            return 0.0
        ordered = sorted(self.costs)
        return ordered[min(max(math.ceil(q * len(ordered)), 1), len(ordered)) - 1]

    def plot(self, ax: matplotlib.axes.Axes | None = None) -> matplotlib.axes.Axes:
        """Plot the sampled cost histogram on ``ax``, or on new axes."""
        from gulps.analysis.viz.report_viz import plot_cost_histogram

        return plot_cost_histogram(self, ax=ax)


def empirical_cost(
    decomposer: GulpsDecomposer, targets: QuantumCircuit | Targets
) -> SampledCost:
    """Select a sentence for every target and summarize the costs.

    Supply ``targets`` as a list of gates, ``Operator`` objects, 4-by-4 unitaries,
    or invariant classes. You can also supply a circuit to evaluate its
    consolidated two-qubit blocks, counting each branch and loop body once.
    """
    if isinstance(targets, QuantumCircuit):
        targets = two_qubit_blocks(targets)
    selected = decomposer.select(list(targets))
    return SampledCost(tuple(cost for cost, _ in selected))


def two_qubit_blocks(circuit: QuantumCircuit) -> list[Gate]:
    """The consolidated two-qubit operations of a copy of ``circuit``."""
    lower = PassManager(
        [
            HighLevelSynthesis(),
            Unroll3qOrMore(),
            ConsolidateBlocks(force_consolidate=True),
        ]
    )
    dag = circuit_to_dag(lower.run(circuit))
    targets = []
    for node in dag.topological_op_nodes():
        if isinstance(node.op, ControlFlowOp):
            for block in node.op.blocks:
                targets.extend(two_qubit_blocks(block))
        elif isinstance(node.op, Gate) and node.op.num_qubits == 2:
            targets.append(node.op)
    return targets
