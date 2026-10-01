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

"""Which fractional powers of a gate to calibrate.

Each candidate set configures a :class:`~gulps.decomposition.GulpsDecomposer`,
which we score by its Haar-average cost or its total cost for a supplied
workload. If a candidate cannot reach the required targets within the depth
bound, it receives infinite cost.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate

from gulps._accelerate import MAX_DEPTH
from gulps.analysis.coverage import Targets, coverage_report, two_qubit_blocks
from gulps.decomposition import GulpsDecomposer, SearchDepthError
from gulps.invariants import LocalEquivalenceClass

if TYPE_CHECKING:
    import matplotlib.axes

#: Default strengths: 55 multiples of 1/60 from 1/10 through 1.
#: Supply strengths that match the device's calibration range instead.
GRID: tuple[float, ...] = tuple((np.arange(6, 61) / 60).tolist())


@dataclass(frozen=True)
class Calibration:
    """Strengths and scores returned by :func:`~gulps.analysis.calibration.calibrate`.

    Args:
        budget_costs: The score at each budget from one through the requested
            maximum.
        budget_strengths: The strengths chosen at each budget.
    """

    budget_costs: tuple[float, ...]
    budget_strengths: tuple[tuple[float, ...], ...]

    @property
    def strengths(self) -> tuple[float, ...]:
        """The strengths chosen at the full budget."""
        return self.budget_strengths[-1]

    @property
    def cost(self) -> float:
        """The score at the full budget."""
        return self.budget_costs[-1]

    def plot(self, ax: matplotlib.axes.Axes | None = None) -> matplotlib.axes.Axes:
        """Plot score against budget on ``ax``, or on new axes."""
        from gulps.analysis.viz.report_viz import plot_calibration

        return plot_calibration(self, ax=ax)


@dataclass(frozen=True)
class StrengthSweep:
    """The score of every single strength on a grid.

    Args:
        strengths: The grid of powers.
        costs: The score at each strength.
    """

    strengths: tuple[float, ...]
    costs: tuple[float, ...]

    @property
    def best(self) -> tuple[float, float]:
        """The cheapest ``(strength, cost)`` on the grid."""
        i = min(range(len(self.costs)), key=self.costs.__getitem__)
        return self.strengths[i], self.costs[i]

    def plot(self, ax: matplotlib.axes.Axes | None = None) -> matplotlib.axes.Axes:
        """Plot cost against strength and mark the best point."""
        from gulps.analysis.viz.report_viz import plot_strength_sweep

        return plot_strength_sweep(self, ax=ax)


def _depth_error(max_depth: int, workload: object) -> SearchDepthError:
    """No candidate covers the Haar measure or the workload within ``max_depth``."""
    covered = "the Haar measure" if workload is None else "the workload"
    return SearchDepthError(
        f"no strength set covers {covered} within max_depth={max_depth}; increase max_depth"
    )


def _scorer(
    base_gate: Gate,
    local_layer_cost: float,
    workload: QuantumCircuit | Targets | None,
    max_depth: int,
    pulse_overhead: float,
) -> Callable[[tuple[float, ...]], float]:
    """Memoized score of a strength set: Haar expected cost, or total cost over ``workload``."""
    if not math.isfinite(pulse_overhead) or pulse_overhead < 0:
        raise ValueError("pulse_overhead must be finite and non-negative")
    if isinstance(workload, QuantumCircuit):
        workload = two_qubit_blocks(workload)
    if workload is not None:
        workload = LocalEquivalenceClass.from_unitaries(workload)
    gate = cache(base_gate.power)

    @cache
    def score(strengths: tuple[float, ...]) -> float:
        decomposer = GulpsDecomposer(
            [gate(k) for k in strengths],
            [k + pulse_overhead for k in strengths],
            local_layer_cost,
            max_depth,
        )
        try:
            return (
                coverage_report(decomposer).expected_cost
                if workload is None
                else sum(cost for cost, _ in decomposer.select(workload))
            )
        except SearchDepthError:
            return math.inf

    return lambda strengths: score(tuple(sorted(strengths)))


def strength_sweep(
    base_gate: Gate,
    strengths: Sequence[float] = GRID,
    local_layer_cost: float = 0.0,
    workload: QuantumCircuit | Targets | None = None,
    max_depth: int = MAX_DEPTH,
    *,
    pulse_overhead: float = 0.0,
) -> StrengthSweep:
    """Score ``base_gate ** k`` for every ``k`` in ``strengths``.

    Args:
        base_gate: The Qiskit gate to power.
        strengths: The powers to score, in ``(0, 1]``.
        local_layer_cost: The cost of one simultaneous local-gate layer. A
            sentence of n pulses has n + 1 such layers.
        workload: A circuit or a list of targets to score instead of the Haar
            average.
        max_depth: Maximum two-qubit sentence depth. Candidates that exceed it
            score infinity; an all-infinite search raises ``SearchDepthError``.
        pulse_overhead: Fixed cost per entangling pulse, added to its strength.
            A pulse of strength k costs k + pulse_overhead.
    """
    grid = tuple(float(k) for k in strengths)
    if not grid:
        raise ValueError("strengths must not be empty")
    score = _scorer(base_gate, local_layer_cost, workload, max_depth, pulse_overhead)
    costs = tuple(score((k,)) for k in grid)
    if all(cost == math.inf for cost in costs):
        raise _depth_error(max_depth, workload)
    return StrengthSweep(grid, costs)


def calibrate(
    base_gate: Gate,
    budget: int = 1,
    strengths: Sequence[float] = GRID,
    local_layer_cost: float = 0.0,
    workload: QuantumCircuit | Targets | None = None,
    max_depth: int = MAX_DEPTH,
    *,
    pulse_overhead: float = 0.0,
) -> Calibration:
    """Choose up to ``budget`` powers of ``base_gate`` from ``strengths``.

    At each step, the search adds the grid point that lowers the score most,
    then revisits each chosen strength once while holding the others fixed.
    It stops adding strengths when it reaches ``budget`` or exhausts the grid.

    Args:
        base_gate: The Qiskit gate to power.
        budget: The maximum number of distinct powers.
        strengths: The powers to choose from, in ``(0, 1]``.
        local_layer_cost: The cost of one simultaneous local-gate layer. A
            sentence of n pulses has n + 1 such layers.
        workload: A circuit or a list of targets to score instead of the Haar
            average.
        max_depth: Maximum two-qubit sentence depth. Candidates that exceed it
            score infinity; an all-infinite search raises ``SearchDepthError``.
        pulse_overhead: Fixed cost per entangling pulse, added to its strength.
            A pulse of strength k costs k + pulse_overhead.
    """
    grid = [float(k) for k in strengths]
    if budget < 1:
        raise ValueError(f"budget must be positive, got {budget}")
    if not grid:
        raise ValueError("strengths must not be empty")
    score = _scorer(base_gate, local_layer_cost, workload, max_depth, pulse_overhead)
    chosen: list[float] = []
    costs: list[float] = []
    sets: list[tuple[float, ...]] = []
    for _ in range(min(budget, len(set(grid)))):
        chosen.append(
            min((k for k in grid if k not in chosen), key=lambda k: score((*chosen, k)))
        )
        for i in range(len(chosen)):
            others = chosen[:i] + chosen[i + 1 :]
            chosen[i] = min(
                (k for k in grid if k not in others),
                key=lambda k: score((*others, k)),
            )
        selected_cost = score(tuple(chosen))
        if selected_cost == math.inf:
            raise _depth_error(max_depth, workload)
        costs.append(selected_cost)
        sets.append(tuple(sorted(chosen)))
    return Calibration(tuple(costs), tuple(sets))
