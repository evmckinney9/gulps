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

"""Which pulses of an interaction to calibrate.

A candidate is a pulse given as ``(cost, gate)``: a fractional power of a gate
(see :func:`power_candidates`), or a point of a measured trajectory. Each set
of candidates configures a :class:`~gulps.decomposition.GulpsDecomposer`,
which we score by its Haar-average cost or its total cost for a supplied
workload. If a set cannot reach the required targets within the depth bound,
it scores infinity.
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


def power_candidates(
    base_gate: Gate, strengths: Sequence[float] = GRID, pulse_overhead: float = 0.0
) -> list[tuple[float, Gate]]:
    """The pulses ``base_gate ** k`` for each ``k`` in ``strengths``.

    Args:
        base_gate: The Qiskit gate to power.
        strengths: The powers, in ``(0, 1]``.
        pulse_overhead: Fixed cost per entangling pulse. A pulse of strength k
            costs k + pulse_overhead.
    """
    if not math.isfinite(pulse_overhead) or pulse_overhead < 0:
        raise ValueError("pulse_overhead must be finite and non-negative")
    return [(float(k) + pulse_overhead, base_gate.power(k)) for k in strengths]


@dataclass(frozen=True)
class Calibration:
    """The candidates chosen by :func:`~gulps.analysis.calibration.calibrate`.

    Args:
        budget_scores: The score at each budget from one through the requested
            maximum.
        budget_choices: The indices of the candidates chosen at each budget.
    """

    budget_scores: tuple[float, ...]
    budget_choices: tuple[tuple[int, ...], ...]

    @property
    def choices(self) -> tuple[int, ...]:
        """The indices of the candidates chosen at the full budget."""
        return self.budget_choices[-1]

    @property
    def score(self) -> float:
        """The score at the full budget."""
        return self.budget_scores[-1]

    def plot(self, ax: matplotlib.axes.Axes | None = None) -> matplotlib.axes.Axes:
        """Plot score against budget on ``ax``, or on new axes."""
        from gulps.analysis.viz.report_viz import plot_calibration

        return plot_calibration(self, ax=ax)


@dataclass(frozen=True)
class StrengthSweep:
    """The score of each candidate on its own.

    Args:
        scores: The score of each candidate, in input order.
    """

    scores: tuple[float, ...]

    @property
    def best(self) -> tuple[int, float]:
        """The ``(index, score)`` of the best candidate."""
        i = min(range(len(self.scores)), key=self.scores.__getitem__)
        return i, self.scores[i]

    def plot(self, ax: matplotlib.axes.Axes | None = None) -> matplotlib.axes.Axes:
        """Plot score against candidate index and mark the best point."""
        from gulps.analysis.viz.report_viz import plot_strength_sweep

        return plot_strength_sweep(self, ax=ax)


def _depth_error(max_depth: int, workload: object) -> SearchDepthError:
    """No candidate covers the Haar measure or the workload within ``max_depth``."""
    covered = "the Haar measure" if workload is None else "the workload"
    return SearchDepthError(
        f"no candidate set covers {covered} within max_depth={max_depth}; increase max_depth"
    )


def _scorer(
    candidates: Sequence[tuple[float, Gate]],
    local_layer_cost: float,
    workload: QuantumCircuit | Targets | None,
    max_depth: int,
) -> Callable[[tuple[int, ...]], float]:
    """Memoized score of a set of candidate indices: Haar or workload cost."""
    if not candidates:
        raise ValueError("candidates must not be empty")
    if isinstance(workload, QuantumCircuit):
        workload = two_qubit_blocks(workload)
    if workload is not None:
        workload = LocalEquivalenceClass.from_unitaries(workload)

    @cache
    def score(chosen: tuple[int, ...]) -> float:
        decomposer = GulpsDecomposer(
            [candidates[i][1] for i in chosen],
            [candidates[i][0] for i in chosen],
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

    return lambda chosen: score(tuple(sorted(chosen)))


def strength_sweep(
    candidates: Sequence[tuple[float, Gate]],
    local_layer_cost: float = 0.0,
    workload: QuantumCircuit | Targets | None = None,
    max_depth: int = MAX_DEPTH,
) -> StrengthSweep:
    """Score each candidate on its own.

    Args:
        candidates: The pulses to score, each as ``(cost, gate)``; see
            :func:`power_candidates`.
        local_layer_cost: The cost of one simultaneous local-gate layer. A
            sentence of n pulses has n + 1 such layers.
        workload: A circuit or a list of targets to score instead of the Haar
            average.
        max_depth: Maximum two-qubit sentence depth. Candidates that exceed it
            score infinity; an all-infinite search raises ``SearchDepthError``.
    """
    score = _scorer(candidates, local_layer_cost, workload, max_depth)
    scores = tuple(score((i,)) for i in range(len(candidates)))
    if all(value == math.inf for value in scores):
        raise _depth_error(max_depth, workload)
    return StrengthSweep(scores)


def calibrate(
    candidates: Sequence[tuple[float, Gate]],
    budget: int = 1,
    local_layer_cost: float = 0.0,
    workload: QuantumCircuit | Targets | None = None,
    max_depth: int = MAX_DEPTH,
) -> Calibration:
    """Choose up to ``budget`` of the candidates.

    At each step, the search adds the candidate that lowers the score most,
    then revisits each chosen candidate once while holding the others fixed.
    It stops adding candidates when it reaches ``budget`` or exhausts them.

    Args:
        candidates: The pulses to choose from, each as ``(cost, gate)``; see
            :func:`power_candidates`.
        budget: The maximum number of candidates.
        local_layer_cost: The cost of one simultaneous local-gate layer. A
            sentence of n pulses has n + 1 such layers.
        workload: A circuit or a list of targets to score instead of the Haar
            average.
        max_depth: Maximum two-qubit sentence depth. Candidates that exceed it
            score infinity; an all-infinite search raises ``SearchDepthError``.
    """
    if budget < 1:
        raise ValueError(f"budget must be positive, got {budget}")
    score = _scorer(candidates, local_layer_cost, workload, max_depth)
    indices = range(len(candidates))
    chosen: list[int] = []
    scores: list[float] = []
    sets: list[tuple[int, ...]] = []
    for _ in range(min(budget, len(candidates))):
        chosen.append(
            min(
                (i for i in indices if i not in chosen),
                key=lambda i: score((*chosen, i)),
            )
        )
        for j in range(len(chosen)):
            others = chosen[:j] + chosen[j + 1 :]
            chosen[j] = min(
                (i for i in indices if i not in others),
                key=lambda i: score((*others, i)),
            )
        selected = score(tuple(chosen))
        if selected == math.inf:
            raise _depth_error(max_depth, workload)
        scores.append(selected)
        sets.append(tuple(sorted(chosen)))
    return Calibration(tuple(scores), tuple(sets))
