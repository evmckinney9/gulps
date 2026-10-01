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

"""Summary report visualizations comparing synthesis results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from gulps.analysis.calibration import Calibration, StrengthSweep
    from gulps.analysis.coverage import CoverageReport, SampledCost

_FIGSIZE = (4.5, 3.0)


def plot_cost_histogram(sample: SampledCost, ax: Axes | None = None) -> Axes:
    """Plot the sampled cost distribution, with the mean marked."""
    if ax is None:
        _fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.hist(sample.costs, bins="auto", color="0.6", edgecolor="white", linewidth=0.4)
    ax.axvline(sample.expected_cost, color="crimson", linewidth=2)
    ax.set_xlabel("cost")
    ax.set_ylabel("targets")
    ax.set_title(
        f"mean {sample.expected_cost:.3f} | median {sample.percentile(0.5):.3f} "
        f"| p90 {sample.percentile(0.9):.3f}"
    )
    return ax


def plot_reach_curve(report: CoverageReport, ax: Axes | None = None) -> Axes:
    """Plot the Haar fraction of two-qubit unitaries reachable at cost ``c`` or less, against ``c``."""
    cdf = report.cost_cdf
    if ax is None:
        _fig, ax = plt.subplots(figsize=_FIGSIZE)
    if cdf:
        costs = [c for c, _ in cdf]
        frac = [f for _, f in cdf]
        ax.step([0.0, *costs], [0.0, *frac], where="post")
        ax.plot(costs, frac, "o", markersize=4)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("cost budget")
    ax.set_ylabel("reachable fraction of SU(4)")
    ax.set_title(
        f"median {report.percentile(0.5):.3f} | p90 {report.percentile(0.9):.3f}"
    )
    return ax


def plot_calibration(calibration: Calibration, ax: Axes | None = None) -> Axes:
    """Plot cost vs calibration budget (number of calibrated gates)."""
    if ax is None:
        _fig, ax = plt.subplots(figsize=_FIGSIZE)
    budgets = list(range(1, len(calibration.budget_costs) + 1))
    ax.plot(budgets, calibration.budget_costs, "-o", markersize=6)
    ax.set_xlabel("calibration budget (# distinct gates)")
    ax.set_ylabel("cost")
    ax.set_xticks(budgets)
    for b, c, ks in zip(
        budgets, calibration.budget_costs, calibration.budget_strengths, strict=True
    ):
        ax.annotate(
            "\n".join(f"{k:.3f}" for k in ks),
            (b, c),
            textcoords="offset points",
            xytext=(6, 0),
            va="center",
            fontsize=7,
        )
    return ax


def plot_strength_sweep(sweep: StrengthSweep, ax: Axes | None = None) -> Axes:
    """Plot cost against calibrated strength ``G^k``, marking the optimum."""
    if ax is None:
        _fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(sweep.strengths, sweep.costs, "-o", markersize=3)
    k, c = sweep.best
    ax.plot([k], [c], marker="*", color="crimson", markersize=14, linestyle="none")
    ax.set_xlabel(r"calibrated strength $k$  ($G^{k}$)")
    ax.set_ylabel("cost")
    ax.set_title(rf"best: $k={k:.3f}$  (cost ${c:.3f}$)")
    return ax
