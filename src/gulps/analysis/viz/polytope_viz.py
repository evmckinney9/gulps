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

"""3-D polytope visualization for reach regions and coverage reports."""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from gulps.analysis.region import ReachableRegion
from gulps.analysis.viz.weyl_chamber import draw_chamber

if TYPE_CHECKING:
    import matplotlib.figure
    from qiskit.circuit import Gate

    from gulps.analysis.coverage import CoverageReport, SentenceCoverage
    from gulps.decomposition import GulpsDecomposer

_MAX_COLS = 3

# Search-tree node colors: a row that adds coverage, a row that earlier rows
# already cover, and a pruned candidate.
_ADDS, _COVERED, _PRUNED = "#1f5fa6", "#e08a1e", "0.6"
# Backward-step colors: the prefix region, the backward region, their
# intersection, and the path of fixed classes.
_PREFIX, _BACKWARD, _MEET, _INK = "tab:blue", "tab:orange", "tab:green", "0.3"


def _short(name: str) -> str:
    """Format a gate power with three significant digits."""
    base, sep, exponent = name.rpartition("^")
    try:
        return f"{base}^{float(exponent):.3g}" if sep else name
    except ValueError:
        return name


def _sentence(names: list[str]) -> str:
    """Write a sentence as in the docs: ``AAB``, or space-separated longer names."""
    return ("" if all(len(name) == 1 for name in names) else " ").join(names)


def _draw_cell(ax: Axes3D, region: ReachableRegion, color: str) -> None:
    """Draw one orientation as a point, a segment, or its faces."""
    vertices = region.vertices
    if vertices is None:
        return
    if len(vertices) == 1:
        ax.scatter3D(*vertices.T, color=color)
        return
    if len(vertices) == 2:
        ax.plot3D(*vertices.T, color=color, linewidth=2)
        return
    faces = region.faces
    coll = Poly3DCollection(faces)
    coll.set_facecolor(color)
    coll.set_edgecolor(color)
    coll.set_alpha(0.2 if len(faces) > 1 else 0.35)
    ax.add_collection3d(coll)


def plot_region(
    region: ReachableRegion, ax: Axes3D | None = None, color: str = "red"
) -> Axes3D:
    """Plot both orientations of a :class:`~gulps.analysis.region.ReachableRegion` in the Weyl chamber."""
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5, 5))
    draw_chamber(ax)
    _draw_cell(ax, region, color)
    _draw_cell(ax, region.rho, color)
    return ax


def plot_coverage_set(
    entries: list[SentenceCoverage],
) -> matplotlib.figure.Figure | None:
    """Show each coverage entry in its own Weyl-chamber subplot.

    Args:
        entries: The cost-ordered entries from a coverage report.

    Returns:
        The figure, or ``None`` when there are no entries.
    """
    if not entries:
        return None
    n = len(entries)
    ncols = min(n, _MAX_COLS)
    nrows = math.ceil(n / ncols)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        squeeze=False,
        subplot_kw={"projection": "3d"},
        figsize=(ncols * 4, nrows * 4),
    )
    axs = axs.ravel()

    cumulative = 0.0
    for i, entry in enumerate(entries):
        ax = plot_region(entry.region, ax=axs[i], color=_ADDS)
        cumulative += entry.fresh_mass
        label = _sentence([_short(gate.label or gate.name) for gate in entry.gates])
        ax.set_title(
            f"{label or 'identity'} | cost {entry.cost:.4g}\n"
            f"new {entry.fresh_mass:.2%} | cumulative {cumulative:.2%}",
            fontsize=11,
        )

    for j in range(n, len(axs)):
        axs[j].set_visible(False)
    plt.close(fig)
    return fig


def _pruned_children(
    report: CoverageReport, sentences: list[tuple[int, ...]]
) -> list[tuple[float, tuple[int, ...]]]:
    """The candidates the search discarded before its last row in ``report``.

    This is the only rule shared with the Rust search: its candidate order. A
    row shorter than ``max_depth`` queues one child per gate at or after its
    own last gate in gate order (cost, then slot), and the search takes candidates in order of
    cost plus a tie per gate, number of gates, parent row, and gate rank. The tie,
    ``max_depth**2 * EPSILON * (largest cost + local_layer_cost)``, matches ``Walk::tie``. Each gate multiset becomes a row at
    most once. A child that is not a row and orders before the last row was
    taken and pruned; one that orders after it was still queued.

    Args:
        report: A coverage report.
        sentences: The gate slots of each row in ``report.rows``.

    Returns:
        The cost and gate slots of each pruned child.
    """
    decomposer = report.decomposer
    costs, local = decomposer.costs, decomposer.local_layer_cost
    tie = decomposer.max_depth**2 * sys.float_info.epsilon * (max(costs) + local)
    order = sorted(range(len(costs)), key=lambda s: (costs[s], s))
    rank = {s: r for r, s in enumerate(order)}
    index = {sentence: i for i, sentence in enumerate(sentences)}

    def key(cost: float, sentence: tuple[int, ...], parent: int | None) -> tuple:
        return (
            cost + len(sentence) * tie,
            len(sentence),
            -1 if parent is None else parent,
            rank[sentence[-1]],
        )

    last = key(report.rows[-1].cost, sentences[-1], index.get(sentences[-1][:-1]))
    pruned = []
    for i, (row, sentence) in enumerate(zip(report.rows, sentences, strict=True)):
        if len(sentence) >= decomposer.max_depth:
            continue
        for s in order[rank[sentence[-1]] :]:
            child, cost = (*sentence, s), row.cost + (costs[s] + local)
            if child not in index and key(cost, child, i) < last:
                pruned.append((cost, child))
    return pruned


def plot_search_tree(
    report: CoverageReport, names: Sequence[str] | None = None
) -> matplotlib.figure.Figure:
    """Draw the rows of a coverage report and the candidates pruned among them.

    Each column holds the sentences with one number of gates, in search order
    from the top. A node shows both orientations of its sentence's region, its
    gate multiset, and its cost. An edge joins a sentence to the sentence
    without its last gate.

    Args:
        report: A coverage report.
        names: A display name for each gate of ``report.decomposer``, in order.
            Defaults to the gates' names.

    Returns:
        The figure.
    """
    from gulps.invariants import LocalEquivalenceClass

    gates = report.decomposer.gates
    if names is None:
        names = [_short(g.name) for g in gates]
    slot = {id(g): i for i, g in enumerate(gates)}
    sentences = [tuple(slot[id(g)] for g in row.gates) for row in report.rows]
    classes = LocalEquivalenceClass.from_unitaries(list(gates))
    nodes = [
        (
            row.cost,
            sentence,
            row.region,
            _ADDS if row.fresh_mass is not None else _COVERED,
        )
        for row, sentence in zip(report.rows, sentences, strict=True)
    ]
    nodes += [
        (cost, child, ReachableRegion.of([classes[s] for s in child]), _PRUNED)
        for cost, child in _pruned_children(report, sentences)
    ]
    columns: dict[int, list] = {}
    for node in sorted(nodes, key=lambda n: (n[0], n[1])):
        columns.setdefault(len(node[1]), []).append(node)
    depth, height = max(columns), max(len(c) for c in columns.values())

    fig = plt.figure(figsize=(max(5.6, 1.6 * depth), 1.5 * (height + 1.2)))
    base = fig.add_axes((0, 0, 1, 1))
    base.set_xlim(0.4, depth + 0.6)
    base.set_ylim(height + 0.9, -0.35)
    base.set_axis_off()
    position = {}
    for d, column in columns.items():
        base.text(d, -0.2, f"{d} gate{'s' if d > 1 else ''}", ha="center", fontsize=11)
        offset = (height - len(column)) / 2
        for i, node in enumerate(column):
            position[node[1]] = (d, offset + i + 0.5)
    for _, sentence, _, color in nodes:
        if len(sentence) > 1:
            (x0, y0), (x1, y1) = position[sentence[:-1]], position[sentence]
            pruned = color == _PRUNED
            base.plot(
                [x0 + 0.22, x1 - 0.22],
                [y0, y1],
                color="0.6" if pruned else "0.35",
                linestyle="--" if pruned else "-",
                linewidth=1,
                zorder=0,
            )

    to_fig = base.transData + fig.transFigure.inverted()
    (x0, y0), (x1, y1) = to_fig.transform([(0, 0), (0.42, 0.78)])
    w, h = x1 - x0, y0 - y1
    for cost, sentence, region, color in nodes:
        x, y = position[sentence]
        fx, fy = to_fig.transform((x, y))
        ax = fig.add_axes((fx - w / 2, fy - h / 2 - 0.01, w, h), projection="3d")
        draw_chamber(ax)
        _draw_cell(ax, region, color)
        _draw_cell(ax, region.rho, color)
        ax.set_axis_off()
        label = _sentence([names[s] for s in sentence])
        symbol = {_ADDS: "●", _COVERED: "□", _PRUNED: "×"}[color]
        base.text(
            x,
            y + 0.24,
            f"{symbol} {label}\ncost {cost:.4g}",
            ha="center",
            va="top",
            fontsize=10,
            color="0.45" if color == _PRUNED else "black",
        )

    legend = [
        (_ADDS, "o", "adds coverage"),
        (_COVERED, "s", "never selected, but not prunable"),
        (_PRUNED, "x", "pruned by a cheaper sentence"),
    ]
    base.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                color=c,
                marker=marker,
                markersize=7,
                linestyle="none",
                markerfacecolor="none" if c == _COVERED else c,
                label=label,
            )
            for c, marker, label in legend
        ],
        loc="lower center",
        ncol=1,
        frameon=False,
        fontsize=9,
    )
    plt.close(fig)
    return fig


def _trajectory(
    decomposer: GulpsDecomposer, target: Gate | Operator | np.ndarray
) -> tuple[list, list[np.ndarray]]:
    """The native gates of the compiled target and the Weyl class after each prefix."""
    from gulps.invariants import LocalEquivalenceClass

    prefix, natives, classes = QuantumCircuit(2), [], [np.zeros(3)]
    for inst in decomposer(target).data:
        prefix.append(inst)
        if inst.operation.num_qubits == 2:
            natives.append(inst.operation)
            classes.append(LocalEquivalenceClass.from_unitary(Operator(prefix)).weyl)
    return natives, classes


def _backward_step(
    natives: list, q: list[np.ndarray], i: int
) -> tuple[ReachableRegion, ReachableRegion, ReachableRegion]:
    """The prefix, backward, and intersection regions that select ``q[i - 1]``.

    The intersection keeps the larger of each pair of lower bounds, so it holds
    ``q[i - 1]`` exactly when both regions do, and each orientation is chosen
    on its own.
    """
    from gulps.invariants import LocalEquivalenceClass

    prefix = ReachableRegion.of(LocalEquivalenceClass.from_unitaries(natives[: i - 1]))
    inverse = LocalEquivalenceClass.from_unitary(natives[i - 1].inverse())
    backward = ReachableRegion.of([LocalEquivalenceClass(q[i]), inverse])
    prefix = prefix if prefix.contains(q[i - 1]) else prefix.rho
    backward = backward if backward.contains(q[i - 1]) else backward.rho
    meet = ReachableRegion(tuple(map(max, prefix._bounds, backward._bounds)))
    return prefix, backward, meet


def _outline(ax: Axes3D, region: ReachableRegion, color: str) -> None:
    """Draw a region as edges over a faint fill, so an intersection stays visible."""
    if region.vertices is None or len(region.vertices) < 3:
        _draw_cell(ax, region, color)
        return
    ax.add_collection3d(
        Poly3DCollection(
            region.faces,
            facecolor=to_rgba(color, 0.05),
            edgecolor=to_rgba(color, 0.55),
            linewidth=0.6,
        )
    )


def plot_waypoints(
    decomposer: GulpsDecomposer, target: Gate | Operator | np.ndarray
) -> matplotlib.figure.Figure:
    """Draw each backward waypoint selection of a compiled target in its own Weyl chamber.

    The panels run from the target toward the identity. Each shows the prefix
    region, the backward region through the inverse of the next native gate,
    their intersection, and the class the compiler chose in it. The last panel
    has no choice: its prefix region is the class of the first native gate,
    joined to the identity by a dashed segment. The classes are read from the
    compiled circuit.

    Args:
        decomposer: The decomposer that compiles ``target``.
        target: A two-qubit unitary that ``decomposer`` accepts.

    Returns:
        The figure.

    Raises:
        ValueError: If the compiled circuit has fewer than two native gates,
            so there is no backward step to draw.
    """
    natives, q = _trajectory(decomposer, target)
    n = len(natives)
    if n < 2:
        raise ValueError(f"the target compiles to {n} native gates; need at least 2")
    fig = plt.figure(figsize=(4.3 * (n - 1), 4.2))
    grid = fig.add_gridspec(1, n - 1, wspace=0.0)
    for col, i in enumerate(range(n, 1, -1)):
        ax = fig.add_subplot(grid[0, col], projection="3d")
        draw_chamber(ax)
        ax.set(xlabel="$c_1$", ylabel="$c_2$", zlabel="$c_3$")
        prefix, backward, meet = _backward_step(natives, q, i)
        _outline(ax, backward, _BACKWARD)
        _outline(ax, prefix, _PREFIX)
        _draw_cell(ax, meet, _MEET)
        ax.plot(*np.array(q[i - 1 :]).T, color=_INK, linewidth=0.8)
        ax.scatter(*np.array(q[i:]).T, color=_INK, s=10)
        ax.scatter(*q[i - 1], color=_MEET, edgecolors="k", s=30, linewidths=0.6)
        if i == 2:
            ax.plot(
                *np.array(q[:2]).T, color=_INK, linewidth=0.8, linestyle=(0, (2, 2))
            )
            ax.scatter(*q[0], facecolors="w", edgecolors=_INK, s=12, linewidths=0.8)
            title = r"$C_1=G_1\in\mathrm{Reach}(C_2,G_2^{-1})$"
        else:
            title = (
                rf"choose $C_{i - 1}\in\mathcal{{R}}_{i - 1}\cap"
                rf"\mathrm{{Reach}}(C_{i},G_{i}^{{-1}})$"
            )
        ax.set_title(title, fontsize=9)
    fig.legend(
        handles=[
            Patch(
                facecolor=_PREFIX, alpha=0.15, edgecolor=_PREFIX, label="prefix region"
            ),
            Patch(
                facecolor=_BACKWARD,
                alpha=0.15,
                edgecolor=_BACKWARD,
                label="backward region",
            ),
            Patch(facecolor=_MEET, alpha=0.35, edgecolor=_MEET, label="intersection"),
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor=_MEET,
                markeredgecolor="k",
                markersize=6,
                label="chosen class",
            ),
        ],
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    plt.close(fig)
    return fig
