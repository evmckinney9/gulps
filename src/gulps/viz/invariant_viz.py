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

"""Weyl-chamber scatter plots for gate invariants."""

import matplotlib.pyplot as plt
import numpy as np

from gulps.core.invariants import GateInvariants
from gulps.core.isa import ContinuousISA, ISAInvariants
from gulps.viz.legend_helpers import (
    add_continuous_legend,
    add_discrete_legend,
    build_color_map_from_isa,
)
from gulps.viz.weyl_chamber import WeylChamber


def scatter_plot(
    invariant_list: list[GateInvariants],
    ax: plt.Axes | None = None,
    **kwargs,
):
    """Scatter plot of a list of GateInvariants in the Weyl chamber.

    Args:
        invariant_list: Gate invariants to plot.
        ax: Optional existing 3D axes for overlaying multiple scatter calls.
        **kwargs: Passed through to ``ax.scatter`` (e.g. ``color``, ``s``, ``label``).
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
        ax.set_proj_type("persp")
        w = WeylChamber()
        w.labels = {}
        w.render(ax)
    else:
        fig = ax.get_figure()

    points = np.array([abs(g.weyl) for g in invariant_list])

    # Depth indicators (vectorized)
    _add_depth_indicators_batch(ax, points)

    kwargs.setdefault("zorder", 1)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], **kwargs)
    return fig, ax


def _add_depth_indicators_batch(
    ax,
    points: np.ndarray,
    color="gray",
    alpha: float = 0.25,
):
    """Vectorized depth indicators for an array of 3D points.

    Draws vertical dashed lines to z=0 and shadow dots in two bulk calls.

    Args:
        ax: Matplotlib 3D axes to plot on.
        points: (N, 3) array of 3D coordinates.
        color: Single color string/tuple or sequence of per-point colors.
        alpha: Opacity for the depth indicators.
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    # Shadows on z=0 plane (single scatter call)
    ax.scatter(
        points[:, 0],
        points[:, 1],
        np.zeros(len(points)),
        color=color,
        s=15,
        alpha=alpha,
        zorder=-10,
        marker="o",
    )

    # Vertical dashed lines from point to z=0 (single LineCollection)
    above = points[:, 2] > 0
    if above.any():
        pts = points[above]
        segments = [[(p[0], p[1], p[2]), (p[0], p[1], 0.0)] for p in pts]
        # Filter colors to match the above-mask if per-point
        if isinstance(color, (list, np.ndarray)):
            line_colors = [c for c, a in zip(color, above) if a]
        else:
            line_colors = color
        lc = Line3DCollection(
            segments,
            colors=line_colors,
            linestyles="--",
            linewidths=0.8,
            alpha=alpha * 1.6,
            zorder=-10,
        )
        ax.add_collection3d(lc)


def _render_path(
    traj_points: list[tuple[float, float, float]],
    colors: list[str],
    ax,
):
    """Low-level path rendering function.

    Args:
        traj_points: List of (x, y, z) points including origin at index 0.
        colors: List of colors for each segment (length = len(traj_points) - 1).
        ax: Matplotlib 3D axes to plot on.
    """
    # Plot all trajectory points
    pts = np.array(traj_points)
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2], zorder=-1, color="gray", s=20, alpha=0.5
    )

    # Add depth indicators with per-point colors
    # Origin gets first segment color; subsequent points get their segment color
    point_colors = (
        [colors[0]] + [colors[i] for i in range(len(colors))]
        if colors
        else ["gray"] * len(pts)
    )
    _add_depth_indicators_batch(ax, pts, color=point_colors)

    # Plot each segment
    for i in range(len(traj_points) - 1):
        p0 = traj_points[i]
        p1 = traj_points[i + 1]
        color = colors[i]

        xs = [p0[0], p1[0]]
        ys = [p0[1], p1[1]]
        zs = [p0[2], p1[2]]

        ax.plot(xs, ys, zs, color=color, linewidth=6, zorder=0)

        # Endpoint marker (filled with segment color, thin black outline)
        ax.scatter(
            [p1[0]],
            [p1[1]],
            [p1[2]],
            color=color,
            edgecolors="black",
            marker="o",
            s=50,
            linewidths=1,
            zorder=1,
            depthshade=False,
        )

    # Mark origin (same color as first segment)
    first_color = colors[0] if colors else "gray"
    ax.scatter(
        [0],
        [0],
        [0],
        color=first_color,
        edgecolors="black",
        marker="o",
        s=50,
        linewidths=1,
        zorder=1,
        depthshade=False,
    )


def plot_decomposition(
    intermediates: list[GateInvariants],
    sentence: list[GateInvariants] | None = None,
    isa: ISAInvariants | None = None,
    parameters: list[float] | None = None,
    ax: plt.Axes | None = None,
):
    """Visualize the decomposition path in the Weyl chamber with colored segments.

    Args:
        intermediates: List of intermediate gate invariants representing the path points.
                      Path goes from origin (0,0,0) -> intermediates[0] -> ... -> intermediates[n].
        sentence: Optional list of basis gates used in the decomposition.
                 If provided, sentence[i] is the basis gate used to reach intermediates[i].
                 Must be accompanied by isa parameter for proper labeling.
        isa: ISA to use for consistent color mapping across multiple calls.
             Required when sentence is provided.
        parameters: Optional list of gate parameters (for continuous ISAs).
                   If provided, parameters[i] is the fractional power of the basis gate
                   used to reach intermediates[i]. Used to select color from colormap.
        ax: Optional matplotlib 3D axes to plot on. If None, creates new figure and axes.

    Returns:
        Matplotlib figure object.

    Notes:
        - If sentence, isa, and parameters are provided for continuous ISA: segments colored
          by continuous colormap where parameter value determines the shade.
        - If sentence and isa provided for discrete ISA: segments colored by gate type with
          consistent discrete colors.
        - If neither provided: segments colored by index with a gradient for visual distinction.
    """
    traj_points = [abs(g.weyl) for g in intermediates]
    traj_points.insert(0, (0.0, 0.0, 0.0))  # Start at origin
    n_segments = len(traj_points) - 1

    # Set up the 3D plot
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    else:
        fig = ax.get_figure()

    w = WeylChamber()
    w.labels = {}
    w.render(ax)

    # Determine coloring scheme
    if sentence and isa:
        # ISA-based consistent coloring with legend
        gate_color_map, legend_labels = build_color_map_from_isa(isa)

        # Check if this is a continuous ISA with parameters
        if isinstance(isa, ContinuousISA) and parameters is not None:
            # Use colormaps with parameter values
            colors = []
            for gate, param in zip(sentence, parameters):
                cmap = gate_color_map.get(gate.name, plt.get_cmap("viridis"))
                # Map parameter to [0.3, 0.9] range for better visibility
                # (avoiding very light colors at 0.0 and very dark at 1.0)
                normalized_param = 0.3 + 0.6 * param
                colors.append(cmap(normalized_param))

            # Create colorbar legend for continuous families
            add_continuous_legend(ax, gate_color_map, legend_labels)
        else:
            # Discrete ISA: use discrete colors
            colors = [gate_color_map.get(g.name, "black") for g in sentence]

            # Create standard legend with discrete colors
            add_discrete_legend(ax, gate_color_map, legend_labels)
    elif sentence:
        # Sentence without ISA - error
        raise ValueError(
            "When sentence is provided, isa must also be provided for consistent color mapping."
        )
    else:
        # Index-based coloring (gradient) - no legend
        cmap = plt.cm.viridis
        colors = [cmap(i / max(1, n_segments - 1)) for i in range(n_segments)]

    _render_path(traj_points, colors, ax)
    return fig
