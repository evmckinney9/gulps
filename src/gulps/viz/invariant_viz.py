from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from weylchamber import WeylChamber

from gulps.core.invariants import GateInvariants
from gulps.core.isa import ContinuousISA, DiscreteISA, ISAInvariants

# Define color palette for basis gates
COLOR_PALETTE = [
    "tab:blue",
    "tab:red",
    "tab:green",
    "tab:purple",
    "tab:orange",
    "tab:cyan",
    "tab:pink",
    "tab:olive",
    "tab:brown",
    "gold",
]


def scatter_plot(invariant_list: List[GateInvariants]):
    """Scatter plot of a list of GateInvariants in the Weyl chamber."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    w = WeylChamber()
    ax.set_proj_type("persp")
    w.labels = {}
    w.render(ax)

    points = np.array([abs(g.weyl) for g in invariant_list])
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], zorder=1)
    return fig, ax


def _build_color_map_from_isa(
    isa: ISAInvariants,
) -> Tuple[Dict[str, str], List[str]]:
    """Build consistent color mapping from ISA gate set.

    Args:
        isa: ISA containing the gate set to map colors to.

    Returns:
        Tuple of (color_map, legend_labels) where:
        - color_map: Maps gate name to color
        - legend_labels: List of legend labels in order
    """
    color_map = {}
    legend_labels = []

    for i, gate in enumerate(isa.gate_set):
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]

        # Handle continuous ISA: label as "base_name(θ)" to show parameterization
        if isinstance(isa, ContinuousISA):
            base_name = gate.name
            color_map[base_name] = color
            legend_labels.append(f"{base_name}(θ)")
        else:
            # Discrete ISA: use gate name directly
            color_map[gate.name] = color
            legend_labels.append(gate.name)

    return color_map, legend_labels


def _render_path(
    traj_points: List[Tuple[float, float, float]],
    colors: List[str],
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
            c=color,
            edgecolors="black",
            marker="o",
            s=50,
            linewidths=1,
            zorder=1,
            depthshade=False,
        )

    # Mark origin (same color as first segment)
    first_color = colors[0] if colors else "black"
    ax.scatter(
        [0],
        [0],
        [0],
        c=first_color,
        edgecolors="black",
        marker="o",
        s=50,
        linewidths=1,
        zorder=1,
        depthshade=False,
    )


def plot_decomposition(
    intermediates: List[GateInvariants],
    sentence: Optional[List[GateInvariants]] = None,
    isa: Optional[ISAInvariants] = None,
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

    Returns:
        Matplotlib figure object.

    Notes:
        - If both sentence and isa are provided: segments colored by gate type with consistent
          colors based on ISA gate_set order. Legend shows all gates in ISA.
        - If neither provided: segments colored by index with a gradient for visual distinction.
    """
    traj_points = [abs(g.weyl) for g in intermediates]
    traj_points.insert(0, (0.0, 0.0, 0.0))  # Start at origin
    n_segments = len(traj_points) - 1

    # Set up the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    w = WeylChamber()
    w.labels = {}
    w.render(ax)

    # Determine coloring scheme
    if sentence and isa:
        # ISA-based consistent coloring with legend
        gate_color_map, legend_labels = _build_color_map_from_isa(isa)
        colors = [gate_color_map.get(g.name, "black") for g in sentence]

        # Create legend with all ISA gates
        legend_colors = [gate_color_map[label.split("(")[0]] for label in legend_labels]
        custom_lines = [Line2D([0], [0], color=color, lw=6) for color in legend_colors]
        ax.legend(
            custom_lines,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            title="Basis Gate",
            frameon=True,
        )
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
