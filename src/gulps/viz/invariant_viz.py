from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from weylchamber import WeylChamber

from gulps.core.invariants import GateInvariants
from gulps.core.isa import ContinuousISA, DiscreteISA, ISAInvariants

# Define color palette for basis gates (used for discrete ISAs)
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

# Define colormaps for continuous ISAs (one per basis gate)
COLORMAP_NAMES = [
    # "viridis",
    "plasma",
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

    # Add depth indicators for each point
    for point in points:
        _add_depth_indicators(ax, point[0], point[1], point[2])

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], zorder=1)
    return fig, ax


def _add_depth_indicators(
    ax,
    x: float,
    y: float,
    z: float,
    color: str = "gray",
    alpha: float = 0.25,
):
    """Add subtle depth indicators to a 3D point for better spatial perception.

    Draws a faint dashed vertical line from the point down to z=0 and a shadow
    on the z=0 plane to help visualize the 3D structure.

    Args:
        ax: Matplotlib 3D axes to plot on.
        x: X coordinate of the point.
        y: Y coordinate of the point.
        z: Z coordinate of the point.
        color: Color for the depth indicators (default: "gray").
        alpha: Opacity for the depth indicators (default: 0.25).
    """
    # Vertical dashed line from point to z=0 plane
    if z > 0:  # Only draw if point is above the plane
        ax.plot(
            [x, x],
            [y, y],
            [z, 0],
            color=color,
            linestyle="--",
            linewidth=0.8,
            alpha=alpha * 1.6,  # Slightly more visible than shadow
            zorder=-10,
        )

    # Shadow on z=0 plane
    ax.scatter(
        [x],
        [y],
        [0],
        color=color,
        s=15,
        alpha=alpha,
        zorder=-10,
        marker="o",
    )


def _build_color_map_from_isa(
    isa: ISAInvariants,
) -> Tuple[Dict[str, any], List[str]]:
    """Build consistent color mapping from ISA gate set.

    Args:
        isa: ISA containing the gate set to map colors to.

    Returns:
        Tuple of (color_map, legend_labels) where:
        - color_map: For discrete ISA, maps gate name to color string.
                    For continuous ISA, maps gate name to colormap object.
        - legend_labels: List of legend labels in order
    """
    color_map = {}
    legend_labels = []

    for i, gate in enumerate(isa.gate_set):
        # Handle continuous ISA: use colormaps for parameterization
        if isinstance(isa, ContinuousISA):
            base_name = gate.name
            cmap_name = COLORMAP_NAMES[i % len(COLORMAP_NAMES)]
            color_map[base_name] = plt.get_cmap(cmap_name)
            legend_labels.append(f"{base_name}(θ)")
        else:
            # Discrete ISA: use discrete colors
            color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
            color_map[gate.name] = color
            legend_labels.append(gate.name)

    return color_map, legend_labels


def _add_colorbar_legend(ax, gate_color_map: Dict[str, any], legend_labels: List[str]):
    """Add colorbar legend for continuous ISA families.

    Args:
        ax: Matplotlib 3D axes to add the legend to.
        gate_color_map: Dictionary mapping gate name to colormap object.
        legend_labels: List of legend labels (e.g., "iSWAP(θ)").
    """
    # Create a figure for the colorbar legend positioned to the right of the plot
    fig = ax.figure

    # Calculate position for colorbars
    n_gates = len(legend_labels)
    colorbar_width = 0.015
    colorbar_spacing = 0.08
    colorbar_height = 0.15

    # Starting position on the right side
    start_x = 0.88
    start_y = 0.75

    for i, label in enumerate(legend_labels):
        gate_name = label.split("(")[0]
        cmap = gate_color_map[gate_name]

        # Create an axis for this colorbar
        y_pos = start_y - i * colorbar_spacing
        cax = fig.add_axes([start_x, y_pos, colorbar_width, colorbar_height])

        # Create a ScalarMappable for the colorbar
        norm = plt.cm.colors.Normalize(vmin=0.0, vmax=1.0)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # Add the colorbar
        cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
        cbar.set_label(label, rotation=0, labelpad=15, ha="left")
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels(["0", "0.5", "1"])


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

    # Add depth indicators for all trajectory points with matching colors
    first_color = colors[0] if colors else "gray"
    # Origin gets the first segment color
    _add_depth_indicators(
        ax, traj_points[0][0], traj_points[0][1], traj_points[0][2], color=first_color
    )
    # Subsequent points get the color of the segment that led to them
    for i in range(1, len(traj_points)):
        point = traj_points[i]
        color = colors[i - 1]  # Color of the segment that reached this point
        _add_depth_indicators(ax, point[0], point[1], point[2], color=color)

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
    parameters: Optional[List[float]] = None,
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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    w = WeylChamber()
    w.labels = {}
    w.render(ax)

    # Determine coloring scheme
    if sentence and isa:
        # ISA-based consistent coloring with legend
        gate_color_map, legend_labels = _build_color_map_from_isa(isa)

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
            _add_colorbar_legend(ax, gate_color_map, legend_labels)
        else:
            # Discrete ISA: use discrete colors
            colors = [gate_color_map.get(g.name, "black") for g in sentence]

            # Create standard legend with discrete colors
            legend_colors = [
                gate_color_map[label.split("(")[0]] for label in legend_labels
            ]
            custom_lines = [
                Line2D([0], [0], color=color, lw=6) for color in legend_colors
            ]
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
