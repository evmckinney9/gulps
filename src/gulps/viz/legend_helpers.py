"""Helper functions for creating legends in visualization plots.

This module contains utilities for creating consistent, visually appealing legends
for both discrete and continuous ISA decomposition visualizations.
"""

from typing import Dict, List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from gulps.core.isa import ContinuousISA, ISAInvariants

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
    "hsv",
    "viridis",
    "plasma",
]


class GradientLineHandler(HandlerBase):
    """Custom legend handler that draws a horizontal gradient bar.

    This handler creates a smooth gradient representation for continuous ISA
    parameters, making the legend visually consistent with discrete ISA legends
    that use solid color lines.
    """

    def __init__(self, cmap, num_samples=50):
        """Initialize the gradient handler.

        Args:
            cmap: Matplotlib colormap to use for the gradient.
            num_samples: Number of color samples to create the gradient (default: 50).
        """
        super().__init__()
        self.cmap = cmap
        self.num_samples = num_samples

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        """Create the gradient bar for the legend entry.

        Args:
            legend: The legend object.
            orig_handle: The original handle (patch).
            xdescent: X offset for the legend entry.
            ydescent: Y offset for the legend entry.
            width: Width of the legend entry.
            height: Height of the legend entry.
            fontsize: Font size for the legend.
            trans: Transformation to apply.

        Returns:
            List of Rectangle artists forming the gradient.
        """
        # Create small rectangles with gradient colors
        rects = []
        segment_width = width / self.num_samples

        for i in range(self.num_samples):
            # Normalize parameter from 0.3 to 0.9 (same range used in plotting)
            t = i / (self.num_samples - 1)
            normalized_param = 0.3 + 0.6 * t
            color = self.cmap(normalized_param)

            rect = Rectangle(
                xy=(xdescent + i * segment_width, ydescent),
                width=segment_width,
                height=height,
                facecolor=color,
                edgecolor="none",
                transform=trans,
            )
            rects.append(rect)

        return rects


def build_color_map_from_isa(
    isa: ISAInvariants,
) -> tuple[Dict[str, any], List[str]]:
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


def add_continuous_legend(ax, gate_color_map: Dict[str, any], legend_labels: List[str]):
    """Add legend for continuous ISA families using horizontal gradient bars.

    Creates a legend with horizontal gradient bars that visually match the style
    of discrete ISA legends, providing a consistent appearance across visualization types.

    Args:
        ax: Matplotlib 3D axes to add the legend to.
        gate_color_map: Dictionary mapping gate name to colormap object.
        legend_labels: List of legend labels (e.g., "fsim(θ)").
    """
    # Create custom legend handles with horizontal gradient representation
    custom_handles = []
    handler_map = {}

    for label in legend_labels:
        gate_name = label.split("(")[0]
        cmap = gate_color_map[gate_name]

        # Create a dummy patch as a handle (the actual rendering is done by the handler)
        patch = mpatches.Patch(label=label)
        custom_handles.append(patch)

        # Map this handle to our custom gradient handler
        handler_map[patch] = GradientLineHandler(cmap, num_samples=50)

    # Add legend in similar position and style to discrete version
    legend = ax.legend(
        handles=custom_handles,
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        title="Basis Gate",
        frameon=True,
        fontsize=10,
        title_fontsize=10,
        handler_map=handler_map,
    )
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.8)


def add_discrete_legend(ax, gate_color_map: Dict[str, str], legend_labels: List[str]):
    """Add legend for discrete ISA with solid color lines.

    Args:
        ax: Matplotlib 3D axes to add the legend to.
        gate_color_map: Dictionary mapping gate name to color string.
        legend_labels: List of legend labels (gate names).
    """
    # Create standard legend with discrete colors
    legend_colors = [gate_color_map[label.split("(")[0]] for label in legend_labels]
    custom_lines = [Line2D([0], [0], color=color, lw=6) for color in legend_colors]
    legend = ax.legend(
        custom_lines,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        title="Basis Gate",
        frameon=True,
        fontsize=10,
        title_fontsize=10,
    )
    # Add border to match continuous legend styling
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.8)
