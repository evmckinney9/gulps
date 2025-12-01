from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from weylchamber import WeylChamber

from gulps.core.invariants import GateInvariants

# colors = cm.tab10(np.linspace(0, 1.0, len(self.gate_set)))
# color_map = {gate: color for gate, color in zip(self.gate_set, colors)}


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


def render_path(
    trajectory_invariants: List[GateInvariants],
):
    """Visualize the decomposition path in the Weyl chamber using named styles and per-gate primary+gap color pairs."""
    # # Define named line styles
    # dash_patterns = {
    #     "solid": None,
    #     "densely dotted": [0.75, 1],
    #     "densely dashed": [2.5, 2],
    #     "densely dashdot": [3, 1.5, 1, 1.5],
    # }

    # primary_colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
    # gap_colors = ["tab:orange", "tab:cyan", "tab:pink", "tab:olive"]

    # # Map each gate to a color + gapcolor + linestyle
    # gates = list(color_map.keys())
    # gate_styles = {
    #     gate: {
    #         "color": primary_colors[i % len(primary_colors)],
    #         "gap": gap_colors[i % len(gap_colors)],
    #         "style": list(dash_patterns.keys())[i % len(dash_patterns)],
    #     }
    #     for i, gate in enumerate(gates)
    # }

    # # Compute trajectory points if not already computed
    # if not self.traj_points:
    #     self.traj_points = [
    #         abs(np.array(monodromy_to_positive_canonical_coordinate(*p))) / (np.pi / 2)
    #         for p in self.mono_points
    #     ]
    # trajectory_invariants.insert(0, GateInvariants(logspec=(0.0, 0.0, 0.0, 0.0)))
    traj_points = [abs(g.weyl) for g in trajectory_invariants]
    traj_points.insert(0, (0.0, 0.0, 0.0))  # Start at origin

    # Set up the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    w = WeylChamber()
    w.labels = {}
    w.render(ax)

    # Plot all trajectory points
    pts = np.array(traj_points)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], zorder=-1)

    # Plot each segment
    for i in range(len(traj_points) - 1):
        p0 = traj_points[i]
        p1 = traj_points[i + 1]
        # gate = isa_sequence[i]
        # styles = gate_styles.get(
        #     gate, {"color": "black", "gap": "gray", "style": "solid"}
        # )
        # dashes = dash_patterns.get(styles["style"])

        xs = [p0[0], p1[0]]
        ys = [p0[1], p1[1]]
        zs = [p0[2], p1[2]]

        (line,) = ax.plot(
            xs,
            ys,
            zs,
            # color=styles["color"],
            linewidth=6,
            zorder=0,
            # gapcolor=styles["gap"],
        )
        # if dashes:
        #     line.set_dashes(dashes)

        # Endpoint marker
        ax.scatter(
            [p1[0]],
            [p1[1]],
            [p1[2]],
            color="black",
            edgecolor="black",
            marker="o",
            s=30,
            zorder=1,
            depthshade=False,
        )

    # Mark origin
    ax.scatter([0], [0], [0], color="black", s=30, zorder=1, depthshade=False)

    # # Legend (just show primary colors and style)
    # custom_lines = []
    # legend_labels = []
    # for gate, styles in gate_styles.items():
    #     line = Line2D([0], [0], color=styles["color"], lw=6, linestyle=styles["style"])
    #     custom_lines.append(line)
    #     legend_labels.append(gate)

    # ax.legend(custom_lines, legend_labels, ncol=2, title="Basis gate")
    # plt.savefig("trajn.svg", bbox_inches="tight")
    return ax
