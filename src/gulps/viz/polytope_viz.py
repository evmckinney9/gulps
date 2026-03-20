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

"""3-D polytope visualization for monodromy coverage sets."""

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as ss
from monodromy.coordinates import monodromy_to_positive_canonical_polytope
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from gulps.viz.weyl_chamber import WeylChamber

# Maximum columns in subplot grid
MAX_COLS = 5

# Color palette for cost levels
COLORS = [
    "red",
    "green",
    "blue",
    "orange",
    "purple",
    "cyan",
    "black",
    "pink",
    "brown",
    "grey",
]


def _plot_polytope(circuit_polytope, ax, w, color="red"):
    """Plot a single circuit polytope on the given axes.

    Args:
        circuit_polytope: CircuitPolytope object to plot.
        ax: Matplotlib 3D axes.
        w: WeylChamber object.
        color: Color for the polytope faces.
    """
    canon = monodromy_to_positive_canonical_polytope(circuit_polytope).reduce()

    for vertex_set in canon.vertices:
        vertices = np.unique([[float(b) for b in a] for a in vertex_set], axis=0)

        if len(vertices) == 1:
            w.ax.scatter3D(*zip(*vertices), color=color)
        elif len(vertices) == 2:
            w.ax.plot3D(*zip(*vertices), color=color, linewidth=2)
        elif len(vertices) == 3:
            triangle = Poly3DCollection([vertices])
            triangle.set_facecolor(color)
            triangle.set_edgecolor("k")
            triangle.set_alpha(0.5)
            w.ax.add_collection3d(triangle)
        else:
            hull = ss.ConvexHull(vertices, qhull_options="QJ")
            faces = Poly3DCollection([vertices[simplex] for simplex in hull.simplices])
            faces.set_facecolor(color)
            faces.set_alpha(0.2)
            faces.set_edgecolor("k")
            w.ax.add_collection3d(faces)


def plot_coverage_set(coverage_set, volume_info=None):
    """Plot a coverage set of 3D polytopes in the Weyl chamber.

    Args:
        coverage_set: List of CircuitPolytope objects. Each gets its own subplot.
        volume_info: Optional list of (cost, depth, unique_volume, cumulative_volume)
            tuples. When provided, depth and volume information are displayed in subplot titles.
    """
    if not coverage_set:
        print("No polytopes to plot.")
        return

    # Skip the identity polytope (no operations, depth 0)
    coverage_set = [p for p in coverage_set if len(p.operations) > 0]

    if not coverage_set:
        print("No non-trivial polytopes to plot.")
        return

    n = len(coverage_set)
    ncols = min(n, MAX_COLS)
    nrows = (n + ncols - 1) // ncols
    _, axs = plt.subplots(
        nrows,
        ncols,
        subplot_kw={"projection": "3d"},
        figsize=(ncols * 5, nrows * 5),
    )
    # Flatten to 1D array for consistent indexing
    if n == 1:
        axs = [axs]
    else:
        axs = np.array(axs).flatten()

    for i, polytope in enumerate(coverage_set):
        ax = axs[i]
        w = WeylChamber()
        w.labels = {}
        w.render(ax)
        color = COLORS[i % len(COLORS)]
        _plot_polytope(polytope, ax, w, color=color)

        # Create title with optional volume information
        if volume_info and i < len(volume_info):
            cost, depth, unique_vol, cumulative_vol = volume_info[i]
            title = (
                f"Cost: {cost:.4f} | Depth: {depth}\n"
                f"New vol: {unique_vol:.4f} | Cum. vol: {cumulative_vol:.4f} \n"
                f""
            )
        else:
            title = f"Cost: {polytope.cost:.4f}"
        w.ax.set_title(title)

    # Hide empty subplots
    for j in range(n, len(axs)):
        axs[j].set_visible(False)

    plt.show()
