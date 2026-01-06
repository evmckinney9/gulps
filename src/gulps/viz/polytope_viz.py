import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as ss
from monodromy.coordinates import monodromy_to_positive_canonical_polytope
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from weylchamber import WeylChamber

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
    polytope_vertices = (
        monodromy_to_positive_canonical_polytope(circuit_polytope).reduce().vertices
    )
    polytope_vertices = np.array([[float(b) for b in a] for a in polytope_vertices[0]])

    left_vertices = polytope_vertices
    right_vertices = np.array([[1 - a[0], a[1], a[2]] for a in polytope_vertices])

    for vertices in [left_vertices, right_vertices]:
        # delete duplicates that might exist
        vertices = np.unique(vertices, axis=0)

        if len(vertices) < 3:
            w.ax.scatter3D(*zip(*vertices), color=color)
        elif len(vertices) == 3:
            triangle = Poly3DCollection([vertices])
            triangle.set_facecolor(color)
            triangle.set_edgecolor("k")
            triangle.set_alpha(0.5)
            w.ax.add_collection3d(triangle)
        else:
            # TODO use Qbk:0Bk:0 - drop dimension k from the input points
            hull = ss.ConvexHull(vertices, qhull_options="QJ")
            faces = Poly3DCollection([vertices[simplex] for simplex in hull.simplices])
            faces.set_facecolor(color)
            faces.set_alpha(0.2)
            faces.set_edgecolor("k")
            w.ax.add_collection3d(faces)


def _organize_by_cost(coverage_set):
    """Group CircuitPolytope objects by their cost.

    Args:
        coverage_set: List of CircuitPolytope objects.

    Returns:
        dict: Mapping from cost to list of polytopes with that cost.
              Zero-cost polytopes are excluded.
    """
    organized = {}
    for polytope in coverage_set:
        cost = polytope.cost
        if cost == 0:
            continue
        if cost not in organized:
            organized[cost] = []
        organized[cost].append(polytope)
    return organized


def plot_coverage_set(coverage_set, overlap=False, volume_info=None):
    """Plot a coverage set of 3D polytopes in the Weyl chamber.

    Args:
        coverage_set: List of CircuitPolytope objects.
        overlap: If True, all polytopes are drawn on the same plot.
            If False, each cost level gets a separate subplot.
        volume_info: Optional dict mapping cost to (unique_volume, cumulative_volume)
            tuples. When provided, volume information is displayed in subplot titles.
    """
    organized_set = _organize_by_cost(coverage_set)

    if not organized_set:
        print("No non-zero cost polytopes to plot.")
        return

    if overlap:
        fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
        w = WeylChamber()
        w.labels = {}
        w.render(ax)
        for i, (cost, polytopes) in enumerate(organized_set.items()):
            color = COLORS[i % len(COLORS)]
            for polytope in polytopes:
                _plot_polytope(polytope, ax, w, color=color)
    else:
        n = len(organized_set)
        ncols = min(n, MAX_COLS)
        nrows = (n + ncols - 1) // ncols
        fig, axs = plt.subplots(
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

        for i, (cost, polytopes) in enumerate(organized_set.items()):
            ax = axs[i]
            w = WeylChamber()
            w.labels = {}
            w.render(ax)
            color = COLORS[i % len(COLORS)]
            for polytope in polytopes:
                _plot_polytope(polytope, ax, w, color=color)

            # Create title with optional volume information
            if volume_info and cost in volume_info:
                unique_vol, cumulative_vol = volume_info[cost]
                title = (
                    f"Cost: {cost}\n"
                    f"Unique vol: {unique_vol:.4f}\n"
                    f"Cumulative vol: {cumulative_vol:.4f}"
                )
            else:
                title = f"Cost: {cost}"
            w.ax.set_title(title)

        # Hide empty subplots
        for j in range(n, len(axs)):
            axs[j].set_visible(False)

    plt.show()
