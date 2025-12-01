import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as ss
from monodromy.coordinates import monodromy_to_positive_canonical_polytope
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from weylchamber import WeylChamber


def _plot_polytope(circuit_polytope, w, color="red"):
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


def _plot_coverage_set(coverage_set, overlap=False):
    """Plot a set of 3D polytopes.

    Args:
        coverage_set (list): a list of CircuitPolytope objects.
        overlap (bool): If True, all polytopes are drawn on the same plot. If False, each polytope is drawn in a separate subplot.
    """
    colors = [
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

    # Preprocess coverage_set to organize CircuitPolytope objects based on their cost
    organized_set = {}
    for circuit_polytope in coverage_set:
        cost = circuit_polytope.cost
        if cost == 0:
            continue
        if cost not in organized_set:
            organized_set[cost] = []
        organized_set[cost].append(circuit_polytope)

    if overlap:
        fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
        w = WeylChamber()
        w.labels = {}
        w.render(ax)
        for i, (cost, polytopes) in enumerate(organized_set.items()):
            color = colors[i % len(colors)]
            for circuit_polytope in polytopes:
                _plot_polytope(circuit_polytope, color=color, w=w)
    else:
        n = len(organized_set)
        fig, axs = plt.subplots(
            1, n, subplot_kw={"projection": "3d"}, figsize=(n * 5, 5)
        )  # Adjust size to avoid crowding
        for i, (cost, polytopes) in enumerate(organized_set.items()):
            ax = axs[i] if n > 1 else axs
            w = WeylChamber()
            w.labels = {}
            w.render(ax)
            color = colors[i % len(colors)]
            for circuit_polytope in polytopes:
                _plot_polytope(circuit_polytope, color=color, w=w)
            w.ax.set_title(f"Cost: {cost}")

    plt.show()
