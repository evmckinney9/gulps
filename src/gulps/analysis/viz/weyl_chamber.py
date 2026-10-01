# Adapted from https://github.com/qucontrol/weylchamber

"""Minimal Weyl-chamber 3-D renderer inlined from the ``weylchamber`` package.

Original source: https://github.com/qucontrol/weylchamber

References:
    Goerz et al., "Optimizing for an arbitrary perfect entangler. II. Application",
    Phys. Rev. A 91, 062307 (2015). https://doi.org/10.1103/PhysRevA.91.062307

    Watts et al., "Optimizing for an arbitrary perfect entangler: I. Functionals",
    Phys. Rev. A 91, 062306 (2015). https://doi.org/10.1103/PhysRevA.91.062306

    Zhang et al., "Geometric theory of nonlocal two-qubit operations",
    Phys. Rev. A 67, 042313 (2003). https://doi.org/10.1103/PhysRevA.67.042313
"""

from __future__ import annotations

import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D

# Named points in the Weyl chamber (coordinates in units of pi)
_WEYL_POINTS = {
    "O": np.array((0.0, 0.0, 0.0)),
    "A1": np.array((1.0, 0.0, 0.0)),
    "A2": np.array((0.5, 0.5, 0.0)),
    "A3": np.array((0.5, 0.5, 0.5)),
    "L": np.array((0.5, 0.0, 0.0)),
    "M": np.array((0.75, 0.25, 0.0)),
    "N": np.array((0.75, 0.25, 0.25)),
    "P": np.array((0.25, 0.25, 0.25)),
    "Q": np.array((0.25, 0.25, 0.0)),
}

# Edges of the Weyl chamber: (point1, point2, foreground?)
_WEYL_EDGES = [
    ("O", "A1", True),
    ("A1", "A2", True),
    ("A2", "A3", True),
    ("A3", "A1", True),
    ("A3", "O", True),
    ("O", "A2", False),
]

# Edges of the perfect-entanglers polyhedron
_PE_EDGES = [
    ("L", "N", True),
    ("L", "P", True),
    ("N", "P", True),
    ("N", "A2", True),
    ("N", "M", True),
    ("M", "L", False),
    ("Q", "L", False),
    ("P", "Q", False),
    ("P", "A2", False),
]

_EDGE_FG = {"color": "black", "linestyle": "-", "lw": 0.5}
_EDGE_BG = {"color": "black", "linestyle": "--", "lw": 0.5}


def draw_chamber(ax: Axes3D) -> None:
    """Draw the Weyl chamber wireframe and the perfect-entangler polyhedron on 3-D axes."""
    ax.view_init(elev=20, azim=-50)
    ax.patch.set_facecolor("None")
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Permute from matplotlib's default, not the current value: a reused
    # axes would otherwise flip the c3 axis on every call.
    d = type(ax.zaxis)._PLANES
    ax.zaxis._PLANES = (d[2], d[3], d[0], d[1], d[4], d[5])
    ax.zaxis.set_rotate_label(False)
    ax.zaxis.label.set_rotation(90)
    ax.grid(False)

    for edges in (_WEYL_EDGES, _PE_EDGES):
        for p1, p2, fg in edges:
            o, e = _WEYL_POINTS[p1], _WEYL_POINTS[p2]
            style = _EDGE_FG if fg else {**_EDGE_BG, "zorder": -1}
            ax.plot([o[0], e[0]], [o[1], e[1]], [o[2], e[2]], **style)

    ax.set_xlabel(r"$c_1$", labelpad=-9)
    ax.set_ylabel(r"$c_2$", labelpad=-14)
    ax.set_zlabel(r"$c_3$", labelpad=-14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.5)
    ax.set_zlim(0, 0.5)
    ax.xaxis.set_ticks([0, 0.25, 0.5, 0.75, 1])
    ax.xaxis.set_ticklabels(["0", "", r"$1/2$", "", "1"])
    ax.yaxis.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.yaxis.set_ticklabels(["0", "", "", "", "", r"$1/2$"])
    ax.zaxis.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.zaxis.set_ticklabels(["0", "", "", "", "", r"$1/2$"])
    ax.tick_params(axis="x", pad=-6.0)
    ax.tick_params(axis="y", pad=-4.0)
    ax.tick_params(axis="z", pad=-6.0)
    for t in ax.get_yticklabels():
        t.set_va("center")
        t.set_ha("left")
    for t in ax.get_zticklabels():
        t.set_va("center")
        t.set_ha("right")
