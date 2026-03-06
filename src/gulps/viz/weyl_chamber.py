"""Minimal Weyl-chamber 3-D renderer inlined from the ``weylchamber`` package.

Original source: https://github.com/qucontrol/weylchamber
"""

import numpy as np

# ---------------------------------------------------------------------------
# Named points in the Weyl chamber (coordinates in units of π)
# ---------------------------------------------------------------------------
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


class WeylChamber:
    """Lightweight Weyl-chamber plotter (subset of ``weylchamber.WeylChamber``).

    Originally from the ``weylchamber`` Python package by the QuControl group.
    """

    def __init__(self):
        """Initialize default label positions and rendering options."""
        self.ax = None
        self.labels: dict = {
            "A_1": _WEYL_POINTS["A1"] + np.array((-0.03, 0.04, 0.00)),
            "A_2": _WEYL_POINTS["A2"] + np.array((0.01, 0.0, -0.01)),
            "A_3": _WEYL_POINTS["A3"] + np.array((-0.01, 0.0, 0.0)),
            "O": _WEYL_POINTS["O"] + np.array((-0.025, 0.0, 0.02)),
            "L": _WEYL_POINTS["L"] + np.array((-0.075, 0.0, 0.01)),
            "M": _WEYL_POINTS["M"] + np.array((0.05, -0.01, 0.0)),
            "N": _WEYL_POINTS["N"] + np.array((-0.075, 0.0, 0.009)),
            "P": _WEYL_POINTS["P"] + np.array((-0.05, 0.0, 0.008)),
            "Q": _WEYL_POINTS["Q"] + np.array((0.0, 0.01, 0.03)),
        }

    # ------------------------------------------------------------------
    def render(self, ax):
        """Render the Weyl chamber wireframe on *ax* (an ``Axes3D``)."""
        self.ax = ax
        ax.view_init(elev=20, azim=-50)
        ax.patch.set_facecolor("None")
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # z-axis on the left
        tmp = ax.zaxis._PLANES
        ax.zaxis._PLANES = (tmp[2], tmp[3], tmp[0], tmp[1], tmp[4], tmp[5])
        ax.zaxis.set_rotate_label(False)
        ax.zaxis.label.set_rotation(90)
        ax.grid(False)

        # background edges
        for p1, p2, fg in _WEYL_EDGES:
            if not fg:
                self._line(ax, p1, p2, zorder=-1, **_EDGE_BG)
        for p1, p2, fg in _PE_EDGES:
            if not fg:
                self._line(ax, p1, p2, zorder=-1, **_EDGE_BG)

        # labels
        for name, coords in self.labels.items():
            ax.text(*coords, f"${name}$", color="black", fontsize="small")

        # foreground edges
        for p1, p2, fg in _WEYL_EDGES:
            if fg:
                self._line(ax, p1, p2, **_EDGE_FG)
        for p1, p2, fg in _PE_EDGES:
            if fg:
                self._line(ax, p1, p2, **_EDGE_FG)

        # axis cosmetics
        ax.set_xlabel(r"$c_1$", labelpad=-9)
        ax.set_ylabel(r"$c_2$", labelpad=-14)
        ax.set_zlabel(r"$c_3$", labelpad=-14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.5)
        ax.set_zlim(0, 0.5)
        ax.xaxis.set_ticks([0, 0.25, 0.5, 0.75, 1])
        ax.xaxis.set_ticklabels(["0", "", r"$\pi/2$", "", r"$\pi$"])
        ax.yaxis.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax.yaxis.set_ticklabels(["0", "", "", "", "", r"$\pi/2$"])
        ax.zaxis.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax.zaxis.set_ticklabels(["0", "", "", "", "", r"$\pi/2$"])
        ax.tick_params(axis="x", pad=-6.0)
        ax.tick_params(axis="y", pad=-4.0)
        ax.tick_params(axis="z", pad=-6.0)
        for t in ax.get_yticklabels():
            t.set_va("center")
            t.set_ha("left")
        for t in ax.get_zticklabels():
            t.set_va("center")
            t.set_ha("right")

    # ------------------------------------------------------------------
    @staticmethod
    def _line(ax, origin, end, **kwargs):
        o = _WEYL_POINTS[origin] if isinstance(origin, str) else origin
        e = _WEYL_POINTS[end] if isinstance(end, str) else end
        ax.plot([o[0], e[0]], [o[1], e[1]], [o[2], e[2]], **kwargs)
