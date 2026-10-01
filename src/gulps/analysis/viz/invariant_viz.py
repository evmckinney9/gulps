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

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D

from gulps.analysis.viz.weyl_chamber import draw_chamber
from gulps.invariants import LocalEquivalenceClass

_SHADOW_COLOR = "gray"
_SHADOW_ALPHA = 0.25
# Drop lines to the shadows are drawn a little darker than the shadows.
_DROP_ALPHA = 0.4


def scatter_plot(
    invariant_list: list[LocalEquivalenceClass],
    ax: Axes3D | None = None,
    **kwargs: Any,  # matplotlib scatter args
) -> Axes3D:
    """Scatter plot of a list of LocalEquivalenceClass in the Weyl chamber.

    Args:
        invariant_list: Gate invariants to plot.
        ax: Optional existing 3D axes for overlaying multiple scatter calls.
        **kwargs: Passed through to ``ax.scatter`` (e.g. ``color``, ``s``, ``label``).

    Returns:
        The three-dimensional axes.
    """
    if ax is None:
        ax = plt.figure().add_subplot(111, projection="3d", computed_zorder=False)
        ax.set_proj_type("persp")
        draw_chamber(ax)

    # |c3| draws each class in the c3 >= 0 half, where the chamber is drawn.
    points = np.array([abs(g.weyl) for g in invariant_list]).reshape(-1, 3)
    # Project each point onto z=0 to show its depth.
    ax.scatter(
        points[:, 0],
        points[:, 1],
        np.zeros(len(points)),
        color=_SHADOW_COLOR,
        s=15,
        alpha=_SHADOW_ALPHA,
        zorder=-10,
        marker="o",
    )
    above = points[points[:, 2] > 0]
    if len(above):
        segments = [[(p[0], p[1], p[2]), (p[0], p[1], 0.0)] for p in above]
        ax.add_collection3d(
            Line3DCollection(
                segments,
                colors=_SHADOW_COLOR,
                linestyles="--",
                linewidths=0.8,
                alpha=_DROP_ALPHA,
                zorder=-10,
            )
        )
    kwargs.setdefault("zorder", 1)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], **kwargs)
    return ax
