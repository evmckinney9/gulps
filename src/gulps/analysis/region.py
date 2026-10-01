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
"""Reachable regions of gate sentences and their Haar measure.

A gate sentence reaches a convex polytope of local-equivalence classes in two
orientations related by the chamber reflection ``rho``. ``ReachableRegion`` is
one orientation, stored as the compiler's 14 lower bounds on phase-subset sums
in monodromy coordinates, with Weyl coordinates at the public interface.
Vertices are the feasible triple-plane intersections.

To compute Haar measure, we divide the polytope into tetrahedra and integrate
the chamber's Haar density (Watts, O'Connor, Vala) over each in closed form.
For a union of regions, inclusion-exclusion accounts for their intersections.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations

import numpy as np

from gulps._accelerate import (
    REGION_FORMS,
    region_contains,
    sentence_bounds,
    weyl_from_monodromy,
)
from gulps.invariants import LocalEquivalenceClass

# Half-space slack for constructing vertices and faces, and for testing whether
# one region's vertices lie in another: vertices are only this accurate.
_FEAS_TOL = 1e-7
# Vertices equal to this many decimals are one vertex.
_VERTEX_DECIMALS = 9
# Nodes at least this far apart use the divided-difference recurrence.
_SEPARATED = 0.1
# A tetrahedron below this volume (in Weyl units) is a degenerate face.
_DEGENERATE_VOLUME = 1e-15
# Rounding floor of inclusion-exclusion over unit-scale masses in double precision.
_MASS_TOL = 1e-12

# A bound ``b`` on row ``k`` is the half-space ``_NORMALS[k] . m >= b + _SHIFTS[k]``
# in monodromy coordinates. The Weyl chamber adds x - w <= 1, x >= y, and y >= z
# as three more rows, and its c3 >= 0 is the row of y + z.
_NORMALS = np.array(
    [normal for normal, _shift in REGION_FORMS]
    + [(-2.0, -1.0, -1.0), (1.0, -1.0, 0.0), (0.0, 1.0, -1.0)]
)
_SHIFTS = np.array([shift for _normal, shift in REGION_FORMS])
# The chamber as right-hand sides over ``_NORMALS``; ``-inf`` leaves a row free.
_CHAMBER = np.full(len(_NORMALS), -np.inf)
_CHAMBER[len(_SHIFTS) :] = (-1.0, 0.0, 0.0)
_CHAMBER[(_NORMALS == (0.0, 1.0, 1.0)).all(axis=1)] = 0.0


def _vertex_system() -> tuple[np.ndarray, np.ndarray]:
    """The row triples with a unique intersection, and their inverse matrices."""
    triples = np.array(list(combinations(range(len(_NORMALS)), 3)))
    systems = _NORMALS[triples]
    # The normals are integers, so a nonzero determinant is at least one.
    keep = np.abs(np.linalg.det(systems)) > 0.5
    return triples[keep], np.linalg.inv(systems[keep])


_TRIPLES, _INVERSES = _vertex_system()


def _rho_rows() -> list[int]:
    """The row of each reflected bound's source.

    ``rho`` maps monodromy ``(x, y, z)`` to ``(z + 1/2, w + 1/2, x - 1/2)``,
    with ``w = -x-y-z``: it permutes the phases x, y, z, w to z, w, x, y and
    offsets them. Each row's shift absorbs the offset of its subset sum, so
    the reflected region's bounds are a permutation of the region's.
    """
    n = _NORMALS[: len(_SHIFTS)]
    image = np.stack([n[:, 2] - n[:, 1], -n[:, 1], n[:, 0] - n[:, 1]], axis=1)
    target = [int(np.flatnonzero((n == row).all(axis=1))[0]) for row in image]
    return np.argsort(target).tolist()


_RHO = _rho_rows()


def _haar_terms() -> tuple[np.ndarray, np.ndarray]:
    """The Haar density as ``sum coef * cos(<a, c>)`` in Weyl coordinates."""
    coefs, avecs = [], []
    for left, right in ((0, 1), (1, 2), (2, 0)):
        for sign in (-1.0, 1.0):
            for coef, (p, q) in ((1.0, (2.0, 4.0)), (-1.0, (4.0, 2.0))):
                a = np.zeros(3)
                a[left], a[right] = p, sign * q
                coefs.append(coef)
                avecs.append(np.pi * a)
    return np.array(coefs), np.array(avecs)


_HAAR_COEF, _HAAR_A = _haar_terms()  # (12,), (12, 3)
# Normalizes the full chamber to mass 1.
_HAAR_NORM = 3.0 * np.pi**2 / 2.0


def _vertices(rhs: np.ndarray) -> np.ndarray | None:
    """Vertices (monodromy) of the half-spaces ``_NORMALS . m >= rhs``."""
    triples, inverses = _TRIPLES, _INVERSES
    finite = np.isfinite(rhs)[triples].all(axis=1)
    x = (inverses[finite] @ rhs[triples[finite]][..., None])[..., 0]
    pts = x[(_NORMALS @ x.T >= rhs[:, None] - _FEAS_TOL).all(0)]
    if not len(pts):
        return None
    _keys, first = np.unique(np.round(pts, _VERTEX_DECIMALS), axis=0, return_index=True)
    return pts[first]


def _faces(rhs: np.ndarray, mono: np.ndarray) -> list[np.ndarray]:
    """Vertex indices of each polygonal face in cyclic order, over ``mono``."""
    seen: set[tuple[int, ...]] = set()
    faces = []
    for n, r in zip(_NORMALS, rhs, strict=True):
        indices = np.flatnonzero(np.abs(mono @ n - r) <= _FEAS_TOL)
        key = tuple(indices.tolist())
        if len(indices) < 3 or key in seen:
            continue  # an edge, a vertex, or the other side of a flat region
        seen.add(key)
        # Dropping the normal's dominant axis projects the face without
        # collapsing it, so angles in the other two axes keep the cyclic order.
        u, v = np.delete(
            mono[indices] - mono[indices].mean(axis=0), np.argmax(np.abs(n)), axis=1
        ).T
        faces.append(indices[np.argsort(np.arctan2(v, u))])
    return faces


def _dd_exp(nodes: np.ndarray) -> np.ndarray:
    """Divided differences exp[t0..t3] for each row of ``nodes`` (M,4).

    Rows with all nodes at least ``_SEPARATED`` apart use the recurrence, whose
    rounding error is of order eps / gap**3. Other rows use entry (0, 3) of the
    exponential of the bidiagonal matrix with the nodes on the diagonal and
    ones above it (Opitz), by scaling and squaring (McCurdy, Ng, and Parlett,
    1984).
    """
    rows, cols = np.triu_indices(4, 1)
    close = np.abs(nodes[:, rows] - nodes[:, cols]).min(axis=1) < _SEPARATED
    out = np.empty(len(nodes), complex)
    far = nodes[~close]
    c = np.exp(far)
    for k in range(1, 4):
        for i in range(3, k - 1, -1):
            c[:, i] = (c[:, i] - c[:, i - 1]) / (far[:, i] - far[:, i - k])
    out[~close] = c[:, 3]
    near = nodes[close]
    # At norm 1/2 or less, 15 Taylor terms reach double precision.
    s = int(np.ceil(np.log2(2.0 * (np.abs(near).max(initial=0.0) + 1.0))))
    m = np.zeros((len(near), 4, 4), complex)
    m[:, range(4), range(4)] = near / 2.0**s
    m[:, range(3), range(1, 4)] = 1.0 / 2.0**s
    e = term = np.broadcast_to(np.eye(4, dtype=complex), m.shape)
    for k in range(1, 16):
        term = term @ m / k
        e = e + term
    for _ in range(s):
        e = e @ e
    out[close] = e[:, 0, 3]
    return out


def _tetra_haar(verts: np.ndarray) -> float:
    """Haar integral over one or more tetrahedra, evaluated as one array."""
    tetra = np.asarray(verts, float).reshape(-1, 4, 3)
    det = np.abs(np.linalg.det(tetra[:, 1:] - tetra[:, :1]))
    keep = det >= _DEGENERATE_VOLUME
    if not keep.any():
        return 0.0
    tetra, det = tetra[keep], det[keep]
    nodes = 1j * np.sort(tetra @ _HAAR_A.T, axis=1).transpose(0, 2, 1)
    terms = _dd_exp(nodes.reshape(-1, 4)).real.reshape(-1, 12)
    return float(det @ (terms @ _HAAR_COEF))


def _mass(rhs: np.ndarray, mono: np.ndarray | None) -> float:
    """Haar mass of one convex region with vertices ``mono``; 0 if empty or flat."""
    if mono is None or len(mono) < 4:
        return 0.0
    weyl = weyl_from_monodromy(mono)
    center = weyl.mean(axis=0)
    tetrahedra = [
        (center, weyl[face[0]], weyl[face[i]], weyl[face[i + 1]])
        for face in _faces(rhs, mono)
        for i in range(1, len(face) - 1)
    ]
    return _HAAR_NORM * _tetra_haar(np.array(tetrahedra)) if tetrahedra else 0.0


class _Geometry:
    """Regions by bounds, so one computation finds each region's vertices once."""

    def __init__(self) -> None:
        self._regions: dict[tuple[float, ...], ReachableRegion] = {}

    def intern(self, region: ReachableRegion) -> ReachableRegion:
        """The region with these bounds seen first."""
        return self._regions.setdefault(region._bounds, region)

    def meet(self, a: ReachableRegion, b: ReachableRegion) -> ReachableRegion:
        """The intersection of ``a`` and ``b``."""
        return self.intern(ReachableRegion(tuple(map(max, a._bounds, b._bounds))))


@dataclass(frozen=True)
class ReachableRegion:
    """One orientation of the local-equivalence classes a gate sentence reaches.

    Use :meth:`reaches` to test whether the sentence can implement a target.
    It checks this orientation and its mirror, ``rho``. Coordinates are Weyl
    ``(c1, c2, c3)`` in units of pi. Construct with :meth:`of`.
    """

    # The compiler's lower bounds on the 14 shifted phase-subset sums; ``-inf``
    # where absent.
    _bounds: tuple[float, ...]

    @classmethod
    def of(cls, sentence: Iterable[LocalEquivalenceClass]) -> ReachableRegion:
        """The region reached by a sentence of these classes, in any order.

        Args:
            sentence: The invariant classes of the two-qubit gates.
        """
        return cls(tuple(sentence_bounds(list(sentence))))

    @cached_property
    def rho(self) -> ReachableRegion:
        """The mirror orientation of the same sentence's reach."""
        return ReachableRegion(tuple(self._bounds[k] for k in _RHO))

    @cached_property
    def _rhs(self) -> np.ndarray:
        """Right-hand sides over ``_NORMALS``, with the chamber."""
        forms = np.asarray(self._bounds) + _SHIFTS
        return np.maximum(np.append(forms, _CHAMBER[len(forms) :]), _CHAMBER)

    @cached_property
    def _mono(self) -> np.ndarray | None:
        return _vertices(self._rhs)

    @property
    def vertices(self) -> np.ndarray | None:
        """The polytope's vertices as an ``(N, 3)`` Weyl array, or ``None`` if empty."""
        return None if self._mono is None else weyl_from_monodromy(self._mono)

    @property
    def faces(self) -> list[np.ndarray]:
        """The polygonal faces, each an ``(k, 3)`` Weyl array in cyclic order."""
        if self._mono is None:
            return []
        return [
            weyl_from_monodromy(self._mono[f]) for f in _faces(self._rhs, self._mono)
        ]

    @cached_property
    def haar_mass(self) -> float:
        """The fraction of Haar-random two-qubit unitaries whose class lies inside."""
        return _mass(self._rhs, self._mono)

    def contains(self, target: LocalEquivalenceClass | np.ndarray) -> bool | np.ndarray:
        """Whether a class lies inside this orientation, by the compiler's test.

        Args:
            target: A :class:`~gulps.invariants.LocalEquivalenceClass`, one Weyl triple, or an
                ``(N, 3)`` array of Weyl triples.

        Returns:
            One bool, or a bool array with one entry per row.
        """
        if isinstance(target, LocalEquivalenceClass):
            target = target.weyl
        pts = np.asarray(target, float)
        inside = region_contains(self._bounds, np.atleast_2d(pts))
        return bool(inside[0]) if pts.ndim == 1 else inside

    def reaches(self, target: LocalEquivalenceClass | np.ndarray) -> bool | np.ndarray:
        """Whether the sentence can implement a target, allowing global phase.

        Unlike :meth:`contains`, this checks both orientations of the region.

        Args:
            target: A :class:`~gulps.invariants.LocalEquivalenceClass`, one Weyl
                triple, or an ``(N, 3)`` array of Weyl triples.

        Returns:
            One bool, or a bool array with one entry per row.
        """
        return self.contains(target) | self.rho.contains(target)


def _contains_region(outer: ReachableRegion, vertices: np.ndarray) -> bool:
    """Whether a convex region with these vertices lies inside ``outer``."""
    return bool((_NORMALS @ vertices.T >= outer._rhs[:, None] - _FEAS_TOL).all())


def _reduce_union(
    pieces: list[ReachableRegion], geometry: _Geometry
) -> list[ReachableRegion]:
    """Drop pieces contained in another piece; the union is unchanged."""
    kept: list[ReachableRegion] = []
    for p in map(geometry.intern, pieces):
        v = p._mono
        if v is None or any(_contains_region(q, v) for q in kept):
            continue
        kept = [q for q in kept if not _contains_region(p, q._mono)]
        kept.append(p)
    return kept


# No bounds: the whole chamber.
_FREE = ReachableRegion((-np.inf,) * len(_SHIFTS))


def _union_mass(pieces: list[ReachableRegion], geometry: _Geometry) -> float:
    """Haar mass of a reduced union, by inclusion-exclusion."""

    def rec(start: int, region: ReachableRegion, depth: int) -> float:
        total = 0.0
        for i in range(start, len(pieces)):
            r = geometry.meet(region, pieces[i])
            m = r.haar_mass
            # Further intersections are subsets and also have zero mass.
            if m == 0.0:
                continue
            total += (m if depth % 2 == 0 else -m) + rec(i + 1, r, depth + 1)
        return total

    return rec(0, _FREE, 0)


def haar_mass(regions: Iterable[ReachableRegion]) -> float:
    """Haar mass of the union of ``regions``, by inclusion-exclusion."""
    geometry = _Geometry()
    return _union_mass(_reduce_union(list(regions), geometry), geometry)


class RegionUnion:
    """A growing union of sentences' reach, for cost-ordered coverage."""

    def __init__(self) -> None:
        """An empty union."""
        self._geometry = _Geometry()
        self._pieces: list[ReachableRegion] = []  # reduced full-dimensional union
        self._shown: list[ReachableRegion] = []  # every added region, both orientations
        self.mass = 0.0

    def add(self, region: ReachableRegion) -> float | None:
        """Add both orientations of ``region``.

        Returns:
            The Haar mass they reach first; ``0.0`` if their reach is new but
            lower-dimensional; ``None`` if earlier regions already cover it.
        """
        pieces = tuple(map(self._geometry.intern, (region, region.rho)))
        massive = [p for p in pieces if p.haar_mass > 0.0]
        if massive:
            candidate = _reduce_union(self._pieces + massive, self._geometry)
            fresh = _union_mass(candidate, self._geometry) - self.mass
            if fresh <= _MASS_TOL:
                return None
            self._pieces, self.mass = candidate, self.mass + fresh
        elif not any(self._outside_shown(p) for p in pieces):
            return None
        else:
            fresh = 0.0
        self._shown.extend(pieces)
        return fresh

    def _outside_shown(self, piece: ReachableRegion) -> bool:
        """Whether ``piece`` is nonempty and outside every single shown region."""
        v = piece._mono
        return v is not None and not any(_contains_region(r, v) for r in self._shown)
