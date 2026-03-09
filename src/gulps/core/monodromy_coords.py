"""Monodromy coordinate helpers inlined from the ``monodromy`` package.

The two public functions :func:`unitary_to_monodromy_coordinate` and
:func:`positive_canonical_to_monodromy_coordinate` were originally provided
by ``monodromy.coordinates``.  They are reproduced here so that the core
GULPS decomposition path has no runtime dependency on the ``monodromy``
package (which is now an optional extra).

References:
    Peterson, Crooks, Smith, "Fixed-Depth Two-Qubit Circuits and the Monodromy
    Polytope", Quantum 4, 247 (2020). https://doi.org/10.22331/q-2020-03-26-247

    Agnihotri, Woodward, "Eigenvalues of products of unitary matrices and quantum
    Schubert calculus", Math. Res. Lett. 5(6), 817-836 (1998).
    https://doi.org/10.4310/MRL.1998.v5.n6.a10
"""

from functools import reduce

import numpy as np

_EPSILON = 1e-6

# σ_y ⊗ σ_y matrix used to build the Γ_Q matrix
_SYSY = np.array(
    [[0, 0, 0, 1], [0, 0, -1, 0], [0, -1, 0, 0], [1, 0, 0, 0]],
    dtype=complex,
)


def _normalize_logspec_A(coordinate):
    """Rotate a sorted LogSpec tuple into its A-normal form.

    Originally ``monodromy.coordinates.normalize_logspec_A``.
    """
    total = sum(coordinate)
    if total > _EPSILON:
        return _normalize_logspec_A([*coordinate[1:], coordinate[0] - 1])
    if total < -_EPSILON:
        raise ValueError(f"Over-rotated: {total}.")
    return coordinate


def _normalize_logspec_AC2(coordinate):
    """Rotate a sorted LogSpec tuple into its A_{C2}-normal form.

    Originally ``monodromy.coordinates.normalize_logspec_AC2``.
    """
    pn = _normalize_logspec_A(coordinate)
    if pn[1] >= -pn[2]:
        return pn
    return [pn[2] + 0.5, pn[3] + 0.5, pn[0] - 0.5, pn[1] - 0.5]


def unitary_to_monodromy_coordinate(unitary: np.ndarray):
    """Compute the alcove (monodromy) coordinate of a 4x4 unitary.

    Originally ``monodromy.coordinates.unitary_to_monodromy_coordinate``.
    """
    unitary = unitary * np.complex128(np.linalg.det(unitary)) ** (-1 / 4)
    gamma_q = reduce(np.dot, [unitary, _SYSY, unitary.T, _SYSY])
    logspec = np.real(np.log(np.linalg.eigvals(gamma_q)) / (2j * np.pi))
    return _normalize_logspec_AC2(sorted(np.mod(logspec, 1.0), reverse=True))


def positive_canonical_to_monodromy_coordinate(x, y, z):
    """Convert unnormalised positive-canonical (x, y, z) to monodromy coords.

    Originally ``monodromy.coordinates.positive_canonical_to_monodromy_coordinate``.
    """
    return (
        (x + y - z) / np.pi,
        (x - y + z) / np.pi,
        (-x + y + z) / np.pi,
    )
