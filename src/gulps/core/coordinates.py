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

"""Ground-truth coordinate definitions for two-qubit gate invariants.

Coordinate systems for a local-equivalence class of U in SU(4):

  Monodromy (m0, m1, m2 [, m3]):  eigenvalue log-spectrum of
      Gamma_Q = U*(sy kron sy)*U^T*(sy kron sy), alcove-normalized.  m3 = -sum.
  Weyl (c1, c2, c3):  c = M_WEYL @ m[:3].  Chamber: c1 >= c2 >= c3 >= 0, c1 <= 0.5.
      Units are fractions of pi; multiply by pi/2 for Qiskit KAK radians.
  Makhlin (Re G1, Im G1, Re G2):  polynomial invariants in magic basis.

rho-reflection identifies two Weyl-chamber representatives of each class:
  Weyl:      (c1, c2, c3)  <->  (1-c1, c2, -c3)
  Monodromy: (a,b,c,d)     <->  (c+0.5, d+0.5, a-0.5, b-0.5)
Geometrically: conjugation by sy kron I on the canonical gate.  Makhlin
invariants are identical for both branches (polynomial, rho-agnostic).
Each pipeline layer resolves rho locally because accumulated floating-point
error can flip which branch KAK assigns.

References:
    Zhang et al., PRA 67, 042313 (2003).
    Peterson, Crooks, Smith, Quantum 4, 247 (2020).
"""

import numpy as np

from gulps.core.monodromy_coords import (
    positive_canonical_to_monodromy_coordinate,
    unitary_to_monodromy_coordinate,
)

# -- shared constants --

# sy kron sy  (real, symmetric, unitary, self-inverse)
SYSY = np.array(
    [[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]],
    dtype=np.complex128,
)

# Magic basis  Q = (H kron I)*CX*(I kron Sdg) -- diagonalizes the canonical gate
MAGIC = np.array(
    [[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]],
    dtype=np.complex128,
) / np.sqrt(2)
MAGIC_DAG = MAGIC.conj().T

# Monodromy -> Weyl linear transform:  c = M_WEYL @ m[:3]
M_WEYL = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.float64)

# -- small Kronecker product --


def kron_2x2(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> None:
    """Write kron(a, b) into pre-allocated 4x4 *out*, where a and b are 2x2."""
    out[0:2, 0:2] = a[0, 0] * b
    out[0:2, 2:4] = a[0, 1] * b
    out[2:4, 0:2] = a[1, 0] * b
    out[2:4, 2:4] = a[1, 1] * b


# Pauli matrices (used in rho-branch KAK corrections)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

# -- monodromy <-> unitary --

# Re-export so callers can use coordinates.py as the single import.
monodromy_from_unitary = unitary_to_monodromy_coordinate


def monodromy_from_weyl(c1: float, c2: float, c3: float) -> tuple:
    """Monodromy coordinates from Weyl (c1, c2, c3) in normalized [0,1] units."""
    return positive_canonical_to_monodromy_coordinate(
        np.pi / 2 * c1, np.pi / 2 * c2, np.pi / 2 * c3
    )


# -- monodromy -> weyl (and back) --


def weyl_from_monodromy(m) -> np.ndarray:
    """Weyl coordinates (c1, c2, c3) from monodromy (m0, m1, m2)."""
    return M_WEYL @ np.asarray(m[:3], dtype=np.float64)


# -- rho-reflection --


def rho_reflect_weyl(c) -> np.ndarray:
    """rho-reflect Weyl coordinates: (c1, c2, c3) -> (1-c1, c2, -c3)."""
    c = np.asarray(c, dtype=np.float64)
    return np.array([1.0 - c[0], c[1], -c[2]], dtype=np.float64)


def rho_reflect_monodromy(logspec) -> tuple:
    """rho-reflect monodromy logspec (a,b,c,d) -> (c+0.5, d+0.5, a-0.5, b-0.5)."""
    a, b, c, d = logspec
    return (
        np.float64(c + 0.5),
        np.float64(d + 0.5),
        np.float64(a - 0.5),
        np.float64(b - 0.5),
    )


# -- canonical matrix --


def canonical_matrix(c1: float, c2: float, c3: float) -> np.ndarray:
    """4x4 canonical unitary exp(-i(a*XX + b*YY + c*ZZ)) from Weyl coords.

    Args:
        c1: float: Weyl coordinate c1 in normalized [0,1] units.
        c2: float: Weyl coordinate c2 in normalized [0,1] units
        c3: float: Weyl coordinate c3 in normalized [0,1] units.
    """
    a = np.pi / 2 * c1
    b = np.pi / 2 * c2
    g = np.pi / 2 * c3
    eig = np.exp(1j * g)
    eig_c = np.exp(-1j * g)
    cam = np.cos(a - b)
    sam = np.sin(a - b)
    cap = np.cos(a + b)
    sap = np.sin(a + b)
    return np.array(
        [
            [eig * cam, 0, 0, 1j * eig * sam],
            [0, eig_c * cap, 1j * eig_c * sap, 0],
            [0, 1j * eig_c * sap, eig_c * cap, 0],
            [1j * eig * sam, 0, 0, eig * cam],
        ],
        dtype=np.complex128,
    )
