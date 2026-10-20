"""Seeded two-qubit targets and their symmetries for parametrized tests."""

import numpy as np
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate
from qiskit.quantum_info import random_unitary


def haar_unitary(seed):
    """A Haar-random U(4) matrix."""
    return random_unitary(4, seed=seed).data


def local_pair(seed):
    """A tensor product of two Haar-random single-qubit unitaries."""
    a = random_unitary(2, seed=2 * seed).data
    b = random_unitary(2, seed=2 * seed + 1).data
    return np.kron(a, b)


# Weyl chamber faces, edges, a repeated spectrum, and a point 1e-7 off a face.
BOUNDARY_COORDINATES = (
    (0.0, 0.0, 0.0),
    (0.5, 0.0, 0.0),
    (0.5, 0.5, 0.0),
    (0.5, 0.5, 0.5),
    (0.2, 0.1, 0.0),
    (0.2, 0.1, 1e-7),
    (0.3, 0.3, 0.1),
    (0.3, 0.1, 0.1),
)


def boundary_unitary(seed):
    """A chamber-boundary class between seeded local gates, built without GULPS."""
    a, b, c = BOUNDARY_COORDINATES[seed % len(BOUNDARY_COORDINATES)]
    canonical = (
        RXXGate(np.pi * a).to_matrix()
        @ RYYGate(np.pi * b).to_matrix()
        @ RZZGate(np.pi * c).to_matrix()
    )
    return local_pair(20000 + seed) @ canonical @ local_pair(30000 + seed)
