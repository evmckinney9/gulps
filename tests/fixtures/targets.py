"""Seeded two-qubit targets and their symmetries for parametrized tests."""

import numpy as np
from qiskit.quantum_info import random_unitary


def haar_unitary(seed):
    """A Haar-random U(4) matrix."""
    return random_unitary(4, seed=seed).data


def local_pair(seed):
    """A tensor product of two Haar-random single-qubit unitaries."""
    a = random_unitary(2, seed=2 * seed).data
    b = random_unitary(2, seed=2 * seed + 1).data
    return np.kron(a, b)
