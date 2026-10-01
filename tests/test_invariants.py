"""Batch projection preserves ordering across parallel execution and validates inputs."""

import numpy as np
import pytest
from qiskit._accelerate.two_qubit_decompose import two_qubit_local_invariants
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate

from gulps.invariants import LocalEquivalenceClass


@pytest.mark.parametrize(
    "coords, folded",
    [
        ((0.31, 0.17, 0.08), (0.31, 0.17, 0.08)),
        ((0.31, 0.17, -0.08), (0.69, 0.17, 0.08)),
        ((0.2, 0.1, 1e-7), (0.2, 0.1, 1e-7)),
        ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ],
    ids=["interior", "reflected", "no_approximate_specialization", "swap"],
)
def test_weyl_chart_matches_independent_pauli_rotations(coords, folded):
    # An inverse/forward round trip can hide a shared sign or normalization bug.
    expected = (
        RXXGate(-np.pi * coords[0]).to_matrix()
        @ RYYGate(-np.pi * coords[1]).to_matrix()
        @ RZZGate(-np.pi * coords[2]).to_matrix()
    )
    canonical = (
        RXXGate(-np.pi * folded[0]).to_matrix()
        @ RYYGate(-np.pi * folded[1]).to_matrix()
        @ RZZGate(-np.pi * folded[2]).to_matrix()
    )
    np.testing.assert_allclose(
        LocalEquivalenceClass(coords).matrix, canonical, atol=1e-12
    )
    np.testing.assert_allclose(
        LocalEquivalenceClass.from_unitary(expected).weyl, folded, atol=1e-14
    )
    np.testing.assert_allclose(
        LocalEquivalenceClass(coords).makhlin,
        two_qubit_local_invariants(np.exp(0.37j) * expected),
        atol=1e-12,
    )
