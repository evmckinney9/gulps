# test_decomposer_end_to_end.py

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import SwapGate, XXPlusYYGate, iSwapGate
from qiskit.quantum_info import Operator, average_gate_fidelity, random_unitary

from gulps.gulps_decomposer import GulpsDecomposer
from gulps.utils.invariants import GateInvariants
from tests.fixtures.isas import get_all_test_isas


@pytest.fixture(params=get_all_test_isas())
def decomposer_fixture(request):
    gates, costs = request.param
    return GulpsDecomposer(gates, costs, precompute_polytopes=True)


def test_swap(decomposer_fixture):
    """Related to issues #2 and #3"""
    target = SwapGate()
    output_circuit = output_circuit = decomposer_fixture._run(target)
    fidelity = average_gate_fidelity(Operator(target), Operator(output_circuit))
    assert fidelity > 1 - 1e-6, f"ISA gate fidelity too low: {fidelity}"


def test_swap_into_fsim():
    """Issue #2"""

    def fsim(theta, phi):
        _fsim = QuantumCircuit(2, name="fsim")
        _fsim.append(XXPlusYYGate(2 * theta), [0, 1])
        _fsim.cp(phi, 0, 1)
        return _fsim.to_gate()

    decomposer = GulpsDecomposer(
        gate_set=[fsim(np.pi / 2, np.pi / 6).power(1 / 2)],
        costs=[1.0],
    )
    u = SwapGate()
    v = Operator(decomposer(u))
    fid = average_gate_fidelity(u, v)
    assert fid > 1 - 1e-6, f"Fidelity too low: {fid}"


def test_random_mirror_into_sq3iswap():
    """Issue #3"""
    decomposer = GulpsDecomposer([iSwapGate().power(1 / 3), SwapGate()], [1 / 3, 0.0])
    u = random_unitary(4, seed=10)
    v = Operator(decomposer(u))
    fid = average_gate_fidelity(u, v)
    assert fid > 1 - 1e-6, f"Fidelity too low: {fid}"
    assert fid > 1 - 1e-6, f"Fidelity too low: {fid}"
    assert fid > 1 - 1e-6, f"Fidelity too low: {fid}"
