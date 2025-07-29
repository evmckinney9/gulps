# test_decomposer_end_to_end.py

import numpy as np
import pytest
from qiskit.circuit.library import iSwapGate
from qiskit.quantum_info import Operator, average_gate_fidelity, random_unitary

from gulps.gulps_decomposer import GulpsDecomposer
from gulps.utils.invariants import GateInvariants
from tests.fixtures.isas import get_all_test_isas

N_tests = 100


@pytest.fixture(params=get_all_test_isas())
def decomposer_fixture(request):
    gates, costs = request.param
    return GulpsDecomposer(gates, costs, precompute_polytopes=True)


@pytest.fixture(params=get_all_test_isas())
def decomposer_fixture_no_precompute(request):
    gates, costs = request.param
    return GulpsDecomposer(gates, costs, precompute_polytopes=False)


def test_decomposer_on_local_unitary():
    # Local-only gate: U\otimes V
    U = random_unitary(2, seed=42).data
    V = random_unitary(2, seed=42).data
    target = np.kron(U, V)

    decomposer = GulpsDecomposer(
        gate_set=[iSwapGate().power(1 / 2)],
        costs=[1.0],
        precompute_polytopes=True,
    )
    output_circuit = decomposer._run(target)
    fidelity = average_gate_fidelity(Operator(target), Operator(output_circuit))
    assert fidelity > 1 - 1e-6, f"Local-only fidelity too low: {fidelity}"


def test_decomposer_on_exact_isa_gate():
    gate = iSwapGate().power(1 / 2)

    decomposer = GulpsDecomposer(
        gate_set=[gate],
        costs=[1.0],
        precompute_polytopes=True,
    )

    output_circuit = decomposer._run(gate)
    fidelity = average_gate_fidelity(Operator(gate), Operator(output_circuit))
    assert fidelity > 1 - 1e-6, f"ISA gate fidelity too low: {fidelity}"


def test_decomposer_fidelity_on_random_unitaries(decomposer_fixture):
    for seed in range(N_tests):
        target_unitary = random_unitary(4, seed=seed)
        output_circuit = decomposer_fixture._run(target_unitary)

        fidelity = average_gate_fidelity(
            Operator(target_unitary), Operator(output_circuit)
        )
        assert fidelity > 1 - 1e-6, f"Fidelity too low at seed {seed}: {fidelity}"


def test_decomposer_fidelity_no_precompute_on_random_unitaries(
    decomposer_fixture_no_precompute,
):
    for seed in range(N_tests):
        target_unitary = random_unitary(4, seed=seed)
        output_circuit = decomposer_fixture_no_precompute._run(target_unitary)

        fidelity = average_gate_fidelity(
            Operator(target_unitary), Operator(output_circuit)
        )
        assert fidelity > 1 - 1e-6, f"Fidelity too low at seed {seed}: {fidelity}"
