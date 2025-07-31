# test_decomposer_end_to_end.py

import numpy as np
import pytest
from qiskit.circuit.library import CXGate, iSwapGate
from qiskit.quantum_info import Operator, average_gate_fidelity, random_unitary

from gulps.gulps_decomposer import GulpsDecomposer
from gulps.utils.invariants import GateInvariants
from tests.fixtures.isas import get_all_test_isas


def test_decomposer_fidelity_on_random_unitaries(decomposer_fixture):
    isa = [
        (CXGate(), 1.0, "cx"),
        (CXGate().power(1 / 2), 1 / 2, "sqrt2cx"),
        (iSwapGate().power(1 / 2), 1 / 2, "sqrt2iswap"),
        (iSwapGate().power(1 / 3), 1 / 3, "sqrt3iswap"),
    ]
    gate_set, costs, names = zip(*isa)
    decomposer = GulpsDecomposer(gate_set=gate_set, costs=costs, names=names)

    hard_seeds = [330, 528, 891]
    for seed in hard_seeds:
        u = random_unitary(4, seed=seed)
        v = Operator(decomposer(u))
        target_unitary = random_unitary(4, seed=seed)
        output_circuit = decomposer_fixture._run(target_unitary)

        fidelity = average_gate_fidelity(
            Operator(target_unitary), Operator(output_circuit)
        )
        assert fidelity > 1 - 1e-6, f"Fidelity too low at seed {seed}: {fidelity}"
