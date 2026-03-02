# test_decomposer_end_to_end.py

import time

import numpy as np
import pytest
from qiskit.circuit.library import CXGate, iSwapGate
from qiskit.quantum_info import Operator, average_gate_fidelity, random_unitary

from gulps.core.invariants import GateInvariants
from gulps.gulps_decomposer import GulpsDecomposer
from tests.fixtures.isas import get_all_test_isas


def test_cx_iswap_on_slow_seeds():
    isa = [
        (CXGate(), 1.0, "cx"),
        (CXGate().power(1 / 2), 1 / 2, "sqrt2cx"),
        (iSwapGate().power(1 / 2), 1 / 2, "sqrt2iswap"),
        (iSwapGate().power(1 / 3), 1 / 3, "sqrt3iswap"),
    ]
    gate_set, costs, names = zip(*isa)
    decomposer = GulpsDecomposer(gate_set=gate_set, costs=costs, names=names)
    # warm up (jit compile)
    decomposer(random_unitary(4, seed=0))

    slow_seeds = [956, 587, 891, 217, 330, 244, 781, 594, 996, 437]
    for seed in slow_seeds:
        target_unitary = random_unitary(4, seed=seed)
        output_circuit = decomposer._run(target_unitary)

        fidelity = average_gate_fidelity(
            Operator(target_unitary), Operator(output_circuit)
        )
        assert fidelity > 1 - 1e-6, f"Fidelity too low at seed {seed}: {fidelity}"
        assert (
            decomposer.last_timing["segments"] < 0.2
        ), f"Numeric timing too high at seed {seed}: {decomposer.last_timing['segments']:.4f} seconds"


def test_sq4iswap_on_slow_seeds():
    isa = [
        (iSwapGate().power(1 / 4), 1 / 4, "sqrt4iswap"),
    ]
    gate_set, costs, names = zip(*isa)
    decomposer = GulpsDecomposer(gate_set=gate_set, costs=costs, names=names)
    # warm up (jit compile)
    decomposer(random_unitary(4, seed=0))

    slow_seeds = [638, 437, 386, 529, 16, 627, 674, 718, 189, 261]
    for seed in slow_seeds:
        target_unitary = random_unitary(4, seed=seed)
        output_circuit = decomposer._run(target_unitary)

        fidelity = average_gate_fidelity(
            Operator(target_unitary), Operator(output_circuit)
        )
        assert fidelity > 1 - 1e-6, f"Fidelity too low at seed {seed}: {fidelity}"
        assert (
            decomposer.last_timing["segments"] < 0.2
        ), f"Numeric timing too high at seed {seed}: {decomposer.last_timing['segments']:.4f} seconds"


def test_sq2iswap_on_slow_seeds():
    isa = [
        (iSwapGate().power(1 / 2), 1 / 2, "sqrt2iswap"),
    ]
    gate_set, costs, names = zip(*isa)
    decomposer = GulpsDecomposer(gate_set=gate_set, costs=costs, names=names)
    # warm up (jit compile)
    decomposer(random_unitary(4, seed=0))

    slow_seeds = [984, 573, 781, 244, 217, 324, 627, 690, 117]
    for seed in slow_seeds:
        target_unitary = random_unitary(4, seed=seed)
        output_circuit = decomposer._run(target_unitary)

        fidelity = average_gate_fidelity(
            Operator(target_unitary), Operator(output_circuit)
        )
        assert fidelity > 1 - 1e-6, f"Fidelity too low at seed {seed}: {fidelity}"
        assert (
            decomposer.last_timing["segments"] < 0.2
        ), f"Numeric timing too high at seed {seed}: {decomposer.last_timing['segments']:.4f} seconds"
