# tests/isas.py
import numpy as np
import pytest
from qiskit.circuit.library import CXGate, XXPlusYYGate, iSwapGate
from qiskit.quantum_info import random_unitary

from gulps.gulps_synthesis import GulpsDecomposer
from gulps.utils.invariants import GateInvariants


def get_all_test_isas():
    """Returns a list of (gate_list, cost_list) tuples for testing."""
    return [
        ([iSwapGate()], [1.0]),
        ([CXGate(), iSwapGate().power(1 / 2)], [1.0, 0.5]),
        ([iSwapGate(), iSwapGate().power(1 / 2)], [1.0, 0.5]),
        ([CXGate(), CXGate().power(1 / 2)], [1.0, 0.5]),
    ]
