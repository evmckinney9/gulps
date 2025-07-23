# test_decomposer_lp_agreement.py

import numpy as np
import pytest
from qiskit.circuit.library import CXGate, XXPlusYYGate, iSwapGate
from qiskit.quantum_info import random_unitary

from gulps.gulps_synthesis import GulpsDecomposer
from gulps.utils.invariants import GateInvariants
from tests.fixtures.isas import get_all_test_isas


@pytest.fixture(params=get_all_test_isas())
def decomposer_fixture(request):
    gates, costs = request.param
    return GulpsDecomposer(gates, costs, precompute_polytopes=True)


def test_lp_agrees_with_polytope_solution(decomposer_fixture):
    # Pick a target from within known ISA coverage
    for seed in range(10):
        target_unitary = random_unitary(4, seed=seed)
        target_inv = GateInvariants.from_unitary(target_unitary, enforce_alcove=True)

        # Step 1: Lookup via polytope
        sentence, rho_bool = decomposer_fixture.isa.polytope_lookup(target_inv)
        assert sentence is not None, "Polytope lookup failed for reachable gate."

        # Step 2: Try LP on that sentence
        sentence_out, intermediates, actual_rho = decomposer_fixture._try_lp(
            sentence, target_inv, rho_bool=rho_bool
        )
        assert sentence_out is not None, (
            "LP failed even though polytope lookup succeeded."
        )
