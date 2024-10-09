"""Test the monodromy linear programming."""

import numpy as np
from monodromy.coordinates import unitary_to_monodromy_coordinate
from qiskit.circuit.library import CXGate, SwapGate, iSwapGate
from qiskit.quantum_info.random import random_unitary
from scipy.optimize import linprog

from hetero_isas.monodromy_lp import (
    MonodromyLPDecomposer,
    MonodromyLPGate,
    MonodromyLPISA,
)


def test_cnot_swap():
    """Check 3*CNOT decomposes a SWAP gate."""
    cx = MonodromyLPGate.from_unitary(CXGate(), cost=1, name="cx")
    isa = MonodromyLPISA(cx, cx, cx, isa_ordered=True, sequence_length=3)
    decomposer = MonodromyLPDecomposer(isa)
    target = SwapGate()
    result = decomposer._best_decomposition(target)
    assert result


def test_sqrtiswap_haar():
    """Check sqrtiSWAP Haar score within 5% over 1000 random samples."""
    sqrtiswap = MonodromyLPGate.from_unitary(
        iSwapGate().power(1 / 2), cost=1, name="isw2"
    )
    isa = MonodromyLPISA(sqrtiswap)
    decomposer = MonodromyLPDecomposer(isa)
    expected_len = 0
    for _ in range(N := 1_000):
        target = random_unitary(4).to_matrix()
        result = decomposer._best_decomposition(target)
        expected_len += result.n
    expected_len /= N
    assert np.abs(expected_len - (13 / 6)) < 0.05


def test_qiskit_vs_monolp():
    """Check that qiskit/monodromyLP agree in expected strengths.

    Uses an ISA of {CX, sqrtCX, sqrt[3]CX}
    """
    pass


# def test_cnot_cnot_iswap():
#     """L(cnot, cnot, iSWAP)."""
#     g = unitary_to_monodromy_coordinate(CXGate())[:-1]
#     t = unitary_to_monodromy_coordinate(iSwapGate())[:-1]
#     c, A, b, A_eq, b_eq = _single_L(g, g, t)
#     ret = linprog(
#         c=c, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=(None, None), method="highs"
#     )
#     assert ret.success


# def test_iswap_cnot_iswap():
#     """L(iswap, cnot, swap)."""
#     ci = unitary_to_monodromy_coordinate(iSwapGate())[:-1]
#     g = unitary_to_monodromy_coordinate(CXGate())[:-1]
#     t = unitary_to_monodromy_coordinate(SwapGate())[:-1]
#     c, A, b, A_eq, b_eq = _single_L(ci, g, t)
#     ret = linprog(
#         c=c, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=(None, None), method="highs"
#     )
#     assert ret.success


# def test_cnot_swap():
#     """L(cnot, cnot, c_2) + L(c_2, cnot, swap); solve for c_2."""
#     G = unitary_to_monodromy_coordinate(CXGate())[:-1]
#     T = unitary_to_monodromy_coordinate(SwapGate())[:-1]
#     c, A, b, A_eq, b_eq = _construct_L(G, T, N=3)
#     ret = linprog(
#         c=c, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=(None, None), method="highs"
#     )
#     ret.x
