"""End-to-end fidelity tests across every ISA in the fixture library."""

import numpy as np
import pytest
from qiskit.circuit.library import iSwapGate
from qiskit.quantum_info import Operator, average_gate_fidelity, random_unitary

from gulps.core.isa import DiscreteISA
from gulps.gulps_decomposer import GulpsDecomposer
from tests.fixtures.isas import get_all_test_isas

N_RANDOM = 20
FIDELITY_TOL = 1 - 1e-8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(params=get_all_test_isas(precompute_polytopes=True))
def decomposer(request):
    return GulpsDecomposer(isa=request.param)


@pytest.fixture(params=get_all_test_isas())
def decomposer_no_precompute(request):
    return GulpsDecomposer(isa=request.param)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _assert_fidelity(target, circuit, label=""):
    fid = average_gate_fidelity(Operator(target), Operator(circuit))
    assert (
        fid > FIDELITY_TOL
    ), f"Fidelity too low{' (' + label + ')' if label else ''}: {fid}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_local_unitary():
    """A tensor-product target (zero entangling power) should decompose perfectly."""
    U = random_unitary(2, seed=42).data
    V = random_unitary(2, seed=43).data
    target = np.kron(U, V)

    isa = DiscreteISA([iSwapGate().power(1 / 2)], [0.5], ["sq2iswap"])
    _assert_fidelity(target, GulpsDecomposer(isa=isa)(target), "local")


def test_exact_isa_gate():
    """A target that *is* a basis gate should round-trip exactly."""
    gate = iSwapGate().power(1 / 2)
    isa = DiscreteISA([gate], [0.5], ["sq2iswap"])
    _assert_fidelity(gate, GulpsDecomposer(isa=isa)(gate), "exact ISA gate")


def test_random_unitaries(decomposer):
    """N_RANDOM random unitaries x every ISA with polytope precompute."""
    for seed in range(N_RANDOM):
        u = random_unitary(4, seed=seed)
        _assert_fidelity(u, decomposer(u), f"seed={seed}")


def test_random_unitaries_no_precompute(decomposer_no_precompute):
    """Same sweep but with on-demand sentence enumeration."""
    for seed in range(N_RANDOM):
        u = random_unitary(4, seed=seed)
        _assert_fidelity(u, decomposer_no_precompute(u), f"seed={seed}")
