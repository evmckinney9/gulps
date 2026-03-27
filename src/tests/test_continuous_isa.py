"""End-to-end correctness tests for ContinuousISA decomposition.

Tests the full CPLEX LP → segment synthesis → recovery pipeline for
CX, iSWAP, and SWAP gate families with continuous power parameterization.
"""

import numpy as np
import pytest
from qiskit.circuit.library import CXGate, SwapGate, iSwapGate
from qiskit.quantum_info import Operator, average_gate_fidelity, random_unitary

from gulps.config import GulpsConfig
from gulps.core.isa import ContinuousISA
from gulps.gulps_decomposer import GulpsDecomposer

try:
    import docplex  # noqa: F401

    HAS_CPLEX = True
except ImportError:
    HAS_CPLEX = False

pytestmark = pytest.mark.skipif(not HAS_CPLEX, reason="CPLEX not installed")

FIDELITY_TOL = 1 - 1e-8
N_RANDOM = 10
_CFG = GulpsConfig(flag_duration=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _assert_fidelity(target, circuit, label=""):
    fid = average_gate_fidelity(Operator(target), Operator(circuit))
    assert fid > FIDELITY_TOL, (
        f"Fidelity too low{' (' + label + ')' if label else ''}: {fid}"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(
    params=[
        ("CX", CXGate()),
        ("iSWAP", iSwapGate()),
        ("SWAP", SwapGate()),
    ],
    ids=lambda p: p[0],
)
def continuous_decomposer(request):
    """GulpsDecomposer wrapping a single-family ContinuousISA."""
    name, gate = request.param
    isa = ContinuousISA.from_base_gate(gate, name=name)
    return GulpsDecomposer(isa=isa, config_options=_CFG)


@pytest.fixture(
    params=[
        ("CX_sqc05", CXGate(), 0.5),
        ("iSWAP_sqc1", iSwapGate(), 1.0),
    ],
    ids=lambda p: p[0],
)
def continuous_decomposer_large_sqc(request):
    """ContinuousISA with large single_qubit_cost."""
    name, gate, sqc = request.param
    isa = ContinuousISA.from_base_gate(gate, name=name, single_qubit_cost=sqc)
    return GulpsDecomposer(isa=isa, config_options=_CFG)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_random_unitaries(continuous_decomposer):
    """Random Haar unitaries should decompose correctly."""
    for seed in range(N_RANDOM):
        u = random_unitary(4, seed=seed)
        _assert_fidelity(u, continuous_decomposer(u), f"seed={seed}")


def test_identity_target(continuous_decomposer):
    """Identity (zero entangling power) should decompose with no 2Q gates."""
    target = np.eye(4, dtype=complex)
    _assert_fidelity(target, continuous_decomposer(target), "identity")


def test_exact_base_gate():
    """The base gate itself should round-trip as a depth-1 decomposition."""
    for Gate, name in [(CXGate, "CX"), (iSwapGate, "iSWAP"), (SwapGate, "SWAP")]:
        gate = Gate()
        isa = ContinuousISA.from_base_gate(gate, name=name)
        decomposer = GulpsDecomposer(isa=isa, config_options=_CFG)
        circ = decomposer(gate)
        _assert_fidelity(gate, circ, f"{name} exact gate")
        depth = sum(1 for op in circ.data if op.operation.num_qubits == 2)
        assert depth == 1, f"{name} exact gate should be depth 1, got {depth}"


def test_fractional_power_target():
    """gate^0.5 should be decomposable as depth 1 with k=0.5."""
    for Gate, name in [(CXGate, "CX"), (iSwapGate, "iSWAP")]:
        gate = Gate()
        half_gate = gate.power(0.5)
        isa = ContinuousISA.from_base_gate(gate, name=name)
        decomposer = GulpsDecomposer(isa=isa, config_options=_CFG)
        circ = decomposer(half_gate)
        _assert_fidelity(half_gate, circ, f"{name}^0.5")
        depth = sum(1 for op in circ.data if op.operation.num_qubits == 2)
        assert depth == 1, f"{name}^0.5 should be depth 1, got {depth}"


def test_large_single_qubit_cost(continuous_decomposer_large_sqc):
    """Decomposition must succeed with large single_qubit_cost."""
    for seed in range(5):
        u = random_unitary(4, seed=seed)
        _assert_fidelity(u, continuous_decomposer_large_sqc(u), f"seed={seed}")


def test_swap_family_targets():
    """SWAP-family targets (c3 > 0) should work with ContinuousISA SWAP base."""
    gate = SwapGate()
    isa = ContinuousISA.from_base_gate(gate, name="SWAP")
    decomposer = GulpsDecomposer(isa=isa, config_options=_CFG)

    half_swap = gate.power(0.5)
    _assert_fidelity(half_swap, decomposer(half_swap), "SWAP^0.5")
