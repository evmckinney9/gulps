"""Tests for non-Haar targets: random_circuit inputs and deterministic Weyl grid."""

import pytest
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Operator, average_gate_fidelity
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition

from gulps import GateInvariants, GulpsDecomposer, GulpsDecompositionPass
from gulps.config import GulpsConfig
from gulps.core.coverage import weyl_linspace
from tests.fixtures.isas import get_random_circuit_isas

FIDELITY_TOL = 1 - 1e-8
N_SEEDS = 5
N_WEYL = 64


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(params=get_random_circuit_isas())
def gulps_pm(request):
    """PassManager wrapping GulpsDecomposer for each random-circuit ISA."""
    decomposer = GulpsDecomposer(isa=request.param, config_options=GulpsConfig())
    return PassManager(
        [
            GulpsDecompositionPass(decomposer),
        ]
    )


@pytest.fixture(params=get_random_circuit_isas())
def decomposer(request):
    """Raw GulpsDecomposer for each random-circuit ISA."""
    return GulpsDecomposer(isa=request.param, config_options=GulpsConfig())


def test_random_circuit(gulps_pm):
    """3-qubit random circuit (depth 20) through the full pass manager."""
    for seed in range(N_SEEDS):
        qc = random_circuit(3, 20, max_operands=2, seed=seed)
        target_op = Operator(qc)
        out = gulps_pm.run(qc)
        fid = average_gate_fidelity(target_op, Operator(out))
        assert fid > FIDELITY_TOL, f"Fidelity {fid:.8f} for seed={seed}"


def test_weyl_linspace(decomposer):
    """Decompose a target at a deterministic Weyl-chamber grid points."""
    for weyl_pt in weyl_linspace(N_WEYL):
        target = GateInvariants.from_weyl(weyl_pt)
        circuit = decomposer(target.unitary)
        fid = average_gate_fidelity(Operator(target.unitary), Operator(circuit))
        assert fid > FIDELITY_TOL, (
            f"Fidelity {fid:.8f} at Weyl point ({weyl_pt[0]:.3f}, {weyl_pt[1]:.3f}, {weyl_pt[2]:.3f})"
        )
