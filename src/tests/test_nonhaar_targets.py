"""Tests for non-Haar targets: random_circuit inputs and deterministic Weyl grid."""

import pytest
from qiskit.circuit.random import random_circuit
from qiskit.transpiler import PassManager

from gulps import GateInvariants, GulpsDecomposer, GulpsDecompositionPass
from gulps.viz.weyl_chamber import weyl_linspace
from tests._common import assert_fidelity
from tests.fixtures.isas import get_all_test_isas

N_SEEDS = 5
N_WEYL = 128


@pytest.fixture(params=get_all_test_isas())
def gulps_pm(request):
    """PassManager wrapping GulpsDecomposer for each random-circuit ISA."""
    decomposer = GulpsDecomposer(
        isa=request.param,
    )
    return PassManager(
        [
            GulpsDecompositionPass(decomposer),
        ]
    )


@pytest.fixture(params=get_all_test_isas())
def decomposer(request):
    """Raw GulpsDecomposer for each random-circuit ISA."""
    return GulpsDecomposer(isa=request.param)


def test_random_circuit(gulps_pm):
    """3-qubit random circuit (depth 20) through the full pass manager."""
    for seed in range(N_SEEDS):
        qc = random_circuit(3, 20, max_operands=2, seed=seed)
        assert_fidelity(qc, gulps_pm.run(qc), f"seed={seed}")


def test_weyl_linspace(decomposer):
    """Decompose a target at a deterministic Weyl-chamber grid points."""
    for weyl_pt in weyl_linspace(N_WEYL):
        target = GateInvariants.from_weyl(weyl_pt)
        assert_fidelity(
            target.matrix,
            decomposer(target.matrix),
            f"weyl=({weyl_pt[0]:.3f},{weyl_pt[1]:.3f},{weyl_pt[2]:.3f})",
        )
