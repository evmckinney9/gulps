"""Pass-level behavior for GulpsDecompositionPass: dedup, identity short-circuit, no-op shapes."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, iSwapGate
from qiskit.quantum_info import random_unitary
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from gulps import GulpsDecomposer, GulpsDecompositionPass
from gulps.comparisons.isa_library.benchmark_xx import (
    build_benchmark_circuits,
    build_target,
)
from gulps.core.isa import DiscreteISA
from tests._common import assert_fidelity, count_2q

N_QUBITS = 4
FAMILIES = ["QFT", "EfficientSU2", "QV", "Random"]


@pytest.fixture(scope="module")
def decomposer():
    return GulpsDecomposer(
        isa=DiscreteISA(
            [iSwapGate().power(1 / 2), CXGate()], [0.5, 1.0], precompute_polytopes=False
        )
    )


@pytest.fixture(scope="module")
def benchmarks():
    return build_benchmark_circuits(N_QUBITS)


def _run(decomposer, qc):
    """Consolidate 2q blocks, then decompose. Returns (out_circuit, pass_ref)."""
    pass_ = GulpsDecompositionPass(decomposer)
    out = PassManager([Collect2qBlocks(), ConsolidateBlocks(), pass_]).run(qc)
    return out, pass_


@pytest.mark.parametrize("family", FAMILIES)
def test_benchmark_fidelity(decomposer, benchmarks, family):
    """Standard Qiskit circuit families round-trip through the pass."""
    qc = benchmarks[family]
    out, _ = _run(decomposer, qc)
    assert_fidelity(qc, out, family)


# EfficientSU2's consolidated blocks all share the CX Weyl class -> n_unique=1.
# QV's blocks are independent random SU(4) -> every block is its own class.
@pytest.mark.parametrize(
    "family, expected_unique",
    [("EfficientSU2", lambda n: 1), ("QV", lambda n: n)],
)
def test_benchmark_dedup(decomposer, benchmarks, family, expected_unique):
    _, pass_ = _run(decomposer, benchmarks[family])
    t = pass_.last_phase_timing
    assert t["n_nodes"] > 1
    assert t["n_unique"] == expected_unique(t["n_nodes"])


def test_identity_class_no_2q(decomposer):
    """kron(b, a) is locally identity; LP returns depth-0 synthetic solution."""
    a = random_unitary(2, seed=10).data
    b = random_unitary(2, seed=11).data
    qc = QuantumCircuit(2)
    qc.unitary(np.kron(b, a), [0, 1])
    pass_ = GulpsDecompositionPass(decomposer)
    out = PassManager([pass_]).run(qc)
    assert_fidelity(qc, out, "identity")
    assert pass_.last_phase_timing["n_unique"] == 1
    assert count_2q(out) == 0


@pytest.mark.parametrize("family", FAMILIES)
def test_plugin_entry_point(benchmarks, family):
    """Plugin is discovered via qiskit.transpiler.translation entry point and preserves fidelity."""
    pm = generate_preset_pass_manager(
        optimization_level=1, target=build_target(N_QUBITS), translation_method="gulps"
    )
    qc = benchmarks[family]
    assert_fidelity(qc, pm.run(qc), f"plugin/{family}")


def test_noop_when_no_2q_present(decomposer):
    """Empty / single-qubit-only circuits skip the pass entirely."""
    qc = QuantumCircuit(3)
    pass_ = GulpsDecompositionPass(decomposer)
    out = PassManager([pass_]).run(qc)
    assert_fidelity(qc, out, "noop")
    assert pass_.last_phase_timing == {}
