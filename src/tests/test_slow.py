"""Regression tests for historically-slow seeds (fidelity + timing)."""

import pytest
from qiskit.circuit.library import CXGate, iSwapGate
from qiskit.quantum_info import Operator, average_gate_fidelity, random_unitary

from gulps.core.isa import DiscreteISA
from gulps.gulps_decomposer import GulpsDecomposer

FIDELITY_TOL = 1 - 1e-6
SEGMENT_TIME_LIMIT = 0.2  # seconds


def _run_slow_seeds(isa, seeds):
    """Shared helper: warm up, then assert fidelity + timing on each seed."""
    decomposer = GulpsDecomposer(isa=isa)
    for seed in seeds:
        u = random_unitary(4, seed=seed)
        v = decomposer(u)

        fid = average_gate_fidelity(Operator(u), Operator(v))
        assert fid > FIDELITY_TOL, f"Fidelity too low at seed {seed}: {fid}"
        assert decomposer.last_timing["segments"] < SEGMENT_TIME_LIMIT, (
            f"Segments too slow at seed {seed}: "
            f"{decomposer.last_timing['segments']:.4f}s"
        )


def test_cx_iswap_on_slow_seeds():
    isa = DiscreteISA(
        [
            CXGate(),
            CXGate().power(1 / 2),
            iSwapGate().power(1 / 2),
            iSwapGate().power(1 / 3),
        ],
        [1.0, 1 / 2, 1 / 2, 1 / 3],
        ["cx", "sq2cx", "sq2iswap", "cb3iswap"],
    )
    _run_slow_seeds(isa, [956, 587, 891, 217, 330, 244, 781, 594, 996, 437])


def test_sq4iswap_on_slow_seeds():
    isa = DiscreteISA([iSwapGate().power(1 / 4)], [1 / 4], ["sq4iswap"])
    _run_slow_seeds(isa, [638, 437, 386, 529, 16, 627, 674, 718, 189, 261])


def test_sq2iswap_on_slow_seeds():
    isa = DiscreteISA([iSwapGate().power(1 / 2)], [1 / 2], ["sq2iswap"])
    _run_slow_seeds(isa, [984, 573, 781, 244, 217, 324, 627, 690, 117])
