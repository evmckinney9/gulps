"""Edge-case tests: SWAP targets, mirror ISAs (zero-cost SWAP), fSim gates."""

import numpy as np
import pytest
from qiskit.circuit.library import CXGate, SwapGate, iSwapGate
from qiskit.quantum_info import Operator, average_gate_fidelity, random_unitary

from gulps.core.isa import DiscreteISA
from gulps.gulps_decomposer import GulpsDecomposer
from gulps.isa_library.fsim import fsim
from tests.fixtures.isas import get_all_test_isas, get_slim_isas

FIDELITY_TOL = 1 - 1e-8


@pytest.fixture(params=get_slim_isas(precompute_polytopes=True))
def decomposer(request):
    return GulpsDecomposer(isa=request.param)


def _assert_fidelity(target, circuit, label=""):
    fid = average_gate_fidelity(Operator(target), Operator(circuit))
    assert fid > FIDELITY_TOL, (
        f"Fidelity too low{' (' + label + ')' if label else ''}: {fid}"
    )


def test_swap(decomposer):
    """SWAP should decompose on every ISA (issues #2, #3)."""
    _assert_fidelity(SwapGate(), decomposer(SwapGate()))


def test_mirror_isa():
    """ISA with a zero-cost SWAP mirror gate."""
    isa = DiscreteISA(
        [CXGate().power(1 / 2), iSwapGate().power(1 / 2), SwapGate()],
        [1 / 2, 1 / 2, 0.0],
        ["sq2cx", "sq2iswap", "swap"],
    )
    u = random_unitary(4, seed=0)
    _assert_fidelity(u, GulpsDecomposer(isa=isa)(u), "mirror ISA")


def test_swap_into_fsim():
    """SWAP target into an fSim-based ISA (issue #2)."""
    isa = DiscreteISA([fsim(np.pi / 2, np.pi / 6).power(1 / 2)], [1.0], ["sq2fsim"])
    _assert_fidelity(SwapGate(), GulpsDecomposer(isa=isa)(SwapGate()), "fsim SWAP")


def test_random_mirror_into_sq3iswap():
    """Random unitary into iSwap^1/3 + zero-cost SWAP (issue #3)."""
    isa = DiscreteISA(
        [iSwapGate().power(1 / 3), SwapGate()], [1 / 3, 0.0], ["cb3iswap", "swap"]
    )
    u = random_unitary(4, seed=10)
    _assert_fidelity(u, GulpsDecomposer(isa=isa)(u), "mirror sq3iswap")
