"""ISA fixtures shared by tests and benchmarks.

All helpers return ``list[DiscreteISA]``.  Extra keyword arguments
(e.g. ``precompute_polytopes=True``) are forwarded to the constructor.
"""

import numpy as np
from qiskit.circuit.library import CXGate, SwapGate, iSwapGate

from gulps.core.isa import DiscreteISA
from gulps.isa_library.fsim import fsim


def get_slim_isas(**isa_kwargs) -> list[DiscreteISA]:
    """Small ISA set for fast benchmarking."""
    return [
        DiscreteISA([iSwapGate().power(1 / 4)], [0.25], ["sq4iswap"], **isa_kwargs),
        DiscreteISA(
            [CXGate().power(1 / 2), iSwapGate().power(1 / 3)],
            [0.5, 1 / 3],
            ["sq2cx", "sq3iswap"],
            **isa_kwargs,
        ),
    ]


def get_all_test_isas(**isa_kwargs) -> list[DiscreteISA]:
    """Full ISA set: slim + random-circuit + mirror (zero-cost SWAP)."""
    return get_slim_isas(**isa_kwargs) + [
        DiscreteISA(
            [iSwapGate(), iSwapGate().power(1 / 2), iSwapGate().power(1 / 3)],
            [1.0, 0.5, 1 / 3],
            ["iswap", "sq2iswap", "sq3iswap"],
            **isa_kwargs,
        ),
        DiscreteISA(
            [CXGate(), CXGate().power(1 / 2), CXGate().power(1 / 4)],
            [1.0, 0.5, 0.25],
            ["cx", "sq2cx", "sq4cx"],
            **isa_kwargs,
        ),
        DiscreteISA(
            [CXGate().power(1 / 3)],
            [1 / 3],
            ["sq3cx"],
            **isa_kwargs,
        ),
        DiscreteISA(
            [fsim(np.pi / 2, np.pi / 6), iSwapGate().power(1 / 4)],
            [1.0, 0.25],
            ["fsim_pi2_pi6", "sq4iswap"],
            **isa_kwargs,
        ),
        DiscreteISA(
            [CXGate(), iSwapGate().power(1 / 2), SwapGate()],
            [1.0, 0.5, 0.0],
            ["cx", "sq2iswap", "swap"],
            **isa_kwargs,
        ),
    ]
