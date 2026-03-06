"""ISA fixtures shared by tests and benchmarks.

All helpers return ``list[DiscreteISA]``.  Extra keyword arguments
(e.g. ``precompute_polytopes=True``) are forwarded to the constructor.
"""

import numpy as np
from qiskit.circuit.library import CXGate, SwapGate, iSwapGate

from gulps.core.isa import DiscreteISA
from gulps.isa_library.fsim import fsim


def get_slim_isas(**isa_kwargs) -> list[DiscreteISA]:
    """Small diverse set for fast A/B speed benchmarking.

    Stresses different code paths to avoid over-optimizing for one ISA:
      - Single weak gate  → moderate depth, simple LP
      - CX + iSwap family → richer LP, different Weyl region
    """
    return [
        DiscreteISA([iSwapGate().power(1 / 4)], [0.25], ["sq4iswap"], **isa_kwargs),
        DiscreteISA(
            [CXGate().power(1 / 2), iSwapGate().power(1 / 3)],
            [0.5, 1 / 3],
            ["sq2cx", "cb3iswap"],
            **isa_kwargs,
        ),
    ]


def get_random_circuit_isas(**isa_kwargs) -> list[DiscreteISA]:
    """ISAs that stress degenerate Weyl faces hit by random_circuit.

    Unlike Haar-random unitaries, ``random_circuit`` composes standard
    library gates that sit exactly on degenerate Weyl faces (e.g. iSwap
    has c₁=c₂, c₃=0, c₁+c₂=1).  These ISAs exercise those faces.
    """
    return [
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
            [CXGate().power(1 / 2), iSwapGate().power(1 / 3)],
            [0.5, 1 / 3],
            ["sq2cx", "sq3iswap"],
            **isa_kwargs,
        ),
    ]


def get_all_test_isas(**isa_kwargs) -> list[DiscreteISA]:
    """All ISAs: slim + random-circuit + mirror (zero-cost SWAP).

    Covers every distinct code path:
      slim        → single weak gate, mixed family
      random_circ → full-strength iSwap, full-strength CX, mixed CX/iSwap
      extras      → fSim (nonzero c₃), mirror (zero-cost SWAP)
    """
    return (
        get_slim_isas(**isa_kwargs)
        + get_random_circuit_isas(**isa_kwargs)
        + [
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
    )
