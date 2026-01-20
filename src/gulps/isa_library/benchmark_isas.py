"""ISA configurations for benchmarking purposes.

This module provides a comprehensive list of ISA configurations for
benchmarking GULPS against other compilation methods. ISAs are defined
with minimal metadata to avoid slow import times.

Usage:
    from benchmark_isas import get_benchmark_isas

    isa_configs = get_benchmark_isas()
    # Returns: List[(name, gate_set)]
    # where gate_set = [(gate, cost, label), ...]
"""

import numpy as np
from qiskit.circuit.library import CXGate, SwapGate, XXPlusYYGate, iSwapGate

from gulps.isa_library.fsim import fsim


def get_benchmark_isas():
    """Return a comprehensive list of ISA configurations for benchmarking.

    Returns:
        List of (name, gate_set) tuples where:
        - name: str, descriptive ISA name
        - gate_set: List[(gate, cost, label), ...]
    """
    configs = []

    # ====================================================================
    # Single-fraction ISAs (simple fractional powers)
    # ====================================================================

    # iSwap fractional powers
    configs.append(
        (
            "iswap^1/2",
            [(iSwapGate().power(1 / 2), 0.5, "iswap^1/2")],
        )
    )

    configs.append(
        (
            "iswap^1/3",
            [(iSwapGate().power(1 / 3), 1 / 3, "iswap^1/3")],
        )
    )

    configs.append(
        (
            "iswap^1/4",
            [(iSwapGate().power(1 / 4), 0.25, "iswap^1/4")],
        )
    )

    configs.append(
        (
            "iswap^1/5",
            [(iSwapGate().power(1 / 5), 0.2, "iswap^1/5")],
        )
    )

    configs.append(
        (
            "iswap^1/6",
            [(iSwapGate().power(1 / 6), 1 / 6, "iswap^1/6")],
        )
    )

    # CX fractional powers
    configs.append(
        (
            "cx^1/2",
            [(CXGate().power(1 / 2), 0.5, "cx^1/2")],
        )
    )

    configs.append(
        (
            "cx^1/3",
            [(CXGate().power(1 / 3), 1 / 3, "cx^1/3")],
        )
    )

    configs.append(
        (
            "cx^1/4",
            [(CXGate().power(1 / 4), 0.25, "cx^1/4")],
        )
    )

    configs.append(
        (
            "cx^1/5",
            [(CXGate().power(1 / 5), 0.2, "cx^1/5")],
        )
    )

    configs.append(
        (
            "cx^1/6",
            [(CXGate().power(1 / 6), 1 / 6, "cx^1/6")],
        )
    )

    # ====================================================================
    # Two-gate combinations
    # ====================================================================

    # iSwap pairs
    configs.append(
        (
            "iswap^{1/2,1/3}",
            [
                (iSwapGate().power(1 / 2), 0.5, "iswap^1/2"),
                (iSwapGate().power(1 / 3), 1 / 3, "iswap^1/3"),
            ],
        )
    )

    configs.append(
        (
            "iswap^{1/2,1/4}",
            [
                (iSwapGate().power(1 / 2), 0.5, "iswap^1/2"),
                (iSwapGate().power(1 / 4), 0.25, "iswap^1/4"),
            ],
        )
    )

    configs.append(
        (
            "iswap^{1/2,1/5}",
            [
                (iSwapGate().power(1 / 2), 0.5, "iswap^1/2"),
                (iSwapGate().power(1 / 5), 0.2, "iswap^1/5"),
            ],
        )
    )

    configs.append(
        (
            "iswap^{1/3,1/4}",
            [
                (iSwapGate().power(1 / 3), 1 / 3, "iswap^1/3"),
                (iSwapGate().power(1 / 4), 0.25, "iswap^1/4"),
            ],
        )
    )

    configs.append(
        (
            "iswap^{1/3,1/5}",
            [
                (iSwapGate().power(1 / 3), 1 / 3, "iswap^1/3"),
                (iSwapGate().power(1 / 5), 0.2, "iswap^1/5"),
            ],
        )
    )

    configs.append(
        (
            "iswap^{1/4,1/5}",
            [
                (iSwapGate().power(1 / 4), 0.25, "iswap^1/4"),
                (iSwapGate().power(1 / 5), 0.2, "iswap^1/5"),
            ],
        )
    )

    # CX pairs
    configs.append(
        (
            "cx^{1/2,1/3}",
            [
                (CXGate().power(1 / 2), 0.5, "cx^1/2"),
                (CXGate().power(1 / 3), 1 / 3, "cx^1/3"),
            ],
        )
    )

    configs.append(
        (
            "cx^{1/3,1/4}",
            [
                (CXGate().power(1 / 3), 1 / 3, "cx^1/3"),
                (CXGate().power(1 / 4), 0.25, "cx^1/4"),
            ],
        )
    )

    configs.append(
        (
            "cx^{1/3,1/5}",
            [
                (CXGate().power(1 / 3), 1 / 3, "cx^1/3"),
                (CXGate().power(1 / 5), 0.2, "cx^1/5"),
            ],
        )
    )

    configs.append(
        (
            "cx^{1/4,1/5}",
            [
                (CXGate().power(1 / 4), 0.25, "cx^1/4"),
                (CXGate().power(1 / 5), 0.2, "cx^1/5"),
            ],
        )
    )

    configs.append(
        (
            "cx^{1/4,1/6}",
            [
                (CXGate().power(1 / 4), 0.25, "cx^1/4"),
                (CXGate().power(1 / 6), 1 / 6, "cx^1/6"),
            ],
        )
    )

    configs.append(
        (
            "cx^{1/5,1/6}",
            [
                (CXGate().power(1 / 5), 0.2, "cx^1/5"),
                (CXGate().power(1 / 6), 1 / 6, "cx^1/6"),
            ],
        )
    )

    # ====================================================================
    # Three-gate combinations
    # ====================================================================

    configs.append(
        (
            "iswap^{1/2,1/3,1/4}",
            [
                (iSwapGate().power(1 / 2), 0.5, "iswap^1/2"),
                (iSwapGate().power(1 / 3), 1 / 3, "iswap^1/3"),
                (iSwapGate().power(1 / 4), 0.25, "iswap^1/4"),
            ],
        )
    )

    configs.append(
        (
            "iswap^{1/2,1/4,1/5}",
            [
                (iSwapGate().power(1 / 2), 0.5, "iswap^1/2"),
                (iSwapGate().power(1 / 4), 0.25, "iswap^1/4"),
                (iSwapGate().power(1 / 5), 0.2, "iswap^1/5"),
            ],
        )
    )

    configs.append(
        (
            "iswap^{1/3,1/4,1/5}",
            [
                (iSwapGate().power(1 / 3), 1 / 3, "iswap^1/3"),
                (iSwapGate().power(1 / 4), 0.25, "iswap^1/4"),
                (iSwapGate().power(1 / 5), 0.2, "iswap^1/5"),
            ],
        )
    )

    configs.append(
        (
            "iswap^{1/2,1/4,1/6}",
            [
                (iSwapGate().power(1 / 2), 0.5, "iswap^1/2"),
                (iSwapGate().power(1 / 4), 0.25, "iswap^1/4"),
                (iSwapGate().power(1 / 6), 1 / 6, "iswap^1/6"),
            ],
        )
    )

    configs.append(
        (
            "cx^{1/2,1/3,1/4}",
            [
                (CXGate().power(1 / 2), 0.5, "cx^1/2"),
                (CXGate().power(1 / 3), 1 / 3, "cx^1/3"),
                (CXGate().power(1 / 4), 0.25, "cx^1/4"),
            ],
        )
    )

    configs.append(
        (
            "cx^{1/3,1/4,1/5}",
            [
                (CXGate().power(1 / 3), 1 / 3, "cx^1/3"),
                (CXGate().power(1 / 4), 0.25, "cx^1/4"),
                (CXGate().power(1 / 5), 0.2, "cx^1/5"),
            ],
        )
    )

    # ====================================================================
    # Four-gate combinations (include full gate)
    # ====================================================================

    configs.append(
        (
            "iswap^{1,1/2,1/3,1/4}",
            [
                (iSwapGate(), 1.0, "iswap"),
                (iSwapGate().power(1 / 2), 0.5, "iswap^1/2"),
                (iSwapGate().power(1 / 3), 1 / 3, "iswap^1/3"),
                (iSwapGate().power(1 / 4), 0.25, "iswap^1/4"),
            ],
        )
    )

    configs.append(
        (
            "cx^{1,1/2,1/3,1/4}",
            [
                (CXGate(), 1.0, "cx"),
                (CXGate().power(1 / 2), 0.5, "cx^1/2"),
                (CXGate().power(1 / 3), 1 / 3, "cx^1/3"),
                (CXGate().power(1 / 4), 0.25, "cx^1/4"),
            ],
        )
    )

    # ====================================================================
    # Mixed gate types (CX + iSwap)
    # ====================================================================

    configs.append(
        (
            "cx+iswap^1/2",
            [
                (CXGate(), 1.0, "cx"),
                (iSwapGate().power(1 / 2), 0.5, "iswap^1/2"),
            ],
        )
    )

    configs.append(
        (
            "cx+iswap^1/3",
            [
                (CXGate(), 1.0, "cx"),
                (iSwapGate().power(1 / 3), 1 / 3, "iswap^1/3"),
            ],
        )
    )

    configs.append(
        (
            "cx^1/2+iswap^1/2",
            [
                (CXGate().power(1 / 2), 0.5, "cx^1/2"),
                (iSwapGate().power(1 / 2), 0.5, "iswap^1/2"),
            ],
        )
    )

    configs.append(
        (
            "cx^1/3+iswap^1/3",
            [
                (CXGate().power(1 / 3), 1 / 3, "cx^1/3"),
                (iSwapGate().power(1 / 3), 1 / 3, "iswap^1/3"),
            ],
        )
    )

    configs.append(
        (
            "mixed_4",
            [
                (CXGate(), 1.0, "cx"),
                (CXGate().power(1 / 2), 0.5, "cx^1/2"),
                (iSwapGate().power(1 / 2), 0.5, "iswap^1/2"),
                (iSwapGate().power(1 / 3), 1 / 3, "iswap^1/3"),
            ],
        )
    )

    # ====================================================================
    # Mixed with SWAP (zero cost)
    # ====================================================================

    configs.append(
        (
            "mixed_swap",
            [
                (CXGate(), 1.0, "cx"),
                (CXGate().power(1 / 2), 0.5, "cx^1/2"),
                (iSwapGate().power(1 / 2), 0.5, "iswap^1/2"),
                (iSwapGate().power(1 / 3), 1 / 3, "iswap^1/3"),
                (SwapGate(), 0.0, "swap"),
            ],
        )
    )

    configs.append(
        (
            "iswap^1/3+swap",
            [
                (iSwapGate().power(1 / 3), 1 / 3, "iswap^1/3"),
                (SwapGate(), 0.0, "swap"),
            ],
        )
    )

    configs.append(
        (
            "cx^1/3+swap",
            [
                (CXGate().power(1 / 3), 1 / 3, "cx^1/3"),
                (SwapGate(), 0.0, "swap"),
            ],
        )
    )

    # ====================================================================
    # fSim gates (various theta, phi)
    # ====================================================================

    configs.append(
        (
            "fsim(π/2,π/6)^1/3",
            [
                (fsim(np.pi / 2, np.pi / 6).power(1 / 3), 1 / 3, "fsim^1/3"),
            ],
        )
    )

    configs.append(
        (
            "fsim(π/2,π/6)^{1/3,1/5}",
            [
                (fsim(np.pi / 2, np.pi / 6).power(1 / 3), 1 / 3, "fsim^1/3"),
                (fsim(np.pi / 2, np.pi / 6).power(1 / 5), 0.2, "fsim^1/5"),
            ],
        )
    )

    configs.append(
        (
            "fsim(π/2,π/6)^{1,1/2,1/3}",
            [
                (fsim(np.pi / 2, np.pi / 6), 1.0, "fsim"),
                (fsim(np.pi / 2, np.pi / 6).power(1 / 2), 0.5, "fsim^1/2"),
                (fsim(np.pi / 2, np.pi / 6).power(1 / 3), 1 / 3, "fsim^1/3"),
            ],
        )
    )

    return configs
