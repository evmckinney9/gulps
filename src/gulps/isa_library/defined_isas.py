# isa_library.py

import logging
from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    CXGate,
    SwapGate,
    UnitaryGate,
    XXPlusYYGate,
    iSwapGate,
)
from qutip import sigmax, sigmay, tensor

from gulps.core.isa import ISAInvariants, expected_costs

logger = logging.getLogger(__name__)


def fsim(theta, phi):
    qc = QuantumCircuit(2, name="fsim")
    qc.append(XXPlusYYGate(2 * theta), [0, 1])
    qc.cp(phi, 0, 1)
    return qc.to_gate()


def snail_conversion_gain(c, g):
    H = 0.5 * (
        (c + g) * tensor(sigmax(), sigmax()) + (c - g) * tensor(sigmay(), sigmay())
    )
    return UnitaryGate((-1j * H).expm().full())


def build_isa(name, gate_set):
    logger.info(f"Building ISA: {name}")
    start_time = perf_counter()

    gates, costs, labels = zip(*gate_set)
    isa = ISAInvariants(
        gate_set=gates,
        costs=costs,
        names=labels,
        precompute_polytopes=True,
        single_qubit_cost=1e-6,
    )
    build_time = perf_counter() - start_time
    logger.info(f"  → Coverage set built in {build_time:.2f} seconds")

    haar_cost, sentence_len, polytope_index = expected_costs(isa.coverage_set)
    logger.info(
        f"  → Expected (Cost: {haar_cost:.2f}, Depth: {sentence_len:.2f}, Polytope Index: {polytope_index})"
    )

    return {
        "name": name,
        "isa": isa,
        "expected_cost": haar_cost,
        "expected_sentence_len": sentence_len,
        "expected_polytope_index": polytope_index,
    }


ISA_LIBRARY = [
    # Single-fraction iSWAPs
    # build_isa(
    #     "iswap_5",
    #     [
    #         (iSwapGate().power(1 / 5), 1 / 5, "sqrt5iswap"),
    #     ],
    # ),
    build_isa(
        "iswap_4",
        [
            (iSwapGate().power(1 / 4), 1 / 4, "sqrt4iswap"),
        ],
    ),
    # build_isa(
    #     "iswap_3",
    #     [
    #         (iSwapGate().power(1 / 3), 1 / 3, "sqrt3iswap"),
    #     ],
    # ),
    # # # 2-gate sets (diverse but practical)
    # build_isa(
    #     "iswap_3_5",
    #     [
    #         (iSwapGate().power(1 / 3), 1 / 3, "sqrt3iswap"),
    #         (iSwapGate().power(1 / 5), 1 / 5, "sqrt5iswap"),
    #     ],
    # ),
    # build_isa(
    #     "iswap_4_5",
    #     [
    #         (iSwapGate().power(1 / 4), 1 / 4, "sqrt4iswap"),
    #         (iSwapGate().power(1 / 5), 1 / 5, "sqrt5iswap"),
    #     ],
    # ),
    # build_isa(
    #     "iswap_3_4",
    #     [
    #         (iSwapGate().power(1 / 3), 1 / 3, "sqrt3iswap"),
    #         (iSwapGate().power(1 / 4), 1 / 4, "sqrt4iswap"),
    #     ],
    # ),
    # build_isa(
    #     "iswap_2_5",
    #     [
    #         (iSwapGate().power(1 / 2), 1 / 2, "sqrt2iswap"),
    #         (iSwapGate().power(1 / 5), 1 / 5, "sqrt5iswap"),
    #     ],
    # ),
    # # 3-gate combinations
    # build_isa(
    #     "iswap_2_4_5",
    #     [
    #         (iSwapGate().power(1 / 2), 1 / 2, "sqrt2iswap"),
    #         (iSwapGate().power(1 / 4), 1 / 4, "sqrt4iswap"),
    #         (iSwapGate().power(1 / 5), 1 / 5, "sqrt5iswap"),
    #     ],
    # ),
    # build_isa(
    #     "iswap_3_4_5",
    #     [
    #         (iSwapGate().power(1 / 3), 1 / 3, "sqrt3iswap"),
    #         (iSwapGate().power(1 / 4), 1 / 4, "sqrt4iswap"),
    #         (iSwapGate().power(1 / 5), 1 / 5, "sqrt5iswap"),
    #     ],
    # ),
    build_isa(
        "cx_3",
        [
            (CXGate().power(1 / 3), 1 / 3, "sqrt3cx"),
        ],
    ),
    # build_isa(
    #     "cx_4",
    #     [
    #         (CXGate().power(1 / 4), 1 / 4, "sqrt4cx"),
    #     ],
    # ),
    # build_isa(
    #     "cx_5",
    #     [
    #         (CXGate().power(1 / 5), 1 / 5, "sqrt5cx"),
    #     ],
    # ),
    # build_isa(
    #     "cx_5_6",
    #     [
    #         (CXGate().power(1 / 5), 1 / 5, "sqrt5cx"),
    #         (CXGate().power(1 / 6), 1 / 6, "sqrt6cx"),
    #     ],
    # ),
    # build_isa(
    #     "iswap_2_4_6",
    #     [
    #         (iSwapGate().power(1 / 2), 1 / 2, "sqrt2iswap"),
    #         (iSwapGate().power(1 / 4), 1 / 4, "sqrt4iswap"),
    #         (iSwapGate().power(1 / 6), 1 / 6, "sqrt6iswap"),
    #     ],
    # ),
    # build_isa(
    #     "cx_3_4",
    #     [
    #         (CXGate().power(1 / 3), 1 / 3, "sqrt3cx"),
    #         (CXGate().power(1 / 4), 1 / 4, "sqrt4cx"),
    #     ],
    # ),
    # build_isa(
    #     "cx_3_5",
    #     [
    #         (CXGate().power(1 / 3), 1 / 3, "sqrt3cx"),
    #         (CXGate().power(1 / 5), 1 / 5, "sqrt5cx"),
    #     ],
    # ),
    # build_isa(
    #     "cx_4_6",
    #     [
    #         (CXGate().power(1 / 4), 1 / 4, "sqrt4cx"),
    #         (CXGate().power(1 / 6), 1 / 6, "sqrt6cx"),
    #     ],
    # ),
    # build_isa(
    #     "iswap_3_5",
    #     [
    #         (iSwapGate().power(1 / 3), 1 / 3, "sqrt3iswap"),
    #         (iSwapGate().power(1 / 5), 1 / 5, "sqrt5iswap"),
    #     ],
    # ),
    # build_isa(
    #     "fsim_3_5",
    #     [
    #         (fsim(np.pi / 2, np.pi / 6).power(1 / 3), 1 / 3, "sqrt3fsim"),
    #         (fsim(np.pi / 2, np.pi / 6).power(1 / 5), 1 / 5, "sqrt5fsim"),
    #     ],
    # ),
    # build_isa(
    #     "cx_1_2_3_4",
    #     [
    #         (CXGate(), 1.0, "cx"),
    #         (CXGate().power(1 / 2), 1 / 2, "sqrt2cx"),
    #         (CXGate().power(1 / 3), 1 / 3, "sqrt3cx"),
    #         (CXGate().power(1 / 4), 1 / 4, "sqrt4cx"),
    #     ],
    # ),
    # build_isa(
    #     "fsim_1_2_3",
    #     [
    #         (fsim(np.pi / 2, np.pi / 6).power(1.0), 1.0, "fsim"),
    #         (fsim(np.pi / 2, np.pi / 6).power(1 / 2), 1 / 2, "sqrt2fsim"),
    #         (fsim(np.pi / 2, np.pi / 6).power(1 / 3), 1 / 3, "sqrt3fsim"),
    #     ],
    # ),
    build_isa(
        "cx_iswap_2_3",
        [
            (CXGate(), 1.0, "cx"),
            (CXGate().power(1 / 2), 1 / 2, "sqrt2cx"),
            (iSwapGate().power(1 / 2), 1 / 2, "sqrt2iswap"),
            (iSwapGate().power(1 / 3), 1 / 3, "sqrt3iswap"),
        ],
    ),
    build_isa(
        "cx_iswap_swap",
        [
            (CXGate(), 1.0, "cx"),
            (CXGate().power(1 / 2), 1 / 2, "sqrt2cx"),
            (iSwapGate().power(1 / 2), 1 / 2, "sqrt2iswap"),
            (iSwapGate().power(1 / 3), 1 / 3, "sqrt3iswap"),
            (SwapGate(), 0.0, "swap"),
        ],
    ),
    # build_isa(
    #     "snail_mckinney23",
    #     [
    #         (snail_conversion_gain(np.pi / 4, np.pi / 4), 1.8, "cx"),
    #         (snail_conversion_gain(np.pi / 8, np.pi / 8), 0.9, "sqrt2cx"),
    #         (snail_conversion_gain(np.pi / 2, 0), 1.0, "iswap"),
    #         (snail_conversion_gain(np.pi / 4, 0), 0.5, "sqrtiswap"),
    #         (snail_conversion_gain(3 * np.pi / 8, np.pi / 8), 1.4, "b"),
    #         (snail_conversion_gain(3 * np.pi / 16, np.pi / 16), 0.7, "sqrtb"),
    #     ],
    # ),
]
