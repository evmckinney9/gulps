"""Behavior-atomic check that precomputed polytope lookup and enumeration agree.

The two decomposition paths (``precompute_polytopes=True`` for O(1) polytope
lookup, ``precompute_polytopes=False`` for priority-queue enumeration breaking
on the first feasible sentence) must select the same optimal gate sentence
for every target. A mismatch indicates that the enumeration order or the
polytope coverage set is inconsistent.
"""

import pytest

pytest.importorskip(
    "monodromy",
    reason="monodromy package not installed; skipping polytope-agreement tests",
)

from qiskit.circuit.library import CXGate, iSwapGate
from qiskit.quantum_info import random_unitary

from gulps import GateInvariants
from gulps.core.isa import DiscreteISA
from gulps.gulps_decomposer import GulpsDecomposer
from gulps.viz.weyl_chamber import weyl_linspace
from tests._common import assert_fidelity, count_2q

N_RANDOM = 8
N_WEYL = 16


@pytest.fixture(
    scope="module",
    params=[
        ("sq2iswap", [iSwapGate().power(1 / 2)], [0.5]),  # single-gate path
        ("cx_iswap", [CXGate(), iSwapGate()], [1.0, 1.0]),  # two-gate path
    ],
    ids=lambda p: p[0],
)
def isa_pair(request):
    """Fast-to-precompute ISAs built both ways: single-gate and two-gate."""
    label, gate_set, costs = request.param
    args = dict(gate_set=gate_set, costs=costs, names=[label])
    if len(gate_set) > 1:
        args["names"] = [f"{label}_{i}" for i in range(len(gate_set))]
    return (
        DiscreteISA(**args, precompute_polytopes=False),
        DiscreteISA(**args, precompute_polytopes=True),
    )


def test_paths_agree_on_depth_and_fidelity(isa_pair):
    """Enum and polytope paths must match depth (and both reach the target)."""
    isa_no_pre, isa_pre = isa_pair
    dec_enum = GulpsDecomposer(isa=isa_no_pre)
    dec_poly = GulpsDecomposer(isa=isa_pre)

    targets = [random_unitary(4, seed=s).data for s in range(N_RANDOM)] + [
        GateInvariants.from_weyl(p).matrix for p in weyl_linspace(N_WEYL)
    ]
    for i, U in enumerate(targets):
        circ_enum = dec_enum(U)
        circ_poly = dec_poly(U)
        assert_fidelity(U, circ_enum, f"enum #{i}")
        assert_fidelity(U, circ_poly, f"poly #{i}")
        d_enum, d_poly = count_2q(circ_enum), count_2q(circ_poly)
        assert d_enum == d_poly, f"Depth mismatch at #{i}: enum={d_enum}, poly={d_poly}"
