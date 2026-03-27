"""Test that precomputed polytope lookup agrees with enumeration-based search.

The two decomposition paths — ``precompute_polytopes=True`` (O(1) polytope
lookup) and ``precompute_polytopes=False`` (priority-queue enumeration, break
on first feasible sentence) — must select the same optimal gate sentence for
every target.  A mismatch indicates that the enumeration order or the polytope
coverage set is inconsistent.
"""

import pytest
from qiskit.quantum_info import Operator, average_gate_fidelity, random_unitary

from gulps import GateInvariants
from gulps.core.isa import DiscreteISA
from gulps.gulps_decomposer import GulpsDecomposer
from gulps.viz.weyl_chamber import weyl_linspace
from tests.fixtures.isas import get_slim_isas

FIDELITY_TOL = 1 - 1e-8
N_RANDOM = 10
N_WEYL = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _assert_fidelity(target, circuit, label=""):
    fid = average_gate_fidelity(Operator(target), Operator(circuit))
    assert fid > FIDELITY_TOL, (
        f"Fidelity too low{' (' + label + ')' if label else ''}: {fid}"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(params=get_slim_isas())
def isa_pair(request):
    """Return (ISA_no_precompute, ISA_with_precompute) for the same gate set."""
    isa_no_pre = request.param
    isa_pre = DiscreteISA(
        gate_set=[g.gate for g in isa_no_pre.gate_set],
        costs=[isa_no_pre.cost_dict[g] for g in isa_no_pre.gate_set],
        names=[g.name for g in isa_no_pre.gate_set],
        precompute_polytopes=True,
        single_qubit_cost=isa_no_pre.single_qubit_cost,
        max_sequence_length=isa_no_pre.max_sequence_length,
    )
    return isa_no_pre, isa_pre


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_sentence_agreement_random(isa_pair):
    """Both paths must produce the same depth for random unitaries."""
    isa_no_pre, isa_pre = isa_pair
    dec_enum = GulpsDecomposer(isa=isa_no_pre)
    dec_poly = GulpsDecomposer(isa=isa_pre)

    for seed in range(N_RANDOM):
        u = random_unitary(4, seed=seed)
        circ_enum = dec_enum(u)
        circ_poly = dec_poly(u)

        _assert_fidelity(u, circ_enum, f"enum seed={seed}")
        _assert_fidelity(u, circ_poly, f"poly seed={seed}")

        depth_enum = sum(1 for op in circ_enum.data if op.operation.num_qubits == 2)
        depth_poly = sum(1 for op in circ_poly.data if op.operation.num_qubits == 2)
        assert depth_enum == depth_poly, (
            f"Depth mismatch at seed={seed}: enum={depth_enum}, poly={depth_poly}"
        )


def test_sentence_agreement_weyl_grid(isa_pair):
    """Both paths must agree on the Weyl chamber grid."""
    isa_no_pre, isa_pre = isa_pair
    dec_enum = GulpsDecomposer(isa=isa_no_pre)
    dec_poly = GulpsDecomposer(isa=isa_pre)

    for weyl_pt in weyl_linspace(N_WEYL):
        target = GateInvariants.from_weyl(weyl_pt)

        circ_enum = dec_enum(target.matrix)
        circ_poly = dec_poly(target.matrix)

        _assert_fidelity(target.matrix, circ_enum, f"enum weyl={weyl_pt}")
        _assert_fidelity(target.matrix, circ_poly, f"poly weyl={weyl_pt}")

        depth_enum = sum(1 for op in circ_enum.data if op.operation.num_qubits == 2)
        depth_poly = sum(1 for op in circ_poly.data if op.operation.num_qubits == 2)
        assert depth_enum == depth_poly, (
            f"Depth mismatch at weyl=({weyl_pt[0]:.3f},{weyl_pt[1]:.3f},{weyl_pt[2]:.3f}): "
            f"enum={depth_enum}, poly={depth_poly}"
        )
