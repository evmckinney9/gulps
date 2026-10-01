"""Coverage reports, sampled costs, and calibration, pinned to the monodromy oracle."""

import os
from itertools import combinations_with_replacement

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, SwapGate, UnitaryGate, iSwapGate

from gulps.analysis.calibration import strength_sweep
from gulps.analysis.coverage import coverage_report, empirical_cost
from gulps.analysis.region import ReachableRegion
from gulps.decomposition import GulpsDecomposer
from gulps.invariants import LocalEquivalenceClass

REPORT_ISA = GulpsDecomposer([CXGate(), iSwapGate().power(0.5)], [1.0, 0.5])
ORACLE_ISAS = {
    "sqiswap": ([iSwapGate().power(1 / 2)], [0.5]),
    "cx": ([CXGate()], [1.0]),
    "cx_sqiswap": ([CXGate(), iSwapGate().power(1 / 2)], [1.0, 0.5]),
}


def rounded_vertices(array):
    return sorted({tuple(round(float(x), 6) for x in vertex) for vertex in array})


@pytest.fixture(scope="module")
def oracle():
    """The monodromy oracle; required in CI, optional elsewhere."""
    if os.environ.get("CI"):
        import monodromy  # noqa: F401  a missing oracle in CI is a failure, not a skip
    pytest.importorskip("monodromy")
    from .fixtures import monodromy_oracle

    return monodromy_oracle


def test_region_vertices_match_the_monodromy_oracle(oracle):
    # The mixed basis includes every pure-gate sentence as well.
    gates, _ = ORACLE_ISAS["cx_sqiswap"]
    for depth in range(1, 4):
        for sentence in combinations_with_replacement(gates, depth):
            region = ReachableRegion.of(LocalEquivalenceClass.from_unitaries(sentence))
            vertices = np.vstack(
                [
                    part.vertices
                    for part in (region, region.rho)
                    if part.vertices is not None
                ]
            )
            assert rounded_vertices(vertices) == oracle.region_vertices(sentence)


@pytest.mark.parametrize("label", list(ORACLE_ISAS))
def test_haar_expected_cost_matches_the_monodromy_oracle(oracle, label):
    isa = GulpsDecomposer(*ORACLE_ISAS[label])
    report = coverage_report(isa)
    assert report.total_coverage == pytest.approx(1.0, abs=1e-7)
    assert report.expected_cost == pytest.approx(oracle.expected_cost(isa), abs=1e-7)


def test_cx_cost_distribution_requires_three_gates_almost_surely():
    # Targets reachable with fewer CX gates have Haar measure zero.
    report = coverage_report(GulpsDecomposer([CXGate()], [1.0]))
    np.testing.assert_allclose(report.cost_cdf, [(3.0, 1.0)], atol=1e-9)
    assert report.percentile(0.5) == 3.0


def test_calibration_preserves_distinct_targets_with_equal_class_keys():
    from gulps.invariants import LocalEquivalenceClass

    targets = [
        LocalEquivalenceClass((0.5, 0.0, 0.0)),
        LocalEquivalenceClass((0.5, 2e-13, 0.0)),
    ]
    assert targets[0] == targets[1]  # Public equality is coarser than reachability.
    isa = GulpsDecomposer([CXGate()], [1.0])
    expected = sum(cost for cost, _ in isa.select(targets))
    assert expected == 3.0
    assert strength_sweep(CXGate(), [1.0], workload=targets).costs == (expected,)


@pytest.mark.parametrize("width", [3])
def test_workload_counts_each_nested_branch_and_loop_body_once(width):
    circuit = QuantumCircuit(width, 1)
    with circuit.for_loop(range(4)):
        with circuit.if_test((circuit.clbits[0], True)) as otherwise:
            circuit.cx(0, 1)
            if width == 3:
                circuit.cx(1, 2)
        with otherwise:
            circuit.iswap(0, 1)
    targets = [CXGate()] * (width - 1) + [iSwapGate()]
    assert empirical_cost(REPORT_ISA, circuit) == empirical_cost(REPORT_ISA, targets)
    assert (
        strength_sweep(BASE, [0.5, 1.0], workload=circuit).costs
        == strength_sweep(BASE, [0.5, 1.0], workload=targets).costs
    )


BASE = iSwapGate()


@pytest.mark.parametrize("search", [strength_sweep])
def test_pulse_overhead_and_local_layers_have_different_counts(search):
    # Identity needs no pulse; CX needs one; SWAP needs three CX pulses.
    workload = [np.eye(4), CXGate(), SwapGate()]
    result = search(
        CXGate(),
        strengths=[1.0],
        workload=workload,
        pulse_overhead=0.2,
        local_layer_cost=0.07,
    )
    cost = result.best[1] if search is strength_sweep else result.cost
    assert cost == pytest.approx(4 * 1.2 + 7 * 0.07)


@pytest.mark.parametrize("workload", [None, [CXGate()]])
def test_unreachable_targets_cannot_win_a_strength_search(workload):
    sweep = strength_sweep(SwapGate(), strengths=[0.5, 1.0], workload=workload)
    assert np.isfinite(sweep.costs[0])
    assert sweep.costs[1] == np.inf
    assert sweep.best[0] == 0.5


def test_covered_corners_do_not_hide_a_cheaper_face():
    isa = GulpsDecomposer(
        [UnitaryGate(np.eye(4)), iSwapGate(), CXGate()], [0.0, 1.0, 0.6]
    )
    target = LocalEquivalenceClass((0.25, 0.1, 0.0))
    selected_cost, _ = isa.select(target)
    report = coverage_report(isa)
    covering = next(
        entry
        for entry in report.entries
        if entry.region.contains(target) or entry.region.rho.contains(target)
    )
    assert selected_cost == pytest.approx(1.2)
    assert covering.cost == pytest.approx(selected_cost)
    assert covering.names == ("cx", "cx")
    assert covering.fresh_mass == 0.0
