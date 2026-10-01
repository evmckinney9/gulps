"""GulpsDecomposer: construction, validation, synthesis, and selection."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from qiskit.circuit import Gate
from qiskit.circuit.library import CXGate, UnitaryGate, iSwapGate
from qiskit.quantum_info import Operator, random_unitary
from qiskit.synthesis import TwoQubitBasisDecomposer

from gulps.decomposition import DecompositionError, GulpsDecomposer, SearchDepthError
from gulps.invariants import LocalEquivalenceClass

from ._common import (
    assert_implements,
    count_2q,
)
from .fixtures.targets import haar_unitary


@pytest.mark.parametrize("scale", [1.0 + 4.9e-9], ids=["roundoff"])
def test_local_target_emits_no_entangler(scale):
    target = scale * np.kron(
        random_unitary(2, seed=42).data, random_unitary(2, seed=43).data
    )
    circuit = GulpsDecomposer([CXGate()], [1.0])(target)
    assert count_2q(circuit) == 0
    assert_implements(target, circuit)


def test_one_isa_serves_concurrent_callers():
    isa = GulpsDecomposer([CXGate().power(1 / 2)], [0.5])
    targets = [random_unitary(4, seed=seed).data for seed in range(4)] * 4
    with ThreadPoolExecutor(max_workers=8) as pool:
        circuits = list(pool.map(isa, targets))
    for target, circuit in zip(targets, circuits, strict=True):
        assert_implements(target, circuit)


def test_decomposition_accepts_array_layouts_and_defined_gates():
    from qiskit import QuantumCircuit

    matrix = random_unitary(4, seed=617).data
    definition = QuantumCircuit(2)
    definition.unitary(matrix, [0, 1])
    compiler = GulpsDecomposer([CXGate()], [1.0])
    for target in (
        matrix.tolist(),
        np.asfortranarray(matrix),
        matrix[::-1].copy()[::-1],
        Operator(matrix),
        definition.to_gate(),
    ):
        assert_implements(matrix, compiler(target))


@pytest.mark.parametrize(
    "target",
    [
        2 * np.eye(4, dtype=complex),
        np.eye(3, dtype=complex),
        np.full((4, 4), np.nan, dtype=complex),
    ],
    ids=["non_unitary", "wrong_shape", "nan"],
)
def test_invalid_target_raises_value_error(target):
    with pytest.raises(ValueError):
        GulpsDecomposer([CXGate()], [1.0])(target)


def test_depth_limit_error_carries_the_target_class():
    local_only = GulpsDecomposer([UnitaryGate(np.eye(4))], [1.0])
    with pytest.raises(SearchDepthError) as caught:
        local_only(CXGate())
    assert isinstance(caught.value, DecompositionError)
    assert caught.value.target == LocalEquivalenceClass.from_unitary(CXGate())


def _renamed(gate, name):
    gate = gate.to_mutable()
    gate.name = name
    return gate


@pytest.mark.parametrize(
    "gate",
    [
        pytest.param(Gate("custom", 2, []), id="custom gate without a definition"),
        pytest.param(_renamed(CXGate(), "native_cx"), id="renamed standard gate"),
        pytest.param(
            type("Bogus", (CXGate,), {"_standard_gate": 999})(), id="bad metadata"
        ),
    ],
)
def test_only_standard_gates_and_unitary_gates_are_emitted(gate):
    """The C API writes standard gates and unitaries; anything else is refused."""
    with pytest.raises(NotImplementedError):
        GulpsDecomposer([gate], [1.0])


@pytest.mark.parametrize("power", [np.nextafter(0.125, 0.0), np.nextafter(0.125, 1.0)])
def test_repeated_native_gate_at_reachability_boundary(power):
    target = CXGate().to_matrix()
    compiler = GulpsDecomposer([CXGate().power(power)], [0.125])
    circuit = compiler(target)
    assert count_2q(circuit) == 8
    assert_implements(target, circuit)


def test_call_matches_qiskit_decomposer_convention():
    from qiskit.converters import dag_to_circuit
    from qiskit.dagcircuit import DAGCircuit

    target = random_unitary(4, seed=5).data
    compiler = GulpsDecomposer([CXGate()], [1.0])
    dag = compiler(target, use_dag=True)
    assert isinstance(dag, DAGCircuit)
    assert_implements(target, dag_to_circuit(dag))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(gates=[CXGate()], costs=[1.0, 2.0]),
        dict(gates=[CXGate()], costs=[-1.0]),
        dict(gates=[CXGate()], costs=[np.inf]),
        dict(gates=[CXGate()], costs=[1.0], local_layer_cost=-1.0),
    ],
    ids=[
        "cost_length",
        "negative_cost",
        "infinite_cost",
        "negative_local_cost",
    ],
)
def test_invalid_construction_raises_value_error(kwargs):
    with pytest.raises(ValueError):
        GulpsDecomposer(**kwargs)


@pytest.mark.parametrize(
    "costs, label", [([1.0, 1.0], "first"), ([2.0, 1.0], "second")]
)
def test_identical_computed_classes_keep_the_cheapest_and_warn(costs, label):
    with pytest.warns(UserWarning):
        isa = GulpsDecomposer([CXGate(label="first"), CXGate(label="second")], costs)
    cost, gates = isa.select(CXGate())
    assert gates[0].label == label
    assert cost == 1.0


def test_nearby_native_classes_are_not_removed_by_approximate_equality():
    from gulps.invariants import LocalEquivalenceClass

    gates = [
        LocalEquivalenceClass((0.5, 0.0, 0.0)),
        LocalEquivalenceClass((0.5, 2e-13, 0.0)),
    ]
    assert gates[0] == gates[1]
    isa = GulpsDecomposer(gates, [1.0, 1.5])
    assert len(isa.gates) == 2
    assert [cost for cost, _ in isa.select(gates)] == [1.0, 1.5]


QISKIT_CX = TwoQubitBasisDecomposer(CXGate())


@pytest.mark.parametrize(
    "target",
    [
        iSwapGate().to_matrix(),
        haar_unitary(0),
    ],
    ids=["two_cx", "three_cx"],
)
def test_cx_count_matches_qiskit_optimal_decomposer(target):
    circuit = GulpsDecomposer([CXGate()], [1.0])(target)
    assert_implements(target, circuit)
    assert count_2q(circuit) == QISKIT_CX.num_basis_gates(target)
