"""Fixed compiler workloads; make test checks results, make bench measures time."""

import os
import subprocess
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    CXGate,
    RXXGate,
    RYYGate,
    RZZGate,
    iSwapGate,
    quantum_volume,
)
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Operator
from qiskit.synthesis import synth_qft_full
from qiskit.transpiler import InstructionProperties, PassManagerConfig, Target

from gulps import _accelerate
from gulps.invariants import LocalEquivalenceClass
from gulps.transpiler import GulpsDecompositionPass, GulpsTranslationPlugin

from ._common import assert_implements, count_2q
from .fixtures.isas import ISA_BUILDERS
from .fixtures.targets import haar_unitary, local_pair

ISA_NAMES = tuple(ISA_BUILDERS)


@pytest.fixture(scope="module")
def measurement_context(request):
    if request.config.getoption("benchmark_disable") and not request.config.getoption(
        "benchmark_enable"
    ):
        return {}
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), *sorted((root / "tests/fixtures").glob("*.py"))]
    return {
        "extension_sha256": sha256(Path(_accelerate.__file__).read_bytes()).hexdigest(),
        "workload_source_sha256": sha256(
            b"".join(p.read_bytes() for p in sources)
        ).hexdigest(),
        "dependencies": {
            p: version(p) for p in ("numpy", "qiskit", "pytest-benchmark")
        },
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "solver_commit": subprocess.check_output(
            ["git", "-C", str(root / "crates/can_sandwich"), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "solver_source_sha256": sha256(
            b"".join(
                p.read_bytes()
                for p in sorted((root / "crates/can_sandwich/src").rglob("*.rs"))
            )
        ).hexdigest(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "PYTHONHASHSEED",
                "RAYON_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "QISKIT_IN_PARALLEL",
                "QISKIT_FORCE_THREADS",
            )
        },
    }


@pytest.fixture(autouse=True)
def record_context(benchmark, measurement_context):
    benchmark.extra_info.update(measurement_context)


@pytest.fixture(params=ISA_NAMES)
def isa_name(request):
    return request.param


@pytest.fixture(scope="module", params=("haar", "boundary", "local"))
def targets(request):
    if request.param == "haar":
        return [haar_unitary(10000 + i) for i in range(32)]
    if request.param == "local":
        return [local_pair(10000 + i) for i in range(32)]
    # Faces, edges, repeated spectra, and nearby points. Generate independently of GULPS.
    coordinates = [
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.5, 0.5, 0.5),
        (0.2, 0.1, 0.0),
        (0.2, 0.1, 1e-7),
        (0.3, 0.3, 0.1),
        (0.3, 0.1, 0.1),
    ]
    matrices = []
    for i in range(32):
        a, b, c = coordinates[i % len(coordinates)]
        canonical = (
            RXXGate(np.pi * a).to_matrix()
            @ RYYGate(np.pi * b).to_matrix()
            @ RZZGate(np.pi * c).to_matrix()
        )
        matrices.append(local_pair(20000 + i) @ canonical @ local_pair(30000 + i))
    return matrices


def test_setup(benchmark, isa_name):
    compiler = benchmark(ISA_BUILDERS[isa_name])
    assert compiler.gates


def test_projection(benchmark, targets):
    record_targets(benchmark, targets)
    classes = benchmark(LocalEquivalenceClass.from_unitaries, targets)
    assert classes == [LocalEquivalenceClass.from_unitary(target) for target in targets]


def record_targets(benchmark, targets):
    benchmark.extra_info["input_sha256"] = sha256(
        np.asarray(targets).tobytes()
    ).hexdigest()
    benchmark.extra_info["target_count"] = len(targets)


@pytest.mark.parametrize("cache", ("cold", "warm"))
@pytest.mark.parametrize("input_kind", ("matrices", "classes"))
def test_selection(benchmark, isa_name, cache, input_kind):
    targets = [haar_unitary(10000 + i) for i in range(128)]
    record_targets(benchmark, targets)
    compiler = ISA_BUILDERS[isa_name]()
    expected = compiler.select(targets)
    # Project outside the timer to separate selection from matrix-to-class work.
    inputs = (
        LocalEquivalenceClass.from_unitaries(targets)
        if input_kind == "classes"
        else targets
    )
    if cache == "cold":
        # Includes compiler construction and initial search on every iteration.
        selected = benchmark(lambda: ISA_BUILDERS[isa_name]().select(inputs))
    else:
        selected = benchmark(compiler.select, inputs)
    assert selected == expected
    benchmark.extra_info["costs"] = [cost for cost, _ in selected]
    benchmark.extra_info["depths"] = [len(gates) for _, gates in selected]
    benchmark.extra_info["input_kind"] = input_kind


@pytest.mark.parametrize("cache", ("cold", "warm"))
def test_decomposition(benchmark, isa_name, targets, cache):
    compiler = ISA_BUILDERS[isa_name]()
    record_targets(benchmark, targets)
    selected = compiler.select(targets)
    if cache == "cold":
        # A fresh compiler with its search warmed but no sentences realized, so
        # every round times the solver and emission.
        def setup():
            fresh = ISA_BUILDERS[isa_name]()
            fresh.select(targets)
            return (fresh,), {}

        circuits = benchmark.pedantic(
            lambda fresh: [fresh(target) for target in targets],
            setup=setup,
            rounds=100,
            warmup_rounds=5,
        )
    else:
        circuits = benchmark(lambda: [compiler(target) for target in targets])
    errors = []
    for target, circuit, (_, gates) in zip(targets, circuits, selected, strict=True):
        errors.append(assert_implements(target, circuit))
        assert count_2q(circuit) == len(gates)
    benchmark.extra_info["max_error"] = max(errors)
    benchmark.extra_info["entanglers"] = [count_2q(circuit) for circuit in circuits]
    benchmark.extra_info["costs"] = [cost for cost, _ in selected]


@pytest.mark.parametrize("dressed", (False, True), ids=("exact", "dressed"))
def test_native_decomposition(benchmark, isa_name, dressed):
    compiler = ISA_BUILDERS[isa_name]()
    target = Operator(compiler.gates[0]).data
    if dressed:
        target = np.exp(0.37j) * local_pair(20000) @ target @ local_pair(30000)
    record_targets(benchmark, [target])
    result = benchmark(compiler, target)
    benchmark.extra_info["max_error"] = assert_implements(target, result)
    assert count_2q(result) == 1


@pytest.fixture(
    scope="module",
    params=[
        (n, sharing)
        for n in (8, 128)
        for sharing in ("unique", "one_class", "repeated")
    ],
    ids=lambda case: f"{case[0]}-{case[1]}",
)
def dag_workload(request):
    blocks, sharing = request.param
    base = haar_unitary(10000)
    targets = [
        local_pair(20000 + i) @ base @ local_pair(30000 + i)
        if sharing == "one_class"
        else (base if sharing == "repeated" else haar_unitary(10000 + i))
        for i in range(blocks)
    ]
    circuit = QuantumCircuit(3, global_phase=0.31)
    for i, target in enumerate(targets):
        circuit.unitary(target, [(0, 1), (2, 1), (1, 0)][i % 3])
    return circuit, targets


def test_dag(benchmark, isa_name, dag_workload):
    compiler = ISA_BUILDERS[isa_name]()
    circuit, targets = dag_workload
    record_targets(benchmark, targets)
    dag = circuit_to_dag(circuit)
    pass_ = GulpsDecompositionPass(compiler)
    pass_.run(deepcopy(dag))
    # The pass mutates its DAG. Copy outside the timer for every round.
    result = benchmark.pedantic(
        pass_.run,
        setup=lambda: ((deepcopy(dag),), {}),
        rounds=100,
        warmup_rounds=5,
    )
    compiled = dag_to_circuit(result)
    assert_implements(Operator(circuit), compiled)
    benchmark.extra_info["entanglers"] = count_2q(compiled)


@pytest.mark.parametrize("distinct_costs", (False, True), ids=("shared", "distinct"))
def test_target_setup(benchmark, distinct_costs):
    target = Target(num_qubits=129)
    for gate, cost in [(CXGate(), 1.0), (RXXGate(np.pi / 4), 0.5)]:
        target.add_instruction(
            gate,
            {
                (i, i + 1): InstructionProperties(
                    duration=cost + (i / 1000 if distinct_costs else 0.0)
                )
                for i in range(128)
            },
        )
    compiler = benchmark(GulpsDecompositionPass, target)
    # The pass compiles the unitaries ConsolidateBlocks makes.
    circuit = QuantumCircuit(2)
    circuit.unitary(CXGate().to_matrix(), [0, 1])
    assert_implements(circuit, dag_to_circuit(compiler.run(circuit_to_dag(circuit))))


def test_routed_dag(benchmark):
    target = Target(num_qubits=3)
    target.add_instruction(CXGate(), {(0, 1): None})
    target.add_instruction(iSwapGate(), {(2, 1): None})
    circuit = QuantumCircuit(3)
    targets = [haar_unitary(10000 + i) for i in range(128)]
    record_targets(benchmark, targets)
    for i, matrix in enumerate(targets):
        circuit.unitary(matrix, [(0, 1), (2, 1), (1, 0), (1, 2)][i % 4])
    dag = circuit_to_dag(circuit)
    pass_ = GulpsDecompositionPass(target)
    pass_.run(deepcopy(dag))
    result = benchmark.pedantic(
        pass_.run,
        setup=lambda: ((deepcopy(dag),), {}),
        rounds=100,
        warmup_rounds=5,
    )
    compiled = dag_to_circuit(result)
    assert_implements(circuit, compiled)
    for inst in compiled.data:
        if inst.operation.num_qubits == 2:
            pair = tuple(compiled.find_bit(q).index for q in inst.qubits)
            assert target.instruction_supported(inst.operation.name, pair)


@pytest.mark.parametrize("workload", ("qft", "quantum_volume"))
def test_translation(benchmark, workload):
    circuit = synth_qft_full(4) if workload == "qft" else quantum_volume(4, seed=42)
    target = Target.from_configuration(
        basis_gates=["cx", "rz", "sx", "x"], num_qubits=4
    )
    manager = GulpsTranslationPlugin().pass_manager(PassManagerConfig(target=target))
    result = benchmark(manager.run, circuit)
    assert_implements(circuit, result)
    assert set(result.count_ops()) <= {"cx", "rz", "sx", "x"}
