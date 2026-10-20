"""Compile workloads on real circuits; make test checks results, make bench measures time.

Each circuit is consolidated once, and routed first for the Target, outside the
timers, so the timed regions are GulpsDecompositionPass on the DAG its
translation stage receives.
"""

import os
import subprocess
from copy import deepcopy
from functools import cache
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest
from qiskit import qasm2
from qiskit.circuit import Parameter
from qiskit.circuit.library import (
    CPhaseGate,
    CZGate,
    ECRGate,
    RXXGate,
    RZGate,
    RZZGate,
    SXGate,
    XGate,
    XXPlusYYGate,
    iSwapGate,
    quantum_volume,
)
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import (
    CouplingMap,
    InstructionProperties,
    PassManager,
    StagedPassManager,
    Target,
)
from qiskit.transpiler.passes import ConsolidateBlocks, Unroll3qOrMore
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from gulps import _accelerate
from gulps.transpiler import GulpsDecompositionPass

from .fixtures.isas import ISA_BUILDERS

ROOT = Path(__file__).resolve().parents[1]
QASM = ROOT / "tests/fixtures/qasm"

# QASMBench and Feynman circuits (via Qiskit benchpress), and a seeded quantum volume.
CIRCUITS = (
    "qft_n29",
    "multiplier_n45",
    "gf2_16_mult",
    "dnn_n51",
    "qugan_n71",
    "qv_n32",
)

# Mixed gate families and strengths cycled over the heavy-hex edges, with durations.
EDGE_GATES = (
    ((CZGate(), 1.0), (CPhaseGate(np.pi / 2), 0.5)),
    ((iSwapGate(), 1.0), (XXPlusYYGate(np.pi / 2), 0.5)),
    ((ECRGate(), 1.0), (RZZGate(np.pi / 6), 1 / 3)),
    ((RXXGate(np.pi / 3), 0.6),),
)


def heavy_hex():
    """A 115-qubit heavy-hex Target, one direction per edge, with four gate sets.

    Durations differ on every edge, as a backend reports them.
    """
    edges = CouplingMap.from_heavy_hex(7, bidirectional=False).get_edges()
    target = Target(num_qubits=1 + max(max(edge) for edge in edges))
    by_gate = {}
    for i, edge in enumerate(edges):
        scale = 1 + (i * 7 % 11) / 100
        for gate, duration in EDGE_GATES[i % len(EDGE_GATES)]:
            by_gate.setdefault(gate.name, (gate, {}))[1][edge] = InstructionProperties(
                duration=duration * scale
            )
    for gate, properties in by_gate.values():
        target.add_instruction(gate, properties)
    qubits = {(q,): None for q in range(target.num_qubits)}
    for gate in (RZGate(Parameter("theta")), SXGate(), XGate()):
        target.add_instruction(gate, qubits)
    return target


HEAVY_HEX = heavy_hex()

# The heavy-hex Target on every circuit; each fixture ISA, as an all-to-all
# decomposer, on a dense and a structured circuit.
GRID = [("heavy_hex", name) for name in CIRCUITS] + [
    (isa, name) for isa in ISA_BUILDERS for name in ("qv_n32", "multiplier_n45")
]


def make_pass(source):
    return GulpsDecompositionPass(
        HEAVY_HEX if source == "heavy_hex" else ISA_BUILDERS[source]()
    )


@cache
def load_circuit(name):
    if name == "qv_n32":
        return quantum_volume(32, seed=42)
    circuit = qasm2.load(
        QASM / f"{name}.qasm", custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS
    )
    circuit.remove_final_measurements()
    # Gates of three or more qubits become two-qubit gates the router can place.
    return PassManager([Unroll3qOrMore(basis_gates=["u", "cx"])]).run(circuit)


@cache
def workload(source, name):
    """The consolidated DAG the translation stage receives.

    Circuits for the heavy-hex Target go through its layout and routing first.
    """
    circuit = load_circuit(name)
    if source == "heavy_hex":
        preset = generate_preset_pass_manager(1, target=HEAVY_HEX, seed_transpiler=0)
        circuit = StagedPassManager(
            ["init", "layout", "routing"],
            init=preset.init,
            layout=preset.layout,
            routing=preset.routing,
        ).run(circuit)
    return circuit_to_dag(
        PassManager([ConsolidateBlocks(force_consolidate=True)]).run(circuit)
    )


@pytest.fixture(scope="module")
def measurement_context(request):
    if request.config.getoption("benchmark_disable") and not request.config.getoption(
        "benchmark_enable"
    ):
        return {}
    sources = [Path(__file__), *sorted((ROOT / "tests/fixtures").rglob("*.*"))]
    return {
        "extension_sha256": sha256(Path(_accelerate.__file__).read_bytes()).hexdigest(),
        "workload_source_sha256": sha256(
            b"".join(p.read_bytes() for p in sources)
        ).hexdigest(),
        "dependencies": {
            p: version(p) for p in ("numpy", "qiskit", "pytest-benchmark")
        },
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
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


def record_workload(benchmark, dag):
    blocks = [node.op.to_matrix() for node in dag.two_qubit_ops()]
    benchmark.extra_info["blocks"] = len(blocks)
    benchmark.extra_info["distinct_blocks"] = len({b.tobytes() for b in blocks})
    benchmark.extra_info["input_sha256"] = sha256(
        b"".join(b.tobytes() for b in blocks)
    ).hexdigest()


def record_output(benchmark, source, circuit):
    """Record the output cost; on the Target, check each gate is native to its pair."""
    entanglers, duration = 0, 0.0
    for inst in circuit.data:
        if inst.operation.num_qubits != 2:
            continue
        entanglers += 1
        if source == "heavy_hex":
            pair = tuple(circuit.find_bit(q).index for q in inst.qubits)
            assert HEAVY_HEX.instruction_supported(inst.operation.name, pair)
            duration += HEAVY_HEX[inst.operation.name][pair].duration
    benchmark.extra_info["entanglers"] = entanglers
    if source == "heavy_hex":
        benchmark.extra_info["duration"] = duration


@pytest.mark.parametrize("source, name", GRID)
def test_compile(benchmark, source, name):
    """A new pass on every round: what each transpile() call pays."""
    dag = workload(source, name)
    record_workload(benchmark, dag)
    result = benchmark.pedantic(
        lambda dag: make_pass(source).run(dag),
        setup=lambda: ((deepcopy(dag),), {}),
        rounds=20,
        warmup_rounds=2,
    )
    record_output(benchmark, source, dag_to_circuit(result))


@pytest.mark.parametrize("source, name", GRID)
def test_recompile(benchmark, source, name):
    """One pass for every round: a pass manager reused across circuits."""
    dag = workload(source, name)
    record_workload(benchmark, dag)
    pass_ = make_pass(source)
    pass_.run(deepcopy(dag))
    result = benchmark.pedantic(
        pass_.run, setup=lambda: ((deepcopy(dag),), {}), rounds=20, warmup_rounds=2
    )
    record_output(benchmark, source, dag_to_circuit(result))


@pytest.mark.parametrize("name", ("qft_n29", "qugan_n71"))
def test_transpile(benchmark, name):
    """The whole optimization level 2 pipeline, mostly Qiskit's time."""
    circuit = load_circuit(name)
    result = benchmark.pedantic(
        lambda: generate_preset_pass_manager(
            2, target=HEAVY_HEX, translation_method="gulps", seed_transpiler=0
        ).run(circuit),
        rounds=5,
        warmup_rounds=1,
    )
    record_output(benchmark, "heavy_hex", result)
