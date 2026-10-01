"""GulpsDecompositionPass and the translation plugin inside Qiskit's transpiler."""

import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import (
    CXGate,
    CZGate,
    RXXGate,
    UnitaryGate,
    iSwapGate,
)
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Operator, SuperOp, random_unitary
from qiskit.synthesis import synth_qft_full
from qiskit.transpiler import (
    InstructionProperties,
    PassManager,
    PassManagerConfig,
    Target,
    TranspilerError,
)

from gulps.decomposition import GulpsDecomposer
from gulps.transpiler import GulpsDecompositionPass, GulpsTranslationPlugin

from ._common import assert_implements, count_2q, two_qubit_names


@pytest.fixture(scope="module")
def decomposer():
    return GulpsDecomposer([iSwapGate().power(1 / 2), CXGate()], [0.5, 1.0])


@pytest.mark.parametrize("num_processes", [2])
def test_compilation_uses_each_pairs_gate_costs(num_processes):
    target = Target(num_qubits=3)
    target.add_instruction(
        CXGate(),
        {
            (0, 1): InstructionProperties(duration=1.0),
            (1, 2): InstructionProperties(duration=3.0),
        },
    )
    target.add_instruction(
        iSwapGate(),
        {
            (0, 1): InstructionProperties(duration=3.0),
            (1, 2): InstructionProperties(duration=1.0),
        },
    )
    circuit = QuantumCircuit(3)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    outputs = PassManager([GulpsDecompositionPass(target)]).run(
        [circuit, circuit], num_processes=num_processes
    )
    for out in outputs:
        assert_implements(circuit, out)
        assert two_qubit_names(out) == ("cx", "iswap", "iswap")


@pytest.mark.parametrize(
    "noise, circuit",
    [(True, synth_qft_full(4))],
    ids=["qft"],
)
def test_plugin_compiles_onto_a_generic_backend(noise, circuit):
    """A fake backend with per-pair durations, or with none, transpiles exactly."""
    from qiskit import transpile
    from qiskit.providers.fake_provider import GenericBackendV2

    backend = GenericBackendV2(
        num_qubits=4, basis_gates=["cz", "rz", "sx", "x"], noise_info=noise, seed=1
    )
    out = transpile(
        circuit, backend=backend, translation_method="gulps", seed_transpiler=1
    )
    assert set(out.count_ops()) <= {"cz", "rz", "sx", "x"}
    assert np.allclose(
        Operator.from_circuit(out).data, Operator(circuit).data, atol=1e-12
    )


def test_variable_bearing_circuit_is_rejected_before_substitution(decomposer):
    circuit = QuantumCircuit(2)
    circuit.unitary(random_unitary(4, seed=93), [0, 1])
    circuit.add_var("flag", False)
    with pytest.raises(NotImplementedError):
        PassManager([GulpsDecompositionPass(decomposer)]).run(circuit)


@pytest.mark.parametrize(
    "field, cx_properties, iswap_properties, pricing, expected_cost",
    [
        (
            "duration",
            {"duration": 3.0, "error": 0.01},
            {"duration": 1.0, "error": 0.03},
            "duration",
            2.0,
        ),
        (
            "error",
            {"duration": 3.0, "error": 0.01},
            {"duration": 1.0, "error": 0.03},
            "error",
            0.01,
        ),
        ("duration", {"error": 0.03}, {"error": 0.01}, None, 1.0),
    ],
)
def test_target_costs_choose_the_cheapest_circuit(
    field, cx_properties, iswap_properties, pricing, expected_cost
):
    target = Target(num_qubits=2)
    properties = {"cx": cx_properties, "iswap": iswap_properties}
    for gate in (CXGate(), iSwapGate()):
        target.add_instruction(
            gate, {(0, 1): InstructionProperties(**properties[gate.name])}
        )
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    output = PassManager([GulpsDecompositionPass(target, cost=field)]).run(circuit)
    assert_implements(circuit, output)
    actual_cost = sum(
        properties[name][pricing] if pricing else 1.0
        for name in two_qubit_names(output)
    )
    assert actual_cost == pytest.approx(expected_cost, rel=1e-9, abs=1e-15)


def test_invalid_target_cost_field_is_rejected():
    with pytest.raises(ValueError):
        GulpsDecompositionPass(Target(num_qubits=2), cost="fidelity")


@pytest.mark.parametrize("value", [np.nan])
def test_invalid_target_costs_are_rejected(value):
    target = Target(num_qubits=2)
    target.add_instruction(CXGate(), {(0, 1): InstructionProperties(duration=value)})
    with pytest.raises(ValueError):
        GulpsDecompositionPass(target)


def test_missing_costs_on_another_pair_do_not_change_the_selected_sentence():
    target = Target(num_qubits=3)
    target.add_instruction(CXGate(), {(0, 1): InstructionProperties(duration=3.0)})
    target.add_instruction(iSwapGate(), {(0, 1): InstructionProperties(duration=1.0)})
    circuit = QuantumCircuit(3)
    circuit.cx(0, 1)
    before = PassManager([GulpsDecompositionPass(target)]).run(circuit)
    target.add_instruction(RXXGate(np.pi / 4), {(1, 2): None})
    after = PassManager([GulpsDecompositionPass(target)]).run(circuit)
    for output in (before, after):
        assert_implements(circuit, output)
        assert two_qubit_names(output) == ("iswap", "iswap")


def test_target_alias_is_rejected_before_emitting_an_unsupported_name():
    target = Target(num_qubits=2)
    target.add_instruction(RXXGate(np.pi / 4), {(0, 1): None}, name="rxx_half")
    with pytest.raises(NotImplementedError):
        GulpsDecompositionPass(target)


def test_target_preserves_each_pairs_native_gate():
    target = Target(num_qubits=4)
    gates = [RXXGate(np.pi / 4), iSwapGate().power(1 / 3)]
    pairs = [(0, 1), (2, 3)]
    circuit = QuantumCircuit(4)
    for gate, pair in zip(gates, pairs, strict=True):
        target.add_instruction(gate, {pair: InstructionProperties(duration=1.0)})
        circuit.unitary(gate.to_matrix(), pair)
    out = PassManager([GulpsDecompositionPass(target)]).run(circuit)
    assert_implements(circuit, out)
    assert count_2q(out) == 2
    for instruction in out.data:
        if instruction.operation.num_qubits == 2:
            pair = tuple(out.find_bit(q).index for q in instruction.qubits)
            assert instruction.operation == gates[pairs.index(pair)]
            assert target.instruction_supported(
                operation_name=instruction.operation.name,
                qargs=pair,
                parameters=instruction.operation.params,
            )


def test_control_flow_is_rejected_before_substitution():
    target = Target(num_qubits=2)
    target.add_instruction(CXGate(), {(0, 1): None})
    body = QuantumCircuit(2)
    body.unitary(iSwapGate().to_matrix(), [0, 1])
    circuit = QuantumCircuit(2, 1)
    circuit.unitary(CXGate().to_matrix(), [0, 1])
    circuit.if_else((circuit.clbits[0], True), body, None, [0, 1], [])
    dag = circuit_to_dag(circuit)
    with pytest.raises(NotImplementedError):
        GulpsDecompositionPass(target).run(dag)
    assert dag_to_circuit(dag) == circuit


@pytest.mark.parametrize("multiple_groups", [False])
def test_target_pass_rejects_an_unsupported_physical_pair(multiple_groups):
    target = Target(num_qubits=3)
    target.add_instruction(CXGate(), {(0, 1): None})
    if multiple_groups:
        target.add_instruction(iSwapGate(), {(0, 2): None})
    circuit = QuantumCircuit(3)
    circuit.unitary(CXGate().to_matrix(), [1, 2])
    with pytest.raises(TranspilerError):
        PassManager([GulpsDecompositionPass(target)]).run(circuit)


def test_bidirectional_target_retains_each_directions_calibration():
    target = Target(num_qubits=4)
    forward, reverse = [(0, 1), (2, 3)], [(1, 0), (3, 2)]
    target.add_instruction(
        CXGate(),
        {
            **dict.fromkeys(forward, InstructionProperties(duration=0.1)),
            **dict.fromkeys(reverse, InstructionProperties(duration=10.0)),
        },
    )
    target.add_instruction(
        iSwapGate(),
        {
            **dict.fromkeys(forward, InstructionProperties(duration=10.0)),
            **dict.fromkeys(reverse, InstructionProperties(duration=0.1)),
        },
    )
    circuit = QuantumCircuit(4)
    circuit.unitary(iSwapGate().to_matrix(), [0, 1])
    circuit.unitary(iSwapGate().to_matrix(), [3, 2])
    # Exercise both ordered pairs directly; these blocks are already consolidated.
    out = dag_to_circuit(GulpsDecompositionPass(target).run(circuit_to_dag(circuit)))
    assert_implements(circuit, out)
    for inst in out.data:
        if inst.operation.num_qubits == 2:
            pair = tuple(out.find_bit(q).index for q in inst.qubits)
            assert (pair, inst.operation.name) in [((0, 1), "cx"), ((3, 2), "iswap")]


@pytest.mark.parametrize("pair_gate", [iSwapGate(), CZGate()])
def test_global_instructions_join_the_native_gates_on_every_pair(pair_gate):
    target = Target(num_qubits=3)
    target.add_instruction(CXGate())
    target.add_instruction(pair_gate, {(1, 2): None})
    circuit = QuantumCircuit(3)
    circuit.unitary(iSwapGate().to_matrix(), [1, 2])
    circuit.unitary(CXGate().to_matrix(), [2, 0])
    out = PassManager([GulpsDecompositionPass(target)]).run(circuit)
    assert_implements(circuit, out)
    for inst in out.data:
        if inst.operation.num_qubits == 2:
            pair = tuple(out.find_bit(q).index for q in inst.qubits)
            assert target.instruction_supported(
                operation_name=inst.operation.name, qargs=pair
            )


@pytest.mark.parametrize(
    "boundary, reverse, basis",
    [
        ("barrier", True, ["rz", "sx", "x"]),
        ("reset", False, ["rz", "sx", "x"]),
    ],
)
def test_translation_plugin_preserves_boundaries(boundary, reverse, basis):
    from qiskit.quantum_info import SuperOp

    pairs = [(0, 1), (1, 2)]
    if reverse:
        pairs = [tuple(reversed(pair)) for pair in pairs]
    target = Target.from_configuration(basis_gates=[*basis, "reset"], num_qubits=3)
    target.add_instruction(CXGate(), dict.fromkeys(pairs))
    matrix = random_unitary(4, seed=192).data
    circuit = QuantumCircuit(3, global_phase=0.37)
    circuit.unitary(matrix, [0, 1])
    if boundary == "barrier":
        circuit.barrier(1)
    elif boundary == "h":
        circuit.h(1)
    elif boundary == "reset":
        circuit.reset(1)
    circuit.unitary(matrix, [1, 2])
    manager = GulpsTranslationPlugin().pass_manager(PassManagerConfig(target=target))
    cleaned = manager.run(circuit)
    if boundary == "reset":
        np.testing.assert_allclose(
            SuperOp(cleaned).data, SuperOp(circuit).data, atol=1e-12
        )
    else:
        assert_implements(circuit, cleaned)
    assert cleaned.count_ops().get("barrier", 0) == circuit.count_ops().get(
        "barrier", 0
    )
    assert cleaned.count_ops().get("reset", 0) == circuit.count_ops().get("reset", 0)
    for instruction in cleaned.data:
        if instruction.operation.name != "barrier":
            pair = tuple(cleaned.find_bit(q).index for q in instruction.qubits)
            assert target.instruction_supported(instruction.operation.name, pair)


@pytest.mark.parametrize("basis, with_entangler", [(["u"], False)])
def test_translation_plugin_handles_single_qubit_wires(basis, with_entangler):
    target = Target.from_configuration(
        basis_gates=[*basis, *(["cx"] if with_entangler else [])], num_qubits=3
    )
    circuit = QuantumCircuit(3, global_phase=0.31)
    circuit.h(0)
    circuit.rx(0.27, 0)
    if with_entangler:
        circuit.cx(0, 1)
    circuit.h(2)
    circuit.rx(-0.48, 2)
    circuit.rz(0.19, 2)
    manager = GulpsTranslationPlugin().pass_manager(PassManagerConfig(target=target))
    output = manager.run(circuit)
    assert_implements(circuit, output)
    assert count_2q(output) == int(with_entangler)
    for instruction in output.data:
        qubits = tuple(output.find_bit(q).index for q in instruction.qubits)
        assert target.instruction_supported(instruction.operation.name, qubits)


def test_locally_equivalent_gates_on_different_pairs_each_emit_their_own_gate():
    # CX and CZ have the same class, so both pairs share one canonical decomposer.
    target = Target(num_qubits=3)
    target.add_instruction(CXGate(), {(0, 1): InstructionProperties(duration=1.0)})
    target.add_instruction(CZGate(), {(1, 2): InstructionProperties(duration=1.0)})
    circuit = QuantumCircuit(3)
    for i in range(4):
        circuit.unitary(random_unitary(4, seed=i).data, [0, 1] if i % 2 else [1, 2])
    out = PassManager([GulpsDecompositionPass(target)]).run(circuit)
    assert_implements(circuit, out)
    for inst in out.data:
        if inst.operation.num_qubits == 2:
            pair = tuple(out.find_bit(q).index for q in inst.qubits)
            assert target.instruction_supported(inst.operation.name, pair)


def test_symbolic_phase_and_parameters_survive_substitution():
    theta = Parameter("theta")
    circuit = QuantumCircuit(3, global_phase=theta + 0.37)
    circuit.rz(theta, 2)
    circuit.unitary(random_unitary(4, seed=921).data, [1, 0])
    compiler = GulpsDecompositionPass(GulpsDecomposer([CXGate()], [1.0]))
    dag = circuit_to_dag(circuit)
    output = dag_to_circuit(compiler.run(dag))
    assert output.parameters == circuit.parameters
    for value in (0.13, -1.2):
        assert_implements(
            circuit.assign_parameters({theta: value}),
            output.assign_parameters({theta: value}),
        )


def test_substitution_preserves_registers_metadata_and_measurements():
    a, b, c = (
        QuantumRegister(1, "a"),
        QuantumRegister(2, "b"),
        ClassicalRegister(2, "c"),
    )
    circuit = QuantumCircuit(a, b, c, name="metadata", global_phase=0.17)
    circuit.metadata = {"experiment": [1, 2]}
    circuit.unitary(random_unitary(4, seed=922).data, [b[1], a[0]])
    circuit.barrier()
    circuit.measure(a[0], c[1])
    circuit.reset(b[0])
    circuit.measure(b[1], c[0])
    dag = circuit_to_dag(circuit)
    compiler = GulpsDecompositionPass(GulpsDecomposer([CXGate()], [1.0]))
    output_dag = compiler.run(dag)
    assert output_dag is dag
    output = dag_to_circuit(output_dag)
    assert output.name == circuit.name
    assert output.metadata == circuit.metadata
    assert output.qregs == circuit.qregs and output.cregs == circuit.cregs
    assert output.qubits == circuit.qubits and output.clbits == circuit.clbits
    measurements = lambda qc: {
        (i.qubits, i.clbits) for i in qc.data if i.operation.name == "measure"
    }
    assert measurements(output) == measurements(circuit)
    # Compare the channel before terminal measurement; reset remains part of it.
    np.testing.assert_allclose(
        SuperOp(output.remove_final_measurements(inplace=False)).data,
        SuperOp(circuit.remove_final_measurements(inplace=False)).data,
        atol=1e-12,
    )


def test_later_group_failure_leaves_input_untouched():
    target = Target(num_qubits=3)
    target.add_instruction(CXGate(), {(0, 1): None})
    target.add_instruction(UnitaryGate(np.eye(4)), {(1, 2): None})
    compiler = GulpsDecompositionPass(target)
    circuit = QuantumCircuit(3, global_phase=0.29)
    circuit.unitary(CXGate().to_matrix(), [0, 1])
    circuit.unitary(CXGate().to_matrix(), [1, 2])
    dag = circuit_to_dag(circuit)
    with pytest.raises(TranspilerError):
        compiler.run(dag)
    assert dag_to_circuit(dag) == circuit


def test_unchecked_unitary_is_rejected_before_substitution():
    circuit = QuantumCircuit(2)
    circuit.unitary(CXGate().to_matrix(), [0, 1])
    circuit.append(UnitaryGate(2 * np.eye(4), check_input=False), [0, 1])
    dag = circuit_to_dag(circuit)
    compiler = GulpsDecompositionPass(GulpsDecomposer([CXGate()], [1.0]))
    with pytest.raises(TranspilerError):
        compiler.run(dag)
    assert dag_to_circuit(dag) == circuit
