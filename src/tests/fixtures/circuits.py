# tests/fixtures/circuits.py

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Operator

from gulps.utils.invariants import GateInvariants


@pytest.fixture
def sampled_reachable_target():
    def _generate(decomposer):
        eligible_sentences = [
            poly for poly in decomposer.isa.coverage_set if len(poly.instructions) >= 2
        ]
        sampled_sentence = np.random.choice(eligible_sentences).instructions

        qc = QuantumCircuit(2)
        intermediates = []

        for i, gate in enumerate(sampled_sentence):
            p = ParameterVector(f"p_{i}", 3)
            qc.rv(p[0], p[1], p[2], 0)
            qc.rv(p[0], p[1], p[2], 1)
            qc.append(gate.unitary, [0, 1])

            # Get current intermediate unitary and convert to invariant
            qc.assign_parameters(
                {param: np.random.uniform(-np.pi, np.pi) for param in qc.parameters},
                inplace=True,
            )
            current_unitary = Operator(qc).data
            intermediates.append(GateInvariants.from_unitary(current_unitary))

        # Final layer of 1Q gates
        p = ParameterVector("p_final", 3)
        qc.rv(p[0], p[1], p[2], 0)
        qc.rv(p[0], p[1], p[2], 1)

        qc.assign_parameters(
            {param: np.random.uniform(-np.pi, np.pi) for param in qc.parameters},
            inplace=True,
        )
        target_unitary = Operator(qc).data

        return target_unitary, sampled_sentence, intermediates

    return _generate
