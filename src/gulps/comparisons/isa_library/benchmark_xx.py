# Copyright 2025-2026 Lev S. Bishop, Evan McKinney
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark utilities for Gulps vs Qiskit transpilation comparison.

Provides:
- ``build_target(num_qubits)`` - discrete {π/2, π/4, π/6} RZZ target
- ``build_gulps_pm(target)`` / ``build_qiskit_pm(target)`` - matched PassManagers
- ``circuit_duration(qc)`` - additive cost model (seconds)
- ``build_benchmark_circuits(num_qubits)`` - standard circuit families
"""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RZZGate, UGate, efficient_su2, quantum_volume
from qiskit.circuit.random import random_circuit
from qiskit.synthesis.qft import synth_qft_full
from qiskit.transpiler import InstructionProperties, PassManager, Target
from qiskit.transpiler.passes import (
    Collect2qBlocks,
    ConsolidateBlocks,
    Optimize1qGatesDecomposition,
    UnitarySynthesis,
    Unroll3qOrMore,
)
from qiskit.transpiler.passmanager_config import PassManagerConfig

from gulps.qiskit_ext.translation_plugin import GulpsTranslationPlugin

# ── Hardware parameters ──────────────────────────────────────────────
DUR_BASE = 500e-9  # RZZ(pi/2) gate duration
DUR_1Q = 5.23e-8  # single-qubit U gate duration
ERR_BASE = 0.001  # RZZ(pi/2) error rate

# Discrete gate set: (custom_name, angle, duration_scale)
DISCRETE_GATES = [
    ("zz", np.pi / 2, 1.0),
    ("sq2_zz", np.pi / 4, 0.5),
    ("sq3_zz", np.pi / 6, 1 / 3),
]

# Duration lookups (gulps emits named gates, Qiskit emits generic "rzz")
_GATE_DURATIONS = {name: DUR_BASE * s for name, _, s in DISCRETE_GATES}
_GATE_DURATIONS["u"] = DUR_1Q
_RZZ_ANGLE_TO_DUR = {round(a, 10): DUR_BASE * s for _, a, s in DISCRETE_GATES}


# ── Target ───────────────────────────────────────────────────────────
def build_target(num_qubits: int) -> Target:
    """Discrete {pi/2, pi/4, pi/6} RZZ target, all-to-all connectivity."""
    t = Target()
    pairs = list(combinations(range(num_qubits), 2))

    for name, angle, dur_scale in DISCRETE_GATES:
        err = 1 - (1 - ERR_BASE) ** dur_scale
        dur = DUR_BASE * dur_scale
        props = {}
        for i, j in pairs:
            props[(i, j)] = InstructionProperties(duration=dur, error=err)
            props[(j, i)] = InstructionProperties(duration=dur, error=err)
        t.add_instruction(RZZGate(angle), props, name=name)

    theta, phi, lam = Parameter("theta"), Parameter("phi"), Parameter("lambda")
    u_props = {
        (i,): InstructionProperties(
            duration=DUR_1Q, error=1 - (1 - ERR_BASE) ** (DUR_1Q / DUR_BASE)
        )
        for i in range(num_qubits)
    }
    t.add_instruction(UGate(theta, phi, lam), u_props)
    return t


# ── Pass managers ────────────────────────────────────────────────────
def build_gulps_pm(target: Target) -> PassManager:
    """Gulps translation plugin pipeline with Unroll3qOrMore."""
    gulps_pm = GulpsTranslationPlugin().pass_manager(PassManagerConfig(target=target))
    pm = PassManager()
    pm.append(Unroll3qOrMore(target=target))
    pm._tasks.extend(gulps_pm._tasks)
    return pm


def build_qiskit_pm(target: Target) -> PassManager:
    """Qiskit UnitarySynthesis pipeline (XXDecomposer internally)."""
    return PassManager(
        [
            Unroll3qOrMore(target=target),
            Collect2qBlocks(),
            ConsolidateBlocks(target=target),
            UnitarySynthesis(target=target),
            Optimize1qGatesDecomposition(target=target),
        ]
    )


# ── Metrics ──────────────────────────────────────────────────────────
def circuit_duration(qc: QuantumCircuit) -> float:
    """Total circuit duration under additive cost model (seconds)."""
    dur = 0.0
    for inst in qc.data:
        name = inst.operation.name
        if inst.operation.num_qubits == 1:
            dur += DUR_1Q
        elif name in _GATE_DURATIONS:
            dur += _GATE_DURATIONS[name]
        elif name == "rzz":
            key = round(abs(float(inst.operation.params[0])), 10)
            dur += _RZZ_ANGLE_TO_DUR.get(key, DUR_BASE)
    return dur


def count_2q(qc: QuantumCircuit) -> int:
    """Count 2-qubit gates."""
    return sum(1 for inst in qc.data if inst.operation.num_qubits == 2)


# ── Benchmark circuits ───────────────────────────────────────────────
def build_benchmark_circuits(
    num_qubits: int, seed: int = 42
) -> dict[str, QuantumCircuit]:
    """Standard circuit families (Benchpress-inspired)."""
    return {
        "QFT": synth_qft_full(num_qubits, approximation_degree=max(0, num_qubits - 4)),
        "EfficientSU2": efficient_su2(num_qubits, reps=3, entanglement="circular"),
        "QV": quantum_volume(num_qubits, depth=num_qubits, seed=seed),
        "Random": random_circuit(
            num_qubits, depth=num_qubits * 4, max_operands=2, seed=seed
        ),
    }
