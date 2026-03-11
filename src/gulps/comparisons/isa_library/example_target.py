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

"""Example Target for testing transpiler plugin."""

from itertools import combinations

from qiskit.circuit import Parameter
from qiskit.circuit.library import UGate, iSwapGate
from qiskit.transpiler import InstructionProperties, Target

mock_iswap_family_target = Target()

# all-to-all connectivity on 4 qubits - one add_instruction per gate name,
# with all qubit pairs in a single dict
pairs = list(combinations(range(4), 2))
iswap_props = {}
siswap_props = {}
s3iswap_props = {}
for i, j in pairs:
    iswap_props[(i, j)] = InstructionProperties(duration=200, error=0.02)
    iswap_props[(j, i)] = InstructionProperties(duration=200, error=0.02)
    siswap_props[(i, j)] = InstructionProperties(duration=100, error=0.02 / 2)
    siswap_props[(j, i)] = InstructionProperties(duration=100, error=0.02 / 2)
    s3iswap_props[(i, j)] = InstructionProperties(duration=100, error=0.02 / 3)
    s3iswap_props[(j, i)] = InstructionProperties(duration=100, error=0.02 / 3)

# Use iSwapGate().power() so the target gates match the convention in
# iSwapGate-based ISA definitions (which yield XXPlusYYGate with negative theta).
mock_iswap_family_target.add_instruction(
    iSwapGate().power(1.0), iswap_props, name="iswap"
)
mock_iswap_family_target.add_instruction(
    iSwapGate().power(1 / 2), siswap_props, name="sq2iswap"
)
mock_iswap_family_target.add_instruction(
    iSwapGate().power(1 / 3), s3iswap_props, name="sq3iswap"
)


theta = Parameter("theta")
phi = Parameter("phi")
lam = Parameter("lambda")
u_props = {(i,): InstructionProperties(duration=0, error=0) for i in range(6)}
mock_iswap_family_target.add_instruction(UGate(theta, phi, lam), u_props)
