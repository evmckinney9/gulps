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

"""Configuration dataclasses for GULPS decomposition pipeline."""

from dataclasses import dataclass


@dataclass
class GulpsConfig:
    """Tunables for the GULPS decomposition pipeline."""

    # Primal/dual feasibility tolerance for the Rust DualSimplex.
    lp_feasibility_tol: float = 1e-10
    # GN restart-loop early-exit residual (Makhlin invariants).
    makhlin_conv_tol: float = 1e-9
    # Weyl-coordinate early-exit + inline-polish residual.
    weyl_conv_tol: float = 1e-5

    # Per-call slow-decomposition warning threshold in seconds; 0 disables.
    flag_duration: float = 0.05

    # Minimum segment count before Rayon parallelizes restarts inside Rust.
    min_batch_size: int = 6
    # Seed GN restart 0 with u0 = u1 = I instead of random init. Basis-
    # dependent; benchmark on the target ISA before enabling.
    identity_warmstart: bool = False
