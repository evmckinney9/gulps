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
    """Configuration for GULPS decomposition pipeline.

    This dataclass consolidates all tunable parameters for the GULPS decomposition
    pipeline in one place, including tolerances, solver parameters, and algorithmic
    settings. Adjust these values to trade off between accuracy and performance.

    Tolerance Attributes:
        lp_feasibility_tol: Linear program primal/dual feasibility tolerance.
            Used in scipy linprog solver. Default: 1e-10
        makhlin_conv_tol: Makhlin invariant convergence tolerance.
            Maximum residual for the GN restart loop's early-exit. Default: 1e-9
        weyl_conv_tol: Weyl coordinate convergence tolerance.
            Maximum residual for Weyl-based early-exit and inline
            polish. Default: 1e-5


    Diagnostics Attributes:
        flag_duration: If a single decomposition exceeds this time (seconds),
            emit a warning.  Set to 0 to disable.  Default: 0.05 (50ms)
    """

    # Tolerances
    lp_feasibility_tol: float = 1e-10
    makhlin_conv_tol: float = 1e-9
    weyl_conv_tol: float = 1e-5

    # Diagnostics
    flag_duration: float = 0.05

    # Solver parameters
    min_batch_size: int = 6
