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

    Cache Attributes:
        segment_cache_size: Size of the LFU cache for each step index. Default: 3
    """

    # Tolerances
    lp_feasibility_tol: float = 1e-10
    makhlin_conv_tol: float = 1e-9
    weyl_conv_tol: float = 1e-5

    # Diagnostics
    flag_duration: float = 0.05

    # Search parameters
    max_depth: int = 8  # Maximum gate sentence length for enumeration/LP

    # Cache parameters
    segment_cache_size: int = 3
