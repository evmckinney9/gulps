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
        makhlin_conv_tol: Stage 1 (Makhlin) convergence tolerance.
            Maximum residual in Makhlin invariant space for the restart
            loop's early-exit condition. Default: 1e-9
        weyl_conv_tol: Stage 2 (Weyl) convergence tolerance.
            Maximum residual in Weyl coordinate space for the restart
            loop's early-exit condition. Default: 1e-5
        makhlin_solver_tol: Inner-loop convergence tolerance for Makhlin
            Gauss-Newton iterations. Controls when a single restart's
            iteration loop terminates. Default: 1e-10
        weyl_solver_tol: Inner-loop convergence tolerance for Weyl
            Levenberg-Marquardt iterations. Controls when a single restart's
            iteration loop terminates. Default: 1e-10
        equiv_recovery_tol: Local equivalence matching tolerance.
            Used when comparing Weyl coordinates in recovery. Default: 1e-5

    Solver Attributes:
        makhlin_restarts: Number of random restarts for Stage 1 (Makhlin) optimization.
            Default: 64
        makhlin_maxiter: Maximum iterations per Stage 1 restart.
            Controls when to give up on a single optimization attempt. Default: 256
        makhlin_restart_patience: Stop the restart loop after this many consecutive
            non-improving restarts.  Prevents wasting time on hard segments where
            the Gauss-Newton residual plateaus above makhlin_conv_tol.
            Set to 0 to disable.  Default: 12
        weyl_restarts: Number of restarts for Stage 2 (Weyl) polishing.
            Warm-started from Makhlin; typically converges on restart 0.
            Default: 16
        weyl_maxiter: Maximum iterations per Stage 2 restart.
            Stage 2 usually converges faster than Stage 1. Default: 64
        weyl_perturb_scale: Perturbation magnitude for Stage 2 restarts.
            Controls how far to perturb from Stage 1 solution. Default: 1e-4

    Cache Attributes:
        segment_cache_size: Size of the LFU cache for each step index. Default: 3
    """

    # Tolerances
    lp_feasibility_tol: float = 1e-10
    makhlin_conv_tol: float = 1e-9
    weyl_conv_tol: float = 1e-5
    makhlin_solver_tol: float = 1e-10
    weyl_solver_tol: float = 1e-10
    equiv_recovery_tol: float = 2e-5

    # Solver parameters
    makhlin_restarts: int = 64
    makhlin_maxiter: int = 256
    makhlin_restart_patience: int = 12
    weyl_restarts: int = 16
    weyl_maxiter: int = 64
    weyl_perturb_scale: float = 1e-4
    strict_convergence_checks: bool = False

    # Search parameters
    max_depth: int = 8  # Maximum gate sentence length for enumeration/LP

    # Cache parameters
    segment_cache_size: int = 3
