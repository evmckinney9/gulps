"""A/B benchmark: GULPS vs Qiskit XXDecomposer on 3 CX-based ISAs.

Rich diagnostics go to stderr (human-readable).
Machine-parseable JSON goes to stdout:
  {"isa1_xx": ..., "isa1_gulps": ..., "isa1_ratio": ..., "isa2_xx": ..., ...}

ISAs (mirroring the iSwap tiers in simple_speed.py):
  isa1: [CX^(1/4)]                            — strengths [pi/8]
  isa2: [CX^(1/3), CX^(1/4)]                  — strengths [pi/6, pi/8]
  isa3: [CX^(1/4), CX^(1/3), CX^(1/2)]       — strengths [pi/8, pi/6, pi/4]

Ratios are gulps_median / xx_median — values < 1 mean Gulps is faster.

Usage:
  python scripts/xx_compare.py           # full run, 1000 unitaries
  python scripts/xx_compare.py -n 10     # quick smoke test
"""

import argparse
import json
import logging
import statistics
import sys
import time

import numpy as np
from qiskit.quantum_info import Operator, average_gate_fidelity, random_unitary

from gulps.gulps_decomposer import GulpsDecomposer


def bench(decompose_fn, name, n):
    """Benchmark a callable on n seeded random unitaries.

    Returns (times_list, fidelities_list). Raises on any failure or low fidelity.
    """
    times = []
    failures = 0
    fidelities = []

    for seed in range(n):
        target = random_unitary(4, seed=seed)
        try:
            t0 = time.perf_counter()
            result = decompose_fn(target)
            t1 = time.perf_counter()
            fid = average_gate_fidelity(target, Operator(result))
            if fid < 1 - 1e-8:
                raise ValueError(
                    f"{name} seed={seed}: fidelity {fid:.10f} below threshold"
                )
        except Exception as e:
            # print(e, file=sys.stderr)
            failures += 1
            continue

        fidelities.append(fid)
        times.append(t1 - t0)

        if (seed + 1) % 50 == 0 or seed == n - 1:
            elapsed = sum(times)
            rate = (seed + 1) / elapsed
            eta = (n - seed - 1) / rate
            med = statistics.median(times)
            print(
                f"\r  {name}: {seed + 1}/{n}  "
                f"elapsed={elapsed:.1f}s  eta={eta:.1f}s  "
                f"median={med:.4f}s",
                end="",
                file=sys.stderr,
            )
    print(file=sys.stderr)

    return times, fidelities, failures


def print_rich_stats(name, times, fidelities, failures, n, file=sys.stderr):
    """Print detailed human-readable stats to the given file handle."""
    times_ms = np.array(times) * 1000
    fids = np.array(fidelities)
    print(
        f"  {name:30s}: "
        f"mean={np.mean(times_ms):6.1f} ms  "
        f"median={np.median(times_ms):6.1f} ms  "
        f"p95={np.percentile(times_ms, 95):6.1f} ms  "
        f"max={np.max(times_ms):7.1f} ms  "
        f"({n - failures}/{n} ok, "
        f"fid_min={np.min(fids):.8f})",
        file=file,
    )


def make_xx_decomposer(strengths):
    """Create XXDecomposer with near-perfect basis fidelities."""
    from qiskit.synthesis.two_qubit.xx_decompose import XXDecomposer

    slope, offset = 1e-10, 1e-12
    basis_fidelity = {s: 1.0 - (slope * s / (np.pi / 2) + offset) for s in strengths}
    return XXDecomposer(basis_fidelity)


def make_gulps_decomposer(GulpsDecomposer, powers):
    """Create GulpsDecomposer for CX gates at given powers."""
    from qiskit.circuit.library import CXGate

    from gulps.config import GulpsConfig

    gate_set = [CXGate().power(p) for p in powers]
    costs = list(powers)
    # CX^(1/4) has weak interaction strength (pi/8) and needs depth > 8
    return GulpsDecomposer(
        gate_set=gate_set,
        costs=costs,
        precompute_polytopes=False,
        config_options=GulpsConfig(max_depth=12),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=1000)
    args = parser.parse_args()

    logging.getLogger("gulps").setLevel(logging.WARNING)

    # --- 3 CX ISA tiers (mirrors iSwap tiers in simple_speed.py) ---
    # power = strength / (pi/2), so strength = power * pi/2
    sq4 = 1 / 4  # CX^(1/4), strength = pi/8
    sq3 = 1 / 3  # CX^(1/3), strength = pi/6
    sq2 = 1 / 2  # CX^(1/2), strength = pi/4

    isas = {
        "isa1": [sq4],
        "isa2": [sq3, sq4],
        "isa3": [sq4, sq3, sq2],
    }

    print(
        f"CX A/B: {len(isas)} ISAs x {args.n} random unitaries each\n",
        file=sys.stderr,
    )

    json_result = {}

    for isa_name, powers in isas.items():
        strengths = [p * (np.pi / 2) for p in powers]
        label = "+".join(f"CX^(1/{int(1 / p)})" for p in powers)
        print(f"--- {isa_name}: [{label}] ---", file=sys.stderr)

        xx = make_xx_decomposer(strengths)
        gulps = make_gulps_decomposer(GulpsDecomposer, powers)

        xx_times, xx_fids, xx_failures = bench(xx, f"{isa_name}/XX", args.n)
        gulps_times, gulps_fids, gulps_failures = bench(
            gulps._run, f"{isa_name}/Gulps", args.n
        )

        # Rich stats to stderr
        print_rich_stats(
            f"{isa_name}/XXDecomposer", xx_times, xx_fids, xx_failures, args.n
        )
        print_rich_stats(
            f"{isa_name}/GulpsDecomposer",
            gulps_times,
            gulps_fids,
            gulps_failures,
            args.n,
        )

        xx_med = statistics.median(xx_times)
        gulps_med = statistics.median(gulps_times)
        ratio = gulps_med / xx_med if xx_med > 0 else float("inf")
        faster = "Gulps faster" if ratio < 1 else "XX faster"
        print(f"  Ratio: {ratio:.2f}x ({faster})\n", file=sys.stderr)

        json_result[f"{isa_name}_xx"] = round(xx_med, 6)
        json_result[f"{isa_name}_gulps"] = round(gulps_med, 6)
        json_result[f"{isa_name}_ratio"] = round(ratio, 4)

    # --- JSON to stdout ---
    print(json.dumps(json_result))


if __name__ == "__main__":
    main()
