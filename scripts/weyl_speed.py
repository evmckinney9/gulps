"""Benchmark GULPS on a deterministic Weyl-chamber grid (weyl_linspace).

Non-Haar targets stress degenerate Weyl faces that random unitaries avoid.

ISAs match ``tests.fixtures.isas.get_random_circuit_isas()``:
  isa1: [iSwap, iSwap^(1/2), iSwap^(1/3)]       - pure iSwap family
  isa2: [CX, CX^(1/2), CX^(1/4)]                - pure CX family
  isa3: [CX^(1/2), iSwap^(1/3)]                  - mixed

Usage:
  python scripts/weyl_speed.py             # full run, N=64 grid
  python scripts/weyl_speed.py -n 32       # smaller grid
"""

import argparse
import json
import logging
import statistics
import sys
import time

import numpy as np


def bench_weyl(decomposer, name, grid_points):
    """Benchmark a decomposer on weyl_linspace grid points.

    Returns (times_list, fidelities_list, failures).
    """
    from qiskit.quantum_info import Operator, average_gate_fidelity

    from gulps import GateInvariants

    times = []
    failures = 0
    fidelities = []
    n = len(grid_points)

    for idx, weyl_pt in enumerate(grid_points):
        target = GateInvariants.from_weyl(weyl_pt)
        try:
            t0 = time.perf_counter()
            result = decomposer(target.gate)
            t1 = time.perf_counter()
            fid = average_gate_fidelity(Operator(target.gate), Operator(result))
            if fid < 1 - 1e-8:
                raise ValueError(
                    f"{name} pt={weyl_pt}: fidelity {fid:.10f} below threshold"
                )
        except Exception:
            failures += 1
            continue

        fidelities.append(fid)
        times.append(t1 - t0)

        if (idx + 1) % 10 == 0 or idx == n - 1:
            elapsed = sum(times) if times else 0.001
            med = statistics.median(times) if times else 0
            print(
                f"\r  {name}: {idx + 1}/{n}  elapsed={elapsed:.1f}s  median={med:.4f}s",
                end="",
                file=sys.stderr,
            )
    print(file=sys.stderr)

    return times, fidelities, failures


def print_rich_stats(name, times, fidelities, failures, n, file=sys.stderr):
    """Print detailed human-readable stats."""
    times_ms = np.array(times) * 1000
    fids = np.array(fidelities) if fidelities else np.array([0.0])
    total_s = np.sum(times) / 1000  # back to seconds
    print(
        f"  {name:40s}: "
        f"mean={np.mean(times_ms):6.1f} ms  "
        f"median={np.median(times_ms):6.1f} ms  "
        f"total={np.sum(times):6.2f} s  "
        f"p95={np.percentile(times_ms, 95):6.1f} ms  "
        f"max={np.max(times_ms):7.1f} ms  "
        f"({n - failures}/{n} ok, "
        f"fid_min={np.min(fids):.8f})",
        file=file,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=64, help="weyl_linspace grid size")
    args = parser.parse_args()

    logging.getLogger("gulps").setLevel(logging.WARNING)

    from qiskit.circuit.library import CXGate, iSwapGate

    from gulps import GulpsDecomposer
    from gulps.viz.weyl_chamber import weyl_linspace
    from gulps.core.isa import DiscreteISA

    grid_points = list(weyl_linspace(args.n))

    isas = {
        "isa1": DiscreteISA(
            [iSwapGate(), iSwapGate().power(1 / 2), iSwapGate().power(1 / 3)],
            [1.0, 0.5, 1 / 3],
            ["iswap", "sq2iswap", "sq3iswap"],
        ),
        "isa2": DiscreteISA(
            [CXGate(), CXGate().power(1 / 2), CXGate().power(1 / 4)],
            [1.0, 0.5, 0.25],
            ["cx", "sq2cx", "sq4cx"],
            max_sequence_length=12,
        ),
        "isa3": DiscreteISA(
            [CXGate().power(1 / 2), iSwapGate().power(1 / 3)],
            [0.5, 1 / 3],
            ["sq2cx", "sq3iswap"],
        ),
        "isa4": DiscreteISA(
            [CXGate().power(1 / 2)],
            [0.5],
            ["sq2cx"],
        ),
        "isa5": DiscreteISA(
            [CXGate().power(1 / 4)],
            [1 / 4],
            ["sq4cx"],
            max_sequence_length=12,
        ),
    }

    print(
        f"Weyl-linspace benchmark: {len(isas)} ISAs x {len(grid_points)} grid points each\n",
        file=sys.stderr,
    )

    json_result = {}

    for isa_key, isa in isas.items():
        label = "+".join(g.name for g in isa.gate_set)
        display_name = f"{isa_key}/{label}"

        decomposer = GulpsDecomposer(isa=isa)
        times, fidelities, failures = bench_weyl(decomposer, display_name, grid_points)

        if not times:
            print(f"  {display_name}: ALL FAILED", file=sys.stderr)
            json_result[isa_key] = {
                "median": None,
                "mean": None,
                "total": None,
                "failures": failures,
            }
            continue

        median_s = statistics.median(times)
        mean_s = statistics.mean(times)
        total_s = sum(times)

        print_rich_stats(display_name, times, fidelities, failures, len(grid_points))

        json_result[isa_key] = {
            "median": round(median_s, 6),
            "mean": round(mean_s, 6),
            "total": round(total_s, 4),
            "failures": failures,
        }

    # Overall summary
    all_totals = [v["total"] for v in json_result.values() if v["total"] is not None]
    all_failures = sum(v["failures"] for v in json_result.values())
    grand_total = sum(all_totals)
    print(
        f"\nOverall: grand total = {grand_total:.2f} s, "
        f"failures = {all_failures}/{len(grid_points) * len(isas)}",
        file=sys.stderr,
    )

    print(json.dumps(json_result))


if __name__ == "__main__":
    main()
