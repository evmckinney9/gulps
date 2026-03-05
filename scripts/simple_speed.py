"""Benchmark GULPS decomposition on seeded random unitaries across 3 iSwap ISAs.

Rich diagnostics go to stderr (human-readable).
Machine-parseable JSON goes to stdout:
  {"isa1_median": ..., "isa2_median": ..., "isa3_median": ...}

ISAs (matching benchmark_task.py for backlog compatibility):
  isa1: [iSwap^(1/4)]
  isa2: [iSwap^(1/3), iSwap^(1/4)]
  isa3: [iSwap^(1/4), iSwap^(1/3), iSwap^(1/2)]

Usage:
  python scripts/simple_speed.py           # full run, 1000 unitaries
  python scripts/simple_speed.py -n 10     # quick smoke test
"""

import argparse
import json
import logging
import statistics
import sys
import time

import numpy as np


def bench(decomposer, name, n):
    """Benchmark a decomposer on n seeded random unitaries.

    Returns (times_list, fidelities_list). Raises on any failure or low fidelity.
    """
    from qiskit.quantum_info import Operator, average_gate_fidelity, random_unitary

    times = []
    failures = 0
    fidelities = []

    for seed in range(n):
        target = random_unitary(4, seed=seed)
        try:
            t0 = time.perf_counter()
            result = decomposer._run(target)
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
    fids = np.array(fidelities) if fidelities else np.array([0.0])
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=1000)
    args = parser.parse_args()

    # Legacy import fallback (supports running at old commits via collect_backlog.sh)
    try:
        from gulps.gulps_decomposer import GulpsDecomposer
    except ModuleNotFoundError:
        try:
            from gulps.synthesis.gulps_decomposer import GulpsDecomposer
        except ModuleNotFoundError:
            from gulps.gulps_synthesis import GulpsDecomposer

    logging.getLogger("gulps").setLevel(logging.WARNING)

    from qiskit.circuit.library import iSwapGate

    # power = 1/N means iSwap^(1/N)
    sq4, sq3, sq2 = 1 / 4, 1 / 3, 1 / 2

    isas = {
        "isa1": [sq4],
        "isa2": [sq3, sq4],
        "isa3": [sq4, sq3, sq2],
    }

    print(
        f"iSwap benchmark: {len(isas)} ISAs x {args.n} random unitaries each\n",
        file=sys.stderr,
    )

    json_result = {}
    all_medians_ms = []

    for isa_key, powers in isas.items():
        gates = [iSwapGate().power(p) for p in powers]
        costs = list(powers)
        label = "+".join(f"iSwap^(1/{int(1 / p)})" for p in powers)
        display_name = f"{isa_key}/{label}"

        decomposer = GulpsDecomposer(
            gate_set=gates,
            costs=costs,
            precompute_polytopes=False,
        )
        times, fidelities, failures = bench(decomposer, display_name, args.n)
        median_s = statistics.median(times)

        # Rich stats to stderr
        print_rich_stats(display_name, times, fidelities, failures, args.n)
        # Collect for JSON (keys stay isa1/isa2/isa3 for backlog CSV compat)
        json_result[isa_key] = round(median_s, 6)
        all_medians_ms.append(median_s * 1000)

    # Summary to stderr
    geo_mean = np.exp(np.mean(np.log(all_medians_ms)))
    print(
        f"\nOverall: geometric mean of medians = {geo_mean:.1f} ms",
        file=sys.stderr,
    )

    # JSON to stdout (compatible with collect_backlog.sh extraction)
    print(json.dumps(json_result))


if __name__ == "__main__":
    main()
