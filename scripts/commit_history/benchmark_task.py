"""Standalone benchmark: time GULPS decomposition on 1000 seeded random unitaries
across 3 ISA configurations.

Outputs a single JSON line to stdout:
  {"isa1_median": ..., "isa2_median": ..., "isa3_median": ...}

ISAs:
  isa1: [iSwap^(1/4)]
  isa2: [iSwap^(1/3), iSwap^(1/4)]
  isa3: [iSwap^(1/4), iSwap^(1/3), iSwap^(1/2)]

Caveats
-------
Machine-specificity:
  Results are only meaningful when compared across commits run on the same machine.
  The benchmark assumes the machine is otherwise idle - heavy background load will
  inflate timings, especially at recent commits where decomposition is fast (~10ms)
  and OS jitter is a larger fraction of the measurement.

Error handling and old commits:
  Individual decomposition failures (solver non-convergence, numerical errors) are
  caught and do NOT abort the run. Failed calls still contribute their wall-clock
  time to the median. This is intentional: a failed decomposition exhausts all
  solver attempts before giving up, so its timing reflects real cost.

  Older commits tend to have higher error rates (sometimes 100+ per 1000 runs) due
  to numerical bugs that were fixed later. This makes their median timing higher than
  it would be for a fully-correct implementation - the timing and the correctness
  regression are two sides of the same bug.

  The progress line shows `errors=N` when failures occur, so you can see at a glance
  how numerically stable a given commit was.
"""

import argparse
import json
import logging
import statistics
import sys
import time


def bench(decomposer, name, n):
    from qiskit.quantum_info import random_unitary

    times = []
    errors = 0
    for seed in range(n):
        target = random_unitary(4, seed=seed)
        t0 = time.perf_counter()
        try:
            decomposer._run(target)
        except Exception:
            errors += 1
        t1 = time.perf_counter()
        times.append(t1 - t0)

        # progress to stderr every 50 iterations
        if (seed + 1) % 50 == 0 or seed == n - 1:
            elapsed = sum(times)
            rate = (seed + 1) / elapsed
            eta = (n - seed - 1) / rate
            med = statistics.median(times)
            err_str = f"  errors={errors}" if errors else ""
            print(
                f"\r  {name}: {seed + 1}/{n}  "
                f"elapsed={elapsed:.1f}s  eta={eta:.1f}s  "
                f"median={med:.4f}s{err_str}",
                end="",
                file=sys.stderr,
            )
    print(file=sys.stderr)  # newline after progress

    return round(statistics.median(times), 6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=1000)
    args = parser.parse_args()

    from qiskit.circuit.library import iSwapGate

    try:
        from gulps.gulps_decomposer import GulpsDecomposer  # post-refactor (current)
    except ModuleNotFoundError:
        try:
            from gulps.synthesis.gulps_decomposer import GulpsDecomposer  # mid-era
        except ModuleNotFoundError:
            from gulps.gulps_synthesis import GulpsDecomposer  # early era

    # Old commits set logging.DEBUG in logging_config.py at import time; silence after import
    logging.getLogger("gulps").setLevel(logging.WARNING)

    sq2 = iSwapGate().power(1 / 2)
    sq3 = iSwapGate().power(1 / 3)
    sq4 = iSwapGate().power(1 / 4)

    isas = {
        "isa1": ([sq4], [0.25]),
        "isa2": ([sq3, sq4], [1 / 3, 0.25]),
        "isa3": ([sq4, sq3, sq2], [0.25, 1 / 3, 0.5]),
    }

    result = {}
    for name, (gates, costs) in isas.items():
        decomposer = GulpsDecomposer(
            gate_set=gates,
            costs=costs,
            precompute_polytopes=False,
        )
        result[name] = bench(decomposer, name, args.n)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
