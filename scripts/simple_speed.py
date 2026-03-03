import numpy as np
from qiskit.quantum_info import Operator, average_gate_fidelity
from qiskit.quantum_info.random import random_unitary

from gulps import logger
from gulps.gulps_decomposer import GulpsDecomposer
from tests.fixtures.isas import get_slim_isas


def bench_isa(isa, N=1000):
    """Benchmark a single DiscreteISA on N random unitaries. Returns (median_ms, failures)."""
    name = "+".join(g.name for g in isa.gate_set)
    decomposer = GulpsDecomposer(isa=isa)

    # JIT warmup
    decomposer(random_unitary(4, seed=9999))

    fidelities = []
    all_timings = []
    failures = 0

    for idx in range(N):
        u = random_unitary(4, seed=idx)
        try:
            fid = average_gate_fidelity(u, Operator(decomposer(u)))
            if fid < 1 - 1e-8:
                raise ValueError(f"Fidelity too low: {fid:.8f}")
            fidelities.append(fid)
            all_timings.append(decomposer.last_timing)
        except Exception as e:
            failures += 1

    fidelities = np.array(fidelities) if fidelities else np.array([0.0])
    times = [t["total"] for t in all_timings]
    return {
        "name": name,
        "n": N,
        "failures": failures,
        "fid_min": float(np.min(fidelities)),
        "mean_ms": float(np.mean(times)) * 1000 if times else float("inf"),
        "median_ms": float(np.median(times)) * 1000 if times else float("inf"),
    }


def main():
    logger.setLevel("INFO")
    N = 1_000
    isas = get_slim_isas()

    print(f"Benchmarking {len(isas)} ISAs × {N} random unitaries each\n")

    results = []
    for isa in isas:
        r = bench_isa(isa, N)
        results.append(r)
        print(
            f"  {r['name']:30s}: "
            f"mean={r['mean_ms']:6.1f} ms  "
            f"median={r['median_ms']:6.1f} ms  "
            f"({r['n'] - r['failures']}/{r['n']} ok, "
            f"fid_min={r['fid_min']:.8f})"
        )

    # Summary line (for commit history tracking)
    medians = [r["median_ms"] for r in results]
    print(
        f"\nOverall: geometric mean of medians = {np.exp(np.mean(np.log(medians))):.1f} ms"
    )


if __name__ == "__main__":
    main()
