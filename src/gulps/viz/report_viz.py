import lovelyplots
import matplotlib.pyplot as plt
import numpy as np
import scienceplots


def report_benchmark_results(fidelities, all_timings, decomposer, N, failures):
    """Generate comprehensive benchmark report with statistics and visualizations."""
    fidelities = np.array(fidelities)
    cache_stats = decomposer._local_synthesis._cache.stats

    # Print summary statistics
    print(f"Benchmark: {len(fidelities)}/{N} successful ({failures} failures)")
    print(
        f"Fidelity: median={np.median(fidelities):.10f}, min={np.min(fidelities):.10f}"
    )

    if all_timings:
        phase_names = list(all_timings[0].keys())
        timing_array = np.array([[t[k] for k in phase_names] for t in all_timings])
        total_avg = timing_array.sum(axis=1).mean() * 1000
        print(f"Avg time: {total_avg:.1f} ms/decomposition")

    print(
        f"Cache: {cache_stats['hit_rate'] * 100:.1f}% hit rate ({cache_stats['hits']}/{cache_stats['hits'] + cache_stats['misses']})"
    )

    # Create visualizations
    lp_times = [d["lp_sentence"] for d in all_timings]
    numeric_times = [d["segments"] for d in all_timings]

    with plt.style.context(["ieee", "science", "use_mathtext"]):
        fig = plt.figure(figsize=(8, 4.5))
        gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.4)

        # Top row: infidelity plot (spans 2 columns)
        ax0 = fig.add_subplot(gs[0, :2])
        ax0.plot(
            1 - fidelities, marker="o", markersize=2, linestyle="None", color="#e74c3c"
        )
        ax0.set_xlabel("Trial Index")
        ax0.set_ylabel("Infidelity")
        ax0.set_yscale("log")
        ax0.grid(True, alpha=0.3)
        ax0.set_title("Infidelity Distribution")

        # Top right: timing breakdown (horizontal bars with shortened labels)
        ax1 = fig.add_subplot(gs[0, 2])
        mean_times = timing_array.mean(axis=0) * 1000
        colors_bar = plt.cm.viridis(np.linspace(0.2, 0.8, len(phase_names)))
        ax1.barh(range(len(phase_names)), mean_times, color=colors_bar)
        ax1.set_yticks(range(len(phase_names)))
        ax1.set_yticklabels(phase_names, fontsize=8)
        ax1.set_xlabel("Time (ms)")
        ax1.set_title("Avg Phase Time")
        ax1.grid(axis="x", alpha=0.3)
        ax1.invert_yaxis()

        # Bottom row: timing histograms
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(lp_times, bins=20, edgecolor="black", color="#3498db")
        ax2.set_xlabel("LP Time (s)")
        ax2.set_ylabel("Count")
        ax2.set_title("LP Solver")
        ax2.grid(axis="y", alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(numeric_times, bins=20, edgecolor="black", color="#9b59b6")
        ax3.set_xlabel("Numeric Time (s)")
        ax3.set_ylabel("Count")
        ax3.set_title("Segment Solver")
        ax3.grid(axis="y", alpha=0.3)

        # Bottom right: cache stats
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.pie(
            [cache_stats["hits"], cache_stats["misses"]],
            labels=["Hits", "Misses"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#2ecc71", "#95a5a6"],
        )
        ax4.set_title("Cache Performance")

        plt.show()
