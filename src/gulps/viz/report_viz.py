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

"""Summary report visualizations comparing synthesis results."""

from collections import Counter

import lovelyplots
import matplotlib.pyplot as plt
import numpy as np
import scienceplots


def compare_continuous_discrete(
    continuous_params,
    continuous_costs,
    discrete_params,
    discrete_costs,
):
    """Compare continuous vs discrete decomposer parameter and cost distributions.

    Args:
        continuous_params: List of parameters from continuous decomposer
        continuous_costs: List of total costs from continuous decomposer
        discrete_params: List of parameters from discrete decomposer
        discrete_costs: List of total costs from discrete decomposer
    """
    N_continuous = len(continuous_costs)
    N_discrete = len(discrete_costs)
    nbins = int(np.sqrt(N_continuous))

    # Compute discrete frequency distribution
    discrete_counter = Counter(discrete_params)
    discrete_vals = sorted(discrete_counter.keys())
    discrete_freqs = np.array([discrete_counter[v] for v in discrete_vals]) / len(
        discrete_params
    )

    with plt.style.context(["ieee", "science", "no-latex"]):
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))

        # Left: Continuous (histogram)
        axes[0].hist(
            continuous_params,
            bins=nbins,
            weights=np.ones(len(continuous_params)) / len(continuous_params),
            color="C0",
            edgecolor="black",
            linewidth=0.5,
        )
        axes[0].set_xlabel("Parameter (k)")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Continuous ISA")
        axes[0].set_xlim(0, 1.0)
        axes[0].grid(axis="y", alpha=0.3)

        # Right: Discrete (bar chart)
        axes[1].bar(
            discrete_vals, discrete_freqs, width=0.05, color="C1", edgecolor="black"
        )
        axes[1].set_xlabel("Parameter (k)")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Discrete ISA")
        axes[1].set_xlim(0, 1.0)
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Print cost statistics
    print("=== Cost Statistics ===")
    print(
        f"Continuous: mean={np.mean(continuous_costs):.4f}, std={np.std(continuous_costs):.4f}"
    )
    print(
        f"Discrete:   mean={np.mean(discrete_costs):.4f}, std={np.std(discrete_costs):.4f}"
    )
    cost_improvement = (
        (np.mean(discrete_costs) - np.mean(continuous_costs))
        / np.mean(discrete_costs)
        * 100
    )
    print(f"Cost improvement: {cost_improvement:.2f}%")


def report_benchmark_results(fidelities, all_timings, decomposer, N, failures):
    """Generate comprehensive benchmark report with statistics and visualizations."""
    fidelities = np.array(fidelities)

    # Print summary statistics
    print(f"Benchmark: {len(fidelities)}/{N} successful ({failures} failures)")
    print(
        f"Fidelity: median={np.median(fidelities):.10f}, min={np.min(fidelities):.10f}"
    )

    preferred_order = ["init", "lp_sentence", "numerics", "stitch", "total"]
    phase_names = [k for k in preferred_order if k in all_timings[0]]
    timing_array = np.array([[t[k] for k in phase_names] for t in all_timings])

    if "total" in phase_names:
        total_idx = phase_names.index("total")
        total_avg = timing_array[:, total_idx].mean() * 1000000
    else:
        total_avg = timing_array.sum(axis=1).mean() * 1000000

    print(f"Avg time: {total_avg:.1f} us/decomposition")

    # Create visualizations
    hist_phases = ["lp_sentence", "numerics", "stitch"]
    hist_data = {k: np.array([d[k] for d in all_timings]) for k in hist_phases}
    hist_titles = {
        "lp_sentence": "LP Sentence",
        "numerics": "Numerics",
        "stitch": "Stitch",
    }

    # Consistent color map for all phase visualizations
    phase_colors = {
        "init": "#2ecc71",
        "lp_sentence": "#3498db",
        "numerics": "#9b59b6",
        "stitch": "#e67e22",
        "total": "#95a5a6",
    }

    with plt.style.context(["ieee", "science", "no-latex"]):
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

        # Top right: timing breakdown (horizontal bars)
        ax1 = fig.add_subplot(gs[0, 2])
        mean_times = timing_array.mean(axis=0) * 1000000
        bar_colors = [phase_colors.get(p, "#999999") for p in phase_names]
        ax1.barh(range(len(phase_names)), mean_times, color=bar_colors)
        ax1.set_yticks(range(len(phase_names)))
        ax1.set_yticklabels(phase_names, fontsize=8)
        ax1.set_xlabel("Time (us)")
        ax1.set_title("Avg Phase Time")
        ax1.grid(axis="x", alpha=0.3)
        ax1.invert_yaxis()

        # Bottom row: timing histograms
        for col, phase in enumerate(hist_phases):
            ax = fig.add_subplot(gs[1, col])
            data = hist_data[phase]
            ax.hist(data, bins=40, edgecolor="black", color=phase_colors[phase])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Count")
            ax.set_title(hist_titles[phase])
            ax.grid(axis="y", alpha=0.3)

        plt.show()
