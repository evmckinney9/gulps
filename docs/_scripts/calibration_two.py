"""Two and three calibrated strengths of iSWAP: savings and pair landscapes.

Runs calibrate with budget 3 over GRID at several values of b + ell for Haar targets
and for the ZZ workload of the calibration page, and scores every GRID pair on Haar
targets at two values of b + ell. Writes the scores to calibration_two.json and draws
two figures into docs/_static. With b + ell passed as pulse_overhead, the costs are
those at ell = 0; a nonzero ell adds ell per target to every entry and changes no
selection.
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from calibration_one import BASE, HERE, STATIC, ZZ, label

from gulps.analysis.calibration import GRID, calibrate
from gulps.analysis.coverage import coverage_report
from gulps.decomposition import GulpsDecomposer

BUDGET_OVERHEADS = np.logspace(-2, 1, 7).tolist()
PAIR_OVERHEADS = (0.03, 1.0)


def budget_runs(workload: list | None) -> list:
    """Run calibrate at budget 3 for each value of b + ell in BUDGET_OVERHEADS."""
    return [
        calibrate(BASE, 3, workload=workload, pulse_overhead=r)
        for r in BUDGET_OVERHEADS
    ]


def pair_cost(pair: list[float], overhead: float) -> float:
    """Haar-average cost of the instruction set of these strengths at b + ell."""
    decomposer = GulpsDecomposer(
        [BASE.power(k) for k in pair], [k + overhead for k in pair]
    )
    return coverage_report(decomposer).expected_cost


def compute() -> dict:
    """Run the budget sweeps and the exhaustive pair scores."""
    data = {"grid": list(GRID), "budget_overheads": BUDGET_OVERHEADS, "budgets": {}}
    for name, workload in (("haar", None), ("zz", ZZ)):
        scale = 1 if workload is None else len(workload)
        runs = budget_runs(workload)
        data["budgets"][name] = {
            "costs": [[c / scale for c in run.budget_costs] for run in runs],
            "strengths": [[list(s) for s in run.budget_strengths] for run in runs],
        }
        print(f"budgets {name}", flush=True)
    data["pairs"] = {}
    for r in PAIR_OVERHEADS:
        costs = np.full((len(GRID), len(GRID)), np.nan)  # row: stronger, column: weaker
        for i, weak in enumerate(GRID):
            for j in range(i, len(GRID)):
                costs[j, i] = pair_cost(sorted({weak, GRID[j]}), r)
        greedy = calibrate(BASE, 2, pulse_overhead=r)
        data["pairs"][str(r)] = {
            "costs": [[None if np.isnan(c) else c for c in row] for row in costs],
            "greedy": list(greedy.strengths),
            "greedy_cost": greedy.cost,
        }
        print(f"pairs {r}", flush=True)
    return data


def plot_budgets(data: dict) -> None:
    """Selected strengths and the saving of each added strength against b + ell."""
    overheads = np.array(data["budget_overheads"])
    panels = (("Haar targets", "haar"), ("ZZ workload", "zz"))
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.6, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1]},
    )
    for column, (title, key) in enumerate(panels):
        costs = np.array(data["budgets"][key]["costs"])
        sets = data["budgets"][key]["strengths"]
        ax, bx = axes[0, column], axes[1, column]
        for k in (1 / 2, 1 / 3, 1 / 4):
            ax.axhline(k, color="0.88", linewidth=0.8, zorder=0)
        ax.plot(overheads, [s[0][0] for s in sets], "o-", color="C0", label="budget 1")
        ax.plot(
            overheads,
            [s[1][0] for s in sets],
            "s--",
            color="C1",
            label=r"budget 2, weaker",
        )
        ax.plot(
            overheads,
            [s[1][1] for s in sets],
            "^--",
            color="C1",
            label=r"budget 2, stronger",
        )
        ax.set(xscale="log", ylim=(0, 1), title=title)
        bx.plot(
            overheads,
            costs[:, 0] - costs[:, 1],
            "o-",
            color="C1",
            label="second strength",
        )
        bx.plot(
            overheads,
            costs[:, 1] - costs[:, 2],
            "o:",
            color="C2",
            label="third strength",
        )
        bx.set(xscale="log", ylim=(0, None), xlabel=r"Fixed cost per pulse $b+\ell$")
    axes[0, 0].set_ylabel("Selected strengths")
    axes[1, 0].set_ylabel("Saving per target")
    axes[0, 0].legend(loc="upper left", fontsize=7)
    axes[1, 0].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(
        STATIC / "calibration_budgets.svg",
        bbox_inches="tight",
        metadata={"Date": None},
    )
    plt.close(fig)


def plot_pairs(data: dict) -> None:
    """Haar cost of every GRID pair relative to the best pair, at two values of b + ell."""
    grid = np.array(data["grid"])
    step = grid[1] - grid[0]
    extent = [grid[0] - step / 2, grid[-1] + step / 2] * 2
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    for ax, (r, pair) in zip(axes, data["pairs"].items()):
        costs = np.array(pair["costs"], dtype=float)
        j, i = np.unravel_index(np.nanargmin(costs), costs.shape)
        single = int(np.argmin(np.diag(costs)))
        image = ax.imshow(
            costs / np.nanmin(costs),
            origin="lower",
            extent=extent,
            cmap="viridis_r",
            vmin=1.0,
            vmax=1.3,
            aspect="equal",
        )
        ax.plot(
            grid[i],
            grid[j],
            "*",
            color="white",
            markeredgecolor="black",
            markersize=13,
            label=f"best pair ({label(grid[i])}, {label(grid[j])})",
        )
        ax.plot(
            grid[single],
            grid[single],
            "D",
            color="white",
            markeredgecolor="black",
            markersize=6,
            label=f"best single {label(grid[single])}",
        )
        weak, strong = pair["greedy"]
        if not np.allclose((weak, strong), (grid[i], grid[j])):
            ax.plot(
                weak,
                strong,
                "o",
                markerfacecolor="none",
                markeredgecolor="C3",
                markersize=9,
                markeredgewidth=1.5,
                label=f"greedy pair ({label(weak)}, {label(strong)})",
            )
        ax.set(title=rf"$b+\ell={float(r):g}$", xlabel=r"Weaker strength $k_1$")
        ax.legend(
            loc="lower right",
            fontsize=7,
            frameon=True,
            framealpha=0.9,
            edgecolor="0.3",
            fancybox=False,
        )
    axes[0].set_ylabel(r"Stronger strength $k_2$")
    colorbar = fig.colorbar(image, ax=axes, shrink=0.85, pad=0.02, extend="max")
    colorbar.set_label("Haar cost / best pair")
    fig.savefig(
        STATIC / "calibration_pairs.svg",
        bbox_inches="tight",
        metadata={"Date": None},
    )
    plt.close(fig)


def report(data: dict) -> None:
    """Print the numbers the calibration page quotes."""
    grid = np.array(data["grid"])
    for key, runs in data["budgets"].items():
        print(key)
        for r, costs, sets in zip(
            data["budget_overheads"], runs["costs"], runs["strengths"]
        ):
            names = [tuple(label(k) for k in s) for s in sets]
            print(
                f"  b + ell = {r:7.4f}  {names}  cost {np.round(costs, 4).tolist()}"
                f"  second saves {costs[0] - costs[1]:.4f}"
                f" ({1 - costs[1] / costs[0]:.1%} at ell = 0),"
                f" third saves {costs[1] - costs[2]:.4f}"
            )
    for r, pair in data["pairs"].items():
        costs = np.array(pair["costs"], dtype=float)
        j, i = np.unravel_index(np.nanargmin(costs), costs.shape)
        diagonal = np.diag(costs)
        print(
            f"pairs b + ell = {r}: best ({label(grid[i])}, {label(grid[j])}) {costs[j, i]:.5f};"
            f" greedy {tuple(label(k) for k in pair['greedy'])} {pair['greedy_cost']:.5f};"
            f" best single {label(grid[diagonal.argmin()])} {diagonal.min():.5f};"
            f" best single / best pair {diagonal.min() / costs[j, i]:.4f}"
        )


def main() -> None:
    """Compute or load the data, then draw the figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot", action="store_true", help="replot from calibration_two.json only"
    )
    path = HERE / "calibration_two.json"
    if parser.parse_args().plot:
        data = json.loads(path.read_text())
    else:
        data = compute()
        path.write_text(json.dumps(data) + "\n")
    report(data)
    plot_budgets(data)
    plot_pairs(data)


if __name__ == "__main__":
    main()
