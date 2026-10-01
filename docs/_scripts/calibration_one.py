"""One calibrated strength of iSWAP: cost landscape, selection against overheads.

Scores single-strength iSWAP^k instruction sets for Haar targets and for the ZZ
workload of the calibration page, writes the scores to calibration_one.json, and
draws four figures into docs/_static. With one strength, the cost of a target is
(k + b + ell) N + ell, where N is its pulse count, so two sweeps (b = 0 and b = 1)
give the selection at every overhead. The run checks that identity at several
(b, ell) points against direct sweeps.
"""

import argparse
import json
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit.library import RZZGate, iSwapGate

from gulps.analysis.calibration import GRID, strength_sweep

BASE = iSwapGate()
ZZ = [RZZGate(angle) for angle in (0.24, 0.44, 0.64) for _ in range(4)]


def time_and_pulses(workload: list | None) -> tuple[np.ndarray, np.ndarray]:
    """Per-target interaction time k N and pulse count N at each GRID strength."""
    scale = 1 if workload is None else len(workload)
    time_ = np.array(strength_sweep(BASE, workload=workload).costs) / scale
    at_one = np.array(strength_sweep(BASE, workload=workload, pulse_overhead=1.0).costs)
    return time_, at_one / scale - time_


HERE = Path(__file__).resolve().parent
STATIC = HERE.parent / "_static"
# A grid finer than GRID and below its floor, so the landscape shows the curve
# between grid points. It contains 1/n for n up to 20.
FINE = np.union1d(np.linspace(0.05, 1.0, 191), 1 / np.arange(1, 21))
LANDSCAPE_OVERHEADS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0)
CHECKS = ((0.004, 0.0), (0.0, 0.1), (0.3, 0.2), (0.05, 2.0), (3.0, 0.5))
# Values of b + ell for the staircase figure and the numbers quoted with it.
STAIRCASE = np.logspace(-3, 1.5, 2000)


def selected(time_: np.ndarray, pulses: np.ndarray, overhead: np.ndarray) -> np.ndarray:
    """Index of the selected GRID strength at each value of b + ell."""
    costs = time_ + np.multiply.outer(np.asarray(overhead), pulses)
    return costs.argmin(axis=-1)


def compute() -> dict:
    """Run the sweeps, check the one-strength identity, and return the data."""
    grid = np.array(GRID)
    data = {"grid": grid.tolist(), "fine": FINE.tolist(), "landscape": {}}
    for r in LANDSCAPE_OVERHEADS:
        data["landscape"][str(r)] = list(
            strength_sweep(BASE, FINE, pulse_overhead=r).costs
        )
        print(f"landscape b + ell = {r}", flush=True)
    for name, workload in (("haar", None), ("zz", ZZ)):
        time_, pulses = time_and_pulses(workload)
        scale = 1 if workload is None else len(workload)
        for b, ell in CHECKS:
            sweep = strength_sweep(
                BASE, local_layer_cost=ell, workload=workload, pulse_overhead=b
            )
            predicted = (grid + b + ell) * pulses + ell
            np.testing.assert_allclose(
                np.array(sweep.costs) / scale, predicted, rtol=0, atol=1e-9
            )
            assert sweep.best[0] == grid[selected(time_, pulses, b + ell)], (
                name,
                b,
                ell,
            )
        data[name] = {"time": time_.tolist(), "pulses": pulses.tolist()}
    return data


def hull(time_: np.ndarray, pulses: np.ndarray) -> tuple[list[int], list[float]]:
    """Strengths selected for some b + ell, in increasing b + ell, and the switch points."""
    i = int(np.lexsort((pulses, time_))[0])
    out, switches = [i], []
    while True:
        fewer = np.flatnonzero(pulses < pulses[i] - 1e-12)
        if not len(fewer):
            return out, switches
        ties = (time_[fewer] - time_[i]) / (pulses[i] - pulses[fewer])
        r = ties.min()
        candidates = fewer[np.isclose(ties, r)]
        i = int(candidates[np.argmin(pulses[candidates])])
        out.append(i)
        switches.append(float(r))


def label(k: float) -> str:
    """A strength as a fraction with denominator at most 60."""
    return str(Fraction(k).limit_denominator(60))


def plot_landscape(data: dict) -> None:
    """Haar cost relative to its minimum against strength, one curve per overhead."""
    fine = np.array(data["fine"])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(data["landscape"])))
    for n in range(2, 6):
        ax.axvline(1 / n, color="0.85", linewidth=0.8, zorder=0)
        ax.text(
            1 / n, 2.02, f"1/{n}", ha="center", va="bottom", fontsize=8, color="0.4"
        )
    for color, (r, costs) in zip(colors, data["landscape"].items()):
        costs = np.array(costs)
        ax.plot(fine, costs / costs.min(), color=color, label=f"{float(r):g}")
        ax.plot(
            fine[costs.argmin()],
            1.0,
            "o",
            color=color,
            markerfacecolor=color,
            markersize=6,
            zorder=3,
        )
    ax.set(
        xlim=(0.05, 1.0),
        ylim=(0.98, 2.0),
        xlabel=r"Strength $k$ of iSWAP$^k$",
        ylabel="Haar cost / its minimum",
    )
    ax.legend(
        title=r"$b+\ell$",
        ncol=6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        fontsize=8,
        title_fontsize=8,
    )
    fig.savefig(
        STATIC / "calibration_landscape.svg",
        bbox_inches="tight",
        metadata={"Date": None},
    )
    plt.close(fig)


def plot_hull(data: dict) -> None:
    """Each strength as a point (pulses, interaction time), with the lower hull."""
    grid = np.array(data["grid"])
    panels = (("Haar targets", "haar", (0.85, 1.6)), ("ZZ workload", "zz", (0.25, 0.6)))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), gridspec_kw={"wspace": 0.3})
    for ax, (title, key, ylim) in zip(axes, panels):
        time_, pulses = np.array(data[key]["time"]), np.array(data[key]["pulses"])
        idx, _ = hull(time_, pulses)
        points = ax.scatter(
            pulses, time_, c=grid, cmap="viridis", s=12, zorder=2, vmin=0, vmax=1
        )
        ax.plot(pulses[idx], time_[idx], color="0.2", linewidth=1.0, zorder=1)
        ax.plot(
            pulses[idx],
            time_[idx],
            "o",
            markersize=7,
            markerfacecolor="none",
            markeredgecolor="C3",
            zorder=3,
        )
        for i in idx:
            if pulses[i] < 6 and (
                key == "zz" or label(grid[i]) in {"1/2", "1/3", "1/4", "1/5", "3/4"}
            ):
                ax.annotate(
                    f"$k={label(grid[i])}$",
                    (pulses[i], time_[i]),
                    xytext=(6, 2) if label(grid[i]) == "3/4" else (6, -12),
                    textcoords="offset points",
                    fontsize=8,
                )
        best = int(np.argmin(time_ + 0.1 * pulses))
        xs = np.array([1.5, 7.0])
        ax.plot(
            xs,
            time_[best] - 0.1 * (xs - pulses[best]),
            "--",
            color="C3",
            linewidth=0.8,
            label=r"slope $-0.1$",
        )
        ax.legend(loc="upper right", fontsize=7)
        ax.set(title=title, xlabel="Mean pulses per target", xlim=(1.9, 6.2), ylim=ylim)
    axes[0].set_ylabel("Mean interaction time per target")
    colorbar = fig.colorbar(points, ax=axes, shrink=0.85, pad=0.02)
    colorbar.set_label(r"Strength $k$")
    fig.savefig(
        STATIC / "calibration_hull.svg",
        bbox_inches="tight",
        metadata={"Date": None},
    )
    plt.close(fig)


def plot_staircase(data: dict) -> None:
    """Selected strength against b + ell, and the ZZ cost of the Haar selection."""
    grid = np.array(data["grid"])
    overhead = STAIRCASE
    haar = selected(
        np.array(data["haar"]["time"]), np.array(data["haar"]["pulses"]), overhead
    )
    zz_time, zz_pulses = np.array(data["zz"]["time"]), np.array(data["zz"]["pulses"])
    zz = selected(zz_time, zz_pulses, overhead)
    zz_cost = zz_time + np.multiply.outer(overhead, zz_pulses)
    rows = np.arange(len(overhead))
    gap = zz_cost[rows, haar] - zz_cost[rows, zz]
    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(7, 4.8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    for k, text in ((1 / 2, "1/2"), (1 / 3, "1/3"), (1 / 4, "1/4"), (3 / 4, "3/4")):
        ax.axhline(k, color="0.85", linewidth=0.8, zorder=0)
        ax.text(
            1.01,
            k,
            text,
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=8,
            color="0.4",
        )
    ax.plot(overhead, grid[haar], label="Haar targets")
    ax.plot(overhead, grid[zz], label="ZZ workload")
    ax.set(xscale="log", ylim=(0, 0.85), ylabel=r"Selected strength $k$")
    ax.legend(loc="upper left")
    bx.plot(overhead, gap, color="0.3")
    bx.set(
        xscale="log",
        ylim=(0, None),
        xlabel=r"Fixed cost per pulse $b+\ell$",
        ylabel="Extra cost per ZZ\nblock, Haar selection",
    )
    fig.tight_layout()
    fig.savefig(
        STATIC / "calibration_staircase.svg",
        bbox_inches="tight",
        metadata={"Date": None},
    )
    plt.close(fig)


def plot_map(data: dict) -> None:
    """Selected strength over (b, ell) on logarithmic axes, for Haar and ZZ."""
    grid = np.array(data["grid"])
    axis = np.logspace(-3, 0.5, 300)
    total = np.add.outer(axis, axis)  # rows: ell, columns: b
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), sharey=True)
    for ax, (title, key) in zip(
        axes, (("Haar targets", "haar"), ("ZZ workload", "zz"))
    ):
        k = grid[
            selected(np.array(data[key]["time"]), np.array(data[key]["pulses"]), total)
        ]
        image = ax.pcolormesh(
            axis,
            axis,
            k,
            cmap="viridis",
            vmin=0,
            vmax=0.75,
            shading="auto",
            rasterized=True,
        )
        for level in sorted(set(k.ravel())):
            if label(level) in {"1/2", "1/3", "1/4", "1/6", "13/60"}:
                diagonal = np.flatnonzero((k == level).diagonal())
                if len(diagonal):
                    m = diagonal[len(diagonal) // 2]
                    ax.text(
                        axis[m],
                        axis[m],
                        f"$k={label(level)}$",
                        fontsize=7,
                        ha="center",
                        va="center",
                        color="white" if level < 0.4 else "black",
                    )
        ax.set(xscale="log", yscale="log", xlabel=r"Pulse overhead $b$", title=title)
    axes[0].set_ylabel(r"Local-layer cost $\ell$")
    colorbar = fig.colorbar(image, ax=axes, shrink=0.85, pad=0.02)
    colorbar.set_label(r"Selected strength $k$")
    fig.savefig(
        STATIC / "calibration_map.svg", bbox_inches="tight", metadata={"Date": None}
    )
    plt.close(fig)


def report(data: dict) -> None:
    """Print the numbers the calibration page quotes."""
    grid = np.array(data["grid"])
    for r, costs in data["landscape"].items():
        costs = np.array(costs)
        print(
            f"landscape b + ell = {r}: minimum at k = {data['fine'][costs.argmin()]:.4f}"
        )
    for key in ("haar", "zz"):
        time_, pulses = np.array(data[key]["time"]), np.array(data[key]["pulses"])
        idx, switches = hull(time_, pulses)
        print(key)
        for a, i in enumerate(idx):
            low = 0.0 if a == 0 else switches[a - 1]
            high = switches[a] if a < len(switches) else np.inf
            print(
                f"  k = {label(grid[i]):>5}  pulses {pulses[i]:.4f}  time {time_[i]:.4f}"
                f"  b + ell in [{low:.4f}, {high:.4f}]"
            )
    overhead = STAIRCASE
    haar = grid[
        selected(
            np.array(data["haar"]["time"]), np.array(data["haar"]["pulses"]), overhead
        )
    ]
    zz = grid[
        selected(np.array(data["zz"]["time"]), np.array(data["zz"]["pulses"]), overhead)
    ]
    shorter = overhead[haar < zz]
    equal = overhead[haar == zz]
    print(
        "Haar shorter than ZZ for b + ell in",
        shorter.min() if len(shorter) else None,
        shorter.max() if len(shorter) else None,
    )
    print("Haar equal to ZZ for b + ell up to", equal.max() if len(equal) else None)


def main() -> None:
    """Compute or load the data, then draw the figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot", action="store_true", help="replot from calibration_one.json only"
    )
    path = HERE / "calibration_one.json"
    if parser.parse_args().plot:
        data = json.loads(path.read_text())
    else:
        data = compute()
        path.write_text(json.dumps(data) + "\n")
    report(data)
    plot_landscape(data)
    plot_hull(data)
    plot_staircase(data)
    plot_map(data)


if __name__ == "__main__":
    main()
