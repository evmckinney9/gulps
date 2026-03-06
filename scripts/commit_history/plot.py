"""Benchmark history plot: full history (top) + zoomed recent commits (bottom)."""

import csv
import subprocess
from datetime import date, datetime
from pathlib import Path

import lovelyplots  # noqa: F401  (registers styles as a side-effect)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

REPO_ROOT = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
)
DATA_DIR = REPO_ROOT / "scripts/commit_history/wsl_desktop"

N_RECENT = 15  # number of most-recent commits shown in the bottom panel

ISA_SERIES = [
    ("isa1_median", r"isa1  [$\sqrt[4]{\mathrm{iSwap}}$]", "tab:blue"),
    (
        "isa2_median",
        r"isa2  [$\sqrt[3]{\mathrm{iSwap}}$, $\sqrt[4]{\mathrm{iSwap}}$]",
        "tab:orange",
    ),
    (
        "isa3_median",
        r"isa3  [$\sqrt[4]{\mathrm{iSwap}}$, $\sqrt[3]{\mathrm{iSwap}}$, $\sqrt[2]{\mathrm{iSwap}}$]",
        "tab:green",
    ),
]

_LATEX_ESCAPES = str.maketrans(
    {"#": r"\#", "$": r"\$", "%": r"\%", "&": r"\&", "_": r"\_"}
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            row["date"] = datetime.fromisoformat(row["date"])
            for k in ("isa1_median", "isa2_median", "isa3_median"):
                row[k] = float(row[k]) if row.get(k) else None
            rows.append(row)
    rows.sort(key=lambda r: r["date"])
    return rows


# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------


def _plot_lines(ax, rows: list[dict], *, label: bool = True) -> None:
    for col, lbl, color in ISA_SERIES:
        ax.plot(
            [r["date"] for r in rows],
            [r[col] * 1000 for r in rows],
            "o-",
            markersize=3,
            linewidth=1,
            label=lbl if label else None,
            color=color,
            alpha=0.85,
            zorder=2,
        )


def _plot_scatter(ax, rows: list[dict]) -> None:
    for col, _, color in ISA_SERIES:
        ax.scatter(
            [r["date"] for r in rows],
            [r[col] * 1000 for r in rows],
            s=6,
            color=color,
            alpha=0.35,
            zorder=1,
            linewidths=0,
        )


def _annotate_commits(ax, rows: list[dict]) -> None:
    for row in rows:
        ax.annotate(
            row["subject"][:30].translate(_LATEX_ESCAPES),
            xy=(row["date"], 1),
            xycoords=("data", "axes fraction"),
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=3.5,
            rotation=90,
            ha="center",
            va="bottom",
            color="gray",
            alpha=0.85,
            clip_on=False,
        )


def _style_axes(ax, *, date_fmt: str, locator) -> None:
    ax.set_ylabel("Median time (ms)")
    ax.set_xlabel("Commit date")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
    ax.xaxis.set_major_locator(locator)
    ax.tick_params(axis="x", labelrotation=30, labelsize=5)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------


def make_plot() -> None:
    pruned = _read_csv(DATA_DIR / "pruned_log.csv")
    backlog = _read_csv(DATA_DIR / "backlog.csv")
    recent = backlog[-N_RECENT:]

    pruned_commits = {r["commit"] for r in pruned}
    earliest = pruned[0]["date"]
    extra = [
        r
        for r in backlog
        if r["commit"] not in pruned_commits and r["date"] >= earliest
    ]

    zoom_start = recent[0]["date"]
    zoom_end = recent[-1]["date"]

    with plt.style.context(["ieee", "science", "use_mathtext"]):
        fig, (ax_top, ax_bot) = plt.subplots(
            2,
            1,
            figsize=(3.5, 5.5),
            gridspec_kw={"height_ratios": [2, 1.4], "hspace": 0.55},
        )

        # --- top: full history ---
        _plot_scatter(ax_top, extra)
        _plot_lines(ax_top, pruned, label=True)
        _annotate_commits(ax_top, pruned)
        ax_top.axvspan(zoom_start, zoom_end, alpha=0.08, color="gray", zorder=0)
        _style_axes(ax_top, date_fmt="%b %Y", locator=mdates.MonthLocator())
        ax_top.legend(fontsize=5, frameon=True, loc="upper right")
        ax_top.set_title("Full history", fontsize=6, pad=2)
        fig.subplots_adjust(top=0.82)

        # --- bottom: recent commits ---
        _plot_lines(ax_bot, recent, label=False)
        _annotate_commits(ax_bot, recent)
        _style_axes(ax_bot, date_fmt="%b %d", locator=mdates.AutoDateLocator())
        ax_bot.set_title(f"Recent ({N_RECENT} commits)", fontsize=6, pad=2)

        # --- dashed verticals at zoom boundaries on both panels ---
        for x_date in (zoom_start, zoom_end):
            for ax in (ax_top, ax_bot):
                ax.axvline(
                    x_date,
                    color="gray",
                    alpha=0.4,
                    linewidth=0.6,
                    linestyle="--",
                    clip_on=False,
                )

        out = DATA_DIR / f"history.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.show()


if __name__ == "__main__":
    make_plot()
