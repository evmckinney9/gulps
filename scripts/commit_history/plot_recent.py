import csv
import subprocess
from datetime import datetime
from pathlib import Path

import lovelyplots
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import scienceplots

REPO_ROOT = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
)

data_set = "wsl_desktop"
data_dir = REPO_ROOT / "scripts/commit_history" / data_set

N = 10  # number of most-recent commits to show


def read_csv(path):
    """Read CSV into a list of dicts, parsing date and median columns."""
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            row["date"] = datetime.fromisoformat(row["date"])
            for k in ("isa1_median", "isa2_median", "isa3_median"):
                row[k] = float(row[k]) if row.get(k) else None
            rows.append(row)
    rows.sort(key=lambda r: r["date"])
    return rows


backlog = read_csv(data_dir / "backlog.csv")
recent = backlog[-N:]

isa_series = [
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

with plt.style.context(["ieee", "science", "use_mathtext"]):
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    for col, label, color in isa_series:
        ax.plot(
            [r["date"] for r in recent],
            [r[col] * 1000 for r in recent],
            "o-",
            markersize=3,
            linewidth=1,
            label=label,
            color=color,
            alpha=0.85,
            zorder=2,
        )

    for row in recent:
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

    ax.set_xlabel("Commit date")
    ax.set_ylabel("Median decomp.\\ time (ms)")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30, ha="right")
    ax.legend(fontsize=5.5, frameon=True)
    ax.grid(True, alpha=0.3)
    fig.subplots_adjust(top=0.5)
    plt.show()
    fig.savefig(data_dir / "recent_plot.pdf", bbox_inches="tight")
