import subprocess
from pathlib import Path

import lovelyplots
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots

REPO_ROOT = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
)

# pruned: the commits we care about (labeled)
data_set = "wsl_desktop"
df = pd.read_csv(REPO_ROOT / "scripts/commit_history" / data_set / "pruned_log.csv")
df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
df = df.sort_values("date").reset_index(drop=True)

# full backlog: scatter the extras as unlabeled background context
df_back = pd.read_csv(REPO_ROOT / "scripts/commit_history" / data_set / "backlog.csv")
df_back["date"] = pd.to_datetime(df_back["date"], utc=True).dt.tz_convert(None)
df_back = df_back.sort_values("date").reset_index(drop=True)

earliest = df["date"].min()
df_extra = df_back[
    ~df_back["commit"].isin(df["commit"]) & (df_back["date"] >= earliest)
].reset_index(drop=True)

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

    # background scatter: unlabeled commits not in pruned set
    for col, _, color in isa_series:
        ax.scatter(
            df_extra["date"],
            df_extra[col] * 1000,
            s=6,
            color=color,
            alpha=0.35,
            zorder=1,
            linewidths=0,
        )

    # foreground: pruned (labeled) commits as connected line
    for col, label, color in isa_series:
        ax.plot(
            df["date"],
            df[col] * 1000,
            "o-",
            markersize=3,
            linewidth=1,
            label=label,
            color=color,
            alpha=0.85,
            zorder=2,
        )

    # all labels rise above the top edge of the axes, clearing the data area entirely
    for _, row in df.iterrows():
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
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate(rotation=30, ha="right")
    ax.legend(fontsize=5.5, frameon=True)
    ax.grid(True, alpha=0.3)
    fig.subplots_adjust(top=0.5)
    plt.show()
