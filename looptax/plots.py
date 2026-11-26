"""Static charts (PNG) for the lab notebook and README. The website draws its own interactive charts from the same CSVs.

    python -m looptax.plots phase1
"""

from __future__ import annotations

import csv
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"

# reference palette (dataviz skill, light mode): categorical slots in fixed order, text inks
SERIES = ["#257de0", "#f47230", "#00b894"]
TEXT, TEXT2, GRID, SURFACE = "#191919", "#666666", "#e6e6e6", "#ffffff"


def style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, color=TEXT2, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT2, fontsize=10)


def read(exp: str) -> list[dict]:
    p = RESULTS / exp / "summary.csv"
    return list(csv.DictReader(open(p))) if p.exists() else []


def phase1():
    rows = [r for r in read("phase1-baselines") if r["role"] == "variant"]
    comm = [r for r in read("phase1-community") if r["role"] == "variant"]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    style(ax, "Uniform quants of Inkling 2 Mini: quality vs size",
          "file size (GB)", "mean KL divergence vs NVFP4 reference (avg of 3 domains)")
    for rs, color, label in ((rows, SERIES[0], "uniform, made by us (ggml-org imatrix)"),
                             (comm, SERIES[1], "community GGUFs as published")):
        if not rs:
            continue
        xs = [float(r["file_gb"]) for r in rs]
        ys = [float(r["kld_mean_avg"]) for r in rs]
        es = [float(r["kld_mean_avg_err"]) for r in rs]
        ax.errorbar(xs, ys, yerr=es, fmt="o", ms=7, color=color, mec=SURFACE, mew=2, ecolor=color, elinewidth=1,
                    capsize=0, label=label, zorder=3)
        for x, y, r in zip(xs, ys, rs):
            ax.annotate(r["variant"], (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8, color=TEXT)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    ax.text(0, -0.16, "Reference: ggml-org Inkling-2-Mini-NVFP4.gguf (17.6 GB). Error bars: standard error of the mean "
            "over scored tokens.", transform=ax.transAxes, fontsize=7.5, color=TEXT2)
    fig.tight_layout()
    out = ASSETS / "phase1_kl_vs_size.png"
    fig.savefig(out, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    for name in sys.argv[1:] or ["phase1"]:
        globals()[name]()
