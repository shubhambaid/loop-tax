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
import matplotlib.ticker  # noqa: E402
import matplotlib.lines  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"

# reference palette (dataviz skill, light mode): categorical slots in fixed order, text inks
SERIES = ["#257de0", "#f47230", "#00b894"]
LABEL_LEFT = {"iq4_xs"}
LABEL_BELOW = {"q4_k_m"}
TEXT, TEXT2, GRID, SURFACE = "#191919", "#666666", "#e6e6e6", "#ffffff"


def style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(which="both", colors=TEXT2, labelsize=9)
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
            left = r["variant"] in LABEL_LEFT
            ax.annotate(r["variant"], (x, y), xytext=(-6 if left else 6, -11 if r["variant"] in LABEL_BELOW else 4), textcoords="offset points", fontsize=8,
                        color=TEXT, ha="right" if left else "left")
    ax.axvline(17.628, color=TEXT2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.annotate("NVFP4 reference, 17.6 GB\n(KL = 0 by definition)", (17.628, 1), xycoords=("data", "axes fraction"),
                xytext=(4, -4), textcoords="offset points", va="top", fontsize=8, color=TEXT2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}" if f"{v:.0e}"[0] in "25" else ""))
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="lower left")
    ax.text(0, -0.16, "Reference: ggml-org Inkling-2-Mini-NVFP4.gguf (17.6 GB). Error bars: standard error of the mean "
            "over scored tokens.", transform=ax.transAxes, fontsize=7.5, color=TEXT2)
    fig.tight_layout()
    out = ASSETS / "phase1_kl_vs_size.png"
    fig.savefig(out, facecolor=SURFACE)
    print(out)


def phase2():
    import json
    import glob
    rows = [r for r in read("phase2-scan") if r["role"] == "variant" and r["family"] == "block"]
    werr = {}
    for f in glob.glob(str(RESULTS / "phase2-scan-weight-error" / "*.json")):
        d = json.load(open(f))
        werr[d["variant"]] = d["weight_error"]["block_total"]["rel_err"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), dpi=150, sharex=True)
    for ax, qt in zip(axes, ("Q2_K", "Q3_K")):
        rs = sorted((r for r in rows if r["qtype"] == qt), key=lambda r: int(float(r["block"])))
        bs = [int(float(r["block"])) for r in rs]
        ys = [float(r["kld_mean_avg"]) / werr[r["variant"]] for r in rs]
        es = [float(r["kld_mean_avg_err"]) / werr[r["variant"]] for r in rs]
        cols = [SERIES[1] if 6 <= b <= 17 else SERIES[0] for b in bs]
        style(ax, f"{qt} on one block (down experts fall back to Q4_0)", "", "KL / relative weight error")
        ax.bar(bs, ys, color=cols, width=0.8, yerr=es, ecolor=TEXT2, error_kw={"elinewidth": 0.8})
        ax.grid(axis="x", visible=False)
    axes[-1].set_xlabel("physical block (orange: loop core 6-17, runs 3x)", color=TEXT2)
    axes[-1].set_xticks(range(24))
    fig.suptitle("Per-block sensitivity of Inkling 2 Mini, normalised by weight error", x=0.02, ha="left",
                 color=TEXT, fontsize=12, fontweight="bold")
    fig.text(0.02, 0.005, "Mean KL over 3 domains (16 chunks) vs NVFP4 reference, divided by the block's relative squared "
             "weight error. Screening run; bars: standard error over tokens.", fontsize=7.5, color=TEXT2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out = ASSETS / "phase2_scan_normalised.png"
    fig.savefig(out, facecolor=SURFACE)
    print(out)


CAT_COLOR = {"loop": SERIES[1], "prelude": "#6a4ce2", "coda": SERIES[0]}
CAT_LABEL = {"loop": "loop (reused every pass)", "prelude": "prelude (injected into every pass)",
             "coda": "coda (runs once, control)"}


def phase3():
    rows = [r for r in read("phase3-recipes") if r["role"] == "variant"]
    fams = {"uniform": ("same type in every block", "#191919"), "loop-hi": ("loop blocks NVFP4, others lower", SERIES[1]),
            "edge-hi": ("prelude + coda NVFP4, loop lower", SERIES[0]), "guided": ("guided by the experts scan", SERIES[2])}
    fam_of = {}
    import yaml
    for v in yaml.safe_load(open(ROOT / "configs" / "phase3_recipes.yaml"))["variants"]:
        fam_of[v["id"]] = v["family"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    style(ax, "Loop-aware vs uniform mixed precision (Inkling 2 Mini)", "file size (GB)", "mean KL vs NVFP4 reference")
    for fam, (lab, c) in fams.items():
        rs = sorted((r for r in rows if fam_of.get(r["variant"]) == fam), key=lambda r: float(r["file_gb"]))
        if not rs:
            continue
        ax.plot([float(r["file_gb"]) for r in rs], [float(r["kld_mean_avg"]) for r in rs], color=c, lw=2, marker="o",
                ms=6, mec=SURFACE, mew=1.5, label=lab)
    other = [(r, "stock llama.cpp recipe") for r in read("phase1-baselines") if r["role"] == "variant" and float(r["file_gb"]) < 20]
    other += [(r, "community GGUF") for r in read("phase1-community") if r["role"] == "variant"]
    for r, kind in other:
        ax.plot(float(r["file_gb"]), float(r["kld_mean_avg"]), marker="s" if kind == "community GGUF" else "^", ms=6,
                color=TEXT2, ls="none")
        ax.annotate(r["variant"], (float(r["file_gb"]), float(r["kld_mean_avg"])), xytext=(5, -3),
                    textcoords="offset points", fontsize=7, color=TEXT2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}" if f"{v:.0e}"[0] in "25" else ""))
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="upper right")
    fig.text(0.01, 0.01, "Grey: stock llama.cpp recipes (triangles) and community GGUFs (squares) for context. "
             "Down experts NVFP4 and attention Q8_0 in every Phase 3 recipe. Screening run, 16 chunks.", fontsize=7, color=TEXT2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = ASSETS / "phase3_pareto.png"
    fig.savefig(out, facecolor=SURFACE)
    print(out)


def headline():
    import json
    fits = json.load(open(RESULTS / "phase4-passes" / "alpha_fit.json"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), dpi=150, gridspec_kw={"width_ratios": [1, 1, 1.15]})
    import math
    for ax, qt in zip(axes[:2], ("Q2_K", "Q4_0")):
        style(ax, f"one block at {qt}", "loop passes N", "KL vs reference at the same N")
        # de-overlap the N = 1 labels in log space
        sub = sorted((r for r in fits if r["qtype"] == qt), key=lambda r: r["kl_mean_by_passes"]["1"])
        pos, last = {}, None
        for r in sub:
            y = math.log10(r["kl_mean_by_passes"]["1"])
            y = y if last is None or y - last > 0.035 else last + 0.035
            pos[r["variant"]], last = 10 ** y, y
        for r in sub:
            ns = sorted(int(n) for n in r["kl_mean_by_passes"])
            ys = [r["kl_mean_by_passes"][str(n)] for n in ns]
            tr = [i for i, n in enumerate(ns) if n <= 5]
            c = CAT_COLOR[r["category"]]
            ax.plot([ns[i] for i in tr], [ys[i] for i in tr], color=c, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5)
            ex = [i for i, n in enumerate(ns) if n > 5]
            ax.plot([ns[i] for i in ex], [ys[i] for i in ex], ls="none", marker="o", ms=5, mfc=SURFACE, mec=c, mew=1.5)
            ax.annotate(f"blk {r['block']}", (ns[0], ys[0]), xytext=(0.93, pos[r["variant"]]), textcoords="data",
                        fontsize=7, color=TEXT, va="center", ha="right")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(0.72, 6.6)
        ax.set_xticks([1, 2, 3, 4, 5, 6]); ax.set_xticklabels(["1", "2", "3", "4", "5", "6*"])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}" if f"{v:.0e}"[0] in "25" else ""))
    ax = axes[2]
    style(ax, "fitted exponent, KL ~ N^alpha", "alpha (N = 1-5, 95% bootstrap CI)", "")
    order = sorted(fits, key=lambda r: ({"loop": 0, "prelude": 1, "coda": 2}[r["category"]], r["qtype"], r["block"]))
    for i, r in enumerate(order):
        c = CAT_COLOR[r["category"]]
        lo, hi = r["alpha_ci95"]
        ax.plot([lo, hi], [i, i], color=c, lw=2)
        ax.plot([r["alpha"]], [i], "o", color=c, ms=6, mec=SURFACE, mew=1.5)
    for x, lab in ((0, "no effect"), (1, "independent"), (2, "coherent")):
        ax.axvline(x, color=TEXT2, lw=0.8, ls=(0, (3, 3)))
        ax.text(x, len(order) - 0.3, lab, fontsize=7, color=TEXT2, ha="center")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"blk {r['block']} {r['qtype']}" for r in order], fontsize=8)
    ax.set_xlim(-0.5, 2.3); ax.set_ylim(-0.7, len(order) + 0.2); ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    handles = [matplotlib.lines.Line2D([], [], color=CAT_COLOR[k], lw=2, marker="o", label=v) for k, v in CAT_LABEL.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=TEXT)
    fig.suptitle("Same block, more loop passes: TML-d24-varloop-research", x=0.01, ha="left", color=TEXT,
                 fontsize=12, fontweight="bold")
    fig.text(0.01, 0.005, "KL averaged over wikitext (8 chunks) and code (4 chunks). 6* = never trained, hollow, not "
             "in the fit. Screening run; held-out confirmation separate.", fontsize=7.5, color=TEXT2)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    out = ASSETS / "headline_passes.png"
    fig.savefig(out, facecolor=SURFACE)
    print(out)


def grid():
    import json
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
    names = [("reference", "NVFP4 (reference)"), ("uniform-iq4_xs", "IQ4_XS experts"),
             ("uniform-q3_k", "Q3_K experts"), ("uniform-q2_k", "Q2_K experts")]
    for k, (v, lab) in enumerate(names):
        xs, kl, tg = [], [], []
        for n in range(1, 6):
            d = json.load(open(RESULTS / "phase4-passes" / f"{v}-p{n}.json"))
            xs.append(n)
            kl.append(sum(m["kld_mean"] for m in d["domains_vs3"].values()) / 2 if n != 3 or v != "reference" else None)
            tg.append(d["speed"]["tg_tps"])
        c = (["#191919"] + SERIES)[k]
        segs, cur = [], []
        for x, y in zip(xs, kl):
            if y:
                cur.append((x, y))
            elif cur:
                segs.append(cur); cur = []
        segs.append(cur)
        for j, seg in enumerate(segs):
            axes[0].plot([p[0] for p in seg], [p[1] for p in seg], color=c, lw=2, marker="o", ms=5, mec=SURFACE,
                         mew=1.5, label=lab if j == 0 else None)
        axes[1].plot([t for t, y in zip(tg, kl) if y], [y for y in kl if y], color=c, lw=2, marker="o", ms=5,
                     mec=SURFACE, mew=1.5, label=lab)
        for t, y, n in zip(tg, kl, xs):
            if y:
                axes[1].annotate(f"{n}p", (t, y), xytext=(3, 3), textcoords="offset points", fontsize=7, color=TEXT2)
    style(axes[0], "distance from the 3-pass reference", "loop passes N", "KL vs NVFP4 reference at N = 3")
    style(axes[1], "quality vs decode speed", "decode tokens/s (M4 Pro, tg128)", "KL vs NVFP4 reference at N = 3")
    for ax in axes:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].legend(frameon=False, fontsize=8, labelcolor=TEXT)
    axes[0].set_xticks([1, 2, 3, 4, 5])
    axes[0].text(4.5, axes[0].get_ylim()[0] * 1.15, "black: reference at N = 3 is\nKL 0 by definition (not shown)",
                 fontsize=7, color=TEXT2, ha="center")
    fig.suptitle("Bits x loop passes (varloop checkpoint; gate/up experts at the stated type, down experts NVFP4)",
                 x=0.01, ha="left", color=TEXT, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = ASSETS / "grid_bits_passes.png"
    fig.savefig(out, facecolor=SURFACE)
    print(out)


def bf16():
    import json
    import glob
    rows = []
    for f in glob.glob(str(RESULTS / "bf16check" / "*.json")):
        d = json.load(open(f))
        if d["role"] != "variant":
            continue
        dm = d["domains"]
        kl = sum(v["kld_mean"] for v in dm.values()) / len(dm)
        err = sum(v["kld_mean_err"] ** 2 for v in dm.values()) ** 0.5 / len(dm)
        rows.append((d["variant"], d["weights"]["file_gb"], kl, err))
    rows.sort(key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)
    style(ax, "Distance from the BF16 weights (Inkling 2 Mini)", "mean KL vs BF16 GGUF (3 domains, 16 chunks)", "")
    ys = range(len(rows))
    ax.barh(list(ys), [r[2] for r in rows], xerr=[r[3] for r in rows], color=[SERIES[1] if r[0] == "ggml-org-nvfp4" else SERIES[0] for r in rows],
            height=0.6, ecolor=TEXT2)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{r[0]} ({r[1]:.1f} GB)" for r in rows], fontsize=8)
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    fig.text(0.01, 0.01, "Orange: the NVFP4 GGUF (original NVFP4 experts). BF16 = released NVFP4/MXFP8 values upcast. "
             "-from-bf16 / -from-nvfp4: quantization source.", fontsize=7, color=TEXT2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = ASSETS / "bf16_check.png"
    fig.savefig(out, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    for name in sys.argv[1:] or ["phase1"]:
        globals()[name]()
