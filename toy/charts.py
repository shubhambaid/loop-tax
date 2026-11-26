import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, INK2, SURF = "#191919", "#666666", "#ffffff"
BLUE, ORANGE, GRID = "#257de0", "#f47230", "#e6e6e6"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "figure.facecolor": SURF})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# ---------- 1. Architecture: 24 stored layers, 48 visits ----------
def arch():
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150)
    ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
    fig.text(0.05, 0.92, "Inkling 2 Mini: 24 stored layers, 48 layer visits per token",
             fontsize=19, weight="bold")
    fig.text(0.05, 0.875, "Layers 6 to 17 share weights and run 3 times. One rounding error, applied three times.",
             fontsize=12.5, color=INK2)

    def box(x, w, label, color, y=24, h=12, txt="#ffffff"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=color, ec=SURF, lw=2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                color=txt, fontsize=12, weight="bold")

    box(3, 16, "Layers 0-5\nrun once", "#9c9c9c")
    for i, x in enumerate([24, 44, 64]):
        box(x, 16, f"Layers 6-17\npass {i + 1}", BLUE)
        if i:
            ax.annotate("", xy=(x, 30), xytext=(x - 4, 30),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6))
    ax.annotate("", xy=(24, 30), xytext=(19, 30), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6))
    ax.annotate("", xy=(85, 30), xytext=(80, 30), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6))
    box(85, 13, "Layers\n18-23\nrun once", "#9c9c9c")

    # shared-weight bracket
    ax.plot([24, 24, 80, 80], [40, 43, 43, 40], color=BLUE, lw=2)
    ax.text(52, 45, "same weights W  (quantized once, error reused 3x)",
            ha="center", color=BLUE, fontsize=12.5, weight="bold")

    ax.text(52, 16, r"Unique layers:  $W_q = W + \epsilon_1,\ W + \epsilon_2,\ W + \epsilon_3$   errors independent, partly cancel",
            ha="center", fontsize=12, color=INK2)
    ax.text(52, 10, r"Looped layers:  $W_q = W + \epsilon,\ W + \epsilon,\ W + \epsilon$          same error, adds up coherently",
            ha="center", fontsize=12, color=INK)
    fig.text(0.05, 0.04, "Architecture per Thinking Machines / Hugging Face model docs. Diagram: Shubham Baid",
             fontsize=9.5, color=INK2)
    fig.savefig("../assets/1_architecture.png"); plt.close(fig)


# ---------- 2. Toy: error vs depth ----------
def depth():
    r = json.load(open("results.json"))["depth"]
    xs = list(range(1, 13))
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150)
    fig.subplots_adjust(left=0.08, right=0.84, top=0.80, bottom=0.14)
    for key, c, lab in [("distinct", "#9c9c9c", "12 unique layers"),
                        ("looped", BLUE, "4 layers looped 3x")]:
        m = r[f"{key}_mean"]; s = r[f"{key}_std"]
        ax.fill_between(xs, [a - b for a, b in zip(m, s)], [a + b for a, b in zip(m, s)],
                        color=c, alpha=0.15, lw=0)
        ax.plot(xs, m, color=c, lw=2.5, marker="o", ms=7, mec=SURF, mew=2)
        ax.text(12.3, m[-1], f"{lab}\n{m[-1]*100:.1f}%", va="center", fontsize=12, color=INK)
    for x in (4.5, 8.5):
        ax.axvline(x, color=GRID, lw=1, ls="--")
    ax.text(2.5, 0.185, "pass 1", ha="center", color=INK2, fontsize=11)
    ax.text(6.5, 0.185, "pass 2", ha="center", color=INK2, fontsize=11)
    ax.text(10.5, 0.185, "pass 3", ha="center", color=INK2, fontsize=11)
    ax.set_xlim(0.5, 12.2); ax.set_ylim(0, 0.2); ax.set_xticks(xs); ax.set_yticks([0, .05, .1, .15, .2])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Layer visits (depth)", fontsize=12)
    ax.set_ylabel("Hidden-state error vs FP32", fontsize=12)
    style(ax)
    fig.text(0.08, 0.92, "Same depth, same 4-bit quantization: looping ends with ~30% more error",
             fontsize=18, weight="bold")
    fig.text(0.08, 0.865, "Identical until the loop restarts, then the reused error stacks up. Mean of 20 seeds, band = 1 std.",
             fontsize=12, color=INK2)
    fig.text(0.08, 0.03, "TOY SIMULATION: random pre-norm residual MLP blocks, d=256, RTN per-channel int4. Not Inkling 2 Mini results.",
             fontsize=9.5, color=INK2)
    fig.savefig("../assets/2_toy_depth.png"); plt.close(fig)


# ---------- 3. Toy: where to spend bits ----------
def budget():
    rows = json.load(open("grid.json"))
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.14)
    uni = sorted([r for r in rows if r["loop_bits"] == r["edge_bits"]], key=lambda r: r["avg_bits"])
    loop_rich = sorted([r for r in rows if r["loop_bits"] == r["edge_bits"] + 1], key=lambda r: r["avg_bits"])
    edge_rich = sorted([r for r in rows if r["edge_bits"] == r["loop_bits"] + 1], key=lambda r: r["avg_bits"])
    for data, c, lab in [(edge_rich, ORANGE, "edges +1 bit"),
                         (uni, "#9c9c9c", "uniform"),
                         (loop_rich, BLUE, "loop +1 bit")]:
        x = [r["avg_bits"] for r in data]; y = [r["err"] for r in data]
        ax.plot(x, y, color=c, lw=2.5, marker="o", ms=8, mec=SURF, mew=2)
        ax.text(x[-1] + 0.12, y[-1], lab, ha="left", va="center", fontsize=12, color=INK)
    ax.set_yscale("log")
    ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(2.8, 8.9)
    a = next(r for r in loop_rich if r["avg_bits"] == 4.5); b = next(r for r in edge_rich if r["avg_bits"] == 4.5)
    ax.annotate(f"4.5 bits:  loop 5 / edges 4 = {a['err']*100:.1f}%\n             loop 4 / edges 5 = {b['err']*100:.1f}%",
                xy=(4.5, a["err"]), xytext=(4.75, 0.42), fontsize=11.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    ax.set_xlabel("Average bits per stored weight (lower = smaller file)", fontsize=12)
    ax.set_ylabel("Final hidden-state error (log)", fontsize=12)
    style(ax)
    fig.text(0.08, 0.92, "Same file size, very different error: spend your bits on the loop",
             fontsize=18, weight="bold")
    fig.text(0.08, 0.865, "Inkling-shaped toy: 24 blocks, blocks 6-17 looped 3x. Each point is one (loop bits, edge bits) mix.",
             fontsize=12, color=INK2)
    fig.text(0.08, 0.03, "TOY SIMULATION: random residual MLP blocks, RTN per-channel quantization, 10 seeds. Not Inkling 2 Mini results.",
             fontsize=9.5, color=INK2)
    fig.savefig("../assets/3_toy_bit_budget.png"); plt.close(fig)


if __name__ == "__main__":
    import sys
    for name in sys.argv[1:]:
        globals()[name]()
