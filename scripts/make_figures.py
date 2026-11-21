"""
Draw the figures of the notes from the files under runs/.

    uv run --group figures scripts/make_figures.py

Every figure shows the spread over the prompts and not only a mean: a dot
per prompt, or a median with the middle half of the prompts around it. The
SVG files carry both a light and a dark set of colors and follow the
reader's system, so one file serves a light page and a dark one.
"""

import json
import re

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from minterp.tasks import TEMPLATES

OUT = "docs/figures"
SEEDS = ("seed0", "seed1")
VERSIONS = ("clean", "flipped", "neutral")

# light, dark. Marks keep their hue and change their lightness with the page.
INK, MUTED, RULE = ("#1f2328", "#e6edf3"), ("#656d76", "#9198a1"), ("#d0d7de", "#3d444d")
BLUE, RED, GREEN = ("#0b6fb8", "#58b0f0"), ("#c8500a", "#f0884a"), ("#0a8f68", "#3ccba0")
COLOR = {"clean": BLUE, "flipped": RED, "neutral": GREEN}
PAIRS = (INK, MUTED, RULE, BLUE, RED, GREEN)
SAYS = {
    "clean": "clean: the first name acts, so B is due",
    "flipped": "flipped: the second name acts, so A is due",
    "neutral": "neutral: a third name acts",
}

plt.rcParams.update({
    "svg.fonttype": "none", "svg.hashsalt": "minterp", "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 9,
    "text.color": INK[0], "axes.labelcolor": INK[0], "axes.edgecolor": MUTED[0],
    "xtick.color": MUTED[0], "ytick.color": MUTED[0], "xtick.labelcolor": INK[0], "ytick.labelcolor": INK[0],
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "legend.fontsize": 8.5,
})


def read(name):
    with open(f"runs/{name}.json") as f:
        return json.load(f)


def save(fig, name):
    path = f"{OUT}/{name}.svg"
    fig.savefig(path, metadata={"Date": None, "Creator": None}, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    with open(path) as f:
        svg = f.read()
    # matplotlib writes colors into style attributes. A rule with !important
    # beats those, so a dark page gets the second color of every pair.
    rules = "".join(
        f'[style*="{prop}: {light}"]{{{prop}:{dark} !important}}' for light, dark in PAIRS for prop in ("fill", "stroke")
    )
    # Hundredths of a point are enough for a path, and a third of its size.
    svg = re.sub(r' d="[^"]*"', lambda m: re.sub(r"(-?\d+\.\d\d)\d+", r"\1", m.group(0)), svg)
    svg = re.sub(r"(<svg[^>]*>)", rf"\1<style>@media (prefers-color-scheme: dark){{{rules}}}</style>", svg, count=1)
    with open(path, "w") as f:
        f.write(svg)
    print("wrote", path)


def short(template):
    """A template by the words that end it, on two lines of about equal length."""
    words = template.split("{S} ")[1].split()
    cut = min(range(1, len(words)), key=lambda i: abs(len(" ".join(words[:i])) - len(" ".join(words[i:]))))
    return " ".join(words[:cut]) + "\n" + " ".join(words[cut:])


def dots(ax, x, y, color, size=2.4):
    """
    One dot per value, as a single path of very short strokes with round
    ends. A marker per value would be an element per value, and a figure of
    four thousand dots would be half a megabyte.
    """
    gap = np.full(len(x), np.nan)
    xs, ys = np.column_stack([x, x + 1e-4, gap]).ravel(), np.column_stack([y, y, gap]).ravel()
    with plt.rc_context({"path.simplify": False}):
        ax.plot(xs, ys, color=color, alpha=0.6, linewidth=size, solid_capstyle="round")


def zero(ax, horizontal=True):
    (ax.axhline if horizontal else ax.axvline)(0, color=RULE[0], linewidth=0.8, zorder=0)


def behavior():
    """Step 1: one dot per prompt, the three versions of every template, both models."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)
    for ax, seed in zip(axes, SEEDS):
        run = read(f"read/behavior-{seed}")
        for t, entry in enumerate(run["templates"]):
            for v, version in enumerate(VERSIONS):
                y = np.array(entry[version]["logit_diff"])
                x = t + (v - 1) * 0.27 + rng.uniform(-0.09, 0.09, len(y))
                dots(ax, x, y, COLOR[version][0])
                ax.plot([t + (v - 1) * 0.27 - 0.11, t + (v - 1) * 0.27 + 0.11], [np.median(y)] * 2, color=INK[0], linewidth=1.4)
        zero(ax)
        ax.set_xticks(range(len(TEMPLATES)), [short(t) for t in TEMPLATES], fontsize=8)
        ax.set_title(f"{seed}: validation loss {run['model']['val_loss']:.3f}")
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("logit of B minus logit of A, last position")
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=4, color=COLOR[v][0], label=SAYS[v]) for v in VERSIONS]
    handles.append(plt.Line2D([], [], color=INK[0], linewidth=1.4, label="median of the 132 prompts"))
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.13))
    save(fig, "read-behavior")


def lens():
    """Step 1: the difference decoded after every block, median and middle half of all prompts."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1), sharey=True)
    for ax, seed in zip(axes, SEEDS):
        run = read(f"read/lens-{seed}")
        depth = len(run["templates"][0]["lens"]["clean"])
        for version in VERSIONS:
            y = np.concatenate([np.array(t["lens"][version]) for t in run["templates"]], axis=1)  # (depth, prompts)
            lo, mid, hi = np.percentile(y, [25, 50, 75], axis=1)
            ax.fill_between(range(depth), lo, hi, color=COLOR[version][0], alpha=0.18, linewidth=0)
            ax.plot(range(depth), mid, color=COLOR[version][0], linewidth=1.6, marker="o", markersize=3, label=SAYS[version])
        zero(ax)
        ax.set_xticks(range(depth), ["emb"] + [str(i) for i in range(1, depth)])
        ax.set_xlabel("stream decoded after block")
        ax.set_title(seed)
    axes[0].set_ylabel("logit of B minus logit of A")
    axes[0].legend(loc="upper left")
    save(fig, "read-lens")


def attribution():
    """Step 1: what every head and MLP writes into the difference on the clean prompts."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
    for ax, seed in zip(axes, SEEDS):
        a = read(f"read/lens-{seed}")["attribution"]["clean"]
        grid = np.concatenate([np.array(a["heads"]["mean"]), np.array(a["mlp"]["mean"])[:, None]], axis=1)  # (blocks, heads + 1)
        top = np.abs(grid).max()
        rows, cols = grid.shape
        for r in range(rows):
            for c in range(cols):
                v = grid[r, c]
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=(BLUE if v >= 0 else RED)[0], alpha=min(1, abs(v) / top) * 0.85, edgecolor="none"))
                if abs(v) >= 0.05:
                    ax.text(c, r, f"{v:+.2f}", ha="center", va="center", fontsize=7.5, color=INK[0])
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)
        ax.set_xticks(range(cols), [f"head {h}" for h in range(cols - 1)] + ["MLP"], fontsize=8)
        ax.set_yticks(range(rows), [str(i + 1) for i in range(rows)])
        ax.set_ylabel("block")
        ax.axvline(cols - 1.5, color=RULE[0], linewidth=0.8)
        ax.tick_params(length=0)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_title(f"{seed}: the parts sum to {grid.sum() + a['embed']['mean']:+.2f}")
    save(fig, "read-attribution")


def versions():
    """Step 1: the eight largest parts of each model, and what they write in each version."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1))
    for ax, seed in zip(axes, SEEDS):
        a = read(f"read/lens-{seed}")["attribution"]
        parts = []
        for b, row in enumerate(a["clean"]["heads"]["mean"]):
            parts += [(abs(v), f"block {b + 1} head {h}", ("heads", b, h)) for h, v in enumerate(row)]
        parts += [(abs(v), f"block {b + 1} MLP", ("mlp", b)) for b, v in enumerate(a["clean"]["mlp"]["mean"])]
        parts = sorted(parts, reverse=True)[:8]
        for i, (_, name, key) in enumerate(parts):
            for v, version in enumerate(VERSIONS):
                get = lambda stat: (a[version][key[0]][stat][key[1]][key[2]] if key[0] == "heads" else a[version][key[0]][stat][key[1]])
                y = i + (v - 1) * 0.22
                ax.plot([get("q25"), get("q75")], [y, y], color=COLOR[version][0], linewidth=2.2, alpha=0.45, solid_capstyle="butt")
                ax.plot([get("median")], [y], marker="o", markersize=4, color=COLOR[version][0], linestyle="", label=SAYS[version] if i == 0 else None)
        zero(ax, horizontal=False)
        ax.set_yticks(range(len(parts)), [name for _, name, _ in parts], fontsize=8)
        ax.set_ylim(len(parts) - 0.5, -0.5)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("written directly into logit of B minus logit of A")
        ax.set_title(seed)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(w_pad=2.5)
    save(fig, "read-versions")


if __name__ == "__main__":
    behavior()
    lens()
    attribution()
    versions()
