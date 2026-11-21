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


ROLE_NAMES = {"A": "first\nname", "B": "second\nname", "S": "acting\nname", "S+1": "the token\nafter it", "end": "last\nposition"}


def probe_run(seed):
    run = read(f"probe/probe-{seed}")
    roles, targets = run["roles"], run["targets"]
    pick = lambda rule, target, role: np.array(run[rule]["accuracy"])[:, :, :, roles.index(role), targets.index(target)]
    return run, pick  # pick(...) is (folds, versions, depth)


def percent(ax, k):
    ax.set_ylim(-3, 103)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.axhline(100 / k, color=RULE[0], linewidth=0.8, zorder=0)


def depth_axis(ax, depth):
    ax.set_xticks(range(depth), ["emb"] + [str(i) for i in range(1, depth)])
    ax.set_xlabel("stream after block")


def probe_end():
    """Step 2: the due name at the last position, by a trained reader and by the lens."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), sharey=True)
    for ax, seed in zip(axes, SEEDS):
        run, pick = probe_run(seed)
        reader, lens_acc = 100 * pick("new", "due", "end"), 100 * np.array(run["new"]["lens"])
        depth = reader.shape[2]
        for v, version in enumerate(VERSIONS[:2]):
            color = COLOR[version][0]
            for d in range(depth):
                dots(ax, d + rng.uniform(-0.12, 0.12, reader.shape[0]), reader[:, v, d], color, size=3.2)
            ax.plot(range(depth), reader[:, v].mean(0), color=color, linewidth=1.6, label=f"reader, {version} prompts")
            ax.plot(range(depth), lens_acc[:, v].mean(0), color=color, linewidth=1.3, linestyle=(0, (4, 2)), label=f"lens, {version} prompts")
        percent(ax, len(run["names"]))
        depth_axis(ax, depth)
        ax.set_title(seed)
    axes[0].set_ylabel("due name right, percent of test prompts")
    handles, names = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=3.5, color=MUTED[0]))
    names.append("one held-out pattern")
    fig.legend(handles, names, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1), columnspacing=1.2, handlelength=1.8)
    fig.tight_layout()
    save(fig, "probe-end")


def probe_where():
    """Step 2: where the due name can be read: depth by position, the worse of the two versions."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for ax, seed in zip(axes, SEEDS):
        run, pick = probe_run(seed)
        roles = run["roles"]
        grid = np.stack([100 * pick("new", "due", r).mean(0).min(0) for r in roles], axis=1)  # (depth, roles)
        for d in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                ax.add_patch(Rectangle((c - 0.5, d - 0.5), 1, 1, facecolor=BLUE[0], alpha=0.85 * grid[d, c] / 100, edgecolor="none"))
                ax.text(c, d, f"{grid[d, c]:.0f}", ha="center", va="center", fontsize=8, color=INK[0])
        ax.set_xlim(-0.5, len(roles) - 0.5)
        ax.set_ylim(grid.shape[0] - 0.5, -0.5)
        ax.set_xticks(range(len(roles)), [ROLE_NAMES[r] for r in roles], fontsize=8)
        ax.set_yticks(range(grid.shape[0]), ["emb"] + [str(i) for i in range(1, grid.shape[0])])
        ax.set_ylabel("stream after block")
        ax.tick_params(length=0)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_title(f"{seed}: due name right, percent, the worse version")
    fig.tight_layout(w_pad=2.5)
    save(fig, "probe-where")


def probe_controls():
    """Step 2: what a reader scores on pairs it trained on, for the due name and for a made-up label."""
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.6), sharey=True)
    for row, seed in zip(axes, SEEDS):
        run, pick = probe_run(seed)
        depth = len(run["new"]["lens"][0][0])
        ax = row[0]
        for rule, style, says in (("seen", "-", "tested on pairs it trained on"), ("new", (0, (4, 2)), "tested on new pairs")):
            ax.plot(range(depth), 100 * pick(rule, "due", "S").mean(0).min(0), color=BLUE[0], linewidth=1.6, linestyle=style, marker="o", markersize=3, label=says)
        ax.set_title(f"{seed}: the due name at the acting name")
        ax = row[1]
        for role, color in (("B", RED[0]), ("S", BLUE[0]), ("end", GREEN[0])):
            ax.plot(range(depth), 100 * pick("seen", "control", role).mean((0, 1)), color=color, linewidth=1.6, marker="o", markersize=3,
                    label=f"at the {ROLE_NAMES[role].replace(chr(10), ' ')}, pairs it trained on")
        ax.plot(range(depth), 100 * pick("new", "control", "B").mean((0, 1)), color=RED[0], linewidth=1.3, linestyle=(0, (4, 2)), label="at the second name, new pairs")
        ax.set_title(f"{seed}: a made-up label, a random name per pair")
        for ax in row:
            percent(ax, len(run["names"]))
            depth_axis(ax, depth)
        row[0].set_ylabel("right, percent of test prompts")
    axes[0][0].legend(loc="upper left")
    axes[0][1].legend(loc="upper right", fontsize=7.5)
    fig.tight_layout(h_pad=1.6)
    save(fig, "probe-controls")


def cells(ax, grid, columns, rows, floor=0.03, top=None):
    """A table of numbers as shaded cells: blue above zero, red below, strength by size."""
    top = top or np.abs(grid).max()
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            v = grid[r, c]
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=(BLUE if v >= 0 else RED)[0], alpha=min(1, abs(v) / top) * 0.85, edgecolor="none"))
            if abs(v) >= floor:
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color=INK[0])
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(grid.shape[0] - 0.5, -0.5)
    ax.set_xticks(range(len(columns)), columns, fontsize=8)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)


def patch_mean(seed, pick):
    run = read(f"patch/patch-{seed}")
    return run, np.mean([pick(t) for t in run["templates"]], axis=0), np.array([pick(t) for t in run["templates"]])


def patch_travel():
    """Step 3: where the acting name and the second name are, by depth and position."""
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6))
    depths = ["emb"] + [str(i) for i in range(1, 9)]
    for col, seed in enumerate(SEEDS):
        run, grid, _ = patch_mean(seed, lambda t: t["travel"]["carry"])
        keep = [run["travel groups"].index(g) for g in ("A", "B", "S", "S+1", "between", "end")]
        cells(axes[0][col], grid[:, keep], ["first\nname", "second\nname", "acting\nname", "token\nafter it", "tokens\nbetween", "last\nposition"], depths, top=1)
        axes[0][col].set_title(f"{seed}: who acts. Clean stream into the flipped run")
        run, grid, _ = patch_mean(seed, lambda t: t["name"])
        cells(axes[1][col], grid, ["second\nname", "up to the\nacting name", "acting\nname", "token\nafter it", "tokens\nbetween", "last\nposition"], depths, top=1)
        axes[1][col].set_title(f"{seed}: the second name. Stream with another name into the clean run")
        for ax in (axes[0][col], axes[1][col]):
            ax.set_ylabel("stream after block")
    fig.tight_layout(h_pad=1.8, w_pad=2.0)
    save(fig, "patch-travel")


def patch_parts():
    """Step 3: every head and MLP moved from the clean prompt into the flipped run."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
    for ax, seed in zip(axes, SEEDS):
        _, grid, _ = patch_mean(seed, lambda t: t["parts"]["carry"]["all"])
        cells(ax, grid, [f"head {h}" for h in range(grid.shape[1] - 1)] + ["MLP"], [str(i + 1) for i in range(grid.shape[0])], top=1)
        ax.axvline(grid.shape[1] - 1.5, color=RULE[0], linewidth=0.8)
        ax.set_ylabel("block")
        ax.set_title(f"{seed}: share of the gap that comes along")
    fig.tight_layout(w_pad=2.5)
    save(fig, "patch-parts")


def patch_choice():
    """Step 3: the same parts replaced in four ways, on the clean run."""
    says = {"flipped": "the flipped prompt's", "neutral": "the neutral prompt's", "mean": "the average over prompts", "zero": "nothing"}
    color = {"flipped": RED[0], "neutral": GREEN[0], "mean": MUTED[0], "zero": INK[0]}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    for ax, seed in zip(axes, SEEDS):
        run = read(f"patch/patch-{seed}")
        per = {c: np.array([t["choice"][c] for t in run["templates"]]) for c in run["choices"]}  # (patterns, blocks, heads + 1)
        size = np.max([np.abs(per[c].mean(0)) for c in run["choices"]], axis=0)
        size[0, -1] = 0  # block 1's MLP carries the token itself and would set the scale
        order = np.dstack(np.unravel_index(np.argsort(-size, axis=None), size.shape))[0][:8]
        for i, (b, h) in enumerate(order):
            for j, c in enumerate(run["choices"]):
                y = i + (j - 1.5) * 0.19
                dots(ax, per[c][:, b, h], np.full(per[c].shape[0], y), color[c], size=3.0)
                ax.plot([per[c][:, b, h].mean()], [y], marker="|", markersize=9, markeredgewidth=1.6, color=color[c], linestyle="", label=says[c] if i == 0 else None)
        zero(ax, horizontal=False)
        heads = size.shape[1] - 1
        ax.set_yticks(range(len(order)), [f"block {b + 1} " + ("MLP" if h == heads else f"head {h}") for b, h in order], fontsize=8)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("change of the logit difference on the clean prompts")
        ax.set_title(seed)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.13), title="the part's output replaced by", title_fontsize=8.5)
    fig.tight_layout(w_pad=2.5)
    save(fig, "patch-choice")


if __name__ == "__main__":
    behavior()
    lens()
    attribution()
    versions()
    probe_end()
    probe_where()
    probe_controls()
    patch_travel()
    patch_parts()
    patch_choice()
