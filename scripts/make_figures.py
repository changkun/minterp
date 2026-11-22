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


def circuit_run(seed):
    run = read(f"circuit/circuit-{seed}")
    name = lambda x: f"block {x['block']} head {x['head']}"
    avg = lambda pick: float(np.mean([pick(t) for t in run["templates"]]))
    return run, name, avg


def circuit_diagram():
    """Step 4: which position each head reads from, and whether it raises or lowers the name it reads."""
    from matplotlib.patches import FancyArrowPatch

    xs = {"A": 0, "B": 1, "stop": 2, "S": 3, "end": 4}
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.4), sharex=True, sharey=True)
    for row, seed in zip(axes, SEEDS):
        run, name, avg = circuit_run(seed)
        for ax, version, says in zip(row, ("clean", "flipped"), ("the first name acts", "the second name acts")):
            acting = "A" if version == "clean" else "B"
            slot = {}  # heads of one block are drawn one above the other
            for x in run["heads"]:
                slot.setdefault(x["block"], []).append(x["head"])
            for x in run["heads"]:
                looks = {k: avg(lambda t: t["attention"][name(x)][version][k]) for k in ("A", "B", "stop", "S")}
                source = max(looks, key=looks.get)
                if looks[source] < 0.25:
                    continue
                if x["acts at"] == "S":
                    color = MUTED[0]
                else:
                    # What it adds to the logit of the name it reads. At the acting name that is the name
                    # that acts, and at the full stop, which holds no name of its own, both names.
                    read = {"S": [acting], "stop": ["A", "B"]}.get(source, [source])
                    color = (BLUE if np.mean([avg(lambda t, k=k: t["writes"][name(x)][version][k]) for k in read]) >= 0 else RED)[0]
                mates = slot[x["block"]]
                y = x["block"] + (mates.index(x["head"]) - (len(mates) - 1) / 2) * 0.36 - 0.08
                ax.add_patch(FancyArrowPatch((xs[source], y), (xs[x["acts at"]], y), arrowstyle="-|>", mutation_scale=9,
                                             linewidth=0.6 + 3.4 * looks[source], color=color, shrinkA=6, shrinkB=6))
                ax.text((xs[source] + xs[x["acts at"]]) / 2, y + 0.07, f"{x['block']}.{x['head']}  reads {looks[source]:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK[0])
            for k, x0 in xs.items():
                ax.axvline(x0, color=RULE[0], linewidth=0.8, zorder=0)
            ax.set_xlim(-0.45, 4.45)
            ax.set_ylim(0.5, 8.9)
            ax.set_xticks(list(xs.values()), ["first\nname", "second\nname", "full\nstop", "acting\nname", "last\nposition"], fontsize=8)
            ax.set_yticks(range(1, 9))
            ax.tick_params(length=0)
            for side in ("left", "bottom"):
                ax.spines[side].set_visible(False)
            ax.set_title(f"{seed}, {version} prompt: {says}")
        row[0].set_ylabel("block")
    handles = [plt.Line2D([], [], color=c, linewidth=2.4, label=l) for c, l in (
        (BLUE[0], "raises the name it reads"), (RED[0], "lowers the name it reads"), (MUTED[0], "writes at the acting name; what it writes was not decoded"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.045))
    fig.tight_layout(h_pad=1.6)
    save(fig, "circuit-diagram")


def circuit_writes():
    """Step 4: what each head at the last position adds to the logit of each name of the prompt."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
    for ax, seed in zip(axes, SEEDS):
        run, name, avg = circuit_run(seed)
        readers = [x for x in run["heads"] if x["acts at"] == "end"]
        columns = [("clean", "A"), ("clean", "B"), ("flipped", "A"), ("flipped", "B"), ("neutral", "A"), ("neutral", "B"), ("neutral", "C")]
        grid = np.array([[avg(lambda t: t["writes"][name(x)][v][k]) for v, k in columns] for x in readers])
        cells(ax, grid, [k for _, k in columns], [f"{x['block']}.{x['head']}" for x in readers], floor=0.1)
        for edge in (1.5, 3.5):
            ax.axvline(edge, color=RULE[0], linewidth=0.8)
        for at, v in ((0.5, "clean:\nA acts"), (2.5, "flipped:\nB acts"), (5, "neutral:\nC acts")):
            ax.text(at, -0.62, v, ha="center", va="bottom", fontsize=8, color=INK[0])
        ax.set_ylabel("head (block.head)")
        ax.set_title(seed, pad=26)
    fig.tight_layout(w_pad=2.5)
    save(fig, "circuit-writes")


def circuit_sets():
    """Step 4: the heads of the set moved together, against random sets of as many heads."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 2.6), sharex=True)
    for ax, seed in zip(axes, SEEDS):
        run, _, avg = circuit_run(seed)
        random = np.mean([t["sets"]["random"] for t in run["templates"]], axis=0)
        dots(ax, random, rng.uniform(-0.28, 0.28, len(random)), MUTED[0], size=3.4)
        for key, color, says in (("the heads", BLUE[0], f"the {len(run['heads'])} heads of the set"), ("all other heads", RED[0], "all other heads")):
            ax.plot([avg(lambda t: t["sets"][key])], [0], marker="D", markersize=7, color=color, linestyle="", label=says)
        ax.plot([], [], marker="o", markersize=3.5, linestyle="", color=MUTED[0], label=f"a random set of {len(run['heads'])} heads")
        zero(ax, horizontal=False)
        ax.set_yticks([])
        ax.set_ylim(-0.6, 0.6)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("share of the gap that comes along")
        ax.set_title(seed)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=2.0)
    fig.legend(handles, [l.replace(f"{len(run['heads'])} heads of", "heads of") for l in labels], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.16), columnspacing=1.6)
    save(fig, "circuit-sets")


def sae_names():
    """Step 5: a latent for each name as the one that acts, and as the other name: how often it fires when it should and when it should not."""
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.6), sharex=True)
    for col, seed in enumerate(SEEDS):
        run = read(f"sae/sae-{seed}")
        for row, (role, says) in enumerate((("acts", "the name acts"), ("other", "the name is the other one"))):
            ax, rows = axes[row][col], run["task"]["names"][role]
            for i, x in enumerate(rows):
                ax.plot([100 * x["fires when it is not"], 100 * x["fires when it is"]], [i, i], color=RULE[0], linewidth=1.2, zorder=1)
                ax.plot([100 * x["fires when it is"]], [i], marker="o", markersize=5, color=BLUE[0], linestyle="", label="when it is" if i == 0 else None)
                ax.plot([100 * x["fires when it is not"]], [i], marker="o", markersize=5, markerfacecolor="none", markeredgewidth=1.3, color=RED[0], linestyle="", label="when it is not" if i == 0 else None)
            ax.set_yticks(range(len(rows)), [x["name"] for x in rows], fontsize=7.5)
            ax.set_ylim(len(rows) - 0.5, -0.5)
            ax.set_xlim(-4, 104)
            ax.tick_params(axis="y", length=0)
            ax.set_title(f"{seed}: {says}; {len({x['latent'] for x in rows})} different latents")
            if row == 1:
                ax.set_xlabel("the latent fires, percent of prompts")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(h_pad=1.4, w_pad=2.0)
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))
    save(fig, "sae-names")


def sae_readers():
    """Step 5: the other name at the acting name, read by step 2's reader from three inputs."""
    rng = np.random.default_rng(0)
    inputs = (("stream", "the stream"), ("rebuilt", "the rebuilt\nstream"), ("missed", "what the\ndictionary misses"))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), sharey=True)
    for ax, seed in zip(axes, SEEDS):
        reader = read(f"sae/sae-{seed}")["task"]["reader"]
        for i, (key, _) in enumerate(inputs):
            by = 100 * np.array(reader[key]["by pattern"])  # (patterns, versions)
            for v, version in enumerate(VERSIONS[:2]):
                x = i + (v - 0.5) * 0.3
                dots(ax, x + rng.uniform(-0.05, 0.05, by.shape[0]), by[:, v], COLOR[version][0], size=3.2)
                ax.plot([x - 0.1, x + 0.1], [by[:, v].mean()] * 2, color=COLOR[version][0], linewidth=2.0, label=f"{version} prompts, mean" if i == 0 else None)
        percent(ax, 12)
        ax.set_xticks(range(len(inputs)), [says for _, says in inputs], fontsize=8)
        ax.tick_params(axis="x", length=0)
        ax.set_title(seed)
    axes[0].set_ylabel("other name right, percent of test rows")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=3.5, color=MUTED[0]))
    labels.append("one held-out pattern")
    fig.tight_layout(w_pad=2.0)
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    save(fig, "sae-readers")


def sae_often():
    """Step 5: how often the latents fire."""
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    edges = np.linspace(-5, 0, 25)
    for seed, color in zip(SEEDS, (BLUE[0], RED[0])):
        counts = np.array(read(f"sae/sae-{seed}")["rebuilds"]["log10 of how often each latent fires"])
        ax.step(edges, np.append(counts, counts[-1]), where="post", color=color, linewidth=1.5, label=seed)
    ax.set_xticks([-4, -3, -2, -1, 0], ["0.01%", "0.1%", "1%", "10%", "100%"])
    ax.set_xlim(-4.2, 0)
    ax.set_xlabel("a latent fires on this share of tokens")
    ax.set_ylabel("latents")
    ax.legend(loc="upper right")
    save(fig, "sae-often")


def features_count():
    """Step 6: how much of the gap the highest-ranked latents carry, estimated and moved for real."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), sharey=True)
    for ax, seed in zip(axes, SEEDS):
        run = read(f"features/features-{seed}")
        counts = run["counts"]
        x = np.arange(len(counts))
        for key, color, says in (("real", BLUE[0], "moved for real"), ("estimated", RED[0], "estimated from the gradient")):
            by = np.array([t["by count"][key] for t in run["templates"]])  # (patterns, counts)
            for i in range(len(counts)):
                dots(ax, i + rng.uniform(-0.08, 0.08, by.shape[0]), by[:, i], color, size=3.0)
            ax.plot(x, by.mean(0), color=color, linewidth=1.6, label=says)
        whole = np.mean([t["parts"]["rebuilt"] for t in run["templates"]])
        ax.axhline(whole, color=MUTED[0], linewidth=1.0, linestyle=(0, (4, 2)), label="all of what the dictionary rebuilds, for real")
        zero(ax)
        ax.set_xticks(x, [str(c) for c in counts])
        ax.set_xlabel("latents moved, highest-ranked first")
        ax.set_title(seed)
    axes[0].set_ylabel("share of the gap that comes along")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=3.5, color=MUTED[0]))
    labels.append("one sentence pattern")
    fig.tight_layout(w_pad=2.0)
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.17))
    save(fig, "features-count")


STEERED = (("pair", INK, "-", "pair: the same two names in the other four patterns"), ("names", BLUE, "-", "names: pair, with a term per name and none for the two together"),
           ("means + order", GREEN, "-", "means + order"), ("means", RED, "-", "means: who acts, by name"), ("order", RED, (0, (4, 2)), "order: the same vector for every pair"),
           ("other names", MUTED, (0, (4, 2)), "control: means for two names not in the prompt"), ("random, long", MUTED, (0, (1, 2)), "control: a random direction as long as pair"))


def steer_who():
    """Step 7: how far a direction added at the acting name takes a clean prompt towards its flipped twin, by dose."""
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.6), sharex=True, sharey="row")
    for col, seed in enumerate(SEEDS):
        run = read(f"steer/steer-{seed}")
        doses, T = np.array(run["doses"]), run["who acts"]["templates"]
        for row, (measure, level, says) in enumerate((("share", "a third name acts, share", "a third name acts"), ("turned", "the flipped prompt turned", "the flipped prompt itself"))):
            ax = axes[row, col]
            for name, color, style, _ in STEERED:
                by = np.array([t["steered"][name][measure] for t in T])  # (patterns, doses)
                ax.fill_between(doses, by.min(0), by.max(0), color=color[0], alpha=0.13, linewidth=0)
                ax.plot(doses, by.mean(0), color=color[0], linewidth=1.5, linestyle=style)
            ax.axhline(np.mean([t["plain"][level] for t in T]), color=MUTED[0], linewidth=0.8)
            ax.annotate(says, (-2.0, np.mean([t["plain"][level] for t in T])), xytext=(0, 3), textcoords="offset points", fontsize=8, color=MUTED[0])
            zero(ax)
            ax.axvline(0, color=RULE[0], linewidth=0.8, zorder=0)
            if row == 0:
                ax.set_title(seed)
            else:
                ax.set_xlabel("dose: 1 is the direction as built")
                ax.set_xticks([-2, -1, 0, 1, 2])
    axes[0, 0].set_ylabel("share of the way from the clean\nanswer to the flipped one")
    axes[1, 0].set_ylabel("share of prompts where A\nis now above B")
    handles = [plt.Line2D([], [], color=color[0], linewidth=1.5, linestyle=style) for _, color, style, _ in STEERED]
    handles.append(Rectangle((0, 0), 1, 1, color=MUTED[0], alpha=0.25, linewidth=0))
    fig.tight_layout(w_pad=2.0, h_pad=1.2)
    fig.legend(handles, [says for *_, says in STEERED] + ["the range over the five sentence patterns"], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.16))
    save(fig, "steer-who")


def steer_kind():
    """Step 7: the pronoun the model goes on with after a sentence about one name, as a direction is added at the name."""
    titles = {"means": "means: She names minus He names", "latent": "latent: the one the six She names share", "random": "control: a random direction"}
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.0), sharex=True, sharey="row")
    for row, seed in enumerate(SEEDS):
        run = read(f"steer/steer-{seed}")
        doses, K = np.array(run["doses"]), run["kind"]
        people = run["names that go on with She"]
        for col, name in enumerate(titles):
            ax = axes[row, col]
            every = np.array([t["steered"][name]["gap by name"] for t in K["templates"]])  # (sentences, names, doses)
            assert every.shape[1] == 2 * len(people)
            for group, color in (("She names", BLUE), ("He names", RED)):
                lines = every[:, [i for i in range(every.shape[1]) if (run["kind"]["order"][i] in people) == (group == "She names")]].reshape(-1, len(doses))
                for line in lines:
                    ax.plot(doses, line, color=color[0], linewidth=0.5, alpha=0.3)
                ax.plot(doses, lines.mean(0), color=color[0], linewidth=1.8, label=f"{group}: the direction is {'taken away' if group == 'She names' else 'added'} (thin: one name in one sentence)")
            zero(ax)
            ax.axvline(0, color=RULE[0], linewidth=0.8, zorder=0)
            ax.set_xticks([-2, -1, 0, 1, 2])
            if row == 0:
                ax.set_title(titles[name])
            else:
                ax.set_xlabel("dose: 1 is the direction as built")
        axes[row, 0].set_ylabel(f"{seed}\nlogit of She minus logit of He")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(w_pad=1.6, h_pad=1.2)
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.07))
    save(fig, "steer-kind")


THREE = (("seed0", BLUE, "seed0, 8 blocks"), ("seed1", RED, "seed1, 8 blocks"), ("qwen", INK, "the open model, 24 blocks"))


def limits_models():
    """Step 8: where who acts is, by depth, and the pair direction by dose, in the two story models and the open model."""
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0))
    for name, color, says in THREE:
        travel = read(f"patch/patch-{name}")
        at = {g: np.array([np.array(t["travel"]["carry"])[:, travel["travel groups"].index(g)] for t in travel["templates"]]) for g in ("S", "end")}  # (patterns, depths)
        depth = np.linspace(0, 1, at["S"].shape[1])
        for group, style in (("S", "-"), ("end", (0, (4, 2)))):
            axes[0].fill_between(depth, at[group].min(0), at[group].max(0), color=color[0], alpha=0.13, linewidth=0)
            axes[0].plot(depth, at[group].mean(0), color=color[0], linewidth=1.5, linestyle=style, label=says if group == "S" else None)
        steered = read(f"open/steer-{name}") if name == "qwen" else read(f"steer/steer-{name}")
        T = steered["templates"] if name == "qwen" else steered["who acts"]["templates"]
        doses = np.array(steered["doses"])
        for ax, measure in zip(axes[1:], ("share", "turned")):
            by = np.array([t["steered"]["pair"][measure] for t in T])
            ax.fill_between(doses, by.min(0), by.max(0), color=color[0], alpha=0.13, linewidth=0)
            ax.plot(doses, by.mean(0), color=color[0], linewidth=1.5)
    for ax in axes:
        zero(ax)
    axes[0].set_xlabel("depth, as a share of the model's blocks")
    axes[0].set_ylabel("share of the gap that comes along")
    axes[0].set_title("who acts, moved at one position")
    axes[0].set_xticks([0, 0.25, 0.5, 0.75, 1])
    for ax, title, label in zip(axes[1:], ("the pair direction, at half depth", ""), ("share of the way to the flipped answer", "share of prompts with A above B")):
        ax.axvline(0, color=RULE[0], linewidth=0.8, zorder=0)
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_xlabel("dose")
        ax.set_ylabel(label)
        ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color=MUTED[0], linewidth=1.5), plt.Line2D([], [], color=MUTED[0], linewidth=1.5, linestyle=(0, (4, 2))), Rectangle((0, 0), 1, 1, color=MUTED[0], alpha=0.25, linewidth=0)]
    labels += ["left: at the acting name", "left: at the last position", "the range over the five sentence patterns"]
    fig.tight_layout(w_pad=1.6)
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.2))
    save(fig, "limits-models")


def limits_parts():
    """Step 8: every head and MLP of the open model moved one at a time, and the largest of them by sentence pattern."""
    run = read("patch/patch-qwen")
    per = {k: np.array([t["parts"][k]["all"] for t in run["templates"]]) for k in ("carry", "break")}  # (patterns, blocks, heads + 1)
    grid = per["carry"].mean(0)
    heads = grid.shape[1] - 1
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 6.2), gridspec_kw={"width_ratios": [1.75, 1]})
    cells(left, grid, [str(h) for h in range(heads)] + ["MLP"], [str(i + 1) for i in range(grid.shape[0])], floor=0.05, top=0.25)
    left.axvline(heads - 0.5, color=RULE[0], linewidth=0.8)
    left.set_xlabel("head")
    left.set_ylabel("block")
    left.set_title("clean part into the flipped run: share of the gap\nthat comes along, mean of the five patterns")
    size = np.abs(grid)
    size[0, -1] = 0  # block 1's MLP carries the token itself, 1.00 in every pattern
    order = [(b, h) for b, h in np.dstack(np.unravel_index(np.argsort(-size, axis=None), size.shape))[0] if size[b, h] >= 0.05]
    for i, (b, h) in enumerate(order):
        for k, color, shift in (("carry", BLUE[0], -0.17), ("break", RED[0], 0.17)):
            dots(right, per[k][:, b, h], np.full(per[k].shape[0], i + shift), color, size=3.0)
            right.plot([per[k][:, b, h].mean()], [i + shift], marker="|", markersize=7, markeredgewidth=1.6, color=color, linestyle="",
                       label={"carry": "clean part into the flipped run", "break": "flipped part into the clean run"}[k] if i == 0 else None)
    zero(right, horizontal=False)
    right.set_yticks(range(len(order)), [f"block {b + 1} " + ("MLP" if h == heads else f"head {h}") for b, h in order], fontsize=8)
    right.set_ylim(len(order) - 0.5, -0.5)
    right.tick_params(axis="y", length=0)
    right.set_xlabel("share of the gap")
    right.set_title("parts of 0.05 or more:\none dot per sentence pattern")
    right.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), fontsize=8)
    fig.tight_layout(w_pad=2.0)
    save(fig, "limits-parts")


def limits_dictionaries():
    """Step 8: two results of steps 6 and 7 with the first dictionary and with a second one trained from another seed."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.2), sharex="col", sharey="col")
    for row, seed in enumerate(SEEDS):
        for tag, color, says in (("", BLUE, "the first dictionary"), ("-s1", RED, "a second, trained from another seed")):
            run = read(f"features/features-{seed}{tag}")
            counts = run["counts"]
            by = np.array([t["by count"]["real"] for t in run["templates"]])  # (patterns, counts)
            for i in range(len(counts)):
                dots(axes[row, 0], i + rng.uniform(-0.08, 0.08, by.shape[0]), by[:, i], color[0], size=3.0)
            axes[row, 0].plot(np.arange(len(counts)), by.mean(0), color=color[0], linewidth=1.6, label=says)
            steered = read(f"steer/steer-{seed}{tag}")
            doses, K = np.array(steered["doses"]), steered["kind"]
            he = [i for i, p in enumerate(K["order"]) if p not in steered["names that go on with She"]]
            lines = np.array([t["steered"]["latent"]["gap by name"] for t in K["templates"]])[:, he].reshape(-1, len(doses))
            for line in lines:
                axes[row, 1].plot(doses, line, color=color[0], linewidth=0.5, alpha=0.25)
            axes[row, 1].plot(doses, lines.mean(0), color=color[0], linewidth=1.8)
        zero(axes[row, 0])
        zero(axes[row, 1])
        axes[row, 1].axvline(0, color=RULE[0], linewidth=0.8, zorder=0)
        axes[row, 0].set_ylabel(f"{seed}\nshare of the gap that comes along")
        axes[row, 1].set_ylabel("logit of She minus logit of He")
    axes[0, 0].set_title("who acts: the highest-ranked latents, moved for real")
    axes[0, 1].set_title("the shared latent added at a He name")
    axes[1, 0].set_xticks(np.arange(len(counts)), [str(c) for c in counts])
    axes[1, 0].set_xlabel("latents moved, highest-ranked first")
    axes[1, 1].set_xticks([-2, -1, 0, 1, 2])
    axes[1, 1].set_xlabel("dose: 1 is the latent's mean value on She names")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], marker="o", linestyle="", markersize=3.5, color=MUTED[0]), plt.Line2D([], [], color=MUTED[0], linewidth=0.6)]
    labels += ["left: one sentence pattern", "right: one name in one sentence"]
    fig.tight_layout(w_pad=2.0, h_pad=1.0)
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    save(fig, "limits-dictionaries")


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
    circuit_diagram()
    circuit_writes()
    circuit_sets()
    sae_names()
    sae_readers()
    sae_often()
    features_count()
    steer_who()
    steer_kind()
    limits_models()
    limits_parts()
    limits_dictionaries()
