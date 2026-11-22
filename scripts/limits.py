"""
Step 8: how much of the seven steps holds elsewhere? See docs/08-limits.md.

The notes so far reported two story models that differ in the seed of
their training, one dictionary per model, and no other model. This script
measures nothing on the behavior: it reads the records of the earlier
steps and sets them side by side.

models         the headline number of each step for the two story models,
               and for the open model wherever a script ran on it
               (scripts/behavior.py, lens.py and patch.py with --model
               qwen, and scripts/open.py)
dictionaries   steps 5 to 7 again with a second dictionary for each story
               model, trained in the same way from another seed
               (scripts/sae.py, features.py and steer.py with --dict s1)
matches        which latents of the first dictionary have a counterpart
               in the second: by the cosine of their decoder directions,
               and for the latents the earlier steps named, by whether the
               counterpart plays the same part. This table loads the
               models, to see where two latents fire.

Two columns are called the same by one rule, applied to every row: a
number when the two differ by at most a fifth of the larger (or by 0.05),
a count or a name when they are equal.

A third seed of the story model (seed2) and a third dictionary for the
first two models (--seed 2 --dict s2) are read by the same rule, which was
written here before either had been run. For three columns the rule is
applied to each of the three pairs, since it does not carry over: a can
be the same as b and b as c without a being the same as c. A row is the
same in all three when all three pairs are. Otherwise the record says
which of the first two the third agrees with. Reported per table: the
rows the same in all three; of the rows the first two agreed on, how many
the third joins; of the rows they differed on, whether the third sides
with the first, the second, both or neither. The third model's dictionary
is trained after the block of its own head at the acting name, which is
how block 4 was chosen for the first two.

Settled after the third model's steps 1 to 3 had run and before their
records were read: "its own head at the acting name" is the head with the
largest share of the gap moved at that position alone (step 3's parts),
whether or not one head or several pass 0.05 there. That gives block 4
for the first two. If it gives another block for the third, the rows of
steps 5 to 7 compare the models at different places, and no run is added
to tell the seed from the place. The third model sorts the twelve names
into the same six and six as the first two, so every count has the base
its row names; the number of heads its circuit has is recorded beside the
row on what they carry together. The agreement of each pair is counted
too, since the rows the same in all three can only be fewer than the rows
the first two agreed on. One model and one dictionary are added to it,
and one dictionary to each of the first two; no row is run again.

One look was added after the rows had been read, and is no row. Step 4's
record of the third model has two heads at the last position that read
the acting name and lower it, with shares of 0.62 and 0.45, where the
first two models have one large one (0.87 against 0.13, 0.69 against
0.14). Step 6's two rows are about the larger. Written here before
scripts/heads.py had been run on the third model: what the smaller, head
5 of block 6, adds to an acting "She" name against the other names of its
kind on clean prompts is set beside the first two models' -0.26 and -0.23
by the same rule, and reported whichever way it falls. The 24 rows and
their tally stay as they are.

    uv run scripts/limits.py
"""

import json
import os

import numpy as np
import torch
from minllm.data import load_tokens

from minterp import sae as S
from minterp import tasks
from minterp.common import MINLLM, load, save
from minterp.tap import Tap

SEEDS, OPEN, TAG = ("seed0", "seed1"), "qwen", "s1"
THIRD, TAG3 = "seed2", "s2"  # a third seed of the model, and a third dictionary for the first two
KINDS = ("both She", "both He", "one of each")


def record(folder, file, name, tag=""):
    path = f"runs/{folder}/{file}-{name}{'-' + tag if tag else ''}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def over(r, pick, key="templates"):
    return float(np.mean([pick(t) for t in r[key]]))


def same(a, b, kind):
    if a is None or b is None:
        return None
    if kind != "number":
        return a == b
    return abs(a - b) <= max(0.2 * max(abs(a), abs(b)), 0.05)


def third(a, b, c, kind):
    """The rule for three: every pair by the rule for two, with c the newcomer."""
    if c is None:
        return None
    with_a, with_b = bool(same(a, c, kind)), bool(same(b, c, kind))
    return {"agrees with": [with_a, with_b], "the same in all three": bool(same(a, b, kind)) and with_a and with_b}


def tally(rows, first_two, verdict):
    """Over the rows that have a third value: how the third sits with the first two."""
    rows = [r for r in rows if verdict(r) is not None]
    agreed, differed = [r for r in rows if first_two(r)], [r for r in rows if not first_two(r)]
    sides = lambda r: {(True, True): "both", (True, False): "the first only", (False, True): "the second only", (False, False): "neither"}[tuple(verdict(r)["agrees with"])]
    return {"rows": len(rows), "the same in all three": sum(verdict(r)["the same in all three"] for r in rows),
            "by pair": {"the first and the second": len(agreed), "the first and the third": sum(verdict(r)["agrees with"][0] for r in rows),
                        "the second and the third": sum(verdict(r)["agrees with"][1] for r in rows)},
            "of the rows the first two agreed on": {"rows": len(agreed), "the third is the same as both": sum(verdict(r)["the same in all three"] for r in agreed)},
            "of the rows they differed on": {"rows": len(differed), **{k: sum(sides(r) == k for r in differed) for k in ("the first only", "the second only", "both", "neither")}}}


# ---------------------------------------------------------------- one number per step, per model
def blocks_of(name):
    return record("read", "behavior", name)["model"]["config"]["n_layers"]


def behavior(version, sign):
    def value(name, tag):
        r = record("read", "behavior", name)
        return float(np.mean([(sign * np.array(t[version]["logit_diff"]) > 0).mean() for t in r["templates"]]))
    return value


def late_blocks(name, tag):
    a = record("read", "lens", name)["attribution"]["clean"]
    b = np.array(a["heads"]["mean"]).sum(1) + np.array(a["mlp"]["mean"])
    return float(b[-len(b) // 4:].sum() / (b.sum() + a["embed"]["mean"]))


def due_at_half_depth(version):
    def value(name, tag):
        r = record("probe", "probe", name)
        if r is None:
            return None
        a = np.array(r["new"]["accuracy"])[:, r["versions"].index(version), len(r["new"]["accuracy"][0][0]) // 2, r["roles"].index("S"), r["targets"].index("due")]
        return float(a.mean())  # new pairs, mean over the held-out patterns
    return value


def leaves_at(name, tag, line=0.5):
    """The last depth at which moving the acting name's position alone still carries `line` of the gap, as a share of the depth."""
    r = record("patch", "patch", name)
    at = np.mean([t["travel"]["carry"] for t in r["templates"]], 0)[:, r["travel groups"].index("S")]
    return float(max(d for d, v in enumerate(at) if v >= line) / (len(at) - 1))


def block_at_name(name):
    """The block of the head with the largest share of the gap at the acting name: where a model's dictionary is trained."""
    at = np.mean([t["parts"]["carry"]["S"] for t in record("patch", "patch", name)["templates"]], 0)[:, :-1]  # (blocks, heads), without the MLPs
    return int(np.unravel_index(at.argmax(), at.shape)[0]) + 1


def at_end_by_kind(name):
    """Every head scripts/heads.py read at the last position: its share, and what it adds to an acting name against the uninvolved names of its kind, on clean prompts."""
    r = record("open", "heads", name)
    if r is None:
        return None
    key = lambda h: f"block {h['block']} head {h['head']}"
    return [{"head": f"{h['block']}.{h['head']}", "carry": h["carry"],
             "adds to an acting name against its own kind": {k: over(r, lambda t: t["by kind"][key(h)]["clean"]["acts"][k]) for k in ("She", "He")},
             "reads the acting name with": over(r, lambda t: t["attention"][key(h)]["clean"]["S"])} for h in r["heads"] if h["acts at"] == "end"]


def veto_head(name, tag):
    r = record("circuit", "circuit", name)
    if r is None:
        return None
    h = max((h for h in r["heads"] if h["acts at"] == "end"), key=lambda h: h["carry"])
    return f"{h['block']}.{h['head']}"


def circuit_sets(name, tag):
    r = record("circuit", "circuit", name)
    return None if r is None else over(r, lambda t: t["sets"]["the heads"])


def sae(pick):
    def value(name, tag):
        r = record("sae", "sae", name, tag)
        return None if r is None else float(pick(r))
    return value


def features(pick):
    def value(name, tag):
        r = record("features", "features", name, tag)
        return None if r is None else pick(r)
    return value


def steered(direction, measure="share", group=None):
    """Step 7's first experiment at dose 1: from runs/steer for a story model, from runs/open for the open one."""
    def value(name, tag):
        r = record("steer", "steer", name, tag)
        T = r["who acts"]["templates"] if r else (record("open", "steer", name) or {}).get("templates")
        if T is None or direction not in T[0]["steered"]:
            return None
        doses = (r or record("open", "steer", name))["doses"]
        i = doses.index(1.0)
        if group:
            return float(np.mean([t["steered"][direction]["moved, in logits, by pair"][group][i] for t in T]) / np.mean([t["pairs"][group]["gap"] for t in T]))
        return float(np.mean([t["steered"][direction][measure][i] for t in T]))
    return value


def gap_ratio(name, tag):
    r = record("steer", "steer", name, tag)
    T = r["who acts"]["templates"] if r else (record("open", "steer", name) or {}).get("templates")
    return None if T is None else float(np.mean([t["pairs"]["both She"]["gap"] for t in T]) / np.mean([t["pairs"]["one of each"]["gap"] for t in T]))


def pronoun(direction, what, dose=1.0, groups=("He names", "She names")):
    def value(name, tag):
        r = record("steer", "steer", name, tag)
        if r is None:
            return None
        i, T = r["doses"].index(dose), r["kind"]["templates"]
        if what == "turned":
            return sum(t["steered"][direction]["turned"][g][i] for t in T for g in groups)
        if what == "moved by group":  # in logits, the mean of the first group
            return float(abs(np.mean([t["steered"][direction]["gap"][groups[0]][i] for t in T]) - np.mean([t["plain"][groups[0]] for t in T])))
        if what == "moved":  # in logits, the larger of the two groups
            return float(max(abs(np.mean([t["steered"][direction]["gap"][g][i] for t in T]) - np.mean([t["plain"][g] for t in T])) for g in groups))
        return float(np.mean([t["steered"][direction][what][groups[0]][i] for t in T]))
    return value


def kind_record(pick):
    def value(name, tag):
        r = record("steer", "steer", name, tag)
        return None if r is None else pick(r["kind"])
    return value


def she_names_sharing(r):
    she = r["names that go on with She"]
    latents = [x["the head's latent"] for x in r["names"] if x["name"] in she]
    return max(latents.count(j) for j in set(latents))


def he_latents(r):
    she = r["names that go on with She"]
    return len({x["the head's latent"] for x in r["names"] if x["name"] not in she})


T = lambda r: r["templates"]
MODELS = [  # step, what, kind of value, how to get it
    (1, "clean prompts with B above A, share", "number", behavior("clean", 1)),
    (1, "flipped prompts with A above B, share", "number", behavior("flipped", -1)),
    (1, "share of the clean logit difference written by the last quarter of the blocks", "number", late_blocks),
    (2, "the due name read at the acting name at half depth, on new pairs, clean prompts, share right", "number", due_at_half_depth("clean")),
    (2, "the same on flipped prompts", "number", due_at_half_depth("flipped")),
    (3, "the last depth at which the acting name alone carries half of the gap, as a share of the depth", "number", leaves_at),
    (4, "the head that carries most at the last position (block.head)", "name", veto_head),
    (4, "the seven heads of the circuit together carry, share of the gap", "number", circuit_sets),
    (5, "variance the dictionary misses on stories, share", "number", sae(lambda r: r["rebuilds"]["unexplained variance"])),
    (5, "the loss rises by, share", "number", sae(lambda r: r["costs"]["loss rises by"])),
    (6, "who acts carried by what the dictionary rebuilds, share of the gap", "number", features(lambda r: over(r, lambda t: t["parts"]["rebuilt"]))),
    (6, "carried by the highest-ranked latent", "number", features(lambda r: over(r, lambda t: t["by count"]["real"][0]))),
    (6, "of the six She names, how many share the veto head's latent", "count", features(she_names_sharing)),
    (6, "the veto head's write to an acting She name against its own kind, logits", "number",
     features(lambda r: over(r, lambda t: t["head"]["by kind of the acting name"]["She"]["against its own kind"]))),
    (7, "the pair direction at dose 1, share of the way", "number", steered("pair")),
    (7, "its part with a term per name", "number", steered("names")),
    (7, "the names' means with the vector every flipped prompt shares", "number", steered("means + order")),
    (7, "the names' means alone", "number", steered("means")),
    (7, "the shared vector alone", "number", steered("order")),
    (7, "a random direction as long as the pair direction", "number", steered("random, long")),
    (7, "the kind alone, on pairs of two kinds, share of their gap", "number", steered("kind", group="one of each")),
    (7, "the gap between two She names against the gap between names of two kinds", "number", gap_ratio),
    (7, "pronouns turned by the difference of means at dose 1, of 72", "count", pronoun("means", "turned")),
    (7, "pronouns turned by the shared latent at its mean value, of 72", "count", pronoun("latent", "turned")),
]
DICTIONARIES = [
    (5, "variance the dictionary misses on stories, share", "number", sae(lambda r: r["rebuilds"]["unexplained variance"])),
    (5, "the loss rises by, share", "number", sae(lambda r: r["costs"]["loss rises by"])),
    (6, "who acts carried by what the dictionary rebuilds", "number", features(lambda r: over(r, lambda t: t["parts"]["rebuilt"]))),
    (6, "carried by the highest-ranked latent", "number", features(lambda r: over(r, lambda t: t["by count"]["real"][0]))),
    (6, "by the eight highest", "number", features(lambda r: over(r, lambda t: t["by count"]["real"][3]))),
    (6, "by the 32 highest", "number", features(lambda r: over(r, lambda t: t["by count"]["real"][5]))),
    (6, "the gradient's estimate against the real effect, correlation", "number", features(lambda r: over(r, lambda t: t["one latent"]["correlation"]))),
    (6, "the two names' latents moved for real, share of the gap", "number", features(lambda r: over(r, lambda t: t["names"]["the head's latents"]["moved for real"]))),
    (6, "of the six She names, how many share the veto head's latent", "count", features(she_names_sharing)),
    (6, "different latents among the six He names", "count", features(he_latents)),
    (6, "the shared latent off: the acting She name against the nine, logits", "number",
     features(lambda r: over(r, lambda t: t["head"]["the shared latent switched off, where a She name acts"]["the acting name against the nine"]))),
    (6, "the same against its own kind", "number",
     features(lambda r: over(r, lambda t: t["head"]["the shared latent switched off, where a She name acts"]["the acting name against its own kind"]))),
    (7, "the two names' latents steered, share of the way", "number", steered("latents")),
    (7, "the shared latent at its mean value moves the pronoun measure after a He name by, logits (mean)", "number", pronoun("latent", "moved by group")),
    (7, "pronouns it turns, of 72", "count", pronoun("latent", "turned")),
    (7, "of them after the 36 He names", "count", pronoun("latent", "turned", groups=("He names",))),
    (7, "and after the 36 She names", "count", pronoun("latent", "turned", groups=("She names",))),
    (7, "at twice that value, of 72", "count", pronoun("latent", "turned", dose=2.0)),
    (7, "the rebuilt part of the difference of means turns, of 72", "count", pronoun("rebuilt", "turned")),
    (7, "the missed part turns, of 72", "count", pronoun("missed", "turned")),
    (7, "the missed part moves the measure by, logits (the larger of the two groups' means)", "number", pronoun("missed", "moved")),
    (7, "the dictionary reads the latent back at, share of what was written", "number",
     pronoun("latent", "the shared latent as the dictionary reads it, against its usual value")),
    (7, "its encoder's reading of one unit of the latent's direction", "number", kind_record(lambda k: k["the encoder's reading of one unit of the latent's direction"])),
    (7, "where the reading starts on a He name, share of the latent's mean value", "number", kind_record(lambda k: k["the reading before the cut on He names, against the usual value"])),
    (7, "latents that differ by more than 0.5 between the kinds (mean of six sentences)", "number",
     kind_record(lambda k: float(np.mean([t["latents that differ by more than half a unit between the kinds"] for t in k["templates"]])))),
]

result = {"rule": "a number is the same when two columns differ by at most a fifth of the larger, or by 0.05; a count or a name when they are equal",
          "models": {"columns": [*SEEDS, OPEN], "rows": []}, "dictionaries": {"columns": [f"{s}, {d}" for s in SEEDS for d in ("first", "second")], "rows": []}}
second = 0
for step, what, kind, value in MODELS:
    values = [value(name, "") for name in (*SEEDS, OPEN)]
    result["models"]["rows"].append({"step": step, "what": what, "kind": kind, "values": values, "the two story models agree": same(values[0], values[1], kind),
                                     "the open model agrees with": None if values[2] is None else [bool(same(values[0], values[2], kind)), bool(same(values[1], values[2], kind))]})
    second += bool(same(value(SEEDS[0], TAG), value(SEEDS[1], TAG), kind))  # the same row with the second dictionaries, where a row has one
result["models"]["rows the two story models agree on"] = {"with the first dictionaries": sum(bool(r["the two story models agree"]) for r in result["models"]["rows"]),
                                                          "with the second dictionaries": second, "of": len(MODELS)}
# The depth row depends on the line drawn in the open model. With other lines:
result["models"]["the depth row with another line"] = {str(line): [leaves_at(name, "", line) for name in (*SEEDS, OPEN)] for line in (0.9, 0.5, 0.25, 0.1)}
# The vector every flipped prompt shares, by depth (scripts/open.py on all three models): the names' means alone, the vector alone, the two together.
result["models"]["by depth"] = {}
for name in (*SEEDS, OPEN):
    r = record("open", "steer", name)
    if r is not None:
        result["models"]["by depth"][name] = {"depths": r["depths"], **{d: np.mean([t["depth"][d]["share"] for t in r["templates"]], 0).tolist() for d in ("pair", "means", "order", "means + order")}}
for step, what, kind, value in DICTIONARIES:
    values = [value(name, tag) for name in SEEDS for tag in ("", TAG)]
    result["dictionaries"]["rows"].append({"step": step, "what": what, "kind": kind, "values": values,
                                           "the two dictionaries agree": [same(values[0], values[1], kind), same(values[2], values[3], kind)]})

# ---------------------------------------------------------------- a third seed, and a third dictionary
# The third model's records, where every step has run on it: the same 24 rows, by the rule for three.
needed = (("read", "behavior"), ("read", "lens"), ("probe", "probe"), ("patch", "patch"), ("circuit", "circuit"), ("sae", "sae"), ("features", "features"), ("steer", "steer"))
if all(record(folder, file, THIRD) is not None for folder, file in needed):
    rows = []
    for (step, what, kind, value), row in zip(MODELS, result["models"]["rows"]):
        v = value(THIRD, "")
        rows.append({"step": step, "what": what, "kind": kind, "value": v, **third(row["values"][0], row["values"][1], v, kind)})
    paired = list(zip(result["models"]["rows"], rows))
    result["models"]["a third seed"] = {
        "name": THIRD, "the block of its dictionary": record("sae", "sae", THIRD)["block"], "the block of the head with the largest share at the acting name": block_at_name(THIRD),
        "the same block for the first two": [block_at_name(s) for s in SEEDS], "heads of the circuit": [len(record("circuit", "circuit", s)["heads"]) for s in (*SEEDS, THIRD)],
        "names that go on with She": record("features", "features", THIRD)["names that go on with She"],
        "rows": rows, "tally": tally(paired, lambda p: p[0]["the two story models agree"], lambda p: p[1]),
        "the depth row with another line": {str(line): leaves_at(THIRD, "", line) for line in (0.9, 0.5, 0.25, 0.1)},
    }
    # The look added after the rows were read: every head at the last position by kind, where scripts/heads.py has run.
    she_row = features(lambda r: over(r, lambda t: t["head"]["by kind of the acting name"]["She"]["against its own kind"]))
    at_end = {s: at_end_by_kind(s) for s in (*SEEDS, THIRD)}
    for h in at_end[THIRD] or []:
        h["an acting She name, beside the first two models' veto heads"] = third(she_row(SEEDS[0], ""), she_row(SEEDS[1], ""), h["adds to an acting name against its own kind"]["She"], "number")
    result["models"]["a third seed"]["the heads at the last position, by kind"] = at_end
    r = record("open", "steer", THIRD)
    if r is not None:
        result["models"]["by depth"][THIRD] = {"depths": r["depths"], **{d: np.mean([t["depth"][d]["share"] for t in r["templates"]], 0).tolist() for d in ("pair", "means", "order", "means + order")}}
# A third dictionary for each of the first two models: the same 25 rows, by the rule for three.
if all(record(step, step, seed, TAG3) is not None for seed in SEEDS for step in ("sae", "features", "steer")):
    rows = []
    for (step, what, kind, value), row in zip(DICTIONARIES, result["dictionaries"]["rows"]):
        values = [value(seed, TAG3) for seed in SEEDS]
        rows.append({"step": step, "what": what, "kind": kind, "values": values,
                     "by model": [third(row["values"][2 * i], row["values"][2 * i + 1], values[i], kind) for i in range(len(SEEDS))]})
    paired = list(zip(result["dictionaries"]["rows"], rows))
    result["dictionaries"]["a third dictionary"] = {
        "columns": [f"{s}, third" for s in SEEDS], "rows": rows,
        "tally": {seed: tally(paired, lambda p, i=i: p[0]["the two dictionaries agree"][i], lambda p, i=i: p[1]["by model"][i]) for i, seed in enumerate(SEEDS)},
    }

# ---------------------------------------------------------------- which latents have a counterpart
def one_to_one(c, floor=0.7):
    """A greedy one-to-one matching of rows to columns by cosine, highest first. Pairs under the floor are left unmatched: -1."""
    rows, cols = (c >= floor).nonzero(as_tuple=True)
    partner, taken = torch.full((c.shape[0],), -1), torch.zeros(c.shape[1], dtype=torch.bool)
    for k in c[rows, cols].argsort(descending=True).tolist():
        i, j = rows[k].item(), cols[k].item()
        if partner[i] < 0 and not taken[j]:
            partner[i], taken[j] = j, True
    return partner


unit = lambda v: v / v.norm(dim=1, keepdim=True)
shares = lambda best: {"at a cosine of 0.9 or more": (best >= 0.9).float().mean().item(), "at 0.7 or more": (best >= 0.7).float().mean().item(),
                       "at 0.5 or more": (best >= 0.5).float().mean().item(), "median cosine": best.median().item()}
result["matches"] = {}
for seed in SEEDS:
    model, tok, meta = load(seed)
    device = next(model.parameters()).device
    one, two = S.load(S.path(seed, 4), "cpu"), S.load(S.path(seed, 4, TAG), "cpu")
    cosine = one.dec.detach() @ two.dec.detach().T  # (width, width): both sets of directions have length one
    best = cosine.max(1)
    inside = one.dec.detach() @ one.dec.detach().T
    inside.fill_diagonal_(-1)
    g = torch.Generator().manual_seed(0)
    noise = unit(torch.randn(one.width, one.dec.shape[1], generator=g)) @ unit(torch.randn(two.width, two.dec.shape[1], generator=g)).T
    # The stricter measure of Paulo and Belrose: one-to-one, the same partner by the encoder and by the decoder, 0.7 or more in both.
    by_dec, by_enc = one_to_one(cosine), one_to_one(unit(one.enc.detach().T) @ unit(two.enc.detach().T).T)
    # How often each latent of the first dictionary fires on held-out stories.
    SEQ = model.cfg.max_seq_len
    data = load_tokens(os.path.join(MINLLM, "data/tinystories/valid.bin"))
    starts = np.random.default_rng(2).integers(0, len(data) - SEQ - 1, 300)
    fires, seen = torch.zeros(one.width), 0
    for i in range(0, 300, 100):
        idx = torch.from_numpy(np.stack([data[k:k + SEQ].astype(np.int64) for k in starts[i:i + 100]])).to(device)
        with torch.no_grad(), Tap(model, keep=("resid",)) as tap:
            model(idx)
        x = tap["resid", 4].float().reshape(-1, one.dec.shape[1]).cpu()
        with torch.no_grad():
            fires += (one.encode(x) > 0).float().sum(0)
        seen += x.shape[0]
    rate = fires / seen
    bins = {"fires on up to 0.1% of story tokens": rate <= 1e-3, "on 0.1% to 1%": (rate > 1e-3) & (rate <= 1e-2), "on more than 1%": rate > 1e-2}

    f1, f2 = record("features", "features", seed), record("features", "features", seed, TAG)
    she = f1["names that go on with She"]
    head1, head2 = ({x["name"]: x["the head's latent"] for x in f["names"]} for f in (f1, f2))
    step5_1, step5_2 = ({x["name"]: x["step 5's latent"] for x in f["names"]} for f in (f1, f2))
    shared1, shared2 = f1["the shared latent"], f2["the shared latent"]
    # Where the two shared latents fire: at the acting name of every clean and flipped prompt.
    people = tasks.names(tok)
    streams, acting = [], []
    for batch in tasks.batches(tok, device, people=people):
        for version, col in (("clean", 0), ("flipped", 1)):
            with torch.no_grad(), Tap(model, keep=("resid",)) as tap:
                model(getattr(batch, version))
            streams.append(tap["resid", 4][:, batch.pos["S"]].float().cpu())
            acting += [t[col] in she for t in batch.names]
    streams, acting = torch.cat(streams), torch.tensor(acting)
    with torch.no_grad():
        z1, z2 = one.encode(streams)[:, shared1], two.encode(streams)[:, shared2]
    fire = lambda a, b: torch.corrcoef(torch.stack([a, b]))[0, 1].item()

    def by_latent(table1, table2):
        """One entry per latent of the first dictionary that a table names: its best counterpart, and whether the second dictionary's table gives that counterpart to most of the same names."""
        out = []
        for latent in sorted(set(table1.values())):
            names = [p for p in people if table1[p] == latent]
            j = best.indices[latent].item()
            out.append({"latent": latent, "names": names, "cosine of the best counterpart": best.values[latent].item(),
                        "the counterpart is the second dictionary's latent for most of these names": sum(table2[p] == j for p in names) * 2 > len(names)})
        return out

    result["matches"][seed] = {
        "latents of the first dictionary": one.width, "latents that never fire on the stories": int((rate == 0).sum()), "story tokens": seen,
        "with a counterpart in the second dictionary": shares(best.values),
        "by how often a latent fires": {label: {"latents": int(m.sum()), **shares(best.values[m])} for label, m in bins.items()},
        "for comparison": {"the nearest other latent of the same dictionary": shares(inside.max(1).values), "two sets of random directions": shares(noise.max(1).values)},
        "one to one": {"at a cosine of 0.7 or more by the decoder": (by_dec >= 0).float().mean().item(),
                       "the same partner by encoder and decoder, 0.7 or more in both": ((by_dec >= 0) & (by_dec == by_enc)).float().mean().item()},
        "the shared latent": {"first": shared1, "second": shared2, "cosine of the best counterpart": best.values[shared1].item(),
                              "the best counterpart is the second dictionary's shared latent": bool(best.indices[shared1].item() == shared2),
                              "correlation of the two over all prompts": fire(z1, z2), "among the prompts where a She name acts": fire(z1[acting], z2[acting]),
                              "share of the other prompts on which each fires": [(z1[~acting] > 0).float().mean().item(), (z2[~acting] > 0).float().mean().item()],
                              "share of the She prompts on which each fires": [(z1[acting] > 0).float().mean().item(), (z2[acting] > 0).float().mean().item()], "prompts": len(acting)},
        "the veto head's latents": by_latent(head1, head2), "step 5's latents": by_latent(step5_1, step5_2),
    }
    # A third dictionary, where one was trained: counterparts of the first and of the second in it, and latents of the first found in both others.
    f3 = record("features", "features", seed, TAG3)
    if f3 is not None and os.path.exists(S.path(seed, 4, TAG3)):
        three = S.load(S.path(seed, 4, TAG3), "cpu")
        in_third, second_in_third = (d.dec.detach() @ three.dec.detach().T for d in (one, two))
        b13, b23 = in_third.max(1), second_in_third.max(1)
        shared3 = f3["the shared latent"]
        result["matches"][seed]["a third dictionary"] = {
            "the first dictionary's latents with a counterpart in the third": shares(b13.values),
            "the second dictionary's latents with a counterpart in the third": shares(b23.values),
            "latents of the first with a counterpart at 0.7 or more in the second and in the third": ((best.values >= 0.7) & (b13.values >= 0.7)).float().mean().item(),
            "of those that fire on more than 1% of story tokens": ((best.values >= 0.7) & (b13.values >= 0.7))[rate > 1e-2].float().mean().item(),
            "the shared latent": {"third": shared3,
                                  "cosine of the first's best counterpart": b13.values[shared1].item(), "which is the third's shared latent": bool(b13.indices[shared1].item() == shared3),
                                  "cosine of the second's best counterpart": b23.values[shared2].item(), "which is the third's shared latent, from the second": bool(b23.indices[shared2].item() == shared3)},
        }

# ---------------------------------------------------------------- print
def show(v):
    return "     —" if v is None else f"{v:>6}" if isinstance(v, (str, int)) and not isinstance(v, bool) else f"{v:6.2f}"


mark = {True: "same", False: "differs", None: ""}
to_both = lambda a: "" if a is None else {(True, True): "same as both", (True, False): "as seed0 only", (False, True): "as seed1 only", (False, False): "as neither"}[tuple(a)]
print(result["rule"] + "\n")
print(f"{'':4}{'':98} {'seed0':>6} {'seed1':>6} {'open':>6}   the seeds   the open model")
for row in result["models"]["rows"]:
    print(f"{row['step']:>2}  {row['what'][:98]:98} " + " ".join(show(v) for v in row["values"]) + f"   {mark[row['the two story models agree']]:9}   {to_both(row['the open model agrees with'])}")
print("the two story models agree on:", result["models"]["rows the two story models agree on"])
print("the depth row with another line (seed0, seed1, open):", {k: [round(x, 2) for x in v] for k, v in result["models"]["the depth row with another line"].items()})
for name, d in result["models"]["by depth"].items():
    print(f"{name} by depth {d['depths']}: " + "; ".join(f"{k} " + " ".join(f"{x:.2f}" for x in d[k]) for k in ("means", "order", "means + order")))
print(f"\n{'':4}{'':90} {'seed0':>13} {'seed1':>13}")
print(f"{'':4}{'':90} {'first':>6} {'second':>6} {'first':>6} {'second':>6}")
for row in result["dictionaries"]["rows"]:
    print(f"{row['step']:>2}  {row['what'][:90]:90} " + " ".join(show(v) for v in row["values"]) + "   " + " / ".join(mark[a] for a in row["the two dictionaries agree"]))
pct = lambda d: f"{100 * d['at a cosine of 0.9 or more']:.0f}% / {100 * d['at 0.7 or more']:.0f}% / {100 * d['at 0.5 or more']:.0f}%, median {d['median cosine']:.2f}"
for seed, m in result["matches"].items():
    s_ = m["the shared latent"]
    print(f"\n{seed}: of {m['latents of the first dictionary']} latents ({m['latents that never fire on the stories']} never fire), with a counterpart at 0.9 / 0.7 / 0.5: {pct(m['with a counterpart in the second dictionary'])}")
    for label, d in m["by how often a latent fires"].items():
        print(f"  {label} ({d['latents']}): {pct(d)}")
    for label, d in m["for comparison"].items():
        print(f"  for comparison, {label}: {pct(d)}")
    print("  one to one: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in m["one to one"].items()))
    print(f"  the shared latent {s_['first']} -> {s_['second']}: cosine {s_['cosine of the best counterpart']:.2f}, mutual: {s_['the best counterpart is the second dictionary' + chr(39) + 's shared latent']}; "
          f"correlation over all {s_['prompts']} prompts {s_['correlation of the two over all prompts']:.2f}, among those where a She name acts {s_['among the prompts where a She name acts']:.2f}; "
          f"fires on {[round(x, 2) for x in s_['share of the She prompts on which each fires']]} of those and {[round(x, 2) for x in s_['share of the other prompts on which each fires']]} of the rest")
    for table in ("the veto head's latents", "step 5's latents"):
        hit = "the counterpart is the second dictionary's latent for most of these names"
        rows_ = m[table]
        print(f"  {table}: {len(rows_)} latents; the counterpart is the second's for the same names for {sum(x[hit] for x in rows_)}; at a cosine of 0.7 or more {sum(x['cosine of the best counterpart'] >= 0.7 for x in rows_)}; "
              + " ".join(f"{'+'.join(x['names'])[:14]} {x['cosine of the best counterpart']:.2f}{'*' if x[hit] else ''}" for x in rows_))

sides = lambda v: "" if v is None else "same in all three" if v["the same in all three"] else {(True, True): "as each, which differ", (True, False): "as the first only", (False, True): "as the second only", (False, False): "as neither"}[tuple(v["agrees with"])]
t3 = result["models"].get("a third seed")
if t3:
    print(f"\na third seed ({t3['name']}, dictionary after block {t3['the block of its dictionary']}; She names: {', '.join(t3['names that go on with She'])})")
    for row, new in zip(result["models"]["rows"], t3["rows"]):
        print(f"{row['step']:>2}  {row['what'][:98]:98} " + " ".join(show(v) for v in (*row["values"][:2], new["value"])) + f"   {sides(new)}")
    print("  ", t3["tally"])
    print("   the depth row with another line:", {k: round(v, 2) for k, v in t3["the depth row with another line"].items()})
    for s, heads_ in t3["the heads at the last position, by kind"].items():
        for h in heads_ or []:
            k = h["adds to an acting name against its own kind"]
            print(f"   {s} head {h['head']}: share {h['carry']:.2f}, reads the acting name with {h['reads the acting name with']:.2f}, adds to an acting name against its own kind: She {k['She']:.2f}, He {k['He']:.2f}"
                  + (f"   {sides(h[chr(34)[:0] + 'an acting She name, beside the first two models' + chr(39) + ' veto heads'])}" if s == THIRD else ""))
d3 = result["dictionaries"].get("a third dictionary")
if d3:
    print(f"\na third dictionary\n{'':4}{'':90} {'seed0':>20} {'seed1':>20}")
    for row, new in zip(result["dictionaries"]["rows"], d3["rows"]):
        v = row["values"]
        print(f"{row['step']:>2}  {row['what'][:90]:90} " + " ".join(show(x) for x in (v[0], v[1], new["values"][0], v[2], v[3], new["values"][1])) + "   " + " / ".join(sides(x) for x in new["by model"]))
    for seed, t in d3["tally"].items():
        print(f"   {seed}:", t)
    for seed, m in result["matches"].items():
        if "a third dictionary" in m:
            a = m["a third dictionary"]
            print(f"   {seed}: the first in the third: {pct(a['the first dictionary' + chr(39) + 's latents with a counterpart in the third'])}; the second in the third: {pct(a['the second dictionary' + chr(39) + 's latents with a counterpart in the third'])}; "
                  f"of the first, in both others at 0.7 or more: {100 * a['latents of the first with a counterpart at 0.7 or more in the second and in the third']:.0f}%; the shared latent: {a['the shared latent']}")

save("runs/limits/limits.json", result)
