"""
Step 8: do single latents come back with another seed, in a current kind of
dictionary? See docs/08-limits.md.

Step 5 trains a plain top-k dictionary: of every vector the k largest
latents stay. Steps 6 to 8 found that what such a dictionary says in
aggregate holds with another seed and that which latent is which mostly
does not: 9% and 13% of the latents of one have a counterpart in the other
at a cosine of 0.7 or more. A newer variant, the batch top-k dictionary of
Bussmann et al. (2024), keeps the largest latents over a whole batch while
it trains and a threshold when it is used (minterp/sae.py, across_batch).
This script sets pairs of dictionaries side by side that differ only in
their seed, by kind:

    plain   top-k, 4,096 latents: the first and the second of step 8
    batch   batch top-k, 4,096 latents       (--across-batch --dict b0, and --seed 1 --dict b1)
    wide    batch top-k, 8,192 latents       (--across-batch --width 8192 --dict w0, and --seed 1 --dict w1)

for seed0 and for seed1, at the same place (after block 4), with 32 active
latents per vector, 4,000 steps and the same training vectors.

What is compared was written here before any batch top-k dictionary was
trained on a model of this repository. The question is one: is the share
of latents with a counterpart in the other seed's dictionary higher than
the plain one's? It is read by step 8's rule, per model: the same when the
two shares differ by at most a fifth of the larger or by 0.05, and
otherwise higher or lower. It is read on two measures of the note's
section on a second dictionary: the best counterpart at a cosine of 0.7 or
more by the decoder, and the stricter one-to-one match with the same
partner by encoder and decoder. Whichever way it falls is reported, at
both widths. Beside it, so that a dictionary that matches better because
it is worse is seen as such: the variance it misses, how much the loss
rises, how many latents never fire on held-out stories, how many latents a
vector has on average, the share among latents that fire on more than 1%
of story tokens, and, from step 6 run on each, how many of the six "She"
names share the veto head's first choice and whether that latent's best
counterpart in the other seed is the other's shared latent. Nothing else
of steps 6 and 7 is run on them, and no row of the earlier tables changes.

    uv run scripts/units.py
"""

import json
import os

import numpy as np
import torch
from minllm.data import load_tokens

from minterp import match
from minterp import sae as S
from minterp.common import MINLLM, load, save
from minterp.tap import Tap

SEEDS = ("seed0", "seed1")
KINDS = {"plain": ("", "s1"), "batch": ("b0", "b1"), "wide": ("w0", "w1")}  # the tags scripts/sae.py gave the two dictionaries of a pair
BLOCK = 4


def record(folder, name, tag):
    path = f"runs/{folder}/{folder}-{name}{'-' + tag if tag else ''}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def same(a, b):
    return abs(a - b) <= max(0.2 * max(abs(a), abs(b)), 0.05)


def against(plain, other):
    return "the same" if same(plain, other) else "higher" if other > plain else "lower"


result = {"kinds": {k: list(v) for k, v in KINDS.items()}, "rule": "a share is the same as the plain one when the two differ by at most a fifth of the larger, or by 0.05",
          "models": {}}
for seed in SEEDS:
    model, tok, meta = load(seed)
    device = next(model.parameters()).device
    # Held-out stories, as scripts/limits.py takes them: where a latent fires, and how many a vector has.
    SEQ = model.cfg.max_seq_len
    data = load_tokens(os.path.join(MINLLM, "data/tinystories/valid.bin"))
    starts = np.random.default_rng(2).integers(0, len(data) - SEQ - 1, 300)
    streams = []
    for i in range(0, 300, 100):
        idx = torch.from_numpy(np.stack([data[k:k + SEQ].astype(np.int64) for k in starts[i:i + 100]])).to(device)
        with torch.no_grad(), Tap(model, keep=("resid",)) as tap:
            model(idx)
        streams.append(tap["resid", BLOCK].float().reshape(-1, model.cfg.dim).cpu())
    streams = torch.cat(streams)

    kinds = {}
    for kind, tags in KINDS.items():
        if not all(os.path.exists(S.path(seed, BLOCK, t)) for t in tags):
            continue
        one, two = (S.load(S.path(seed, BLOCK, t), "cpu") for t in tags)
        with torch.no_grad():
            on = [(d.encode(streams) > 0) for d in (one, two)]
        rate = on[0].float().mean(0)
        top = match.best(one, two)
        strict, by_decoder = match.shared(one, two)
        often = rate > 1e-2
        sae_records, step6 = [record("sae", seed, t) for t in tags], [record("features", seed, t) for t in tags]
        entry = {
            "width": one.width, "batch top-k": bool(one.across_batch),
            "with a counterpart in the other seed": match.shares(top[0]),
            "one to one": {"at a cosine of 0.7 or more by the decoder": by_decoder, "the same partner by encoder and decoder, 0.7 or more in both": strict},
            "among latents that fire on more than 1% of story tokens": {"latents": int(often.sum()), **match.shares(top[0][often])},
            "for comparison": {"the nearest other latent of the same dictionary": match.shares(match.inside(one)),
                               "two sets of random directions": match.shares(match.by_chance(one.width, two.width, one.dec.shape[1]))},
            "latents that never fire on the stories": [int((o.sum(0) == 0).sum()) for o in on], "story tokens": int(streams.shape[0]),
            "latents on per vector, mean": [o.sum(1).float().mean().item() for o in on],
            "variance missed": [r["rebuilds"]["unexplained variance"] for r in sae_records if r],
            "the loss rises by": [r["costs"]["loss rises by"] for r in sae_records if r],
        }
        if all(step6):
            she = step6[0]["names that go on with She"]
            count = lambda f: max(sum(x["the head's latent"] == j for x in f["names"] if x["name"] in she) for j in {x["the head's latent"] for x in f["names"]})
            a, b = step6[0]["the shared latent"], step6[1]["the shared latent"]
            entry["step 6"] = {"She names that share the veto head's first choice": [count(f) for f in step6], "the shared latents": [a, b],
                               "cosine of the first's best counterpart": top[0][a].item(), "which is the second's shared latent": bool(top[1][a].item() == b)}
        # For the figure: how the best cosines are spread, in 40 bins from 0 to 1 (the pair, and two sets of random directions).
        bins = lambda v: torch.histc(v.clamp(0, 1), bins=40, min=0, max=1).int().tolist()
        entry["best cosines, counts in 40 bins from 0 to 1"] = {"the pair": bins(top[0]), "random directions": bins(match.by_chance(one.width, two.width, one.dec.shape[1]))}
        kinds[kind] = entry
    # Looked at after the result, and not part of the question: a dictionary of another kind from the same seed.
    # The seed sets the first directions, so the plain first dictionary and the batch one called b0 start alike.
    if all(os.path.exists(S.path(seed, BLOCK, t)) for t in ("b0", "b1")):
        first = S.load(S.path(seed, BLOCK), "cpu")
        look = {}
        for label, tag in (("the batch dictionary from the same seed", "b0"), ("the batch dictionary from the other seed", "b1")):
            other = S.load(S.path(seed, BLOCK, tag), "cpu")
            top = match.best(first, other)
            look[label] = {**match.shares(top[0]), "the best counterpart has the same number": (top[1] == torch.arange(first.width)).float().mean().item()}
        kinds["plain"]["set beside another kind, looked at after the result"] = look
    for kind in ("batch", "wide"):  # the question, by the rule
        if kind in kinds and "plain" in kinds:
            kinds[kind]["against the plain dictionaries"] = {
                "a counterpart at 0.7 or more": against(kinds["plain"]["with a counterpart in the other seed"]["at 0.7 or more"], kinds[kind]["with a counterpart in the other seed"]["at 0.7 or more"]),
                "the stricter one-to-one match": against(kinds["plain"]["one to one"]["the same partner by encoder and decoder, 0.7 or more in both"],
                                                         kinds[kind]["one to one"]["the same partner by encoder and decoder, 0.7 or more in both"])}
    result["models"][seed] = kinds

    # Looked at after an independent check of the result, and not part of the question. A dictionary starts from random
    # directions that its seed sets (minterp/sae.py); they are made again here, through the same constructor, and set
    # beside where each latent ended. Nothing above depends on what follows.
    after, kept = {}, {}
    for kind, tags in KINDS.items():
        if kind not in kinds:
            continue
        pair = [S.load(S.path(seed, BLOCK, t), "cpu") for t in tags]
        to_start = []
        for t, d in zip(tags, pair):
            r = record("sae", seed, t)
            # The first dictionaries' records were written before a record named its seed; they were trained with the default of S.train, 0.
            s = r["seed"] if "seed" in r else 0 if t == "" else None
            assert s is not None, f"no seed recorded for {seed} {t}"
            torch.manual_seed(s)
            first = S.SAE(d.dec.shape[1], d.width, d.k, d.across_batch).dec.detach()
            to_start.append((match.unit(d.dec.detach().float()) * first).sum(1))
            assert (to_start[-1] >= 0.9).any(), f"no latent of {seed} {t} is near its start: the wrong seed?"
        with torch.no_grad():
            fires = sum((pair[0].encode(streams[i:i + 8192]) > 0).sum(0) for i in range(0, streams.shape[0], 8192))
        rate = fires.float() / streams.shape[0]
        top, back = match.best(*pair), match.best(pair[1], pair[0])
        stayed = to_start[0] >= 0.7
        left = ~stayed
        peak = (top[0] >= 0.15) & (top[0] < 0.25)
        often = rate.argsort(descending=True)[:500 * pair[0].width // 4096]
        e = {
            "share of the latents at a cosine to their first direction of 0.9, 0.7 and 0.5 or more, each dictionary": [[(c >= x).float().mean().item() for x in (0.9, 0.7, 0.5)] for c in to_start],
            "latents at 0.7 or more to their first direction, each dictionary": [int((c >= 0.7).sum()) for c in to_start],
            "first dictionary, how often a latent fires on story tokens, median: near its first direction, the others": [rate[stayed].median().item(), rate[left].median().item()],
            "first dictionary, the share of all firings that latents near their first direction make": (fires[stayed].sum() / fires.sum()).item(),
            "latents near their first direction with a counterpart at 0.7 or more": int((top[0][stayed] >= 0.7).sum()),
            "median best cosine of the latents near their first direction": top[0][stayed].median().item(),
            "among the latents that left their first direction": {"latents": int(left.sum()), **match.shares(top[0][left])},
            "best cosines from 0.15 to 0.25: latents, and of them near their first direction": [int(peak.sum()), int((peak & stayed).sum())],
            "best cosines of the latents that left, counts in 40 bins from 0 to 1": torch.histc(top[0][left].clamp(0, 1), bins=40, min=0, max=1).int().tolist(),
            "the latents of the first dictionary that fire most often, 500 of every 4,096, with a counterpart at 0.7 or more": (top[0][often] >= 0.7).float().mean().item(),
            # After a second check: the latents that left, at other lines to their start, and how often they fire.
            "among the latents under another line to their first direction, with a counterpart at 0.7 or more":
                {str(line): {"latents": int((to_start[0] < line).sum()), "share": (top[0][to_start[0] < line] >= 0.7).float().mean().item()} for line in (0.9, 0.5)},
            "the latents that left, how often they fire: the median, and the share on more than 1% of story tokens": [rate[left].median().item(), (rate[left] > 1e-2).float().mean().item()],
        }
        kept[kind] = (rate.clamp_min(1e-6).log10(), left, top[0])
        step6 = [record("features", seed, t) for t in tags]
        if all(step6):
            a, b = step6[0]["the shared latent"], step6[1]["the shared latent"]
            e["the second's shared latent, its best counterpart in the first"] = {"latent": back[1][b].item(), "cosine": back[0][b].item(),
                                                                                  "which is the first's shared latent": bool(back[1][b].item() == a),
                                                                                  "cosine to the first's shared latent": match.cosines(*pair)[a, b].item()}
        after[kind] = e
    fifth = lambda p, o: abs(p - o) <= 0.2 * max(p, o)
    for kind in ("batch", "wide"):
        if kind in after and "plain" in after:
            p, o = (after[k]["among the latents that left their first direction"]["at 0.7 or more"] for k in ("plain", kind))
            after[kind]["among the latents that left, against the plain dictionaries"] = {"by the rule": against(p, o), "within a fifth of the larger": fifth(p, o)}
            under = "among the latents under another line to their first direction, with a counterpart at 0.7 or more"
            after[kind]["under another line, against the plain dictionaries by the rule"] = {line: against(after["plain"][under][line]["share"], after[kind][under][line]["share"]) for line in ("0.9", "0.5")}
            q = kinds[kind]["against the plain dictionaries"]
            after[kind]["the question's two comparisons, within a fifth of the larger alone"] = {
                "a counterpart at 0.7 or more": fifth(kinds["plain"]["with a counterpart in the other seed"]["at 0.7 or more"], kinds[kind]["with a counterpart in the other seed"]["at 0.7 or more"]),
                "the stricter one-to-one match": fifth(kinds["plain"]["one to one"]["the same partner by encoder and decoder, 0.7 or more in both"],
                                                      kinds[kind]["one to one"]["the same partner by encoder and decoder, 0.7 or more in both"]),
                "by the rule, as above": q}
    # After a second check: the batch top-k latents that left, weighted to how often the plain ones that left fire (six bins of
    # the log of the firing rate). Only at the same width, since a rate does not mean the same rank in a dictionary twice as wide.
    if "batch" in kept and "plain" in kept:
        edges = (-6, -3.5, -3, -2.5, -2, -1.5, 0)
        (rate0, left0, _), (rate1, left1, best1) = kept["plain"], kept["batch"]
        num = den = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            weight = ((rate0 >= lo) & (rate0 < hi) & left0).sum().item()
            cell = (rate1 >= lo) & (rate1 < hi) & left1
            if cell.any():
                num, den = num + weight * (best1[cell] >= 0.7).float().mean().item(), den + weight
        after["batch"]["the latents that left, with a counterpart at 0.7 or more, weighted to the plain ones' firing rates"] = num / den
    if "set beside another kind, looked at after the result" in kinds.get("plain", {}):
        look = kinds["plain"]["set beside another kind, looked at after the result"]
        kind_only, both = (look[k]["at 0.7 or more"] for k in ("the batch dictionary from the same seed", "the batch dictionary from the other seed"))
        seed_only = kinds["plain"]["with a counterpart in the other seed"]["at 0.7 or more"]
        after["the plain first dictionary, a counterpart at 0.7 or more"] = {
            "another kind from the same seed": kind_only, "the same kind from another seed": seed_only, "another kind from another seed": both,
            "the first two by the rule": "the same" if same(kind_only, seed_only) else "differ"}
    result.setdefault("looked at after the check", {})[seed] = after
    del model

pct = lambda d: f"{100 * d['at a cosine of 0.9 or more']:.0f}% / {100 * d['at 0.7 or more']:.0f}% / {100 * d['at 0.5 or more']:.0f}%, median {d['median cosine']:.2f}"
for seed, kinds in result["models"].items():
    for kind, e in kinds.items():
        print(f"{seed} {kind:5s} ({e['width']} latents): a counterpart at 0.9 / 0.7 / 0.5: {pct(e['with a counterpart in the other seed'])}; "
              f"strict {100 * e['one to one']['the same partner by encoder and decoder, 0.7 or more in both']:.1f}%; "
              f"firing on more than 1% ({e['among latents that fire on more than 1% of story tokens']['latents']}): {100 * e['among latents that fire on more than 1% of story tokens']['at 0.7 or more']:.0f}%")
        print(f"      never fire {e['latents that never fire on the stories']}, on per vector {[round(x, 1) for x in e['latents on per vector, mean']]}, "
              f"variance missed {[round(x, 3) for x in e['variance missed']]}, the loss rises by {[round(x, 3) for x in e['the loss rises by']]}"
              + (f", step 6: {e['step 6']}" if "step 6" in e else "") + (f"  -> {e['against the plain dictionaries']}" if "against the plain dictionaries" in e else ""))
for seed, after in result.get("looked at after the check", {}).items():
    for kind, e in after.items():
        if kind not in KINDS:
            print(f"{seed} {kind}: {e}")
            continue
        left, shared = e["among the latents that left their first direction"], e.get("the second's shared latent, its best counterpart in the first")
        print(f"{seed} {kind:5s} near their first direction (0.9 / 0.7 / 0.5): {[[round(100 * x) for x in s] for s in e['share of the latents at a cosine to their first direction of 0.9, 0.7 and 0.5 or more, each dictionary']]}%; "
              f"of those left ({left['latents']}): {100 * left['at 0.7 or more']:.1f}% at 0.7, median {left['median cosine']:.2f}; "
              f"0.15 to 0.25: {e['best cosines from 0.15 to 0.25: latents, and of them near their first direction']}; "
              f"most often: {100 * e['the latents of the first dictionary that fire most often, 500 of every 4,096, with a counterpart at 0.7 or more']:.1f}%"
              + (f"; shared from the second: {shared}" if shared else ""))
save("runs/limits/units.json", result)
