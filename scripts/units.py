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
        kinds[kind] = entry
    for kind in ("batch", "wide"):  # the question, by the rule
        if kind in kinds and "plain" in kinds:
            kinds[kind]["against the plain dictionaries"] = {
                "a counterpart at 0.7 or more": against(kinds["plain"]["with a counterpart in the other seed"]["at 0.7 or more"], kinds[kind]["with a counterpart in the other seed"]["at 0.7 or more"]),
                "the stricter one-to-one match": against(kinds["plain"]["one to one"]["the same partner by encoder and decoder, 0.7 or more in both"],
                                                         kinds[kind]["one to one"]["the same partner by encoder and decoder, 0.7 or more in both"])}
    result["models"][seed] = kinds
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
save("runs/limits/units.json", result)
