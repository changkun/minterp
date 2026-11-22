"""
Step 8, the open model part by part: where the heads that step 3's patch
found look, what they write, and which of them the head at the acting name
works through. See docs/08-limits.md.

The heads are not chosen by hand. They are every head whose share of the
gap between the clean and the flipped prompt is 0.05 or more in either
direction, in the record of scripts/patch.py. For each of them, per
sentence pattern and version of the prompt, this records

attention   from the position where it acts (the acting name if step 3
            found its share there, the last position otherwise), how much
            it reads from the first token, the first name, the second name,
            the full stop that ends the first sentence, the acting name,
            the token after it and the last position; and the position it
            reads most from among all the others
writes      for a head read at the last position, what it adds to the
            logit of A, of B and of C, each against the mean of the nine
            other names
by kind     the same on clean and flipped prompts for the name that acts
            and the name that is due, each against the uninvolved names of
            its own kind, where a kind is the names the model follows with
            " She" or with " He" (step 6)
holds       for a head that acts at the acting name and every later head:
            its share alone, and what is left of it when the later head is
            held at the flipped run's own output (step 4)

The first two and the last are measures of step 4 (scripts/circuit.py), on
another choice of heads: step 4 takes the heads with a positive share or a
large direct write. tests/test_limits.py checks that on the story models
the two scripts give the same numbers for the heads they share.

    uv run scripts/heads.py                  # the open model: under a minute
    uv run scripts/heads.py --model seed0    # a story model, as a check against step 4
"""

import argparse
import json

import torch

from minterp import attention, patch, tasks
from minterp.common import eot, load, save
from minterp.tap import Tap, head_writes

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="qwen", help="qwen, seed0, seed1 or seed2: scripts/patch.py must have run on it, with its parts")
args = parser.parse_args()

model, tok, meta = load(args.model)
device = next(model.parameters()).device
n, H = model.cfg.n_layers, model.cfg.n_heads
VERSIONS = ("clean", "flipped", "neutral")
PLACES = ("first token", "A", "B", "stop", "S", "S+1", "end")

with open(f"runs/patch/patch-{meta['name']}.json") as f:
    step3 = json.load(f)["templates"]
carry = {s: torch.tensor([t["parts"]["carry"][s] for t in step3]).mean(0)[:, :H] for s in ("all", "S")}  # (blocks, heads)
heads = [{
    "block": b + 1, "head": h, "carry": round(carry["all"][b, h].item(), 3),
    # A head acts at the acting name if step 3 found its share there; all others are read at the last position.
    "acts at": "S" if carry["S"][b, h].abs() >= 0.05 else "end",
} for b in range(n) for h in range(H) if carry["all"][b, h].abs() >= 0.05]
name_of = lambda x: f"block {x['block']} head {x['head']}"
key = lambda x: ("heads", x["block"])
people = tasks.names(tok)
ids = torch.tensor([tok.encode_ordinary(" " + p)[0] for p in people], device=device)
stop_id = tok.encode_ordinary(".")[0]

# kinds, as scripts/open.py finds them: after a sentence about one name, does the model go on with " She" or " He"?
she_id, he_id = (tok.encode_ordinary(w)[0] for w in (" She", " He"))
with torch.no_grad():
    after = model(torch.tensor([[eot(tok)] + tok.encode_ordinary(f"One day, {p} went to the park.") for p in people], device=device))[0][:, -1].float()
she = after[:, she_id] > after[:, he_id]

result = {"model": meta, "heads": heads, "versions": VERSIONS, "places": PLACES,
          "names that go on with She": [p for p, g in zip(people, she.tolist()) if g], "templates": []}
print("the heads:", ", ".join(f"{name_of(x)} (at {x['acts at']})" for x in heads))

for batch in tasks.batches(tok, device, people=people):
    p = batch.pos
    entry = {"template": batch.template, "attention": {}, "elsewhere": {}, "writes": {}, "by kind": {}, "holds": []}
    # The full stop that ends the first sentence: the first one after the second name.
    stop = next(i for i in range(p["B"] + 1, p["S"]) if batch.clean[0, i].item() == stop_id)
    place = {"first token": 0, "A": p["A"], "B": p["B"], "stop": stop, "S": p["S"], "S+1": p["S"] + 1, "end": p["end"]}
    index = {name: i for i, name in enumerate(people)}
    rows = torch.arange(len(batch), device=device)
    who = {k: torch.tensor([index[t[i]] for t in batch.names], device=device) for i, k in enumerate(("A", "B", "C"))}
    uninvolved = torch.ones(len(batch), len(people), dtype=torch.bool, device=device)
    for k in who:
        uninvolved[rows, who[k]] = False

    for version in VERSIONS:
        with torch.no_grad(), Tap(model, keep=("resid", "heads")) as tap:
            model(getattr(batch, version))
        final = tap["resid", n][:, -1].float()
        size = torch.rsqrt(final.pow(2).mean(-1, keepdim=True) + model.norm.eps)
        weights = {}
        for x in heads:
            b = x["block"]
            if b not in weights:
                with torch.no_grad():
                    weights = {b: attention.weights(model, model.layers[b - 1].attn, model.layers[b - 1].attn_norm(tap["resid", b - 1])).float()}
            row = weights[b][:, x["head"], p[x["acts at"]]]  # (prompts, seq): where it reads from
            entry["attention"].setdefault(name_of(x), {})[version] = {k: row[:, i].mean().item() for k, i in place.items()}
            # The most-read position that is none of the seven places.
            others = [i for i in range(p[x["acts at"]] + 1) if i not in place.values()]
            most = max(others, key=lambda i: row[:, i].mean().item())
            entry["elsewhere"].setdefault(name_of(x), {})[version] = {"token": batch.labels[most], "weight": row[:, most].mean().item()}
            if x["acts at"] == "end":
                write = head_writes(model, tap, b)[:, -1, x["head"]].float()
                logits = (write * model.norm.weight.float() * size) @ model.lm_head.weight[ids].float().T  # (prompts, names)
                rest = (logits * uninvolved).sum(1) / uninvolved.sum(1)
                entry["writes"].setdefault(name_of(x), {})[version] = {k: (logits[rows, who[k]] - rest).mean().item() for k in who}
                if version != "neutral":
                    # Step 6's question: against the uninvolved names of the name's own kind, by that kind.
                    acts, due = (who["A"], who["B"]) if version == "clean" else (who["B"], who["A"])
                    split = {}
                    for role, target in (("acts", acts), ("due", due)):
                        own = uninvolved & (she[None, :] == she[target][:, None])
                        against_own = logits[rows, target] - (logits * own).sum(1) / own.sum(1).clamp(min=1)
                        usable = own.sum(1) > 0
                        split[role] = {label: against_own[mask & usable].mean().item() for label, mask in (("She", she[target]), ("He", ~she[target]))}
                    entry["by kind"].setdefault(name_of(x), {})[version] = split
        del tap, weights

    # holds: the sender's clean output goes into the flipped run; the later head keeps what it had there.
    ld = lambda logits: tasks.logit_diff(logits.float(), batch.right, batch.wrong)
    clean, clean_logits = patch.record(model, batch.clean, keep=("heads",))
    flipped, flipped_logits = patch.record(model, batch.flipped, keep=("heads",))
    base, goal = ld(flipped_logits), ld(clean_logits)
    for s in heads:
        if s["acts at"] != "S":
            continue
        send = {key(s): patch.swap(clean[key(s)], head=s["head"])}
        alone = patch.share(ld(patch.run(model, batch.flipped, send)), base, goal)
        for r in heads:
            if r["block"] <= s["block"]:
                continue
            edits = dict(send)
            edits[key(r)] = patch.swap(flipped[key(r)], head=r["head"])  # another block, so another key
            entry["holds"].append({"sender": name_of(s), "receiver": name_of(r), "alone": alone,
                                   "held": patch.share(ld(patch.run(model, batch.flipped, edits)), base, goal)})
    del clean, flipped
    result["templates"].append(entry)

T = result["templates"]
mean = lambda pick: sum(pick(t) for t in T) / len(T)
print(f"\ngoes on with She after {', '.join(result['names that go on with She'])}")
print("\nwhere each head reads from, at the position where it acts (clean | flipped | neutral): " + ", ".join(PLACES) + "; then the most-read other token on clean prompts, by pattern")
for x in heads:
    print(f"  {name_of(x):17s} {x['carry']:6.2f} at {x['acts at']:3s}  " + "  |  ".join(
        " ".join(f"{mean(lambda t, v=v, k=k: t['attention'][name_of(x)][v][k]):.2f}" for k in PLACES) for v in VERSIONS)
        + "   " + ", ".join(f"{t['elsewhere'][name_of(x)]['clean']['token'].strip()} {t['elsewhere'][name_of(x)]['clean']['weight']:.2f}" for t in T))
print("\nwhat the heads at the last position add to the logit of A, B, C against the other names (clean | flipped | neutral)")
for x in heads:
    if x["acts at"] == "end":
        print(f"  {name_of(x):17s}  " + "  |  ".join(
            " ".join(f"{mean(lambda t, v=v, k=k: t['writes'][name_of(x)][v][k]):+.2f}" for k in ("A", "B", "C")) for v in VERSIONS))
print("\nthe same against the uninvolved names of the name's own kind: the name that acts if She, if He; the name that is due if She, if He (clean | flipped)")
for x in heads:
    if x["acts at"] == "end":
        print(f"  {name_of(x):17s}  " + "  |  ".join(
            " ".join(f"{mean(lambda t, v=v, r=r, g=g: t['by kind'][name_of(x)][v][r][g]):+.2f}" for r in ("acts", "due") for g in ("She", "He")) for v in ("clean", "flipped")))
print("\nholds: the share of a head at the acting name alone, and what is left when a later head is held at the flipped run's own output")
for i, h in enumerate(T[0]["holds"]):
    print(f"  {h['sender']} alone {mean(lambda t: t['holds'][i]['alone']):.2f}; with {h['receiver']:17s} held {mean(lambda t: t['holds'][i]['held']):.2f}")

save(f"runs/open/heads-{meta['name']}.json", json.loads(json.dumps(result), parse_float=lambda s: round(float(s), 4)))
