"""
Step 4: what do the parts found in step 3 do, and which works through which? See docs/04-circuit.md.

The heads are not chosen by hand. They are every head that step 3 found to
carry at least 0.05 of the gap between the clean and the flipped prompt,
and every head that step 1 found to write at least 0.3 logits into the
answer in some version. For each of them this records

attention   from the position where it acts (the acting name or the last
            position, as step 3 found), how much it reads from the first
            name, the second name, the full stop that ends the first
            sentence, the acting name and the first token
writes      at the last position, what it adds to the logit of A, of B and
            of C, each against the mean of the nine other names
sources     the same writes split by the position they were read from
holds       for every earlier head and later head of the set: how much of
            the earlier one's share is left when the later one is held at
            the target run's own output, and how much comes along when the
            later head alone is given what it wrote in the patched run
steers      for the heads that read from the names: where they look in the
            flipped run when one earlier head is given its clean output
sets        the share carried by all heads of the set moved together, by
            all other heads moved together, and by random sets of as many

    uv run scripts/circuit.py
    uv run scripts/circuit.py --model seed1
"""

import argparse
import json

import torch

from minterp import attention, patch, tasks
from minterp.common import load, rounded, save
from minterp.tap import Tap, head_writes

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="seed0", help="seed0 or seed1: the steps before must have run on it")
parser.add_argument("--random-sets", type=int, default=200)
args = parser.parse_args()

model, tok, meta = load(args.model)
device = next(model.parameters()).device
n, H = model.cfg.n_layers, model.cfg.n_heads
VERSIONS = ("clean", "flipped", "neutral")

# The heads, from the records of steps 1 and 3.
with open(f"runs/patch/patch-{meta['name']}.json") as f:
    step3 = json.load(f)["templates"]
with open(f"runs/read/lens-{meta['name']}.json") as f:
    step1 = json.load(f)["attribution"]
carry = {s: torch.tensor([t["parts"]["carry"][s] for t in step3]).mean(0)[:, :H] for s in ("all", "last", "S")}  # (blocks, heads)
wrote = torch.stack([torch.tensor(step1[v]["heads"]["mean"]) for v in VERSIONS]).abs().max(0).values
chosen = [(b, h) for b in range(n) for h in range(H) if carry["all"][b, h] >= 0.05 or wrote[b, h] >= 0.3]
heads = [{
    "block": b + 1, "head": h, "carry": round(carry["all"][b, h].item(), 3),
    # A head acts at the acting name if step 3 found its share there; all others are read at the last position.
    "acts at": "S" if carry["S"][b, h] >= 0.05 else "end",
    "writes": [round(step1[v]["heads"]["mean"][b][h], 3) for v in VERSIONS],
} for b, h in chosen]
name_of = lambda x: f"block {x['block']} head {x['head']}"
people = tasks.names(tok)
ids = torch.tensor([tok.encode_ordinary(" " + p)[0] for p in people], device=device)
stop_id = tok.encode_ordinary(".")[0]
PLACES = ("A", "B", "stop", "S", "first token")
result = {"model": meta, "heads": heads, "versions": VERSIONS, "places": PLACES, "templates": []}
print("the heads:", ", ".join(f"{name_of(x)} (at {x['acts at']})" for x in heads))

key = lambda x: ("heads", x["block"])
readers = [x for x in heads if x["acts at"] == "end"]


def as_tap(store):
    return type("Store", (), {"__getitem__": lambda self, k: store[k]})()


def run_tapped(tokens, edits):
    with Tap(model, edit=edits) as tap, torch.no_grad():
        logits, _ = model(tokens)
    return tap, logits


generator = torch.Generator().manual_seed(0)
random_sets = [torch.randperm(n * H, generator=generator)[: len(heads)].tolist() for _ in range(args.random_sets)]
result["random sets"] = [[f"block {i // H + 1} head {i % H}" for i in sorted(s)] for s in random_sets]

for batch in tasks.batches(tok, device):
    p = batch.pos
    ld = lambda logits: tasks.logit_diff(logits.float(), batch.right, batch.wrong)
    store, value = {}, {}
    for version in VERSIONS:
        store[version], value[version] = patch.record(model, getattr(batch, version))
    base, goal = ld(value["flipped"]), ld(value["clean"])
    entry = {"template": batch.template, "attention": {}, "writes": {}, "sources": {}}
    # The full stop that ends the first sentence: the first one after the second name.
    stop = next(i for i in range(p["B"] + 1, p["S"]) if batch.clean[0, i].item() == stop_id)
    place = {"A": p["A"], "B": p["B"], "stop": stop, "S": p["S"], "first token": 0}

    index = {name: i for i, name in enumerate(people)}
    rows = torch.arange(len(batch), device=device)
    who = {k: torch.tensor([index[t[i]] for t in batch.names], device=device) for i, k in enumerate(("A", "B", "C"))}
    uninvolved = torch.ones(len(batch), len(people), dtype=torch.bool, device=device)
    for k in who:
        uninvolved[rows, who[k]] = False

    for version in VERSIONS:
        tap = as_tap(store[version])
        w = attention.patterns(model, tap)  # (prompts, blocks, heads, seq, seq)
        final = store[version]["resid", n][:, -1]
        size = torch.rsqrt(final.pow(2).mean(-1, keepdim=True) + model.norm.eps)
        for x in heads:
            row = w[:, x["block"] - 1, x["head"], p[x["acts at"]]]  # (prompts, seq): where it reads from
            entry["attention"].setdefault(name_of(x), {})[version] = {k: row[:, i].mean().item() for k, i in place.items()}
            if x["acts at"] == "end":
                write = head_writes(model, tap, x["block"])[:, -1, x["head"]]
                logits = ((write * model.norm.weight * size) @ model.lm_head.weight[ids].T).float()  # (prompts, names)
                rest = (logits * uninvolved).sum(1) / uninvolved.sum(1)
                entry["writes"].setdefault(name_of(x), {})[version] = {k: (logits[rows, who[k]] - rest).mean().item() for k in who}
                # The same write, split by where it was read: the weight on a position times that position's value.
                attn = model.layers[x["block"] - 1].attn
                v = attention.values(model, attn, model.layers[x["block"] - 1].attn_norm(store[version]["resid", x["block"] - 1]))
                hd = attn.head_dim
                O = attn.o_proj.weight[:, x["head"] * hd:(x["head"] + 1) * hd]  # (dim, head_dim)
                moved = (row[:, :, None] * v[:, x["head"]]) @ O.T  # (prompts, seq, dim): what came from each position
                by_pos = ((moved * model.norm.weight * size[:, None]) @ model.lm_head.weight[ids].T).float()  # (prompts, seq, names)
                by_pos = by_pos - (by_pos * uninvolved[:, None]).sum(2, keepdim=True) / uninvolved.sum(1)[:, None, None]
                split = {}
                for k in ("A", "B", "stop", "S"):
                    split[k] = {m: by_pos[rows, place[k], who[m]].mean().item() for m in ("A", "B")}
                total = {m: by_pos[rows, :, who[m]].sum(1).mean().item() for m in ("A", "B")}
                split["elsewhere"] = {m: total[m] - sum(split[k][m] for k in ("A", "B", "stop", "S")) for m in ("A", "B")}
                entry["sources"].setdefault(name_of(x), {})[version] = split

    # holds: the sender's clean output goes into the flipped run; the receiver keeps what it had there.
    entry["holds"] = []
    for s in heads:
        send = {key(s): patch.swap(store["clean"][key(s)], head=s["head"])}
        sent, logits = run_tapped(batch.flipped, send)
        alone = patch.share(ld(logits), base, goal)
        for r in heads:
            if r["block"] <= s["block"]:
                continue
            edits = dict(send)
            edits[key(r)] = patch.swap(store["flipped"][key(r)], head=r["head"])  # another block, so another key
            held = patch.share(ld(patch.run(model, batch.flipped, edits)), base, goal)
            # The converse: the earlier head is left alone, and the later head gets what it wrote when the earlier one was patched.
            relay = {key(r): patch.swap(sent[key(r)], head=r["head"])}
            relayed = patch.share(ld(patch.run(model, batch.flipped, relay)), base, goal)
            entry["holds"].append({"sender": name_of(s), "receiver": name_of(r), "alone": alone, "held": held, "relayed": relayed})

    # steers: where a reader of names looks in the flipped run, plain and with one earlier head given its clean output
    entry["steers"] = []
    plain = attention.patterns(model, as_tap(store["flipped"]))
    clean_w = attention.patterns(model, as_tap(store["clean"]))
    for r in readers:
        look = lambda w: {k: w[:, r["block"] - 1, r["head"], p["end"], place[k]].mean().item() for k in ("A", "B", "stop", "S")}
        for s in heads:
            if s["block"] >= r["block"]:
                continue
            tap, _ = run_tapped(batch.flipped, {key(s): patch.swap(store["clean"][key(s)], head=s["head"])})
            # What the reader then adds to the two names, on the scale of the patched run's final stream.
            final = tap["resid", n][:, -1]
            size = torch.rsqrt(final.pow(2).mean(-1, keepdim=True) + model.norm.eps)
            write = head_writes(model, tap, r["block"])[:, -1, r["head"]]
            logits = ((write * model.norm.weight * size) @ model.lm_head.weight[ids].T).float()
            rest = (logits * uninvolved).sum(1) / uninvolved.sum(1)
            entry["steers"].append({"reader": name_of(r), "sender": name_of(s), "flipped": look(plain), "clean": look(clean_w),
                                    "patched": look(attention.patterns(model, tap)),
                                    "patched writes": {k: (logits[rows, who[k]] - rest).mean().item() for k in ("A", "B")}})

    # sets: several heads moved together from the clean prompt into the flipped run
    def together(flat):
        edits = {}
        for i in flat:
            b, h = divmod(i, H)
            k = ("heads", b + 1)
            edit = patch.swap(store["clean"][k], head=h)
            edits[k] = (lambda x, first=edits[k], second=edit: second(first(x))) if k in edits else edit
        return patch.share(ld(patch.run(model, batch.flipped, edits)), base, goal)

    mine = [(x["block"] - 1) * H + x["head"] for x in heads]
    carriers = [(x["block"] - 1) * H + x["head"] for x in heads if x["carry"] >= 0.05]
    entry["sets"] = {
        "the heads": together(mine), "the heads that carry": together(carriers),
        "all other heads": together([i for i in range(n * H) if i not in mine]), "all heads": together(list(range(n * H))),
        "random": [together(s) for s in random_sets],
    }
    result["templates"].append(entry)
    print(batch.template, {k: round(v, 2) for k, v in entry["sets"].items() if k != "random"})

mean = lambda values: sum(values) / len(values)
T = result["templates"]
print("\nwhere each head reads from, at the position where it acts (clean | flipped | neutral)")
for x in heads:
    a = lambda v, k: mean([t["attention"][name_of(x)][v][k] for t in T])
    print(f"  {name_of(x):15s} at {x['acts at']:3s}  " + " | ".join(" ".join(f"{k} {a(v, k):.2f}" for k in ("A", "B", "stop", "S")) for v in VERSIONS))
print("\nwhat each head at the last position adds to a name's logit, against the nine names that are none of A, B and C (clean | flipped | neutral)")
for x in readers:
    wv = lambda v, k: mean([t["writes"][name_of(x)][v][k] for t in T])
    print(f"  {name_of(x):15s}  " + " | ".join(" ".join(f"{k} {wv(v, k):+.2f}" for k in ("A", "B", "C")) for v in VERSIONS))
print("\nwhere the writes of the heads at the last position come from, on clean and flipped prompts (to A, to B)")
for x in readers:
    sv = lambda v, k, m: mean([t["sources"][name_of(x)][v][k][m] for t in T])
    print(f"  {name_of(x):15s}  " + " | ".join(" ".join(f"{k} {sv(v, k, 'A'):+.2f},{sv(v, k, 'B'):+.2f}" for k in ("A", "B", "stop", "S", "elsewhere")) for v in ("clean", "flipped")))
print("\nholds: an earlier head's share alone, with a later head held, and relayed by the later head alone (only where held changes by 0.03 or more)")
for i in range(len(T[0]["holds"])):
    alone, held, relayed = (mean([t["holds"][i][k] for t in T]) for k in ("alone", "held", "relayed"))
    if abs(alone - held) >= 0.03:
        print(f"  {T[0]['holds'][i]['sender']:15s} {alone:5.2f} -> {held:5.2f} with {T[0]['holds'][i]['receiver']} held; relayed {relayed:5.2f}")
print("\nsteers: where a head at the last position looks in the flipped run (A, B), with one earlier head from the clean run (only shifts of 0.05 or more)")
for i in range(len(T[0]["steers"])):
    g = lambda which, k: mean([t["steers"][i][which][k] for t in T])
    if abs(g("patched", "A") - g("flipped", "A")) + abs(g("patched", "B") - g("flipped", "B")) >= 0.05:
        s = T[0]["steers"][i]
        print(f"  {s['reader']:15s} flipped A {g('flipped', 'A'):.2f} B {g('flipped', 'B'):.2f} | with {s['sender']} clean: A {g('patched', 'A'):.2f} B {g('patched', 'B'):.2f} | clean run A {g('clean', 'A'):.2f} B {g('clean', 'B'):.2f}")
sets = {k: mean([t["sets"][k] for t in T]) for k in T[0]["sets"] if k != "random"}
rand = torch.tensor([t["sets"]["random"] for t in T]).mean(0)
print(f"\nsets of heads moved together: {', '.join(f'{k} {v:.2f}' for k, v in sets.items())}")
print(f"  {len(random_sets)} random sets of {len(heads)} heads: median {rand.median():.2f}, 95th percentile {rand.quantile(0.95):.2f}, largest {rand.max():.2f}")

for t in T:  # the record keeps means over the prompts; round them
    t["sets"]["random"] = rounded(torch.tensor(t["sets"]["random"]), 3)
save(f"runs/circuit/circuit-{meta['name']}.json", json.loads(json.dumps(result), parse_float=lambda s: round(float(s), 4)))
