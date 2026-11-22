"""
Step 3: which places does the answer depend on? See docs/03-patch.md.

Four experiments, each per sentence pattern, on every ordered pair of names.

travel   The stream at one depth and one group of positions, moved between
         the clean and the flipped prompt, in both directions:
           carry   clean into the flipped run: does the clean answer come along?
           break   flipped into the clean run: does the clean answer go?
         Measured on the logit of B minus the logit of A, as the share of
         the way from the target run's value to the source run's.

name     The stream moved from the prompt with C in place of B into the
         clean run, measured on the logit of C minus the logit of B: where
         is the second name, at each depth? And its mirror: the stream of
         the prompt with C in place of A moved into the flipped run,
         measured on the logit of C minus the logit of A.

parts    Every head and MLP, its output moved between the clean and the
         flipped prompt: at all positions, at the last one only, at all but
         the last, and at the acting name only. The four parts with the
         largest share are also moved two at a time.

choice   Every head and MLP of the clean run replaced in four ways: by the
         flipped prompt's, by the neutral prompt's, by the average over the
         three versions, and by nothing. Measured as the change of the logit
         difference, in logits.

    uv run scripts/patch.py
    uv run scripts/patch.py --model seed1
    uv run scripts/patch.py --model qwen --stream-only   # step 8: travel and name only, minutes
    uv run scripts/patch.py --model qwen                 # step 8: with parts and choice, hours
"""

import argparse

import torch

from minterp import patch, tasks
from minterp.common import load, rounded, save

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="seed0", help="seed0, seed1, qwen, or the path of a checkpoint")
parser.add_argument("--stream-only", action="store_true", help="travel and name only, without the heads and MLPs: for a larger model (step 8)")
args = parser.parse_args()

model, tok, meta = load(args.model)
device = next(model.parameters()).device
n, heads = model.cfg.n_layers, model.cfg.n_heads
PARTS = [("heads", b, h) for b in range(1, n + 1) for h in range(heads)] + [("mlp", b, None) for b in range(1, n + 1)]
TRAVEL = ("before", "A", "B", "S", "S+1", "between", "end", "all")
NAME = ("B", "B to S", "S", "S+1", "between", "end")
FIRST = ("A", "A to S", "S", "S+1", "between", "end")
SCOPES = ("all", "last", "not last", "S")
CHOICES = ("flipped", "neutral", "mean", "zero")
result = {"model": meta, "travel groups": TRAVEL, "name groups": NAME, "first name groups": FIRST, "scopes": SCOPES, "choices": CHOICES, "templates": []}


def grid(parts_values):
    """A value per part -> (blocks, heads + 1), the MLP in the last column."""
    t = torch.tensor(parts_values)
    return torch.cat([t[: n * heads].view(n, heads), t[n * heads:, None]], dim=1)


def swap_part(part, source, positions=None):
    kind, block, head = part
    return {(kind, block): patch.swap(source[kind, block], positions=positions, head=head)}


for batch in tasks.batches(tok, device):
    p, T = batch.pos, batch.clean.shape[1]
    ld = lambda logits: tasks.logit_diff(logits.float(), batch.right, batch.wrong)  # B minus A
    cb = lambda logits: tasks.logit_diff(logits.float(), batch.third, batch.right)  # C minus B
    ca = lambda logits: tasks.logit_diff(logits.float(), batch.third, batch.wrong)  # C minus A
    rms = lambda x: x.float().pow(2).mean(-1).sqrt().mean().item()
    store, value = {}, {}
    for version in ("clean", "flipped", "neutral", "other", "other_first"):
        store[version], logits = patch.record(model, getattr(batch, version))
        value[version] = logits
    entry = {"template": batch.template, "logit_diff": {v: ld(value[v]).mean().item() for v in ("clean", "flipped", "neutral")}}

    # travel
    groups = {
        "before": [i for i in range(p["S"]) if i not in (p["A"], p["B"])], "A": [p["A"]], "B": [p["B"]], "S": [p["S"]],
        "S+1": [p["S"] + 1], "between": list(range(p["S"] + 2, p["end"])), "end": [p["end"]], "all": list(range(T)),
    }
    entry["travel"] = {}
    for name, src, dst in (("carry", "clean", "flipped"), ("break", "flipped", "clean")):
        entry["travel"][name] = rounded(torch.tensor([[
            patch.share(ld(patch.run(model, getattr(batch, dst), {("resid", d): patch.swap(store[src]["resid", d], positions=groups[g])})),
                        ld(value[dst]), ld(value[src]))
            for g in TRAVEL] for d in range(n + 1)]), 3)  # (depth, groups)

    # name
    groups = {
        "B": [p["B"]], "B to S": list(range(p["B"] + 1, p["S"])), "S": [p["S"]], "S+1": [p["S"] + 1],
        "between": list(range(p["S"] + 2, p["end"])), "end": [p["end"]],
    }
    entry["name"] = rounded(torch.tensor([[
        patch.share(cb(patch.run(model, batch.clean, {("resid", d): patch.swap(store["other"]["resid", d], positions=groups[g])})),
                    cb(value["clean"]), cb(value["other"]))
        for g in NAME] for d in range(n + 1)]), 3)
    entry["name measure"] = {"clean": cb(value["clean"]).mean().item(), "other": cb(value["other"]).mean().item()}
    groups = {
        "A": [p["A"]], "A to S": list(range(p["A"] + 1, p["S"])), "S": [p["S"]], "S+1": [p["S"] + 1],
        "between": list(range(p["S"] + 2, p["end"])), "end": [p["end"]],
    }
    entry["first name"] = rounded(torch.tensor([[
        patch.share(ca(patch.run(model, batch.flipped, {("resid", d): patch.swap(store["other_first"]["resid", d], positions=groups[g])})),
                    ca(value["flipped"]), ca(value["other_first"]))
        for g in FIRST] for d in range(n + 1)]), 3)
    entry["first name measure"] = {"flipped": ca(value["flipped"]).mean().item(), "other": ca(value["other_first"]).mean().item()}
    # How large the stream is at the acting name: the embedding, the stream after block 1, and what block 1 added.
    at = store["clean"]
    entry["size at S"] = {"embedding": rms(at["resid", 0][:, p["S"]]), "after block 1": rms(at["resid", 1][:, p["S"]]),
                          "attention 1": rms(at["attn", 1][:, p["S"]]), "MLP 1": rms(at["mlp", 1][:, p["S"]])}

    if args.stream_only:
        result["templates"].append(entry)
        print(batch.template, {k: round(v, 2) for k, v in entry["logit_diff"].items()})
        continue

    # parts
    scopes = {"all": None, "last": [p["end"]], "not last": list(range(p["end"])), "S": [p["S"]]}
    entry["parts"] = {}
    for name, src, dst in (("carry", "clean", "flipped"), ("break", "flipped", "clean")):
        entry["parts"][name] = {
            scope: rounded(grid([
                patch.share(ld(patch.run(model, getattr(batch, dst), swap_part(part, store[src], positions))), ld(value[dst]), ld(value[src]))
                for part in PARTS]), 3)
            for scope, positions in scopes.items()
        }

    entry["carry by part"] = [patch.share(ld(patch.run(model, batch.flipped, swap_part(part, store["clean"]))), ld(value["flipped"]), ld(value["clean"])) for part in PARTS]

    # choice
    three = [store[v] for v in ("clean", "flipped", "neutral")]
    entry["choice"] = {}
    for choice in CHOICES:
        changes = []
        for part in PARTS:
            key = part[:2]
            source = {"flipped": store["flipped"][key], "neutral": store["neutral"][key],
                      "mean": patch.mean(three, key).expand_as(store["clean"][key]), "zero": patch.zero(store["clean"][key])}[choice]
            changes.append((ld(patch.run(model, batch.clean, swap_part(part, {key: source}))) - ld(value["clean"])).mean().item())
        entry["choice"][choice] = rounded(grid(changes), 3)
    result["templates"].append(entry)
    print(batch.template, {k: round(v, 2) for k, v in entry["logit_diff"].items()})

if not args.stream_only:
    # The four parts with the largest share, block 1's MLP aside, moved two at a time: do their shares add?
    single = torch.tensor([t.pop("carry by part") for t in result["templates"]]).mean(0)
    single[PARTS.index(("mlp", 1, None))] = 0
    top = [PARTS[i] for i in single.abs().argsort(descending=True)[:4].tolist()]
    name_of = lambda part: f"block {part[1]} " + ("MLP" if part[0] == "mlp" else f"head {part[2]}")
    pairs = [(a, b) for i, a in enumerate(top) for b in top[i + 1:]]
    together = {pair: [] for pair in pairs}
    for batch in tasks.batches(tok, device):
        ld = lambda logits: tasks.logit_diff(logits.float(), batch.right, batch.wrong)
        clean, clean_logits = patch.record(model, batch.clean)
        _, flipped_logits = patch.record(model, batch.flipped)
        for a, b in pairs:
            edits = {}
            for part in (a, b):  # two heads of one block share a key, so their edits are chained
                key, edit = next(iter(swap_part(part, clean).items()))
                edits[key] = (lambda x, first=edits[key], second=edit: second(first(x))) if key in edits else edit
            together[a, b].append(patch.share(ld(patch.run(model, batch.flipped, edits)), ld(flipped_logits), ld(clean_logits)))
    result["pairs"] = [{"parts": [name_of(a), name_of(b)], "each": [round(single[PARTS.index(a)].item(), 3), round(single[PARTS.index(b)].item(), 3)],
                        "together": round(sum(v) / len(v), 3)} for (a, b), v in together.items()]

mean_of = lambda pick: torch.tensor([pick(t) for t in result["templates"]]).mean(0)
show = lambda row: " ".join(f"{v:6.2f}" for v in row.tolist())
for name in ("carry", "break"):
    print(f"\ntravel, {name}: share of the gap, mean of the patterns (rows: after block; columns: {', '.join(TRAVEL)})")
    for d, row in enumerate(mean_of(lambda t: t["travel"][name])):
        print(f"  {'emb' if d == 0 else d:>3}  " + show(row))
print(f"\nname: share of the way from B to C (columns: {', '.join(NAME)})")
for d, row in enumerate(mean_of(lambda t: t["name"])):
    print(f"  {'emb' if d == 0 else d:>3}  " + show(row))
print(f"\nfirst name: share of the way from A to C (columns: {', '.join(FIRST)})")
for d, row in enumerate(mean_of(lambda t: t["first name"])):
    print(f"  {'emb' if d == 0 else d:>3}  " + show(row))
if args.stream_only:
    save(f"runs/patch/patch-{meta['name']}.json", result)
    raise SystemExit(0)
print("\ntwo parts moved together, against the sum of their shares")
for pair in result["pairs"]:
    print(f"  {pair['parts'][0]:16s} + {pair['parts'][1]:16s} together {pair['together']:.2f}, sum {sum(pair['each']):.2f}")
for name in ("carry", "break"):
    print(f"\nparts, {name}, all positions (rows: blocks; columns: heads, then the MLP)")
    for b, row in enumerate(mean_of(lambda t: t["parts"][name]["all"]), start=1):
        print(f"  {b:>3}  " + show(row))
carry = {s: mean_of(lambda t, s=s: t["parts"]["carry"][s]) for s in SCOPES}
choice = {c: mean_of(lambda t, c=c: t["choice"][c]) for c in CHOICES}
ranked = sorted(((carry["all"][b, h].abs().item(), b, h) for b in range(n) for h in range(heads + 1)), reverse=True)[:10]
print("\nthe ten parts with the largest carry, where they act, and the four replacements on the clean run (logits)")
print(f"  {'':16s} {'carry':>6} {'last':>6} {'before':>6} | " + " ".join(f"{c:>8}" for c in CHOICES))
for _, b, h in ranked:
    label = f"block {b + 1} " + ("MLP" if h == heads else f"head {h}")
    print(f"  {label:16s} {carry['all'][b, h]:6.2f} {carry['last'][b, h]:6.2f} {carry['not last'][b, h]:6.2f} | "
          + " ".join(f"{choice[c][b, h]:+8.2f}" for c in CHOICES))

save(f"runs/patch/patch-{meta['name']}.json", result)
