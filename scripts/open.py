"""
Step 8: the same task in a published model. See docs/08-limits.md.

Qwen2.5-0.5B-Instruct, read from its published weights by minllm's stage 5
loader: 24 blocks, 14 heads, width 896, 0.5 billion parameters. It goes
into the same Transformer class as the story model, so the scripts of
steps 1 and 3 run on it as they are:

    uv run scripts/behavior.py --model qwen
    uv run scripts/lens.py --model qwen
    uv run scripts/patch.py --model qwen --stream-only   # minutes; without the flag every part is moved, in hours
    uv run scripts/heads.py                              # what the heads found by the full run read and write

This script repeats the first experiment of step 7 there, without a
dictionary, at the acting name's position.

kinds    which names the model follows with " She" and which with " He",
         found as step 5 found them
steer    on a clean prompt, after the block in the middle of the model
         (--block): the directions of step 7 that need no dictionary and
         its controls, at doses from -2 to 2. The random control is as
         long as the pair direction.
depth    at dose 1 after every second block: the pair direction, beside
         what step 3's patch at the acting name carries at that depth, and
         the names' means, the vector every flipped prompt shares, and the
         two together

    uv run scripts/open.py
    uv run scripts/open.py --model seed0 --block 4   # the story models, for the comparison by depth
    uv run scripts/open.py --model seed1 --block 4
"""

import argparse
import json

import torch

from minterp import steer, tasks
from minterp.common import eot, load, rounded, save
from minterp.tap import Tap

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="qwen")
parser.add_argument("--block", type=int, default=None, help="where to steer: by default after the block in the middle")
args = parser.parse_args()

model, tok, meta = load(args.model)
device = next(model.parameters()).device
n = model.cfg.n_layers
block = args.block if args.block is not None else n // 2
people = tasks.names(tok)
k = len(people)
name_ids = torch.tensor([tok.encode_ordinary(" " + p)[0] for p in people], device=device)
DOSES = (-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
generator = torch.Generator().manual_seed(0)
more = torch.Generator().manual_seed(1)  # for a control added later, so that the first keeps its draws
series = lambda values: rounded(torch.tensor(values))

# kinds: after a sentence about one name, does the model go on with " She" or " He"?
she_id, he_id = (tok.encode_ordinary(w)[0] for w in (" She", " He"))
with torch.no_grad():
    after = model(torch.tensor([[eot(tok)] + tok.encode_ordinary(f"One day, {p} went to the park.") for p in people], device=device))[0][:, -1].float()
pronoun = after[:, she_id] - after[:, he_id]
she = pronoun > 0
result = {"model": meta, "block": block, "doses": DOSES, "names": people, "names that go on with She": [p for p, g in zip(people, she.tolist()) if g],
          "logit of She minus logit of He after each name": rounded(pronoun, 2)}

batches = tasks.batches(tok, device, people=people)
A, B, C = (torch.tensor([people.index(t[i]) for t in batches[0].names], device=device) for i in range(3))
rows = torch.arange(len(A), device=device)
acts = torch.stack([A, B])
D, E = steer.stand_ins(A, B, C, she)
apart = torch.ones(len(A), k, dtype=torch.bool, device=device)
for names in (A, B, C, D, E):
    apart[rows, names] = False
own_kind = {"A": apart & (she[None] == she[A][:, None]), "B": apart & (she[None] == she[B][:, None])}
additive = steer.term_per_name(A, B, k)
kinds = {"both She": she[A] & she[B], "both He": ~she[A] & ~she[B], "one of each": she[A] != she[B]}
result["pairs"] = {g: int(m.sum()) for g, m in kinds.items()}

# The stream at the acting name after every block, and the plain answers, for the clean and the flipped prompt.
at_name, plain, third = [], [], []
for batch in batches:
    per = []
    for version in ("clean", "flipped"):
        with torch.no_grad(), Tap(model, keep=("resid",)) as tap:
            logits, _ = model(getattr(batch, version))
        per.append((torch.stack([tap["resid", d][:, batch.pos["S"]].float() for d in range(n + 1)]), logits[:, -1].float()))
    at_name.append(torch.stack([x for x, _ in per]))
    plain.append(torch.stack([l for _, l in per]))
    with torch.no_grad():
        third.append(model(batch.neutral)[0][:, -1].float())
at_name, plain = torch.stack(at_name), torch.stack(plain)  # (patterns, versions, depths, prompts, dim) and (patterns, versions, prompts, vocab)


def directions_at(p, depth, controls=True):
    """The directions of step 7 for pattern p after one block, built on the other four patterns."""
    others = [q for q in range(len(batches)) if q != p]
    x = at_name[others][:, :, depth]  # (patterns, versions, prompts, dim)
    actor = acts[None].expand(len(others), -1, -1)
    mean_when = torch.stack([x[actor == i].mean(0) for i in range(k)])
    pair = (x[:, 1] - x[:, 0]).mean(0)
    order = pair.mean(0, keepdim=True).expand_as(pair)
    means = mean_when[B] - mean_when[A]
    if not controls:
        return {"pair": pair, "means": means, "order": order, "means + order": means + order}
    here = at_name[p][:, depth]
    return {"own": here[1] - here[0], "pair": pair, "names": additive @ pair, "means + order": means + order, "means": means, "order": order,
            "kind": steer.kind_only(mean_when, A, B, she), "other names": mean_when[D] - mean_when[E],
            "random, long": steer.like(pair, generator), "means + random": means + steer.like(order, more), "wipe": here.mean((0, 1))[None] - here[0]}


OUTSIDE = "to the flipped prompt, outside the two names"
NAMES = ("own", "pair", "names", "means + order", "means", "order", "kind", "other names", "random, long", "means + random", "wipe")
BY_DEPTH = ("pair", "means", "order", "means + order")
result["directions"], result["templates"] = NAMES, []
for p, batch in enumerate(batches):
    ld = lambda logits: tasks.logit_diff(logits, batch.right, batch.wrong)
    clean, flipped = ld(plain[p, 0]), ld(plain[p, 1])
    gap = (clean - flipped).mean()
    names_out = torch.stack([batch.right, batch.wrong], dim=1)
    own = {x: lambda logits, x=x, t=t: steer.against(logits[:, name_ids], t, own_kind[x]) for x, t in (("A", A), ("B", B))}
    between = steer.elsewhere(plain[p, 1], plain[p, 0], names_out)
    directions = directions_at(p, block)
    entry = {"template": batch.template, "gap": gap.item(), "stream size at the acting name": at_name[p, 0, block].norm(dim=1).mean().item(),
             "pairs": {g: {"gap": (clean - flipped)[m].mean().item()} for g, m in kinds.items()},
             "length": {name: {"all": d.norm(dim=1).mean().item(), **{g: d[m].norm(dim=1).mean().item() for g, m in kinds.items()}} for name, d in directions.items()},
             "plain": {"clean": clean.mean().item(), "flipped": flipped.mean().item(), "a third name acts, share": ((clean - ld(third[p])).mean() / gap).item(),
                       "the flipped prompt turned": (flipped < 0).float().mean().item(), "the clean prompt turned": (clean < 0).float().mean().item(),
                       "from the clean prompt to the flipped, outside the two names": between.mean().item(),
                       "the same, by pair": {g: between[m].mean().item() for g, m in kinds.items()}},
             "steered": {}, "depth": {name: {"share": [], "turned": []} for name in BY_DEPTH}}
    for name, d in directions.items():
        out = {m: [] for m in ("share", "turned", "A against its own kind", "B against its own kind", OUTSIDE)}
        by = {g: [] for g in kinds}
        for dose in DOSES:
            logits = steer.steered(model, batch.clean, block, d, [batch.pos["S"]], dose)
            moved = clean - ld(logits)
            out["share"].append((moved.mean() / gap).item())
            out["turned"].append((ld(logits) < 0).float().mean().item())
            for x in ("A", "B"):
                out[f"{x} against its own kind"].append(((own[x](logits) - own[x](plain[p, 0])).mean() / (own[x](plain[p, 1]) - own[x](plain[p, 0])).mean()).item())
            out[OUTSIDE].append(steer.elsewhere(plain[p, 1], logits, names_out).mean().item())
            for g, m in kinds.items():
                by[g].append(moved[m].mean().item())
        entry["steered"][name] = {**{m: series(v) for m, v in out.items()}, "moved, in logits, by pair": {g: series(v) for g, v in by.items()}}
    # depth: the pair direction at dose 1 after every second block
    for depth in range(0, n + 1, 2):
        for name, d in directions_at(p, depth, controls=False).items():
            logits = steer.steered(model, batch.clean, depth, d, [batch.pos["S"]], 1.0)
            entry["depth"][name]["share"].append(((clean - ld(logits)).mean() / gap).item())
            entry["depth"][name]["turned"].append((ld(logits) < 0).float().mean().item())
    result["templates"].append(entry)
    print(batch.template, f"clean {clean.mean():+.2f}, flipped {flipped.mean():+.2f}")
result["depths"] = list(range(0, n + 1, 2))
try:  # what step 3's patch at the acting name carries at the same depths, if that script has run on this model
    with open(f"runs/patch/patch-{meta['name']}.json") as f:
        travel = json.load(f)
    at = travel["travel groups"].index("S")
    result["the patch at the acting name, by depth"] = rounded(torch.tensor([[t["travel"]["break"][d][at] for d in result["depths"]] for t in travel["templates"]]).mean(0), 3)
except FileNotFoundError:
    pass

T = result["templates"]
mean = lambda pick: sum(pick(t) for t in T) / len(T)
row = lambda values: " ".join(f"{v:6.2f}" for v in values)
at = DOSES.index(1.0)
print(f"\n{meta['name']}: {n} blocks; goes on with She after {', '.join(result['names that go on with She'])}; steering at the acting name after block {block}, where the stream's size is {mean(lambda t: t['stream size at the acting name']):.1f}")
print(f"plain: clean {mean(lambda t: t['plain']['clean']):+.2f}, flipped {mean(lambda t: t['plain']['flipped']):+.2f}; a third name acts {mean(lambda t: t['plain']['a third name acts, share']):.2f} of the way; A above B in "
      f"{100 * mean(lambda t: t['plain']['the flipped prompt turned']):.1f}% of flipped and {100 * mean(lambda t: t['plain']['the clean prompt turned']):.1f}% of clean prompts; outside the two names the clean prompt is "
      f"{mean(lambda t: t['plain']['from the clean prompt to the flipped, outside the two names']):.2f} nats from the flipped (" + ", ".join(f"{g} {mean(lambda t, g=g: t['plain']['the same, by pair'][g]):.2f}" for g in kinds) + ")")
for measure in ("share", "turned"):
    print(f"  {measure}\n    dose           " + row(DOSES))
    for name in NAMES:
        print(f"    {name:14s} " + row([mean(lambda t, i=i: t["steered"][name][measure][i]) for i in range(len(DOSES))]))
print("  at dose 1: A and B against their own kind; nats from the flipped prompt; length; share of the gap by pair (" + ", ".join(f"{g} {mean(lambda t, g=g: t['pairs'][g]['gap']):.2f}" for g in kinds) + ")")
for name in NAMES:
    print(f"    {name:14s} {mean(lambda t: t['steered'][name]['A against its own kind'][at]):+.2f} {mean(lambda t: t['steered'][name]['B against its own kind'][at]):+.2f}; {mean(lambda t: t['steered'][name][OUTSIDE][at]):.2f}; "
          f"{mean(lambda t: t['length'][name]['all']):.1f}; " + ", ".join(f"{mean(lambda t, g=g: t['steered'][name]['moved, in logits, by pair'][g][at]) / mean(lambda t, g=g: t['pairs'][g]['gap']):+.2f}" for g in kinds))
print("  at dose 1 by depth, share of the way   " + " ".join(f"{d:>5}" for d in result["depths"]))
for name in BY_DEPTH:
    print(f"    {name:34s} " + " ".join(f"{mean(lambda t, i=i: t['depth'][name]['share'][i]):5.2f}" for i in range(len(result["depths"]))))
print("    pair, A above B                    " + " ".join(f"{mean(lambda t, i=i: t['depth']['pair']['turned'][i]):5.2f}" for i in range(len(result["depths"]))))
if "the patch at the acting name, by depth" in result:
    print("    the patch of step 3                " + " ".join(f"{v:5.2f}" for v in result["the patch at the acting name, by depth"]))

save(f"runs/open/steer-{meta['name']}.json", result)
