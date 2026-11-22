"""
Step 7: does adding a direction change the answer as the description predicts? See docs/07-steer.md.

Two things are steered, both at a name's position after block 4, with the
directions the earlier steps suggest.

who acts    On a clean prompt (A acts, B is due), add a direction that
            should turn "A acts" into "B acts", and see whether the answer
            becomes the flipped prompt's. Unless it says otherwise, a
            direction is built on four sentence patterns and used on the
            fifth:
              own            the difference between the prompt's own two
                             streams, flipped minus clean: at dose 1 this
                             is a patch of step 3, the most a direction
                             can do
              pair           the same difference for the same two names,
                             averaged over the other four patterns
              names          `pair` with a term for the first-mentioned
                             name, a term for the second-mentioned one,
                             and nothing that belongs to the two together
              means          the mean stream when B acts minus the mean
                             when A acts, over both versions of the prompt
              order          the mean of `pair` over all pairs: what every
                             flipped prompt has that its clean twin lacks,
                             whoever the names are
              means + order  the sum of the last two
              reader         the direction in which a reader of the acting
                             name (step 2) raises B's score over A's, at
                             the length of `means`: a reader has no length
                             of its own
              latents        step 6's latent for B at its usual value and
                             A's taken away, through the dictionary's
                             decoder. The values are from four patterns;
                             the choice of latent is step 6's, on all five
            and, as controls:
              kind           the mean stream over the "She" names minus
                             the mean over the "He" names, A and B left
                             out, towards B's kind: nothing where A and B
                             are of one kind
              other names    `means` for two names that are not in the
                             prompt, of the same kinds as B and A
              random         a random direction of the length of `means`
              random, long   a random direction of the length of `pair`
              means + random `means` plus a random direction of the length
                             of `order`
              wipe           the mean stream of the prompt's own pattern,
                             over all pairs and both versions, in place of
                             the prompt's stream at dose 1
            A direction can lower the answer by damaging what the position
            holds, so the doses run below zero as well.

kind        After a sentence about one name the model goes on with " She"
            or " He". Add to a "He" name the direction that should make it
            a "She" name, and take it from a "She" name. Unless it says
            otherwise, a direction is built from the other five sentences
            and without the name it is used on:
              means          the mean stream over "She" names minus the
                             mean over "He" names
              rebuilt        the part of `means` that the dictionary
                             rebuilds
              missed         the part it misses
              rebuilt, without the shared latent
                             and its two halves: the latents that are
                             higher on "He" names, and those higher on
                             "She" names
              latent         the one latent of step 6 that the six "She"
                             names share, at its mean value over the other
                             five sentences, through the decoder
              random         a random direction of the length of `means`

For every dose it records the effect, and how far the whole next-token
distribution is from the one the steering aims at, outside the tokens that
were meant to move (a KL divergence, with the aimed-at distribution as the
reference).

    uv run scripts/steer.py
    uv run scripts/steer.py --model seed1
"""

import argparse
import json

import torch

from minterp import probe, steer, tasks
from minterp import sae as S
from minterp.common import eot, load, rounded, save
from minterp.tap import Tap

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="seed0", help="seed0, seed1 or seed2: the steps before must have run on it")
parser.add_argument("--block", type=int, default=4)
parser.add_argument("--dict", default="", help="another dictionary of the same place, by the name scripts/sae.py gave it (step 8)")
args = parser.parse_args()

model, tok, meta = load(args.model)
device = next(model.parameters()).device
sae = S.load(S.path(meta["name"], args.block, args.dict), device)
tag = f"-{args.dict}" if args.dict else ""  # the records of another dictionary are kept beside the first
key = ("resid", args.block)
with open(f"runs/features/features-{meta['name']}{tag}.json") as f:
    step6 = json.load(f)
head_latent = {x["name"]: x["the head's latent"] for x in step6["names"]}
# The latent the veto head rests on most for the "She" names: one for all six in the first dictionary. Another dictionary
# may give one name a latent of its own, so it is the one most of them share.
theirs = [head_latent[q] for q in step6["names that go on with She"]]
shared = max(sorted(set(theirs)), key=theirs.count)
people = tasks.names(tok)
k = len(people)
name_ids = torch.tensor([tok.encode_ordinary(" " + p)[0] for p in people], device=device)
she = torch.tensor([p in step6["names that go on with She"] for p in people], device=device)
DOSES = (-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
generator = torch.Generator().manual_seed(0)  # the two controls called `random`
more = torch.Generator().manual_seed(1)  # the other random controls, so that adding one does not move the first
result = {"model": meta, "block": args.block, "dictionary": args.dict or "first", "doses": DOSES, "names that go on with She": step6["names that go on with She"]}
series = lambda values: rounded(torch.tensor(values))
cosine = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=1)


@torch.no_grad()
def stream_at(tokens, position):
    with Tap(model, keep=("resid",)) as tap:
        logits, _ = model(tokens)
    return tap[key][:, position].float(), logits[:, -1].float()


# ---------------------------------------------------------------- who acts
batches = tasks.batches(tok, device, people=people)
A, B, C = (torch.tensor([people.index(t[i]) for t in batches[0].names], device=device) for i in range(3))
rows = torch.arange(len(A), device=device)
acts = torch.stack([A, B])  # (versions, prompts): who acts on the clean and on the flipped prompt
# Two names that are not in the prompt, of the same kinds as B and A: what `means` does when it is about other people.
D, E = steer.stand_ins(A, B, C, she)
# The yardstick for a name: the names of its kind that are in no version of the prompt and are not the control's stand-ins.
apart = torch.ones(len(A), k, dtype=torch.bool, device=device)
for names in (A, B, C, D, E):
    apart[rows, names] = False
own_kind = {"A": apart & (she[None] == she[A][:, None]), "B": apart & (she[None] == she[B][:, None])}
# A term for the first-mentioned name, a term for the second, and a constant: the projection that keeps only those.
additive = steer.term_per_name(A, B, k)  # (prompts, prompts)

at_name, plain, third = [], [], []
for batch in batches:
    per = [stream_at(getattr(batch, v), batch.pos["S"]) for v in ("clean", "flipped")]
    at_name.append(torch.stack([x for x, _ in per]))
    plain.append(torch.stack([l for _, l in per]))
    third.append(stream_at(batch.neutral, batch.pos["S"])[1])
at_name, plain = torch.stack(at_name), torch.stack(plain)  # (patterns, versions, prompts, dim) and (..., vocab)
kinds = {"both She": she[A] & she[B], "both He": ~she[A] & ~she[B], "one of each": she[A] != she[B]}
NAMES = ("own", "pair", "names", "means + order", "means", "order", "reader", "latents", "kind", "other names", "random", "random, long", "means + random", "wipe")
who = {"directions": NAMES, "pairs": {n: int(m.sum()) for n, m in kinds.items()}, "templates": []}
OUTSIDE, FROM = "to the flipped prompt, outside the two names", "from the clean prompt, outside the two names"
for p, batch in enumerate(batches):
    others = [q for q in range(len(batches)) if q != p]
    x = at_name[others]  # (patterns, versions, prompts, dim): what the directions are built from
    actor = acts[None].expand(len(others), -1, -1)
    mean_when = torch.stack([x[actor == i].mean(0) for i in range(k)])  # (names, dim): the mean stream when each name acts
    pair = (x[:, 1] - x[:, 0]).mean(0)
    order = pair.mean(0, keepdim=True).expand_as(pair)
    means = mean_when[B] - mean_when[A]
    directions = {"own": at_name[p, 1] - at_name[p, 0], "pair": pair, "names": additive @ pair, "means + order": means + order, "means": means, "order": order}
    length = means.norm(dim=1, keepdim=True)

    # the reader's direction: where its score for B rises fastest against its score for A
    W, _, _, spread = probe.weights(x.reshape(-1, x.shape[-1]).cpu(), actor.reshape(-1).cpu(),
                                    torch.tensor(others)[:, None, None].expand(len(others), 2, len(A)).reshape(-1), k)
    grad = (W / spread.T).T.float().to(device)  # (names, dim): the gradient of each name's score with respect to the stream
    directions["reader"] = grad[B] - grad[A]
    directions["reader"] = directions["reader"] / directions["reader"].norm(dim=1, keepdim=True) * length

    # the latents' direction: B's latent at its usual value and A's taken away. Nothing where the two share a latent.
    with torch.no_grad():
        z = sae.encode(x.reshape(-1, x.shape[-1]))
    usual = torch.stack([z[actor.reshape(-1) == i, head_latent[people[i]]].mean() for i in range(k)])  # each name's latent when it acts
    column = torch.stack([steer.writes(sae, head_latent[q]) for q in people])  # (names, dim)
    directions["latents"] = usual[B, None] * column[B] - usual[A, None] * column[A]
    # Step 6 set the two names' latents in the flipped run to their clean values. Its mirror, in this prompt's own two streams:
    with torch.no_grad():
        change = sae.encode(at_name[p, 1]) - sae.encode(at_name[p, 0])  # (prompts, width)
    la, lb = (torch.tensor([head_latent[people[i]] for i in who_.tolist()], device=device) for who_ in (A, B))
    only = torch.zeros_like(change)
    only[rows, la], only[rows, lb] = change[rows, la], change[rows, lb]
    mirror = (only @ sae.dec.detach()) * sae.scale
    apart_ = la != lb  # where the two names share a latent there is nothing to compare
    # the controls
    directions["kind"] = steer.kind_only(mean_when, A, B, she)
    directions["other names"] = mean_when[D] - mean_when[E]
    directions["random"] = steer.like(means, generator)
    directions["random, long"] = steer.like(pair, more)
    directions["means + random"] = means + steer.like(order, more)
    directions["wipe"] = at_name[p].mean((0, 1))[None] - at_name[p, 0]  # at dose 1 the stream becomes the pattern's mean

    ld = lambda logits: tasks.logit_diff(logits, batch.right, batch.wrong)
    clean, flipped = ld(plain[p, 0]), ld(plain[p, 1])
    gap = (clean - flipped).mean()
    names_out = torch.stack([batch.right, batch.wrong], dim=1)
    own = {n: lambda logits, n=n, t=t: steer.against(logits[:, name_ids], t, own_kind[n]) for n, t in (("A", A), ("B", B))}
    between = steer.elsewhere(plain[p, 1], plain[p, 0], names_out)
    entry = {"template": batch.template, "gap": gap.item(), "stream size at the acting name": at_name[p, 0].norm(dim=1).mean().item(),
             "pairs": {n: {"gap": (clean - flipped)[m].mean().item()} for n, m in kinds.items()},
             "length": {name: {**{"all": d.norm(dim=1).mean().item()}, **{n: d[m].norm(dim=1).mean().item() for n, m in kinds.items()}} for name, d in directions.items()},
             "cosine with own": {name: cosine(d, directions["own"]).mean().item() for name, d in directions.items()},
             "cosine with means": {name: cosine(d, means).mean().item() for name, d in directions.items()},
             "the latents' direction against the mirror of step 6's move": {
                 "cosine": cosine(directions["latents"][apart_], mirror[apart_]).mean().item(),
                 "length ratio": (directions["latents"][apart_].norm(dim=1).mean() / mirror[apart_].norm(dim=1).mean()).item()},
             # What the measures read on prompts nobody steered.
             "plain": {"a third name acts, share": ((clean - ld(third[p])).mean() / gap).item(),
                       "the flipped prompt turned": (flipped < 0).float().mean().item(), "the clean prompt turned": (clean < 0).float().mean().item(),
                       "from the clean prompt to the flipped, outside the two names": between.mean().item(),
                       "the same, by pair": {n: between[m].mean().item() for n, m in kinds.items()}},
             "steered": {}}
    for name, d in directions.items():
        out = {m: [] for m in ("share", "turned", "newly turned", "A against its own kind", "B against its own kind", OUTSIDE, FROM)}
        by, far = {n: [] for n in kinds}, {n: [] for n in kinds}
        grouped = {x: {n: [] for n in kinds} for x in ("A", "B")}
        for dose in DOSES:
            logits = steer.steered(model, batch.clean, args.block, d, [batch.pos["S"]], dose)
            moved = clean - ld(logits)
            out["share"].append((moved.mean() / gap).item())
            out["turned"].append((ld(logits) < 0).float().mean().item())  # A is above B
            out["newly turned"].append(((ld(logits) < 0) & (clean >= 0)).float().mean().item())  # and was not before
            for n in ("A", "B"):
                way = (own[n](logits) - own[n](plain[p, 0])).mean() / (own[n](plain[p, 1]) - own[n](plain[p, 0])).mean()
                out[f"{n} against its own kind"].append(way.item())
            short = steer.elsewhere(plain[p, 1], logits, names_out)
            out[OUTSIDE].append(short.mean().item())
            out[FROM].append(steer.elsewhere(plain[p, 0], logits, names_out).mean().item())
            for n, m in kinds.items():
                by[n].append(moved[m].mean().item())
                far[n].append(short[m].mean().item())
                for x in ("A", "B"):
                    grouped[x][n].append(((own[x](logits) - own[x](plain[p, 0]))[m].mean() / (own[x](plain[p, 1]) - own[x](plain[p, 0]))[m].mean()).item())
        entry["steered"][name] = {**{m: series(v) for m, v in out.items()}, "moved, in logits, by pair": {n: series(v) for n, v in by.items()},
                                  f"{OUTSIDE}, by pair": {n: series(v) for n, v in far.items()},
                                  **{f"{x} against its own kind, by pair": {n: series(v) for n, v in grouped[x].items()} for x in ("A", "B")}}
    who["templates"].append(entry)
result["who acts"] = who

# ---------------------------------------------------------------- kind
# Not the sentence step 5 sorted the names with.
SENTENCES = ("Once upon a time, {N} found a red ball.", "One morning, {N} was very happy.", "Yesterday, {N} played in the garden.",
             "After lunch, {N} saw a little dog.", "Today, {N} wanted to bake a cake.", "That night, {N} could not sleep.")
she_id, he_id = (tok.encode_ordinary(w)[0] for w in (" She", " He"))
# Every form of the two pronouns that is one token, with and without a space and a capital.
forms = sorted({v for w in ("he", "she", "her", "his", "him", "hers", "himself", "herself") for v in (w, " " + w, w.capitalize(), " " + w.capitalize()) if len(tok.encode_ordinary(v)) == 1})
LEAVE = {"the two pronouns and the names": torch.cat([torch.tensor([she_id, he_id], device=device), name_ids]),
         "the two pronouns, His and Her, and the names": torch.cat([torch.tensor([she_id, he_id] + [tok.encode_ordinary(w)[0] for w in (" His", " Her")], device=device), name_ids]),
         "every form of the pronouns and the names": torch.cat([torch.tensor([tok.encode_ordinary(v)[0] for v in forms], device=device), name_ids])}
LEAVE = {n: ids[None].expand(k, -1) for n, ids in LEAVE.items()}
gap = lambda l: l[:, she_id] - l[:, he_id]
cases = []
for sentence in SENTENCES:
    tokens = torch.tensor([[eot(tok)] + tok.encode_ordinary(sentence.format(N=q)) for q in people], device=device)
    position = (tokens[0] == name_ids[0]).nonzero()[0, 0].item()
    assert (tokens[:, position] == name_ids).all()
    cases.append((tokens, position, *stream_at(tokens, position)))
streams = torch.stack([x for _, _, x, _ in cases])  # (sentences, names, dim)
with torch.no_grad():
    every = sae.encode(streams)  # (sentences, names, width)
    codes = every[..., shared]  # the shared latent at the name
    before = ((streams / sae.scale - sae.b) @ sae.enc + sae.b_enc)[..., shared]  # the same before the top-k cut and the clamp at zero
same = she[:, None] == she[None, :]
unlike = ~torch.eye(k, dtype=torch.bool, device=device)
KIND = ("means", "rebuilt", "missed", "rebuilt, without the shared latent", "latent", "random", "latents higher on He names", "latents higher on She names, without the shared one")
kind = {"directions": KIND, "sentences": SENTENCES, "the shared latent": shared, "names that share it": [q for q in step6["names that go on with She"] if head_latent[q] == shared], "names per kind": int(she.sum()), "order": people, "pronoun forms left out": forms,
        "the latent fires on": {"She names": (codes[:, she] > 0).float().mean().item(), "He names": (codes[:, ~she] > 0).float().mean().item()},
        "the latent on She names": {"mean": codes[:, she].mean().item(), "largest": codes[:, she].max().item()},
        # Why the dictionary reads back less than was written: its encoder's reading of one unit of the latent's own direction,
        # and where that reading starts on a "He" name, against the latent's usual value on "She" names.
        "the encoder's reading of one unit of the latent's direction": (sae.dec[shared] @ sae.enc[:, shared]).item(),
        "the reading before the cut on He names, against the usual value": (before[:, ~she].mean() / codes[:, she].mean()).item(), "templates": []}
for s, (tokens, position, x, logits) in enumerate(cases):
    rest = torch.tensor([q for q in range(len(SENTENCES)) if q != s], device=device)
    pool, coded = streams[rest], every[rest]
    means, rebuilt, without, lower, higher = [], [], [], [], []
    for i in range(k):
        keep = torch.ones(k, dtype=torch.bool, device=device)
        keep[i] = False
        means.append(pool[:, keep & she].mean((0, 1)) - pool[:, keep & ~she].mean((0, 1)))
        differ = coded[:, keep & she].mean((0, 1)) - coded[:, keep & ~she].mean((0, 1))  # (width,): the same difference in latents
        rebuilt.append((differ @ sae.dec.detach()) * sae.scale)
        differ[shared] = 0
        without.append((differ @ sae.dec.detach()) * sae.scale)
        lower.append((differ.clamp(max=0) @ sae.dec.detach()) * sae.scale)  # the two halves of `without`, by sign
        higher.append((differ.clamp(min=0) @ sae.dec.detach()) * sae.scale)
    means, rebuilt, without, lower, higher = (torch.stack(v) for v in (means, rebuilt, without, lower, higher))  # (names, dim): towards "She"
    usual = codes[rest][:, she].mean()
    directions = {"means": means, "rebuilt": rebuilt, "missed": means - rebuilt, "rebuilt, without the shared latent": without,
                  "latent": (usual * steer.writes(sae, shared)).expand(k, -1), "random": steer.like(means, generator),
                  "latents higher on He names": lower, "latents higher on She names, without the shared one": higher}
    sign = torch.where(she, -1.0, 1.0)[:, None]  # add to a "He" name, take from a "She" name

    def between(reference, changed, ids):
        """From each name's reference distribution to each name's changed one: (reference j, name i)."""
        return torch.stack([steer.elsewhere(reference[j].expand(k, -1), changed, ids) for j in range(k)])

    entry = {"sentence": SENTENCES[s], "stream size at the name": x.norm(dim=1).mean().item(),
             "length": {name: d.norm(dim=1).mean().item() for name, d in directions.items()},
             "cosine with means": {name: cosine(d, means).mean().item() for name, d in directions.items()},
             "latents that differ by more than half a unit between the kinds": int(((coded[:, she].mean((0, 1)) - coded[:, ~she].mean((0, 1))).abs() > 0.5).sum()),
             "the latent that differs most between the kinds": int((coded[:, she].mean((0, 1)) - coded[:, ~she].mean((0, 1))).abs().argmax()),
             "plain": {"She names": gap(logits)[she].mean().item(), "He names": gap(logits)[~she].mean().item(),
                       "names on the side step 5 sorted them to": ((gap(logits) > 0) == she).float().mean().item(),
                       "between two names of one kind": {n: between(logits, logits, ids)[same & unlike].mean().item() for n, ids in LEAVE.items()},
                       "between two names of two kinds": {n: between(logits, logits, ids)[~same].mean().item() for n, ids in LEAVE.items()}},
             "steered": {}}
    for name, d in directions.items():
        out = {m: {"She names": [], "He names": []} for m in ("gap", "turned", "from the name itself")}
        far = {n: {"She names": [], "He names": []} for n in LEAVE}
        back, each = {"She names": [], "He names": []}, []
        for dose in DOSES:
            steered = steer.steered(model, tokens, args.block, sign * d, [position], dose)
            to_other = {n: (between(logits, steered, ids) * ~same).sum(0) / (~same).sum(0) for n, ids in LEAVE.items()}
            away = steer.elsewhere(logits, steered, LEAVE["the two pronouns and the names"])
            g = gap(steered)
            each.append(g)
            with torch.no_grad():
                seen = sae.encode(x + dose * sign * d)[:, shared]  # what the dictionary reads at the name after the change
            for group, m, flips in (("She names", she, g < 0), ("He names", ~she, g > 0)):
                out["gap"][group].append(g[m].mean().item())
                out["turned"][group].append(int(flips[m].sum()))
                out["from the name itself"][group].append(away[m].mean().item())
                back[group].append((seen[m].mean() / usual).item())
                for n in LEAVE:
                    far[n][group].append(to_other[n][m].mean().item())
        entry["steered"][name] = {**{m: {g: series(v) if m != "turned" else v for g, v in groups.items()} for m, groups in out.items()},
                                  "to a name of the other kind": {n: {g: series(v) for g, v in groups.items()} for n, groups in far.items()},
                                  "gap by name": rounded(torch.stack(each, dim=1), 2),  # (names, doses), names in the order of tasks.names
                                  "the shared latent as the dictionary reads it, against its usual value": {g: series(v) for g, v in back.items()}}
    kind["templates"].append(entry)
result["kind"] = kind

# ---------------------------------------------------------------- summary
mean = lambda items, pick: sum(pick(t) for t in items) / len(items)
row = lambda values, form="6.2f": " ".join(f"{v:{form}}" for v in values)
W_T, K_T, n = who["templates"], kind["templates"], range(len(DOSES))
at = DOSES.index(1.0)
print("who acts: steering a clean prompt at the acting name, by dose")
print(f"  plain: a third name acts {mean(W_T, lambda t: t['plain']['a third name acts, share']):.2f} of the way; the flipped prompt has A above B in {100 * mean(W_T, lambda t: t['plain']['the flipped prompt turned']):.0f}%, "
      f"the clean one in {100 * mean(W_T, lambda t: t['plain']['the clean prompt turned']):.1f}%; outside the two names the clean prompt is {mean(W_T, lambda t: t['plain']['from the clean prompt to the flipped, outside the two names']):.2f} nats from the flipped ("
      + ", ".join(f"{g} {mean(W_T, lambda t, g=g: t['plain']['the same, by pair'][g]):.2f}" for g in kinds) + ")")
for measure in ("share", "turned", "A against its own kind", "B against its own kind", OUTSIDE):
    print(f"  {measure}\n    dose           " + row(DOSES))
    for name in NAMES:
        print(f"    {name:14s} " + row([mean(W_T, lambda t, i=i: t["steered"][name][measure][i]) for i in n]))
print("  length, all and by pair; cosine with own, with means")
for name in NAMES:
    print(f"    {name:14s} " + ", ".join(f"{g} {mean(W_T, lambda t, g=g: t['length'][name][g]):.1f}" for g in ("all", *kinds)) + f"; {mean(W_T, lambda t: t['cosine with own'][name]):+.2f}, {mean(W_T, lambda t: t['cosine with means'][name]):+.2f}")
print(f"  the stream's size there {mean(W_T, lambda t: t['stream size at the acting name']):.1f}")
print("  at dose 1 by pair: share of that pair's gap (" + ", ".join(f"{g} {mean(W_T, lambda t, g=g: t['pairs'][g]['gap']):.2f} logits, {who['pairs'][g]} pairs" for g in kinds) + "); nats from the flipped prompt")
for name in NAMES:
    print(f"    {name:14s} " + ", ".join(f"{g} {mean(W_T, lambda t, g=g: t['steered'][name]['moved, in logits, by pair'][g][at]) / mean(W_T, lambda t, g=g: t['pairs'][g]['gap']):+.2f}" for g in kinds)
          + "; " + ", ".join(f"{mean(W_T, lambda t, g=g: t['steered'][name][OUTSIDE + ', by pair'][g][at]):.2f}" for g in kinds))

total = kind["names per kind"] * len(SENTENCES)
print(f"\nkind: the logit of She minus the logit of He after a sentence about one name; plain {mean(K_T, lambda t: t['plain']['She names']):+.2f} for She names, {mean(K_T, lambda t: t['plain']['He names']):+.2f} for He names; "
      f"{100 * mean(K_T, lambda t: t['plain']['names on the side step 5 sorted them to']):.0f}% of names on the side step 5 sorted them to")
for leave in LEAVE:
    print(f"  outside {leave}: two names of one kind are {mean(K_T, lambda t: t['plain']['between two names of one kind'][leave]):.2f} nats apart, of two kinds {mean(K_T, lambda t: t['plain']['between two names of two kinds'][leave]):.2f}"
          + "; means at dose 1 to a name of the other kind " + f"{(mean(K_T, lambda t: t['steered']['means']['to a name of the other kind'][leave]['He names'][at]) + mean(K_T, lambda t: t['steered']['means']['to a name of the other kind'][leave]['She names'][at])) / 2:.2f}")
print("  dose                                  " + row(DOSES))
for name in KIND:
    for group in ("He names", "She names"):
        print(f"  {name[:24]:24s} {group:9s}    " + row([mean(K_T, lambda t, i=i: t["steered"][name]["gap"][group][i]) for i in n])
              + f"   turned of {total}: " + " ".join(f"{sum(t['steered'][name]['turned'][group][i] for t in K_T):2d}" for i in n))
back = "the shared latent as the dictionary reads it, against its usual value"
for name in ("latent", "means"):
    for group in ("He names", "She names"):
        print(f"  {name}, the shared latent read back, {group:9s} " + row([mean(K_T, lambda t, i=i: t["steered"][name][back][group][i]) for i in n]))
print("  length: " + ", ".join(f"{name} {mean(K_T, lambda t, name=name: t['length'][name]):.1f}" for name in KIND)
      + "; cosine with means: " + ", ".join(f"{name} {mean(K_T, lambda t, name=name: t['cosine with means'][name]):+.2f}" for name in KIND)
      + f"; the stream's size at the name {mean(K_T, lambda t: t['stream size at the name']):.1f}; the latent on She names {kind['the latent on She names']}; "
      f"latents that differ by more than half a unit {mean(K_T, lambda t: t['latents that differ by more than half a unit between the kinds']):.0f}; pronoun forms left out {len(forms)}")

save(f"runs/steer/steer-{meta['name']}{tag}.json", result)
