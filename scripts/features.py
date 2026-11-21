"""
Step 6: which latents does the answer depend on? See docs/06-feature-circuit.md.

Uses the dictionary of step 5 (out/sae/<model>-b4.pt), the heads of step 4
and the latents step 5 found for the names. Everything happens at the
acting name's position after block 4, where step 3 found that the answer's
dependence on who acts sits.

parts       the stream there is what the dictionary rebuilds plus what it
            misses. Each part is moved from one prompt into another, for
            real: how much of who acts does each carry, and how much of a
            name (step 3's two name experiments)?
estimate    the same for every latent at once, estimated from one gradient
check       the latents the estimate ranks highest, moved for real, one to
            thirty-two of them, and the estimate set against the real
            effect latent by latent
head        what the veto head of step 4 adds to the acting name: against
            the nine uninvolved names, as in step 4, and against the
            uninvolved names of its own kind, where a kind is the six names
            the model follows with "She" or the six it follows with "He";
            what the gradient says this rests on, and what really happens
            when latents are switched off
names       for each name, two latents: the one step 5 picked by how often
            it fires, and the one the veto head rests on most. What moving
            just the two names' latents carries, by the kinds of the pair

    uv run scripts/features.py
    uv run scripts/features.py --model seed1
"""

import argparse
import json

import torch

from minterp import attribute, tasks
from minterp import sae as S
from minterp.common import load, rounded, save
from minterp.tap import Tap, head_writes

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="seed0", help="seed0 or seed1: the steps before must have run on it")
parser.add_argument("--block", type=int, default=4)
parser.add_argument("--dict", default="", help="another dictionary of the same place, by the name scripts/sae.py gave it (step 8)")
args = parser.parse_args()

model, tok, meta = load(args.model)
device = next(model.parameters()).device
n = model.cfg.n_layers
sae = S.load(S.path(meta["name"], args.block, args.dict), device)
tag = f"-{args.dict}" if args.dict else ""  # the records of another dictionary are kept beside the first
key = ("resid", args.block)
with open(f"runs/sae/sae-{meta['name']}{tag}.json") as f:
    step5 = json.load(f)["task"]
name_latent = {row["name"]: row["latent"] for row in step5["names"]["acts"]}  # step 5's latent for each name as the one that acts
with open(f"runs/circuit/circuit-{meta['name']}.json") as f:
    veto = max((h for h in json.load(f)["heads"] if h["acts at"] == "end"), key=lambda h: h["carry"])
people = tasks.names(tok)
name_ids = torch.tensor([tok.encode_ordinary(" " + p)[0] for p in people], device=device)
she = torch.tensor([p in step5["names that go on with She"] for p in people], device=device)  # the kind of every name
COUNTS = (1, 2, 4, 8, 16, 32)
batches = tasks.batches(tok, device, people=people)
result = {"model": meta, "block": args.block, "dictionary": args.dict or "first", "counts": COUNTS, "veto head": f"block {veto['block']} head {veto['head']}",
          "names that go on with She": step5["names that go on with She"], "templates": []}


def index(batch, i):
    return torch.tensor([people.index(t[i]) for t in batch.names], device=device)


def head_names(tap):
    """What the veto head adds at the last position to the logit of each of the twelve names: (prompts, names), on the
    scale of the final stream, whose size is held fixed so that the gradient is that of the head's write alone."""
    final = tap["resid", n][:, -1].detach()
    size = torch.rsqrt(final.pow(2).mean(-1, keepdim=True) + model.norm.eps)
    write = head_writes(model, tap, veto["block"])[:, -1, veto["head"]]
    return ((write * model.norm.weight * size) @ model.lm_head.weight[name_ids].T).float()


def masks(batch):
    """For every prompt: its row index, and the nine names that are none of A, B and C."""
    rows = torch.arange(len(batch), device=device)
    nine = torch.ones(len(batch), len(people), dtype=torch.bool, device=device)
    for i in range(3):
        nine[rows, index(batch, i)] = False
    return rows, nine


def against(names, target, pool):
    """A name's logit minus the mean over a pool of names, per prompt."""
    rows = torch.arange(names.shape[0], device=names.device)
    return names[rows, target] - (names * pool).sum(1) / pool.sum(1)


def adds_to(batch, target):
    """A measure for attribute.through: what the veto head adds to one name per prompt, against the nine uninvolved names."""
    _, nine = masks(batch)
    return lambda logits, tap: against(head_names(tap), target, nine)


# For each name: the latent the veto head rests on most when that name acts, over the clean and flipped prompts of all
# patterns, and how many latents mark the name as the acting one by firing (on 90% of its prompts and 10% of the others).
tops, actors, fired, where = [], [], [], []
for p, batch in enumerate(batches):
    for version, col in (("clean", 0), ("flipped", 1)):
        actor = index(batch, col)
        z, _, g, _, _ = attribute.through(model, sae, getattr(batch, version), args.block, adds_to(batch, actor), keep=("heads", "resid"))
        at = batch.pos["S"]
        tops.append((z[:, at] * g[:, at]).abs().argmax(1).cpu())
        actors.append(actor.cpu())
        fired.append((z[:, at] > 0).cpu())
        where.append(torch.full((len(batch),), p))
tops, actors, fired, where = torch.cat(tops), torch.cat(actors), torch.cat(fired), torch.cat(where)
head_latent, result["names"] = {}, []
for i, name in enumerate(people):
    mine = actors == i
    mode = tops[mine].mode().values.item()
    marks = (fired[mine].float().mean(0) >= 0.9) & (fired[~mine].float().mean(0) <= 0.1)
    head_latent[name] = mode
    # Found on four patterns, does it hold on the fifth?
    left_out = []
    for p in range(len(batches)):
        seen, unseen = mine & (where != p), mine & (where == p)
        left_out.append((tops[unseen] == tops[seen].mode().values).float().mean().item())
    result["names"].append({
        "name": name, "step 5's latent": name_latent[name], "the head's latent": mode,
        "the head rests on it most in": (tops[mine] == mode).float().mean().item(),
        "on a pattern left out": sum(left_out) / len(left_out),
        "latents that mark the name by firing": int(marks.sum()),
        "the head's latent marks the name by firing": bool(marks[mode]), "step 5's latent marks the name by firing": bool(marks[name_latent[name]]),
        "names for which the head's latent fires on 90% or more": [people[j] for j in range(len(people)) if fired[actors == j][:, mode].float().mean() >= 0.9],
    })
she_latents = {head_latent[p] for p in step5["names that go on with She"]}
result["one latent for the six"] = sorted(she_latents)
# One for all six in the first dictionary. Another dictionary may give a name a latent of its own: the shared one is then
# the one most of the six have, and it alone is switched off below.
theirs = [head_latent[p] for p in step5["names that go on with She"]]
the_shared = max(sorted(she_latents), key=theirs.count)
result["the shared latent"] = the_shared

for batch in batches:
    s = batch.pos["S"]
    rows, nine = masks(batch)
    A, B, C = (index(batch, i) for i in range(3))
    ld = lambda logits, tap=None: tasks.logit_diff(logits.float(), batch.right, batch.wrong)
    zc, ec, _, _, vc = attribute.through(model, sae, batch.clean, args.block, ld)
    zf, ef, gz, ge, vf = attribute.through(model, sae, batch.flipped, args.block, ld)  # gradients in the flipped run, the target
    gap = (vc.mean() - vf.mean()).item()

    @torch.no_grad()
    def run(tokens, z, err, keep=()):
        """A forward pass with the stream after the block set to decode(z) + err."""
        with Tap(model, edit={key: attribute.rebuilt_with(sae, z, err)}, keep=keep) as tap:
            logits, _ = model(tokens)
        return logits, tap

    moved = lambda z, err: ld(run(batch.flipped, z, err)[0])
    share = lambda values, pick=slice(None): ((values[pick].mean() - vf[pick].mean()) / (vc[pick].mean() - vf[pick].mean())).item()

    def at_s(target, source):
        out = target.clone()
        out[:, s] = source[:, s]
        return out

    entry = {"template": batch.template, "gap": gap}
    # parts: what the dictionary rebuilds, what it misses, and both, moved for real
    entry["parts"] = {"rebuilt": share(moved(at_s(zf, zc), ef)), "missed": share(moved(zf, at_s(ef, ec))), "both": share(moved(at_s(zf, zc), at_s(ef, ec)))}
    # The same split for a name, as in step 3: the prompt with C in place of a name, moved into the prompt it was made from.
    entry["name parts"] = {}
    for label, source, target, old in (("second name", batch.other, batch.clean, B), ("first name", batch.other_first, batch.flipped, A)):
        to_c = lambda logits, old=old: tasks.logit_diff(logits.float(), name_ids[C], name_ids[old])  # token ids, not places in the list
        zo, eo, _, _, _ = attribute.through(model, sae, source, args.block, lambda logits, tap: to_c(logits))
        zt, et, _, _, _ = attribute.through(model, sae, target, args.block, lambda logits, tap: to_c(logits))
        base, goal = to_c(run(target, zt, et)[0]).mean(), to_c(run(source, zo, eo)[0]).mean()
        way = lambda z, err: ((to_c(run(target, z, err)[0]).mean() - base) / (goal - base)).item()
        entry["name parts"][label] = {"rebuilt": way(at_s(zt, zo), et), "missed": way(zt, at_s(et, eo)), "both": way(at_s(zt, zo), at_s(et, eo))}

    # estimate: every latent at the acting name at once
    est = (zc[:, s] - zf[:, s]) * gz[:, s]  # (prompts, width): the estimated change of the logit difference, in logits
    est_missed = ((ec[:, s] - ef[:, s]) * ge[:, s]).sum(1)
    on_c, on_f = zc[:, s] > 0, zf[:, s] > 0
    order = est.abs().argsort(dim=1, descending=True)
    entry["estimate"] = {"rebuilt": (est.sum(1).mean() / gap).item(), "missed": (est_missed.mean() / gap).item(),
                         "latents that differ": (zc[:, s] != zf[:, s]).sum(1).float().mean().item(),
                         "on in the clean prompt only": (on_c & ~on_f).sum(1).float().mean().item(),
                         "on in the flipped prompt only": (~on_c & on_f).sum(1).float().mean().item(),
                         "on in both": (on_c & on_f & (zc[:, s] != zf[:, s])).sum(1).float().mean().item()}

    # check: the highest-ranked latents moved for real
    by_count = {"estimated": [], "real": []}
    for count in COUNTS:
        top = order[:, :count]
        z = zf.clone()
        z[rows[:, None], s, top] = zc[rows[:, None], s, top]
        by_count["real"].append(share(moved(z, ef)))
        by_count["estimated"].append((est.gather(1, top).sum(1).mean() / gap).item())
    entry["by count"] = by_count
    # latent by latent: the estimate against the real effect, for each prompt's five highest
    pairs = []
    for rank in range(5):
        j = order[:, rank]
        z = zf.clone()
        z[rows, s, j] = zc[rows, s, j]
        pairs.append(torch.stack([est[rows, j], moved(z, ef) - vf]))  # (2, prompts): estimated and real change, in logits
    pairs = torch.stack(pairs)  # (ranks, 2, prompts)
    flat = pairs.permute(1, 0, 2).reshape(2, -1)
    entry["one latent"] = {"estimated": rounded(pairs[:, 0].mean(1), 3), "real": rounded(pairs[:, 1].mean(1), 3),
                           "correlation": torch.corrcoef(flat)[0, 1].item(), "same sign": (flat[0].sign() == flat[1].sign()).float().mean().item()}

    # names: the two names' latents, by step 5's choice and by the head's, overall and by the kinds of the pair
    kinds = {"both She": she[A] & she[B], "both He": ~she[A] & ~she[B], "one of each": she[A] != she[B]}
    entry["pairs"] = {k: {"prompts": int(m.sum()), "gap": (vc[m].mean() - vf[m].mean()).item()} for k, m in kinds.items()}
    entry["names"] = {}
    for choice, table in (("step 5's latents", name_latent), ("the head's latents", head_latent)):
        la = torch.tensor([table[a] for a, _, _ in batch.names], device=device)  # the first name acts on the clean prompt
        lb = torch.tensor([table[b] for _, b, _ in batch.names], device=device)  # the second on the flipped one
        z = zf.clone()
        z[rows, s, la] = zc[rows, s, la]
        z[rows, s, lb] = zc[rows, s, lb]
        values = moved(z, ef)
        hit = (order[:, 0] == la) | (order[:, 0] == lb)
        entry["names"][choice] = {"moved for real": share(values), "estimated": ((est[rows, la] + est[rows, lb]).mean() / gap).item(),
                                  "the highest is one of them": hit.float().mean().item(),
                                  "by pair": {k: {"moved for real": share(values, m), "the highest is one of them": hit[m].float().mean().item()} for k, m in kinds.items()}}

    # head: what the veto head adds to the acting name on clean prompts, against two pools of uninvolved names
    own = nine & (she[None, :] == she[A][:, None])  # the uninvolved names of the acting name's kind
    other_kind = nine & ~own
    ha = torch.tensor([head_latent[a] for a, _, _ in batch.names], device=device)
    z, err, g, gerr, value = attribute.through(model, sae, batch.clean, args.block, adds_to(batch, A), keep=("heads", "resid"))
    rest_on = z[:, s] * g[:, s]  # what the number would lose, to first order, if a latent were off
    ranked = rest_on.abs().argsort(dim=1, descending=True)
    plain = head_names(run(batch.clean, z, err, keep=("heads", "resid"))[1])

    def off(mask):
        """The head's write with some latents at the acting name switched off, for real."""
        cut = z.clone()
        cut[:, s] = cut[:, s] * (~mask)
        return head_names(run(batch.clean, cut, err, keep=("heads", "resid"))[1])

    one_hot = lambda j: torch.zeros_like(z[:, s], dtype=torch.bool).scatter_(1, j[:, None], True)
    everything = torch.ones_like(z[:, s], dtype=torch.bool)
    shared = torch.zeros_like(z[:, s], dtype=torch.bool)
    shared[:, the_shared] = True
    acts_she = she[A]
    kind_gap = lambda names: (names * own).sum(1) / own.sum(1) - (names * other_kind).sum(1) / other_kind.sum(1)
    head = {"by kind of the acting name": {}}
    for label, m in (("She", acts_she), ("He", ~acts_she)):
        head["by kind of the acting name"][label] = {
            "prompts": int(m.sum()),
            "against the nine": against(plain, A, nine)[m].mean().item(),
            "against its own kind": against(plain, A, own)[m].mean().item(),
            "its own kind against the other": kind_gap(plain)[m].mean().item(),
        }
    cut = off(shared)
    m = acts_she
    head["the shared latent switched off, where a She name acts"] = {
        "the acting name against the nine": (against(cut, A, nine) - against(plain, A, nine))[m].mean().item(),
        "estimated": -rest_on[:, the_shared][m].mean().item(),
        "the acting name against its own kind": (against(cut, A, own) - against(plain, A, own))[m].mean().item(),
        "its own kind against the other": (kind_gap(cut) - kind_gap(plain))[m].mean().item(),
    }
    head["adds to the acting name"] = value.mean().item()
    head["estimated to rest on"] = {
        "the latents at the acting name": rest_on.sum(1).mean().item(),
        "what the dictionary misses there": (err[:, s] * gerr[:, s]).sum(1).mean().item(),
        "the other positions": ((z * g).sum((1, 2)) + (err * gerr).sum((1, 2)) - rest_on.sum(1) - (err[:, s] * gerr[:, s]).sum(1)).mean().item(),
        "its highest latent": rest_on.gather(1, ranked[:, :1]).mean().item(),
        "the latents below zero": rest_on.clamp(max=0).sum(1).mean().item(), "the latents above zero": rest_on.clamp(min=0).sum(1).mean().item(),
    }
    head["really lost when switched off"] = {
        "its highest latent": -(against(off(one_hot(ranked[:, 0])), A, nine) - value).mean().item(),
        "all latents at the acting name": -(against(off(everything), A, nine) - value).mean().item(),
    }
    head["its highest latent is the head's latent for the name"] = (ranked[:, 0] == ha).float().mean().item()
    # The other name: what the head adds, by whether the two names are of one kind.
    z2, err2, g2, _, value2 = attribute.through(model, sae, batch.clean, args.block, adds_to(batch, B), keep=("heads", "resid"))
    rest2 = z2[:, s] * g2[:, s]
    top2 = rest2.gather(1, rest2.abs().argsort(dim=1, descending=True)[:, :1])[:, 0]
    same = she[A] == she[B]
    cut_all = against(off(everything), B, nine)
    head["the other name"] = {
        "adds": value2.mean().item(), "adds, same kind": value2[same].mean().item(), "adds, other kind": value2[~same].mean().item(),
        "estimated to rest on the latents at the acting name": rest2.sum(1).mean().item(),
        "really lost when all latents there are switched off": -(cut_all - value2).mean().item(),
        "its highest latent, size": top2.abs().mean().item(), "its highest latent, same kind": top2[same].mean().item(), "its highest latent, other kind": top2[~same].mean().item(),
    }
    entry["head"] = head
    result["templates"].append(entry)
    print(batch.template, {k: round(v, 2) for k, v in entry["parts"].items()})

T = result["templates"]
mean = lambda pick: sum(pick(t) for t in T) / len(T)
show = lambda d, digits=2: ", ".join(f"{k} {v:+.{digits}f}" for k, v in d.items() if isinstance(v, float))
print(f"\nat the acting name after block {args.block} (shares of a gap, moved for real):")
print(f"  who acts: what the dictionary rebuilds {mean(lambda t: t['parts']['rebuilt']):.2f}, what it misses {mean(lambda t: t['parts']['missed']):.2f}, both {mean(lambda t: t['parts']['both']):.2f}; "
      f"estimated {mean(lambda t: t['estimate']['rebuilt']):.2f} and {mean(lambda t: t['estimate']['missed']):.2f}")
for label in ("second name", "first name"):
    print(f"  the {label}: rebuilt {mean(lambda t: t['name parts'][label]['rebuilt']):.3f}, missed {mean(lambda t: t['name parts'][label]['missed']):.3f}, both {mean(lambda t: t['name parts'][label]['both']):.3f}")
e = {k: mean(lambda t, k=k: t["estimate"][k]) for k in T[0]["estimate"]}
print(f"  {e['latents that differ']:.0f} latents differ: {e['on in the clean prompt only']:.0f} on in the clean prompt only, {e['on in the flipped prompt only']:.0f} in the flipped only, {e['on in both']:.0f} in both")
print("  the highest-ranked latents:  " + " ".join(f"{c:>6}" for c in COUNTS))
print("    estimated                  " + " ".join(f"{mean(lambda t, i=i: t['by count']['estimated'][i]):6.2f}" for i in range(len(COUNTS))))
print("    moved for real             " + " ".join(f"{mean(lambda t, i=i: t['by count']['real'][i]):6.2f}" for i in range(len(COUNTS))))
print(f"  one latent at a time: correlation {mean(lambda t: t['one latent']['correlation']):.2f}, same sign {100 * mean(lambda t: t['one latent']['same sign']):.1f}%")
for choice in T[0]["names"]:
    nm = T[0]["names"][choice]
    print(f"  {choice}: moved for real {mean(lambda t: t['names'][choice]['moved for real']):.2f}, the highest is one of them {100 * mean(lambda t: t['names'][choice]['the highest is one of them']):.0f}%; by pair: "
          + "; ".join(f"{k} {mean(lambda t, k=k: t['names'][choice]['by pair'][k]['moved for real']):+.2f} ({100 * mean(lambda t, k=k: t['names'][choice]['by pair'][k]['the highest is one of them']):.0f}%)" for k in nm["by pair"]))
print("  gap by pair: " + ", ".join(f"{k} {mean(lambda t, k=k: t['pairs'][k]['gap']):.2f} ({T[0]['pairs'][k]['prompts']} prompts)" for k in T[0]["pairs"]))
print("  per name: step 5's latent / the head's (share of prompts; on a pattern left out), names the head's latent fires for")
for x in result["names"]:
    print(f"    {x['name']:5s} {x['step 5' + chr(39) + 's latent']:4d} / {x['the head' + chr(39) + 's latent']:4d} ({100 * x['the head rests on it most in']:.0f}%; {100 * x['on a pattern left out']:.0f}%) {', '.join(x['names for which the head' + chr(39) + 's latent fires on 90% or more'])}")
for label in ("She", "He"):
    print(f"  where a {label} name acts, {result['veto head']} adds to it: " + show({k: mean(lambda t, k=k: t['head']['by kind of the acting name'][label][k]) for k in T[0]['head']['by kind of the acting name'][label] if k != 'prompts'}))
k1 = "the shared latent switched off, where a She name acts"
print(f"  {k1}: " + show({k: mean(lambda t, k=k: t['head'][k1][k]) for k in T[0]['head'][k1]}))
print(f"  adds to the acting name {mean(lambda t: t['head']['adds to the acting name']):+.2f}; estimated to rest on: " + show({k: mean(lambda t, k=k: t['head']['estimated to rest on'][k]) for k in T[0]['head']['estimated to rest on']}))
print("  really lost when switched off: " + show({k: mean(lambda t, k=k: t['head']['really lost when switched off'][k]) for k in T[0]['head']['really lost when switched off']})
      + f"; its highest latent is the head's latent for the name in {100 * mean(lambda t: t['head']['its highest latent is the head' + chr(39) + 's latent for the name']):.0f}%")
print("  the other name: " + show({k: mean(lambda t, k=k: t['head']['the other name'][k]) for k in T[0]['head']['the other name']}))

save(f"runs/features/features-{meta['name']}{tag}.json", result)
