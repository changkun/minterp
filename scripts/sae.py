"""
Step 5: in what units is the stream written? See docs/05-features.md.

Trains a sparse autoencoder on the stream after one block, on ordinary
story text, and asks four things of it.

rebuilds    how much of the stream's variance it misses on text it did not
            train on, against 32 fixed directions; how often latents fire
costs       the model's loss on that text with the rebuilt stream in place
            of the real one, against the plain model and against a stream
            replaced by its mean
fires on    for a sample of latents taken at evenly spaced ranks of how
            often they fire, the text on which each fires most
the task    at the acting name's position, where step 4 found the other
            name being fetched:
            - for each of the twelve names, the latent that best marks it
              as the name that acts, and as the other name
            - how well step 2's reader reads the other name from the
              stream, from the rebuilt stream and from what the dictionary
              misses, and how the score of the reader fitted on the stream
              splits over the two parts
            - what the answer and the veto head of step 4 do when the
              stream there is rebuilt, and when the other name is averaged
              out of it

    uv run scripts/sae.py                    # the model of minllm's stage 3, after block 4
    uv run scripts/sae.py --model seed1
    uv run scripts/sae.py --seed 1 --dict s1 # a second dictionary, for step 8
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from minllm.data import load_tokens

from minterp import probe, tasks
from minterp import sae as S
from minterp.common import MINLLM, eot, load, rounded, save
from minterp.tap import Tap, head_writes

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="seed0", help="seed0, seed1 or seed2: the steps before must have run on it")
parser.add_argument("--block", type=int, default=4, help="the stream after this block")
parser.add_argument("--width", type=int, default=4096)
parser.add_argument("--k", type=int, default=32)
parser.add_argument("--steps", type=int, default=4000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dict", default="", help="a name for another dictionary of the same place, kept beside the first (step 8)")
args = parser.parse_args()

model, tok, meta = load(args.model)
device = next(model.parameters()).device
SEQ, key, n = model.cfg.max_seq_len, ("resid", args.block), model.cfg.n_layers


@torch.no_grad()
def collect(split, windows, seed):
    """The stream after the block at every position of random windows of story text, and the tokens."""
    data = load_tokens(os.path.join(MINLLM, f"data/tinystories/{split}.bin"))
    starts = np.random.default_rng(seed).integers(0, len(data) - SEQ - 1, windows)
    vectors, tokens = [], []
    for i in range(0, windows, 100):
        idx = torch.from_numpy(np.stack([data[s:s + SEQ].astype(np.int64) for s in starts[i:i + 100]])).to(device)
        with Tap(model, keep=("resid",)) as tap:
            model(idx)
        vectors.append(tap[key].float().cpu())
        tokens.append(idx.cpu())
    return torch.cat(vectors), torch.cat(tokens)


train_vectors, _ = collect("train", 4000, seed=1)
held, held_tokens = collect("valid", 300, seed=2)  # (windows, seq, dim), (windows, seq)
dim = train_vectors.shape[-1]
flat_train, flat = train_vectors.reshape(-1, dim), held.reshape(-1, dim)
print(f"{flat_train.shape[0]:,} training vectors, {flat.shape[0]:,} held out")
sae = S.train(flat_train, args.width, args.k, steps=args.steps, seed=args.seed, device=device,
              log=lambda step, loss: print(f"  step {step:5d}  loss {loss:.3f}"))
os.makedirs("out/sae", exist_ok=True)
S.save(sae, S.path(meta["name"], args.block, args.dict))
rebuild = lambda x: sae(x.float())[0].to(x.dtype)

result = {"model": meta, "block": args.block, "width": args.width, "k": args.k, "steps": args.steps, "seed": args.seed,
          "training vectors": flat_train.shape[0], "held-out vectors": flat.shape[0]}

# rebuilds
with torch.no_grad():
    z = torch.cat([sae.encode(flat[i:i + 8192].to(device)).cpu() for i in range(0, flat.shape[0], 8192)])  # (vectors, width)
often = (z > 0).float().mean(0)  # how often each latent fires
story_mean = flat_train[:200_000].mean(0)
# A yardstick: the best 32 fixed directions (principal components of the training vectors), against 32 chosen per vector.
centered = (flat_train[:200_000] - story_mean).double()
basis = torch.linalg.eigh(centered.T @ centered).eigenvectors[:, -args.k:].float()  # (dim, k)
off = flat - story_mean
result["rebuilds"] = {
    "log10 of how often each latent fires": rounded(torch.histc(often.clamp_min(1e-6).log10(), bins=24, min=-5, max=0), 0),
    "unexplained variance": S.unexplained(sae, flat),
    "unexplained variance with the same number of fixed directions": ((off - off @ basis @ basis.T).pow(2).sum() / (flat - flat.mean(0)).pow(2).sum()).item(),
    "latents that never fire": int((often == 0).sum()),
    "fires on share of tokens": {q: often.quantile(p).item() for q, p in (("5%", 0.05), ("median", 0.5), ("95%", 0.95), ("most", 1.0))},
    "active per token": (z > 0).sum(1).float().mean().item(),
}


# costs
@torch.no_grad()
def loss_with(edit=None):
    idx = held_tokens[:100].to(device)
    with Tap(model, edit={key: edit} if edit else None, keep=()):
        logits, _ = model(idx)
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), idx[:, 1:].reshape(-1)).item()


mean_vector = story_mean.to(device)
plain, spliced, meaned = loss_with(), loss_with(rebuild), loss_with(lambda x: mean_vector.to(x.dtype).expand_as(x).clone())
result["costs"] = {"loss": plain, "loss with the rebuilt stream": spliced, "loss with the mean stream": meaned,
                   "share of the loss recovered": (meaned - spliced) / (meaned - plain), "loss rises by": spliced / plain - 1}

# fires on: latents at evenly spaced ranks of how often they fire, so that the sample is not picked by eye
ids = held_tokens.reshape(-1)
alive = torch.nonzero(often > 0)[:, 0]
order = alive[often[alive].argsort(descending=True)]


def strongest(j, count):
    return z[:, j].topk(min(count, int((z[:, j] > 0).sum()))).indices


def one_token(j):
    """Of a latent's 30 strongest activations (or all, if it has fewer), the share on its commonest token."""
    at = strongest(j, 30)
    return torch.bincount(ids[at], minlength=model.cfg.vocab_size).max().item() / len(at)


def examples(j, count):
    out = []
    for at in strongest(j, count).tolist():
        start = max(at - 8, (at // SEQ) * SEQ)
        out.append([tok.decode(ids[start:at].tolist()).replace("\n", " "), tok.decode([ids[at].item()])])
    return out


result["fires on"] = []
for q in np.linspace(0.02, 0.98, 16):
    j = order[int(q * (len(order) - 1))].item()
    fired = ids[z[:, j] > 0]
    shown = ids[strongest(j, 1)[0]].item()  # the token of its strongest example
    result["fires on"].append({"latent": j, "fires on share of tokens": often[j].item(), "examples": examples(j, 8),
                               "share of its 30 strongest on one token": one_token(j),
                               "share of all its firings on its commonest token": torch.bincount(fired).max().item() / len(fired),
                               "share of all its firings on the token of its strongest example": (fired == shown).float().mean().item()})
purity = torch.tensor([one_token(j) for j in order.tolist()])
result["one token"] = {"latents whose 30 strongest are one token in 80% or more": (purity >= 0.8).float().mean().item(),
                       "in under 30%": (purity < 0.3).float().mean().item()}

# ---------------------------------------------------------------- the task, at the acting name
people = tasks.names(tok)
k = len(people)
name_ids = torch.tensor([tok.encode_ordinary(" " + p)[0] for p in people], device=device)
batches = tasks.batches(tok, device, people=people)
first = torch.tensor([people.index(a) for a, _, _ in batches[0].names])
second = torch.tensor([people.index(b) for _, b, _ in batches[0].names])
third = torch.tensor([people.index(c) for _, _, c in batches[0].names])
acts = torch.stack([first, second])  # (versions, prompts): the first name acts on clean prompts, the second on flipped
other = torch.stack([second, first])

# Two kinds of names, as the model sorts them: after a sentence about the name, does it go on with " She" or " He"?
she, he = (tok.encode_ordinary(w)[0] for w in (" She", " He"))
with torch.no_grad():
    after = model(torch.tensor([[eot(tok)] + tok.encode_ordinary(f"One day, {p} went to the park.") for p in people], device=device))[0][:, -1]
kind = (after[:, she] > after[:, he]).cpu()  # (names,): True where the model goes on with " She"

# The head of step 4 that is studied at the last position and carries most: 7.0 in seed0, 8.0 in seed1.
with open(f"runs/circuit/circuit-{meta['name']}.json") as f:
    veto = max((h for h in json.load(f)["heads"] if h["acts at"] == "end"), key=lambda h: h["carry"])

at_name = {x: [] for x in ("stream", "rebuilt", "missed", "averaged", "latents")}  # per pattern and version: (prompts, dim)
effects = {}
MEASURES = ("logit difference", "right side", "change per prompt", "veto head adds to the acting name",
            "veto head adds to the other name", "other name largest in what the veto head writes")
for batch in batches:
    s_pos = batch.pos["S"]
    rows = torch.arange(len(batch), device=device)
    for v, version in enumerate(("clean", "flipped")):
        tokens = getattr(batch, version)
        with torch.no_grad(), Tap(model, keep=("resid",)) as tap:
            model(tokens)
        x = tap[key][:, s_pos].float()
        at_name["stream"].append(x.cpu())
        at_name["rebuilt"].append(sae(x)[0].cpu())
        at_name["missed"].append((x - sae(x)[0]).cpu())
        at_name["latents"].append(sae.encode(x).cpu())
        # The other name averaged out: every prompt gets the mean stream of the prompts of its pattern and version in
        # which the same name acts. Those eleven prompts differ in one token, the other name.
        same_actor = torch.stack([x[(acts[v] == a).to(device)].mean(0) for a in acts[v].tolist()])
        at_name["averaged"].append(same_actor.cpu())

        def put(replacement, s_pos=s_pos):
            def edit(stream):
                stream = stream.clone()
                stream[:, s_pos] = replacement(stream[:, s_pos])
                return stream
            return edit

        ways = {"plain": None, "rebuilt at the acting name": put(rebuild), "rebuilt everywhere": rebuild,
                "the other name averaged out at the acting name": put(lambda cur, m=same_actor: m.to(cur.dtype))}
        base = None
        for where, edit in ways.items():
            with torch.no_grad(), Tap(model, edit={key: edit} if edit else None) as t2:
                logits, _ = model(tokens)
            ld = tasks.logit_diff(logits.float(), batch.right, batch.wrong)
            base = ld if where == "plain" else base
            final = t2["resid", n][:, -1]
            size = torch.rsqrt(final.pow(2).mean(-1, keepdim=True) + model.norm.eps)
            write = head_writes(model, t2, veto["block"])[:, -1, veto["head"]]
            names = ((write * model.norm.weight * size) @ model.lm_head.weight[name_ids].T).float()  # (prompts, names)
            nine = torch.ones_like(names, dtype=torch.bool)
            for who in (first, second, third):  # against the nine names that are none of A, B and C, as in step 4
                nine[rows, who.to(device)] = False
            rest = (names * nine).sum(1) / nine.sum(1)
            a, o = acts[v].to(device), other[v].to(device)
            without_actor = names.clone()
            without_actor[rows, a] = float("-inf")
            sign = 1 if version == "clean" else -1
            d = effects.setdefault(where, {}).setdefault(version, {m: [] for m in MEASURES})
            d["logit difference"].append(ld.mean().item())
            d["right side"].append((sign * ld > 0).float().mean().item())
            d["change per prompt"].append((ld - base).abs().mean().item())
            d["veto head adds to the acting name"].append((names[rows, a] - rest).mean().item())
            d["veto head adds to the other name"].append((names[rows, o] - rest).mean().item())
            d["other name largest in what the veto head writes"].append((without_actor.argmax(1) == o).float().mean().item())

patterns, prompts = len(batches), len(first)
stacked = {x: torch.stack(parts).view(patterns, 2, prompts, -1) for x, parts in at_name.items()}  # (patterns, versions, prompts, dim)
result["task"] = {"prompts": patterns * 2 * prompts, "veto head": f"block {veto['block']} head {veto['head']}",
                  "names that go on with She": [p for p, g in zip(people, kind.tolist()) if g],
                  "effects": {where: {v: {m: {"mean": float(np.mean(x)), "by pattern": rounded(torch.tensor(x), 3)} for m, x in d.items()}
                                      for v, d in by.items()} for where, by in effects.items()}}

# How much of the stream the dictionary misses at this place, where the stories' figure need not hold.
xs, miss = stacked["stream"].reshape(-1, dim), stacked["missed"].reshape(-1, dim)
result["task"]["misses at the acting name"] = {
    "of the variance around the stories' mean": (miss.pow(2).sum() / (xs - story_mean).pow(2).sum()).item(),
    "of the variance among these prompts": (miss.pow(2).sum() / (xs - xs.mean(0)).pow(2).sum()).item(),
    "of the stream's size": (miss.norm(dim=1) / xs.norm(dim=1)).mean().item(),
}

# A latent for each name: as the name that acts, and as the other name.
Z = stacked["latents"].reshape(-1, args.width)
fires = Z > 0
role = {"acts": acts[None].expand(patterns, -1, -1).reshape(-1), "other": other[None].expand(patterns, -1, -1).reshape(-1)}
busy = torch.nonzero(fires.float().mean(0) >= 0.02)[:, 0]  # latents that fire on some of these prompts: the candidates for the size test
result["task"]["names"] = {}
for r, lab in role.items():
    table = []
    for i, name in enumerate(people):
        yes, no = lab == i, lab != i
        gap = fires[yes].float().mean(0) - fires[no].float().mean(0)  # fires when the name has the role, and not otherwise
        j = gap.argmax().item()
        # By size instead of firing: how often a prompt with the name in the role has a larger value than one without.
        larger = (Z[yes][:, None, busy] > Z[no][None, :, busy]).float().mean((0, 1))
        equal = (Z[yes][:, None, busy] == Z[no][None, :, busy]).float().mean((0, 1))
        auroc = larger + equal / 2
        on_token = ids == name_ids[i].item()
        both = ((z[:, j] > 0) & on_token).sum().item()
        table.append({
            "name": name, "latent": j, "fires when it is": fires[yes, j].float().mean().item(), "fires when it is not": fires[no, j].float().mean().item(),
            "fires on share of story tokens": often[j].item(),
            "of its story firings, on the name's token": both / max(1, int((z[:, j] > 0).sum())),
            "of the name's story tokens, fires on": both / max(1, int(on_token.sum())),
            "the name's story tokens": int(on_token.sum()),
            "share of its 30 strongest on one token": one_token(j),
            "share of its 30 strongest on the name's token": (ids[strongest(j, 30)] == name_ids[i].item()).float().mean().item(),
            "a latent that fires on 80% with and under 20% without": bool(((fires[yes].float().mean(0) > 0.8) & (fires[no].float().mean(0) < 0.2)).any()),
            "best separation by size": auroc.max().item(), "its latent": busy[auroc.argmax()].item(),
        })
    result["task"]["names"][r] = table

# The other name, which at this position is the due name of step 2, read by step 2's reader:
# fitted on four patterns, tested on the fifth, on name pairs it did not train on.
held_pairs = (first + second) % 3 == 0
result["task"]["reader"] = {}


def fold(X, f):
    others = [q for q in range(patterns) if q != f]
    xtr = X[others][:, :, ~held_pairs].reshape(-1, X.shape[-1])
    ltr = other[:, ~held_pairs][None].expand(len(others), -1, -1).reshape(-1)
    gtr = torch.tensor(others)[:, None, None].expand(len(others), 2, int((~held_pairs).sum())).reshape(-1)
    return xtr, ltr, gtr, X[f][:, held_pairs]  # the test rows stay (versions, pairs, dim)


truth = other[:, held_pairs]  # (versions, pairs)
for name in ("stream", "rebuilt", "missed", "averaged"):  # the last is the floor: a stream that holds nothing of the other name
    X = stacked[name].double()
    right, same_kind = [], []
    for f in range(patterns):
        xtr, ltr, gtr, xte = fold(X, f)
        guess = probe.read(xtr[None], ltr[None], gtr, xte.reshape(1, -1, X.shape[-1]), k)[0].argmax(-1)[0, :, 0].view(2, -1)
        right.append((guess == truth).double().mean(1))
        same_kind.append((kind[guess] == kind[truth]).double().mean(1))
    right, same_kind = torch.stack(right), torch.stack(same_kind)  # (patterns, versions)
    result["task"]["reader"][name] = {"clean": right[:, 0].mean().item(), "flipped": right[:, 1].mean().item(), "by pattern": rounded(right),
                                      "a name of the right kind": rounded(same_kind.mean(0))}

# The reader fitted on the stream, applied to the two parts of the stream. Its scores are linear, so they add.
split = {m: [] for m in ("from the missed part: share of the margin", "right from the rebuilt part alone", "right from the missed part alone")}
for f in range(patterns):
    xtr, ltr, gtr, _ = fold(stacked["stream"].double(), f)
    W, b, mean, spread = probe.weights(xtr, ltr, gtr, k)
    kept = ((stacked["rebuilt"][f][:, held_pairs].double() - mean) / spread) @ W + b  # (versions, pairs, names)
    lost = (stacked["missed"][f][:, held_pairs].double() / spread) @ W

    def margin(scores):  # the true name's score above the mean of the other eleven
        true = scores.gather(2, truth[:, :, None])[:, :, 0]
        return (true - (scores.sum(2) - true) / (k - 1)).mean(1)

    split["from the missed part: share of the margin"].append(margin(lost) / (margin(kept) + margin(lost)))
    split["right from the rebuilt part alone"].append((kept.argmax(2) == truth).double().mean(1))
    split["right from the missed part alone"].append((lost.argmax(2) == truth).double().mean(1))
result["task"]["reader fitted on the stream"] = {m: rounded(torch.stack(x).mean(0)) for m, x in split.items()}  # (versions,)

# What the latents that stand for the other name fire on in stories.
shared = {}
for row in result["task"]["names"]["other"]:
    shared.setdefault(row["latent"], []).append(row["name"])
result["task"]["other-name latents"] = [{"latent": j, "names": names, "fires on share of story tokens": often[j].item(), "examples": examples(j, 6)}
                                        for j, names in shared.items()]

# ---------------------------------------------------------------- what it found
r, c, t = result["rebuilds"], result["costs"], result["task"]
pc = lambda x: f"{100 * x:.0f}"
print(f"\nmisses {100 * r['unexplained variance']:.1f}% of the variance on held-out text ({args.k} fixed directions miss "
      f"{100 * r['unexplained variance with the same number of fixed directions']:.0f}%); {r['latents that never fire']} latents never fire; "
      f"a latent fires on {100 * r['fires on share of tokens']['median']:.2f}% of tokens (median)")
print(f"loss {c['loss']:.3f}, with the rebuilt stream {c['loss with the rebuilt stream']:.3f} (+{100 * c['loss rises by']:.1f}%), with the mean stream "
      f"{c['loss with the mean stream']:.3f}: {100 * c['share of the loss recovered']:.1f}% recovered")
print(f"one-token latents {pc(result['one token']['latents whose 30 strongest are one token in 80% or more'])}%")
for row in result["fires on"]:
    print(f"  latent {row['latent']:4d}, {100 * row['fires on share of tokens']:.2f}% of tokens, {pc(row['share of its 30 strongest on one token'])}% of its strongest and "
          f"{pc(row['share of all its firings on its commonest token'])}% of all on one token: " + " | ".join(f"{a[-24:]}[{b}]" for a, b in row["examples"][:3]))
m = list(t["misses at the acting name"].values())
print(f"\nat the acting name the dictionary misses {100 * m[0]:.1f}% of the variance around the stories' mean, "
      f"{100 * m[1]:.1f}% of the variance among the prompts, {100 * m[2]:.0f}% of the stream's size")
print("the model goes on with She after:", ", ".join(t["names that go on with She"]))
for role_name in ("acts", "other"):
    rows_ = t["names"][role_name]
    print(f"a latent for each name when it {'acts' if role_name == 'acts' else 'is the other name'}: fires when it is / is not; best separation by size")
    print("  " + "  ".join(f"{x['name']} {pc(x['fires when it is'])}/{pc(x['fires when it is not'])} {x['best separation by size']:.2f}" for x in rows_))
    print(f"  {len({x['latent'] for x in rows_})} different latents; names with a latent at 80% and under 20%: {sum(x['a latent that fires on 80% with and under 20% without'] for x in rows_)}; "
          f"with separation by size of 0.9 or more: {sum(x['best separation by size'] >= 0.9 for x in rows_)}")
print("  on stories, the latents for the acting name: share of their firings on the name's token: " + " ".join(pc(x["of its story firings, on the name's token"]) for x in t["names"]["acts"]))
print("the other name read by step 2's reader (clean / flipped, percent right; a name of the right kind):")
for name, v in t["reader"].items():
    print(f"  from the {name:8s} {pc(v['clean'])}/{pc(v['flipped'])}; kind {pc(v['a name of the right kind'][0])}/{pc(v['a name of the right kind'][1])}")
print("the reader fitted on the stream: " + "; ".join(f"{name} {pc(v[0])}/{pc(v[1])}" for name, v in t["reader fitted on the stream"].items()))
print(f"what happens to the answer and to {t['veto head']} (clean | flipped):")
for where, by in t["effects"].items():
    g = lambda v, name: by[v][name]["mean"]
    print(f"  {where:46s} " + " | ".join(
        f"difference {g(v, 'logit difference'):+.2f} (right {pc(g(v, 'right side'))}%, moves {g(v, 'change per prompt'):.2f} a prompt), "
        f"acting {g(v, 'veto head adds to the acting name'):+.2f}, other {g(v, 'veto head adds to the other name'):+.2f}, "
        f"other largest {pc(g(v, 'other name largest in what the veto head writes'))}%" for v in ("clean", "flipped")))

save(f"runs/sae/sae-{meta['name']}{'-' + args.dict if args.dict else ''}.json", result)
