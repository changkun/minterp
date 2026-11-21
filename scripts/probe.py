"""
Step 2: where can the answer be read out? See docs/02-probe.md.

For every depth (after the embedding and after every block) and five
positions named by their role (A, B, S, the token after S, and the last
one), this fits linear readers from the stream to four properties of the
prompt, each a name out of twelve:

    due      the name that is due          acts     the name that acts
    first    the first name, A             second   the second name, B

and to a fifth that is made up, `control`: a random name drawn once for
every ordered pair, the same in both versions and all patterns. Nothing in
the text determines it, so a reader can only get it right by remembering
pairs it was trained on.

Readers are trained on four sentence patterns and tested on the fifth, each
pattern in turn, under two rules:

    new    tested on name pairs that were held out of training as well
    seen   tested on the same pairs it was trained on

It also records, on the same test prompts, how often the lens of step 1
ranks the due name first among the twelve; how well the due name is read by
combining the readers for first, second and acts without training on it;
and how well it can be read from the three tokens' identities alone, with
no model at all.

    uv run scripts/probe.py
    uv run scripts/probe.py --model seed1
"""

import argparse
import os

import torch

from minterp import lens, probe, tasks
from minterp.common import load, rounded, save
from minterp.tap import Tap

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="seed0", help="seed0, seed1, or the path of a checkpoint")
args = parser.parse_args()

ROLES = ("A", "B", "S", "S+1", "end")
TARGETS = ("due", "acts", "first", "second", "control")
VERSIONS = ("clean", "flipped")

model, tok, meta = load(args.model, device="cpu")  # float64 on the CPU: the same numbers on every machine
people = tasks.names(tok)
k, depth = len(people), model.cfg.n_layers + 1
name_ids = torch.tensor([tok.encode_ordinary(" " + n)[0] for n in people])
batches = tasks.batches(tok, people=people)

# The stream at the five positions, and the lens's pick among the twelve names at the last one.
streams, picks = [], []
for batch in batches:
    at = [batch.pos["A"], batch.pos["B"], batch.pos["S"], batch.pos["S"] + 1, batch.pos["end"]]
    per_version, per_pick = [], []
    for version in VERSIONS:
        with Tap(model, keep=("resid",)) as tap, torch.no_grad():
            model(getattr(batch, version))
        stream = torch.stack([tap["resid", i][:, at] for i in range(depth)], dim=1)  # (prompts, depth, roles, dim)
        per_version.append(stream)
        per_pick.append(lens.decode(model, stream[:, :, -1])[..., name_ids].argmax(-1))  # (prompts, depth)
    streams.append(torch.stack(per_version))
    picks.append(torch.stack(per_pick))
X = torch.stack(streams).double()  # (patterns, versions, prompts, depth, roles, dim)
picks = torch.stack(picks)  # (patterns, versions, prompts, depth)
patterns, prompts = X.shape[0], X.shape[2]

# Labels, as indices into `people`. They are the same in every pattern.
index = {n: i for i, n in enumerate(people)}
first = torch.tensor([index[a] for a, _, _ in batches[0].names])
second = torch.tensor([index[b] for _, b, _ in batches[0].names])
control = torch.randint(0, k, (prompts,), generator=torch.Generator().manual_seed(0)).repeat(2, 1)
labels = torch.stack([
    torch.stack([second, first]),  # due: B on the clean prompt, A on the flipped one
    torch.stack([first, second]),  # acts
    torch.stack([first, first]),
    torch.stack([second, second]),
    control,
])  # (targets, versions, prompts)

# A third of the pairs, in both orders, is kept out of training under the rule "new".
held = (first + second) % 3 == 0
assert set(first[~held].tolist()) == set(second[~held].tolist()) == set(range(k)), "a name never trains"


def rows(which, mask):
    """The chosen patterns and pairs as rows: streams (places, rows, dim), labels, pattern and version of every row."""
    x = X[which][:, :, mask]  # (patterns, versions, pairs, depth, roles, dim)
    p, v, m = x.shape[:3]
    x = x.permute(3, 4, 0, 1, 2, 5).reshape(depth * len(ROLES), p * v * m, -1)
    lab = labels[:, :, mask][:, None].expand(-1, p, -1, -1).reshape(len(TARGETS), -1)
    group = torch.tensor(which)[:, None, None].expand(p, v, m).reshape(-1)
    version = torch.arange(v)[None, :, None].expand(p, v, m).reshape(-1)
    return x, lab, group, version


def by_version(right, version):
    """(places, rows, ...) of hits -> (versions, depth, roles, ...) of accuracies."""
    out = torch.stack([right[:, version == v].double().mean(1) for v in range(len(VERSIONS))])
    return out.unflatten(1, (depth, len(ROLES)))


result = {
    "model": meta, "names": people, "roles": ROLES, "targets": TARGETS, "versions": VERSIONS, "penalties": probe.LAMS,
    "pairs": {"held out": int(held.sum()), "trained": int((~held).sum())},
}
every = torch.ones(prompts, dtype=torch.bool)
for rule, train_mask, test_mask in (("new", ~held, held), ("seen", every, every)):
    acc, composite, lens_acc, chosen = [], [], [], []
    for fold in range(patterns):
        others = [p for p in range(patterns) if p != fold]
        xtr, ltr, gtr, _ = rows(others, train_mask)
        xte, lte, _, vte = rows([fold], test_mask)
        out, pick = probe.read(xtr, ltr, gtr, xte, k)  # (places, rows, targets, k)
        acc.append(by_version(out.argmax(-1) == lte.T, vte))  # (versions, depth, roles, targets)
        chosen.append(pick.unflatten(0, (depth, len(ROLES))))
        # The due name without a reader of its own: it is one of the two names, and not the one that acts.
        t = TARGETS.index
        combined = out[:, :, t("first")] + out[:, :, t("second")] - out[:, :, t("acts")]
        composite.append(by_version(combined.argmax(-1) == lte[t("due")], vte))  # (versions, depth, roles)
        due = labels[t("due")][:, test_mask]  # (versions, pairs)
        lens_acc.append((picks[fold][:, test_mask] == due[:, :, None]).double().mean(1))  # (versions, depth)
    result[rule] = {
        "train rows": xtr.shape[1], "test rows": xte.shape[1],
        "accuracy": rounded(torch.stack(acc)),  # (folds, versions, depth, roles, targets)
        "composite": rounded(torch.stack(composite)),  # (folds, versions, depth, roles)
        "lens": rounded(torch.stack(lens_acc)),  # (folds, versions, depth)
        "penalty": torch.stack(chosen)[..., 0].tolist(),  # for the due name: (folds, depth, roles), an index into penalties
    }

# No model at all: can a linear reader produce the due name from the identities of the tokens?
eye = torch.eye(k, dtype=torch.float64)
tokens = {
    "first, second and acts": lambda v: torch.cat([eye[first], eye[second], eye[labels[1, v]]], dim=1),
    "first and second": lambda v: torch.cat([eye[first], eye[second]], dim=1),
}
result["tokens only"] = {}
for name, make in tokens.items():
    feats = torch.stack([make(v) for v in range(2)])  # (versions, prompts, dim)
    xtr, xte = feats[:, ~held].reshape(1, -1, feats.shape[-1]), feats[:, held].reshape(1, -1, feats.shape[-1])
    ltr, lte = labels[:1, :, ~held].reshape(1, -1), labels[0][:, held]
    groups = torch.arange(xtr.shape[1]) % 4
    out = probe.read(xtr, ltr, groups, xte, k)[0][0, :, 0]  # (test rows, k)
    right = (out.argmax(-1).view(2, -1) == lte).double()
    # Without the acting name the two versions of a prompt are the same input,
    # so the reader gives the same answer to both and is right on one. Which
    # one is a tie between two equal scores, so only the two together count.
    top2 = out.topk(2, dim=-1).values
    result["tokens only"][name] = {
        "clean": right[0].mean().item(), "flipped": right[1].mean().item(), "both": right.mean().item(),
        "tied": bool(((top2[:, 0] - top2[:, 1]).abs() < 1e-9).all()),
    }

path = save(f"runs/probe/probe-{meta['name']}.json", result)

# The reader for the due name at the last position, fitted on every prompt, for later steps. Not part of the record.
x, lab, _, _ = rows(list(range(patterns)), every)
end = [d * len(ROLES) + ROLES.index("end") for d in range(depth)]
mean, spread = x[end].mean(1, keepdim=True), x[end].std(1, keepdim=True).clamp_min(1e-8)
W, b = probe.fit((x[end] - mean) / spread, probe.onehot(lab[:1], k), 0.1)
os.makedirs("out/probe", exist_ok=True)
torch.save({"W": W, "b": b, "mean": mean, "spread": spread, "names": people}, f"out/probe/due-{meta['name']}.pt")

new, seen = torch.tensor(result["new"]["accuracy"]).mean(0), torch.tensor(result["seen"]["accuracy"]).mean(0)
t, e = TARGETS.index, ROLES.index("end")
show = lambda row: " ".join(f"{100 * v:4.0f}" for v in row.tolist())
print(f"\nthe due name at the last position, percent right (chance {100 / k:.0f}), mean of {patterns} held-out patterns")
print("  after block             emb    1    2    3    4    5    6    7    8")
for v, version in enumerate(VERSIONS):
    print(f"  reader, new pairs, {version:8s}" + show(new[v, :, e, t("due")]))
for v, version in enumerate(VERSIONS):
    print(f"  lens, {version:8s}             " + show(torch.tensor(result["new"]["lens"]).mean(0)[v]))
for v, version in enumerate(VERSIONS):
    print(f"  combined readers, {version:8s} " + show(torch.tensor(result["new"]["composite"]).mean(0)[v, :, e]))
for v, version in enumerate(VERSIONS):
    print(f"  seen pairs: due, {version:8s}  " + show(seen[v, :, e, t("due")]))
print("  seen pairs: control           " + show(seen[:, :, e, t("control")].mean(0)))
print("  new pairs: control            " + show(new[:, :, e, t("control")].mean(0)))
print("\nthe worse of the two versions, new pairs, by depth (rows) and position (columns: " + ", ".join(ROLES) + ")")
for target in TARGETS[:4]:
    print(f"  {target}")
    worse = new[:, :, :, t(target)].min(0).values
    for d in range(depth):
        print(f"    {'emb' if d == 0 else d:>3}  " + show(worse[d]))
print("\nfrom the tokens' identities alone, no model:", result["tokens only"])
