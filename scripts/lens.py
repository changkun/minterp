"""
Step 1: where in the model does the answer show? See docs/01-read.md.

For every template, and for each of the three versions of its prompts,
this records

- the logit of the second name (B) minus that of the first (A), decoded from
  the stream after the embedding and after every block, as if the model
  stopped there (the logit lens), and
- the final difference split into what the embedding, every head and every
  MLP wrote directly into it (direct logit attribution).

The difference is always B minus A, also where A is the answer, so that a
part which writes the same number whoever acts can be told from a part
whose number follows the acting name.

    uv run scripts/lens.py
    uv run scripts/lens.py --model seed1
"""

import argparse

import torch

from minterp import lens, tasks
from minterp.common import load, rounded, save
from minterp.tap import Tap

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="seed0", help="seed0, seed1, or the path of a checkpoint")
args = parser.parse_args()

model, tok, meta = load(args.model)
device = next(model.parameters()).device
n = model.cfg.n_layers
result = {"model": meta, "templates": []}
VERSIONS = ("clean", "flipped", "neutral")
pooled = {v: [] for v in VERSIONS}

for batch in tasks.batches(tok, device):
    entry = {"template": batch.template, "prompts": len(batch), "lens": {}}
    print(batch.template)
    print("  decoded after  " + " ".join(f"{'emb' if i == 0 else i:>6}" for i in range(n + 1)))
    for version in VERSIONS:
        with Tap(model) as tap, torch.no_grad():
            logits, _ = model(getattr(batch, version))
        final = tasks.logit_diff(logits.float(), batch.right, batch.wrong)
        depth = torch.stack([
            tasks.logit_diff(lens.decode(model, tap["resid", i][:, -1]).float(), batch.right, batch.wrong) for i in range(n + 1)
        ])  # (depth, prompts), depth 0 = after the embedding
        parts = {k: v.float() for k, v in lens.attribute(model, tap, batch.right, batch.wrong).items()}
        total = parts["embed"] + parts["heads"].sum((1, 2)) + parts["mlp"].sum(1)
        assert torch.allclose(total, final, atol=1e-2), "the attribution does not add up"
        pooled[version].append(parts)
        entry["lens"][version] = rounded(depth, 3)
        if version == "clean":
            # The stream's root mean square at the last position, by depth. The
            # final norm divides by it, so the lens reads a small early stream
            # at full size.
            entry["size"] = rounded(torch.stack([tap["resid", i][:, -1].float().pow(2).mean(-1).sqrt().mean() for i in range(n + 1)]), 3)
        print(f"  {version:8s} B - A " + " ".join(f"{v:+6.2f}" for v in depth.mean(1).tolist()))
    result["templates"].append(entry)


def summary(x):
    """Mean and quartiles over the prompts, per part."""
    q = torch.quantile(x, torch.tensor([0.25, 0.5, 0.75], device=x.device), dim=0)
    return {"mean": rounded(x.mean(0), 3), "q25": rounded(q[0], 3), "median": rounded(q[1], 3), "q75": rounded(q[2], 3)}


result["attribution"] = {}
for version in VERSIONS:
    cat = {k: torch.cat([p[k] for p in pooled[version]]) for k in ("embed", "heads", "mlp")}
    result["attribution"][version] = {"prompts": len(cat["embed"]), **{k: summary(v) for k, v in cat.items()}}

mean = {v: result["attribution"][v] for v in VERSIONS}
heads = {v: torch.tensor(mean[v]["heads"]["mean"]) for v in VERSIONS}
mlp = {v: torch.tensor(mean[v]["mlp"]["mean"]) for v in VERSIONS}
print(f"\nwritten directly into B - A on the clean prompts, mean over {mean['clean']['prompts']} prompts")
print("  block  " + " ".join(f"head {h}" for h in range(model.cfg.n_heads)) + "     MLP")
for i in range(n):
    print(f"  {i + 1:5d}  " + " ".join(f"{v:+6.2f}" for v in heads["clean"][i].tolist()) + f"  {mlp['clean'][i].item():+6.2f}")
ranked = [(abs(heads["clean"][i, h].item()), f"block {i + 1} head {h}", ("heads", i, h)) for i in range(n) for h in range(model.cfg.n_heads)]
ranked += [(abs(mlp["clean"][i].item()), f"block {i + 1} MLP", ("mlp", i)) for i in range(n)]
ranked.sort(reverse=True)
print("\n  the largest parts, and what they write in each version")
print(f"  {'':16s} {'clean':>7} {'flipped':>8} {'neutral':>8}")
for _, name, key in ranked[:8]:
    get = lambda v: (heads[v][key[1], key[2]] if key[0] == "heads" else mlp[v][key[1]]).item()
    print(f"  {name:16s} {get('clean'):+7.2f} {get('flipped'):+8.2f} {get('neutral'):+8.2f}")

save(f"runs/read/lens-{meta['name']}.json", result)
