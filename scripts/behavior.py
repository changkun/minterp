"""
Step 1: does the model do the task, and how well? See docs/01-read.md.

For every template and every ordered pair of names, in the three versions
of minterp/tasks.py, this records the logit of the second name (B) minus
the logit of the first (A) at the last position, the probability of each,
and the token the model holds likeliest.

    uv run scripts/behavior.py                  # the model of minllm's stage 3
    uv run scripts/behavior.py --model seed1    # the same recipe, another seed
"""

import argparse
from collections import Counter

import torch

from minterp import tasks
from minterp.common import load, rounded, save

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="seed0", help="seed0, seed1, or the path of a checkpoint")
args = parser.parse_args()

model, tok, meta = load(args.model)
device = next(model.parameters()).device
people = tasks.names(tok)
result = {"model": meta, "names": people, "templates": []}

for batch in tasks.batches(tok, device, people=people):
    entry = {"template": batch.template, "prompts": len(batch), "labels": batch.labels, "pos": batch.pos}
    for version in ("clean", "flipped", "neutral"):
        with torch.no_grad():
            logits, _ = model(getattr(batch, version))
        last = logits[:, -1].float()
        probs = last.softmax(-1)
        top = last.argmax(-1)
        entry[version] = {
            # Always B minus A: above 0 means the model prefers the second name.
            "logit_diff": rounded(tasks.logit_diff(last, batch.right, batch.wrong)),
            "p_B": rounded(probs.gather(1, batch.right[:, None])[:, 0]),
            "p_A": rounded(probs.gather(1, batch.wrong[:, None])[:, 0]),
            "top_is_B": (top == batch.right).float().mean().item(),
            "top_is_A": (top == batch.wrong).float().mean().item(),
            "top_tokens": Counter(tok.decode([t]) for t in top.tolist()).most_common(5),
        }
    result["templates"].append(entry)
    ld = {v: torch.tensor(entry[v]["logit_diff"]) for v in ("clean", "flipped", "neutral")}
    print(f"{batch.template}\n"
          f"  clean    B - A = {ld['clean'].mean():+.2f}, above 0 in {100 * (ld['clean'] > 0).float().mean():.0f}%, "
          f"B is the top token in {100 * entry['clean']['top_is_B']:.0f}%, p(B) = {torch.tensor(entry['clean']['p_B']).mean():.2f}\n"
          f"  flipped  B - A = {ld['flipped'].mean():+.2f}, below 0 in {100 * (ld['flipped'] < 0).float().mean():.0f}%, "
          f"A is the top token in {100 * entry['flipped']['top_is_A']:.0f}%\n"
          f"  neutral  B - A = {ld['neutral'].mean():+.2f}; top tokens {entry['neutral']['top_tokens'][:3]}")

save(f"runs/read/behavior-{meta['name']}.json", result)
