"""
Step 9: the same numbers from a library. See docs/09-stack.md.

minterp writes its own hooks, lens, patches and attention so that nothing
is hidden. Libraries exist for each of these. This script runs one of
them, TransformerLens, on the open model of step 8 and sets its numbers
beside minterp's, piece by piece: the same weights (the files minllm
downloaded), the same token ids.

logits     minllm's loader against Hugging Face transformers, before the
           library touches that model; then the library's wrapped model
           against both
pieces     every piece a Tap records, against the library's hook of that
           name, in three blocks
pattern    the attention weights minterp recomputes, against the library's
attribute  step 1's split of the logit difference, with its heads summed
           within each block, against the library's helper for that
           split: on the weights as published, with the final norm's gain
           multiplied in first, and in the library's compatibility mode,
           which folds the norms into the weights
mode       what else compatibility mode changes: the logits, and what the
           heads record
patch      the stream of the flipped run put into the clean run at the
           acting name, by minterp/patch.py and by the library's hooks
steer      a vector added there, by minterp/steer.py and by the library's
           hooks

The library is not a dependency of minterp. Run this with it laid over the
project for one command, at the versions the record was made with:

    uv run --with transformer-lens==4.4.0 --with transformers==5.24.0 scripts/stack.py
"""

import os
from importlib.metadata import version

import torch

from minterp import attention, lens, patch, steer, tasks
from minterp.common import MINLLM, OPEN, load, rounded, save
from minterp.tap import Tap

try:
    from transformer_lens.model_bridge import TransformerBridge
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise SystemExit("TransformerLens is not installed. Run:  uv run --with transformer-lens==4.4.0 --with transformers==5.24.0 scripts/stack.py")

NAME, COUNT = "qwen", 16
folder, repo = os.path.join(MINLLM, OPEN[NAME][0]), OPEN[NAME][1]
model, tok, meta = load(NAME, device="cpu")  # the CPU for both, so that the difference is not the device's
n = model.cfg.n_layers
half = n // 2
batch = tasks.batches(tok, "cpu", people=tasks.names(tok))[0]
clean, flipped, s = batch.clean[:COUNT], batch.flipped[:COUNT], batch.pos["S"]
right, wrong = batch.right[:COUNT], batch.wrong[:COUNT]
ld = lambda logits: tasks.logit_diff(logits[:, -1].float(), right, wrong).mean().item()
gap = lambda a, b: (a - b).abs().max().item()  # the largest absolute difference, over every entry
size = lambda a: a.abs().max().item()  # the largest absolute entry

with torch.no_grad(), Tap(model) as tap:
    mine, _ = model(clean)

def counted(hf_model):
    """Count the calls of transformers' own attention and MLP modules, on the objects as they are before any wrapping."""
    calls = {"attention": 0, "MLP": 0}

    def count(kind):
        def hook(module, args, output):
            calls[kind] += 1
        return hook

    for layer in hf_model.model.layers:
        layer.self_attn.register_forward_hook(count("attention"))
        layer.mlp.register_forward_hook(count("MLP"))
    return calls


# The published weights again, read by transformers. Its logits are taken before the library wraps that very model,
# because wrapping changes it in place: the library then computes the attention itself, from transformers' weights.
hf = AutoModelForCausalLM.from_pretrained(folder, torch_dtype=torch.float32)
hf_tok = AutoTokenizer.from_pretrained(folder)
paths = {"model.layers": len(hf.model.layers), "lm_head": hasattr(hf, "lm_head"), "transformer.h": hasattr(hf, "transformer")}  # what a wrapper without names of its own exposes
before = hf.config._attn_implementation
calls = counted(hf)
with torch.no_grad():
    theirs = hf(clean).logits
unwrapped = dict(calls)
bridge = TransformerBridge.boot_transformers(repo, hf_model=hf, tokenizer=hf_tok, device="cpu")
calls.update(attention=0, MLP=0)
with torch.no_grad():
    logits, cache = bridge.run_with_cache(clean)  # token ids, not text
wrapped = dict(calls)
with torch.no_grad():
    _, other = bridge.run_with_cache(flipped)

# What a Tap records, and the library's name for it. Blocks are counted from 1 in minterp and from 0 there.
PIECES = (("resid", "blocks.{}.hook_resid_post"), ("attn", "blocks.{}.hook_attn_out"), ("mlp", "blocks.{}.hook_mlp_out"), ("heads", "blocks.{}.attn.hook_z"))
first = [t[0] for t in batch.names[:COUNT]]
result = {"model": meta, "versions": {p: version(p) for p in ("transformer-lens", "transformers", "torch")},
          "prompts": {"count": COUNT, "of": len(batch), "pattern": batch.template, "first names": {p: first.count(p) for p in sorted(set(first))}},
          "the tokenizers decode the prompt alike": hf_tok.decode(clean[0].tolist()) == tok.decode(clean[0].tolist()),
          "a string gets no first token from the library": len(bridge.to_tokens("One day")[0]) == len(tok.encode_ordinary("One day")),
          "module paths of the transformers model": paths, "the attention setting in transformers' config": {"before wrapping": before, "after": hf.config._attn_implementation},
          "calls of transformers' own modules in one pass": {"before wrapping": unwrapped, "wrapped": wrapped},
          "hook points in the cache": len(cache),
          "logits": {"largest entry": size(mine), "minllm's loader against transformers, before wrapping": gap(mine, theirs),
                     "the library against minterp": gap(logits, mine), "the library against transformers, before wrapping": gap(logits, theirs)},
          "pieces": {'("resid", 0) against hook_embed': {"difference": gap(tap["resid", 0], cache["hook_embed"]), "largest entry": size(tap["resid", 0])}},
          "names": {'("resid", 0)': "hook_embed"}, "canonical names": {"hook_embed": bridge.hook_dict["hook_embed"].name}}
for kind, theirs_name in PIECES:
    result["names"][f'("{kind}", i)'] = theirs_name.format("i-1")
    result["canonical names"][theirs_name.format("i")] = bridge.hook_dict[theirs_name.format(0)].name.replace(".0.", ".i.")
    for block in (1, half, n):
        result["pieces"][f'("{kind}", {block}) against {theirs_name.format(block - 1)}'] = {"difference": gap(tap[kind, block], cache[theirs_name.format(block - 1)]),
                                                                                           "largest entry": size(tap[kind, block])}
result["names"]["attention.patterns(model, tap)[:, i-1]"] = "blocks.i-1.attn.hook_pattern"
weights = attention.patterns(model, tap)
result["pattern"] = {f"block {block}": gap(weights[:, block - 1], cache[f"blocks.{block - 1}.attn.hook_pattern"]) for block in (1, half, n)}

# Step 1's attribution, and the library's helper for the same split. lens.attribute goes down to single heads; the
# helper stops at a block's attention, so the heads are summed within each block.
parts = lens.attribute(model, tap, right, wrong)
named = {"embed": parts["embed"], **{f"{i}_attn_out": parts["heads"].sum(2)[:, i] for i in range(n)}, **{f"{i}_mlp_out": parts["mlp"][:, i] for i in range(n)}}
worst = lambda attrs, labels: max(gap(attrs[j], named[label]) for j, label in enumerate(labels))
with torch.no_grad():
    stack, labels = cache.decompose_resid(pos_slice=-1, return_labels=True)  # (parts, prompts, dim) at the last position
    raw = cache.logit_attrs(stack, tokens=right, incorrect_tokens=wrong, pos_slice=-1)
    # The helper divides by the final norm's scale and expects the norm's learned gain to be in the unembedding already.
    gained = cache.logit_attrs(stack * model.norm.weight, tokens=right, incorrect_tokens=wrong, pos_slice=-1)
    second = AutoModelForCausalLM.from_pretrained(folder, torch_dtype=torch.float32)
    calls_in_mode = counted(second)
    folded = TransformerBridge.boot_transformers(repo, hf_model=second, tokenizer=hf_tok, device="cpu")
    folded.enable_compatibility_mode()
    folded_logits, folded_cache = folded.run_with_cache(clean)
    result["calls of transformers' own modules in one pass"]["wrapped, in compatibility mode"] = dict(calls_in_mode)
    folded_stack, folded_labels = folded_cache.decompose_resid(pos_slice=-1, return_labels=True)
    in_mode = folded_cache.logit_attrs(folded_stack, tokens=right, incorrect_tokens=wrong, pos_slice=-1)
    one_token = folded_cache.logit_attrs(folded_stack, tokens=right, pos_slice=-1).sum(0)  # one token's logit, not a difference
mean_parts = lambda attrs: torch.stack([attrs[j].mean() for j in range(len(labels))])
ours = torch.stack([named[label].mean() for label in labels])
published = mine[:, -1].gather(1, right[:, None])[:, 0]
result["attribute"] = {
    "parts": len(labels), "parts of lens.attribute before the heads are summed": 1 + parts["heads"].shape[1] * parts["heads"].shape[2] + parts["mlp"].shape[1],
    "the logit difference": ld(mine), "minterp's parts sum to": sum(v.mean().item() for v in named.values()),
    "the library's helper on the published weights": {"sums to": raw.sum(0).mean().item(), "largest difference to minterp, per part": worst(raw, labels),
                                                       "correlation of its mean parts with minterp's": torch.corrcoef(torch.stack([mean_parts(raw), ours]))[0, 1].item()},
    "the final norm's gain": {"smallest": model.norm.weight.min().item(), "largest": model.norm.weight.max().item(), "mean": model.norm.weight.mean().item()},
    "with the stack multiplied by the gain first": {"sums to": gained.sum(0).mean().item(), "largest difference to minterp, per part": worst(gained, labels)},
    "in compatibility mode": {"sums to": in_mode.sum(0).mean().item(), "largest difference to minterp, per part": worst(in_mode, folded_labels),
                              "for one token's logit it sums to": one_token.mean().item(), "that token's published logit": published.mean().item(),
                              "the published logit less its mean over the vocabulary": (published - mine[:, -1].mean(-1)).mean().item()},
}

# What else the mode changes. The logits move by one number per prompt and position, their mean over the vocabulary.
shift = mine - folded_logits  # (prompts, positions, vocabulary)
result["mode"] = {
    "logits, largest difference to the published model's": gap(folded_logits, mine), "log-probabilities, largest difference": gap(folded_logits.log_softmax(-1), mine.log_softmax(-1)),
    "the shift within one prompt and position, largest spread over the vocabulary": (shift.max(-1).values - shift.min(-1).values).max().item(),
    "the shift over prompts and positions": {"smallest": shift.mean(-1).min().item(), "largest": shift.mean(-1).max().item()},
    "at the last position": {"smallest": shift.mean(-1)[:, -1].min().item(), "largest": shift.mean(-1)[:, -1].max().item()},
    "the shift against the mean of the published logits over the vocabulary": gap(shift.mean(-1), mine.mean(-1)),
    f"pieces after block {half} against minterp's, on the published weights": {kind: gap(tap[kind, half], folded_cache[name.format(half - 1)]) for kind, name in PIECES},
    "largest entry of the heads there": size(tap["heads", half]),
}
# What the heads record changes by the same vector at every prompt and position: the attention's value bias.
moved = tap["heads", half] - folded_cache[f"blocks.{half - 1}.attn.hook_z"]  # (prompts, positions, heads, head_dim)
result["mode"]["the heads' change"] = {"largest entry": size(moved), "largest spread over prompts and positions": (moved.amax((0, 1)) - moved.amin((0, 1))).max().item(),
                                       "largest entry of that block's value bias": size(model.layers[half - 1].attn.v_proj.bias)}
# The name a hook function is given, for a hook registered under a name of the table: as wrapped, and in the mode.
seen = {}


def note(label):
    def fn(activation, hook):
        seen[label] = hook.name
        return activation
    return fn


asked = ("hook_embed", f"blocks.{half - 1}.hook_resid_post", f"blocks.{half - 1}.attn.hook_z", f"blocks.{half - 1}.attn.hook_pattern")
with torch.no_grad():
    for label, which in (("as wrapped", bridge), ("in compatibility mode", folded)):
        which.run_with_hooks(clean[:1], fwd_hooks=[(name, note(f"{label}: {name}")) for name in asked])
result["the name a hook function sees"] = seen
result["sizes"] = {"the stream after the middle block, without the first position": size(tap["resid", half][:, 1:]), "attention weights": size(weights)}

# A patch: the flipped run's stream after the middle block, at the acting name, in the clean run.
there = f"blocks.{half - 1}.hook_resid_post"
store, _ = patch.record(model, flipped)
by_minterp = patch.run(model, clean, {("resid", half): patch.swap(store["resid", half], positions=[s])})


def put(activation, hook):  # the library passes the second argument by the name `hook`
    activation[:, s] = other[there][:, s]
    return activation


with torch.no_grad():
    by_library = bridge.run_with_hooks(clean, fwd_hooks=[(there, put)])
each = tasks.logit_diff(mine[:, -1].float(), right, wrong)
result["patch"] = {"where": f"after block {half}, at the acting name", "plain": ld(mine), "plain, per prompt": [each.min().item(), each.max().item()],
                   "by minterp": ld(by_minterp), "by the library": ld(by_library), "logits, largest difference": gap(by_minterp, by_library)}

# A vector added there: half of the difference between the two runs' streams at that place, one vector per prompt.
direction = 0.5 * (store["resid", half][:, s] - tap["resid", half][:, s])
steered = steer.steered(model, clean, half, direction, [s], 1.0)


def add(activation, hook):
    activation[:, s] = activation[:, s] + direction
    return activation


with torch.no_grad():
    steered_there = bridge.run_with_hooks(clean, fwd_hooks=[(there, add)])[:, -1]
pick = lambda last: (last.gather(1, right[:, None]) - last.gather(1, wrong[:, None])).mean().item()
result["steer"] = {"where": result["patch"]["where"], "by minterp": pick(steered), "by the library": pick(steered_there.float()), "logits, largest difference": gap(steered, steered_there)}

L, A, M = result["logits"], result["attribute"], result["mode"]
print(f"{repo} with " + ", ".join(f"{p} {v}" for p, v in result["versions"].items()) + f"; the first {COUNT} of {len(batch)} clean prompts, first names {result['prompts']['first names']}")
print(f"logits up to {L['largest entry']:.1f}: " + "; ".join(f"{k} {v:.1e}" for k, v in L.items() if k != "largest entry"))
print("calls of transformers' own attention and MLP modules in one pass:", result["calls of transformers' own modules in one pass"], "| the config's attention setting", before, "->", hf.config._attn_implementation)
print("the name a hook function sees:", seen)
print("sizes:", {k: round(v, 2) for k, v in result["sizes"].items()}, "| the heads' change in the mode:", {k: round(v, 5) for k, v in M["the heads' change"].items()})
print("pieces: largest difference, and the largest entry")
for name, value in result["pieces"].items():
    print(f"  {name:55s} {value['difference']:.1e}   {value['largest entry']:8.2f}")
print("attention weights: " + ", ".join(f"{name} {value:.1e}" for name, value in result["pattern"].items()))
print("canonical names:", result["canonical names"])
print(f"attribution over {A['parts']} parts ({A['parts of lens.attribute before the heads are summed']} before the heads are summed): the logit difference {A['the logit difference']:+.3f}; minterp's parts sum to {A['minterp' + chr(39) + 's parts sum to']:+.3f}")
for key in ("the library's helper on the published weights", "with the stack multiplied by the gain first", "in compatibility mode"):
    print(f"  {key}: sums to {A[key]['sums to']:+.3f}, per part within {A[key]['largest difference to minterp, per part']:.2e} of minterp")
print(f"  the gain runs from {A['the final norm' + chr(39) + 's gain']['smallest']:.3f} to {A['the final norm' + chr(39) + 's gain']['largest']:.2f}; on the published weights the helper's mean parts correlate {A['the library' + chr(39) + 's helper on the published weights']['correlation of its mean parts with minterp' + chr(39) + 's']:.2f} with minterp's")
print(f"  in the mode, one token's logit: the parts sum to {A['in compatibility mode']['for one token' + chr(39) + 's logit it sums to']:.3f}; the published logit is {A['in compatibility mode']['that token' + chr(39) + 's published logit']:.3f}, less its mean over the vocabulary {A['in compatibility mode']['the published logit less its mean over the vocabulary']:.3f}")
print("in compatibility mode: " + "; ".join(f"{k} {v if not isinstance(v, float) else round(v, 5)}" for k, v in M.items()))
print(f"a patch {result['patch']['where']}: the logit difference {result['patch']['plain']:+.3f} becomes {result['patch']['by minterp']:+.3f} by minterp and {result['patch']['by the library']:+.3f} by the library")
print(f"a vector added there: {result['steer']['by minterp']:+.3f} by minterp and {result['steer']['by the library']:+.3f} by the library")
result["pattern"] = rounded(torch.tensor(list(result["pattern"].values())), 8)
save(f"runs/stack/stack-{NAME}.json", result)
