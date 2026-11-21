"""
Patching: run a prompt with a piece of another prompt's run in its place.

Reading (step 1) and probing (step 2) look at a run and leave it alone. A
patch changes it. Record the activations of a source prompt, run a target
prompt, and at one place put the source's activation where the target's own
would be. Everything downstream computes on. If the answer moves towards
the source's answer, the model was carrying, at that place, something the
answer depends on.

    source, _ = record(model, batch.clean)
    at = [batch.pos["S"]]  # the acting name
    logits = run(model, batch.flipped, {("resid", 6): swap(source["resid", 6], positions=at)})

The two prompts must line up token by token, which is why the versions of
minterp/tasks.py differ in a single token.

Three things can stand in for an activation, and they ask different
questions (docs/03-patch.md):

    another prompt's   swap(source[key])        what if this place had seen the other prompt?
    the average        swap(mean(stores, key))  what if this place carried nothing about this prompt?
    nothing            swap(zero(source[key]))  what if this place wrote nothing at all?
"""

import torch

from .tap import KINDS, Tap


@torch.no_grad()
def record(model, tokens, keep=KINDS):
    """One forward pass, with everything a patch may need. Returns the store of the Tap and the logits."""
    with Tap(model, keep=keep) as tap:
        logits, _ = model(tokens)
    return tap.store, logits


@torch.no_grad()
def run(model, tokens, edits):
    """The logits of a forward pass with some activations replaced. `edits` maps a Tap key to a function."""
    with Tap(model, edit=edits, keep=()):
        logits, _ = model(tokens)
    return logits


def swap(source, positions=None, head=None):
    """
    An edit that puts `source` in place of the activation: everywhere, at
    some positions only, or for one head only (for a ("heads", block) key,
    whose activation is (batch, seq, heads, head_dim)).
    """
    def edit(x):
        x = x.clone()
        at = slice(None) if positions is None else positions
        if head is None:
            x[:, at] = source[:, at]
        else:
            x[:, at, head] = source[:, at, head]
        return x
    return edit


def mean(stores, key):
    """The average of an activation over the prompts of several runs, position by position."""
    return torch.cat([s[key] for s in stores]).mean(0, keepdim=True)


def zero(activation):
    return torch.zeros_like(activation)


def share(patched, target, source):
    """
    How far a patch moved a measure from the target run's value towards the
    source run's: 0 is no movement, 1 is all the way. The three arguments
    are the measure per prompt; the share is taken on their means, because a
    ratio per prompt blows up wherever the two runs happen to agree.
    """
    return ((patched.mean() - target.mean()) / (source.mean() - target.mean())).item()
