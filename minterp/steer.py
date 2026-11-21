"""
Steering: write to the stream, and see whether the model does what the description said it would.

The steps so far read the model. Each ended with a description: this
position holds who acts, this latent stands for a kind of name. A
description that is right about what the model goes by makes a prediction:
change that thing, and the behavior changes in a known way. Steering tests
the prediction by adding a vector to the stream at one place while the
model runs,

    x  ->  x + dose * direction        at one depth, at chosen positions,

and looking at what the model says next. It is the patch of step 3 with a
vector one has built in place of a vector taken from another prompt.

Three things are worth knowing about any steering result, and the
functions here return them together: how the effect grows with the dose,
what a random direction of the same length does, and what else changed.
"""

import torch

from .tap import Tap


def add(direction, positions, dose=1.0):
    """An edit that adds dose * direction to the stream at some positions. direction is (dim,) or (prompts, dim)."""
    def edit(x):
        x = x.clone()
        x[:, positions] += (dose * direction).to(x.dtype).unsqueeze(-2) if direction.dim() == 2 else (dose * direction).to(x.dtype)
        return x
    return edit


@torch.no_grad()
def steered(model, tokens, block, direction, positions, dose):
    """The logits at the last position with the direction added after `block`."""
    with Tap(model, edit={("resid", block): add(direction, positions, dose)}, keep=()):
        logits, _ = model(tokens)
    return logits[:, -1].float()


def writes(sae, j):
    """What one unit of latent j adds to the stream: decode(z + e_j) - decode(z), whatever z is."""
    return sae.dec[j].detach() * sae.scale


def like(direction, generator):
    """A random direction of the same length as each given one: the control every steering result needs."""
    noise = torch.randn(direction.shape, generator=generator).to(direction.device, direction.dtype)
    return noise / noise.norm(dim=-1, keepdim=True) * direction.norm(dim=-1, keepdim=True)


def elsewhere(plain, changed, leave_out):
    """
    What else changed: the divergence between the two next-token
    distributions after the tokens in `leave_out` (prompts, some) are taken
    out of both, in nats per prompt. Zero means that apart from those tokens
    the model says what it said.
    """
    keep = torch.ones_like(plain, dtype=torch.bool).scatter_(1, leave_out, False)
    p = plain.masked_fill(~keep, float("-inf")).log_softmax(-1)
    q = changed.masked_fill(~keep, float("-inf")).log_softmax(-1)
    return (p.exp() * (p - q).masked_fill(~keep, 0)).sum(-1)
