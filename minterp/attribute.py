"""
Which latents does a number depend on? Gradients first, then a check.

Step 3 answered "which part does the answer depend on" by moving a part
from one run into another and running the model again: one run per part.
A dictionary has thousands of latents at every position, and one run per
latent is too many. The gradient gives all of them at once, approximately.

Put the dictionary into the forward pass without changing it. At one depth
the stream is written as what the dictionary rebuilds plus what it misses,

    x = decode(z) + err,

with the latents z and the error err as the things one can change. For a
number m computed from the logits (a logit difference, say), the gradient
of m with respect to z says how m would move if a latent moved a little.
Moving a latent from its value in one run to its value in another is not
little, but to first order

    m(with latent j from the other run) - m  ≈  (z_other[j] - z[j]) * dm/dz[j],

for every latent of every position from one backward pass. This is
attribution patching (Syed et al., 2023), applied to latents as in Marks et
al. (2024). It is an estimate. The model is not linear, and the dictionary
least of all: a latent that is off stays off for small changes of the
stream. So the estimate is used to choose what to look at, and what it
chooses is then patched for real.
"""

import torch

from .tap import Tap


def through(model, sae, tokens, block, measure, keep=()):
    """
    One forward and one backward pass with the stream after `block` written
    as decode(z) + err. `measure` maps (logits, tap) to one number per
    prompt. Returns z (prompts, seq, width), err (prompts, seq, dim), the
    gradients of the summed measure with respect to both, and the measure.
    The logits are those of the plain model, since decode(z) + err is x.
    """
    cache = {}

    def edit(x):
        z = sae.encode(x.float()).detach().requires_grad_(True)
        rebuilt = sae.decode(z)
        err = (x.float() - rebuilt).detach().requires_grad_(True)
        cache.update(z=z, err=err)
        return (rebuilt + err).to(x.dtype)

    with torch.enable_grad(), Tap(model, edit={("resid", block): edit}, keep=keep, detach=False) as tap:
        logits, _ = model(tokens)
        value = measure(logits, tap)
        value.sum().backward()
    return cache["z"].detach(), cache["err"].detach(), cache["z"].grad, cache["err"].grad, value.detach()


def rebuilt_with(sae, z, err):
    """An edit that sets the stream to decode(z) + err: for patching latents, or the error, for real."""
    def edit(x):
        return (sae.decode(z) + err).to(x.dtype)
    return edit
