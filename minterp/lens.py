"""
Two ways to read the stream in the model's own vocabulary.

The model turns its final stream into logits with one normalization and one
matrix: logits = lm_head(norm(x)). Both readings here reuse that last step.

decode() applies it to the stream after an earlier block, as if the model
stopped there (the "logit lens"). It answers: if this were the end, what
would the model say?

attribute() takes the final logit difference between two tokens apart. The
final stream is a sum,

    x = embedding + sum over blocks of (each head's write + the MLP's write),

and the logit difference is, up to the norm, a dot product with one fixed
direction. A dot product with a sum is the sum of the dot products, so
every head and every MLP gets a number, and the numbers add up to the logit
difference exactly ("direct logit attribution").

The norm is the one place where this is not a plain sum: RMSNorm divides
the stream by its own size. attribute() computes that size once, from the
whole final stream, and treats it as a constant. The split is then exact
for the forward pass that was run. What it does not say is how the logits
would change if a piece were removed: the size would change, and so would
every later block that read the piece. A piece's number is what it wrote
straight into the answer, not what it caused.
"""

import torch

from .tap import head_writes


@torch.no_grad()
def decode(model, stream):
    """Logits from a stream at any depth, as if it were the final one."""
    return model.lm_head(model.norm(stream))


@torch.no_grad()
def attribute(model, tap, right, wrong, position=-1):
    """
    Split the logit difference logit[right] - logit[wrong] at one position
    into the direct contribution of the embedding, of every head and of
    every MLP. `tap` holds a forward pass recorded with all kinds kept.

    Returns {"embed": (batch,), "heads": (batch, blocks, heads),
    "mlp": (batch, blocks)}. Their sum is the logit difference.
    """
    n = model.cfg.n_layers
    final = tap["resid", n][:, position]  # (batch, dim)
    # RMSNorm: x * rsqrt(mean(x^2) + eps) * gain. The first factor is the
    # stream's size, one number per prompt, held fixed here.
    size = torch.rsqrt(final.pow(2).mean(-1, keepdim=True) + model.norm.eps)
    W = model.lm_head.weight
    direction = (W[right] - W[wrong]) * model.norm.weight * size  # (batch, dim)

    embed = (tap["resid", 0][:, position] * direction).sum(-1)
    heads = torch.stack([
        torch.einsum("bho,bo->bh", head_writes(model, tap, i)[:, position], direction) for i in range(1, n + 1)
    ], dim=1)
    mlp = torch.stack([(tap["mlp", i][:, position] * direction).sum(-1) for i in range(1, n + 1)], dim=1)
    return {"embed": embed, "heads": heads, "mlp": mlp}
