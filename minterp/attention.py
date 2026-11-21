"""
What a head looks at: the attention weights, which the model never keeps.

A head moves information between positions in two steps. From the stream
at every position it computes a query and a key. The query of the position
being written is compared with the keys of all positions up to it, and the
comparison, through a softmax, becomes weights that sum to one. The head
then writes a weighted sum of the positions' values. The weights are its
attention pattern: from each position, how much it reads from each earlier
one.

minllm computes attention with a library call that returns the result and
not the weights. This module computes the weights again, from the same
inputs and the same projections, and a test checks that the weighted sum of
values built from them is exactly what the model's attention wrote.
"""

import torch
from minllm.model import apply_rope


@torch.no_grad()
def patterns(model, tap):
    """
    The attention weights of every head, from a recorded pass:
    (batch, blocks, heads, seq, seq), where [b, i, h, t, s] is how much head
    h of block i + 1, writing at position t, reads from position s.
    """
    cfg = model.cfg
    out = []
    for i, block in enumerate(model.layers, start=1):
        x = block.attn_norm(tap["resid", i - 1])  # what the attention of block i receives
        out.append(weights(model, block.attn, x))
    return torch.stack(out, dim=1)


def weights(model, attn, x):
    """The weights of one block's heads for its normalized input x: (batch, heads, seq, seq)."""
    B, T, _ = x.shape
    H, KV, hd = attn.n_heads, attn.n_kv_heads, attn.head_dim
    q = attn.q_proj(x).view(B, T, H, hd).transpose(1, 2)
    k = attn.k_proj(x).view(B, T, KV, hd).transpose(1, 2)
    # Positions enter here: queries and keys are rotated by an angle that grows with the position.
    cos, sin = model.rope_cos[:T], model.rope_sin[:T]
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    k = k.repeat_interleave(H // KV, dim=1)  # each key head serves a group of query heads
    scores = q @ k.transpose(-1, -2) / hd ** 0.5
    future = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
    return scores.masked_fill(future, float("-inf")).softmax(-1)


@torch.no_grad()
def values(model, attn, x):
    """What each position offers to be read, per head: (batch, heads, seq, head_dim)."""
    B, T, _ = x.shape
    v = attn.v_proj(x).view(B, T, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
    return v.repeat_interleave(attn.n_heads // attn.n_kv_heads, dim=1)
