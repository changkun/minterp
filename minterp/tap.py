"""
Read what flows through the model, and optionally replace it.

Every block of the model adds two things to one vector per token, the
residual stream: what its attention computed and what its MLP computed.

    x = embed(tokens)
    for each block:  x = x + attention(norm(x));  x = x + mlp(norm(x))
    logits = lm_head(norm(x))

A Tap records these pieces during a forward pass, through PyTorch's hooks,
so nothing in minllm/model.py has to change:

    ("resid", 0)   the stream after the embedding
    ("resid", i)   the stream after block i, for i = 1 .. n_layers
    ("attn", i)    what the attention of block i added
    ("mlp", i)     what the MLP of block i added
    ("heads", i)   the output of every head of block i before the block's
                   output projection mixes them, (batch, seq, heads, head_dim)

    with Tap(model) as tap:
        logits, _ = model(tokens)
    stream = tap["resid", 4]

An edit replaces a piece while the pass runs, and everything downstream
sees the replacement:

    with Tap(model, edit={("resid", 4): lambda x: torch.zeros_like(x)}):
        changed, _ = model(tokens)

Blocks are counted from 1, as in the notes. The hooks are removed when the
`with` block ends, and the model is as it was.
"""

import torch

KINDS = ("resid", "attn", "mlp", "heads")


class Tap:
    def __init__(self, model, edit=None, keep=KINDS):
        self.model, self.edit, self.keep = model, dict(edit or {}), set(keep)
        self.store, self.handles = {}, []
        for key in self.edit:
            assert key[0] in KINDS and 0 <= key[1] <= model.cfg.n_layers, key
            assert key[1] > 0 or key[0] == "resid", f"{key}: only the stream has a point 0, after the embedding"

    def __enter__(self):
        cfg = self.model.cfg
        heads, head_dim = cfg.n_heads, cfg.dim // cfg.n_heads

        def output(key):
            # A forward hook sees a module's output and may return another one.
            def hook(module, args, out):
                if key in self.edit:
                    out = self.edit[key](out)
                if key[0] in self.keep:
                    self.store[key] = out.detach()
                return out
            return hook

        def per_head(key):
            # The output projection receives all heads side by side in one
            # vector, so a hook on its input is the last place they are apart.
            def hook(module, args):
                x = args[0].unflatten(-1, (heads, head_dim))
                if key in self.edit:
                    x = self.edit[key](x)
                if key[0] in self.keep:
                    self.store[key] = x.detach()
                return (x.flatten(-2),)
            return hook

        add = self.handles.append
        add(self.model.embed.register_forward_hook(output(("resid", 0))))
        for i, block in enumerate(self.model.layers, start=1):
            add(block.register_forward_hook(output(("resid", i))))
            add(block.attn.register_forward_hook(output(("attn", i))))
            add(block.mlp.register_forward_hook(output(("mlp", i))))
            add(block.attn.o_proj.register_forward_pre_hook(per_head(("heads", i))))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __getitem__(self, key):
        return self.store[key]


def head_writes(model, tap, block):
    """
    What each head of a block added to the stream, (batch, seq, heads, dim).
    The output projection has no bias, so it applies to each head's slice
    separately, and the slices' results sum to the attention's whole output.
    """
    y = tap["heads", block]  # (batch, seq, heads, head_dim)
    W = model.layers[block - 1].attn.o_proj.weight  # (dim, heads * head_dim)
    W = W.unflatten(1, (y.shape[-2], y.shape[-1]))  # (dim, heads, head_dim)
    return torch.einsum("bthd,ohd->btho", y, W)
