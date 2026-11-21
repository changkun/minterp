import torch

from minterp import attention
from minterp.tap import Tap


def test_the_weights_rebuild_what_the_attention_wrote(model):
    torch.manual_seed(2)
    tokens = torch.randint(0, model.cfg.vocab_size, (3, 14))
    with Tap(model) as tap:
        model(tokens)
    w = attention.patterns(model, tap)  # (batch, blocks, heads, seq, seq)
    assert w.shape == (3, model.cfg.n_layers, model.cfg.n_heads, 14, 14)
    assert torch.allclose(w.sum(-1), torch.ones(3, model.cfg.n_layers, model.cfg.n_heads, 14), atol=1e-5)
    assert (w.triu(1) == 0).all()  # no position reads from a later one
    for i, block in enumerate(model.layers, start=1):
        x = block.attn_norm(tap["resid", i - 1])
        v = attention.values(model, block.attn, x)  # (batch, heads, seq, head_dim)
        heads = (w[:, i - 1] @ v).transpose(1, 2)  # (batch, seq, heads, head_dim): each head's weighted sum
        assert torch.allclose(heads, tap["heads", i], atol=1e-5)
        assert torch.allclose(block.attn.o_proj(heads.flatten(-2)), tap["attn", i], atol=1e-5)
