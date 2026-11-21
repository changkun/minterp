import pytest
import torch

from minterp.tap import Tap, head_writes


@pytest.fixture
def tokens(model):
    torch.manual_seed(1)
    return torch.randint(0, model.cfg.vocab_size, (3, 12))


def n_hooks(model):
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def test_reading_changes_nothing_and_leaves_no_hooks(model, tokens):
    plain, _ = model(tokens)
    with Tap(model) as tap:
        assert n_hooks(model) > 0
        seen, _ = model(tokens)
    assert torch.equal(plain, seen)
    assert n_hooks(model) == 0
    assert tap["resid", 0].shape == (3, 12, model.cfg.dim)
    assert tap["heads", 1].shape == (3, 12, model.cfg.n_heads, model.cfg.dim // model.cfg.n_heads)


def test_the_stream_is_the_sum_of_what_was_written(model, tokens):
    with Tap(model) as tap:
        logits, _ = model(tokens)
    for i in range(1, model.cfg.n_layers + 1):
        rebuilt = tap["resid", i - 1] + tap["attn", i] + tap["mlp", i]
        assert torch.allclose(rebuilt, tap["resid", i], atol=1e-5)
    final = tap["resid", model.cfg.n_layers]
    assert torch.allclose(model.lm_head(model.norm(final)), logits, atol=1e-5)


def test_the_heads_sum_to_the_attention(model, tokens):
    with Tap(model) as tap:
        model(tokens)
    for i in range(1, model.cfg.n_layers + 1):
        writes = head_writes(model, tap, i)
        assert writes.shape == (3, 12, model.cfg.n_heads, model.cfg.dim)
        assert torch.allclose(writes.sum(2), tap["attn", i], atol=1e-5)


def test_an_edit_reaches_everything_after_it_and_nothing_before(model, tokens):
    plain, _ = model(tokens)

    def at_position_5(x):
        x = x.clone()
        x[:, 5] = 0
        return x

    with Tap(model, edit={("resid", 1): at_position_5}) as tap:
        changed, _ = model(tokens)
    assert torch.equal(tap["resid", 1][:, 5], torch.zeros(3, model.cfg.dim))  # the store holds the edited stream
    assert torch.allclose(changed[:, :5], plain[:, :5], atol=1e-6)  # attention only looks back
    assert not torch.allclose(changed[:, 5:], plain[:, 5:], atol=1e-3)
    with Tap(model, edit={("resid", 1): lambda x: x}):
        same, _ = model(tokens)
    assert torch.equal(same, plain)


def test_editing_heads_changes_what_the_attention_writes(model, tokens):
    with Tap(model, edit={("heads", 2): torch.zeros_like}) as tap:
        model(tokens)
    assert torch.equal(tap["attn", 2], torch.zeros_like(tap["attn", 2]))

    def without_head_0(y):
        y = y.clone()
        y[:, :, 0] = 0
        return y

    with Tap(model) as plain:
        model(tokens)
    with Tap(model, edit={("heads", 2): without_head_0}) as cut:
        model(tokens)
    removed = plain["attn", 2] - cut["attn", 2]
    assert torch.allclose(removed, head_writes(model, plain, 2)[:, :, 0], atol=1e-5)


def test_keep_limits_what_is_stored(model, tokens):
    with Tap(model, keep=("resid",)) as tap:
        model(tokens)
    assert {k for k, _ in tap.store} == {"resid"}
    with pytest.raises(AssertionError):
        Tap(model, edit={("attn", 0): lambda x: x})
