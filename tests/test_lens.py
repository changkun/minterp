import torch

from minterp import lens, tasks
from minterp.tap import Tap


def test_decoding_the_final_stream_gives_the_logits(model, tok, people):
    batch = tasks.batches(tok, people=people)[0]
    with Tap(model) as tap:
        logits, _ = model(batch.clean)
    assert torch.allclose(lens.decode(model, tap["resid", model.cfg.n_layers]), logits, atol=1e-5)


def test_the_attribution_adds_up_to_the_logit_difference(model, tok, people):
    for batch in tasks.batches(tok, people=people):
        with Tap(model) as tap:
            logits, _ = model(batch.clean)
        parts = lens.attribute(model, tap, batch.right, batch.wrong)
        n, cfg = len(batch), model.cfg
        assert parts["embed"].shape == (n,)
        assert parts["heads"].shape == (n, cfg.n_layers, cfg.n_heads) and parts["mlp"].shape == (n, cfg.n_layers)
        total = parts["embed"] + parts["heads"].sum((1, 2)) + parts["mlp"].sum(1)
        assert torch.allclose(total, tasks.logit_diff(logits, batch.right, batch.wrong), atol=1e-4)


def test_the_attribution_of_a_position_other_than_the_last(model, tok, people):
    batch = tasks.batches(tok, people=people)[0]
    with Tap(model) as tap:
        logits, _ = model(batch.clean)
    parts = lens.attribute(model, tap, batch.right, batch.wrong, position=4)
    total = parts["embed"] + parts["heads"].sum((1, 2)) + parts["mlp"].sum(1)
    assert torch.allclose(total, tasks.logit_diff(logits[:, 4], batch.right, batch.wrong), atol=1e-4)
