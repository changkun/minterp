import torch

from minterp import attribute, tasks
from minterp import sae as S
from minterp.tap import Tap


def setup(model, tok, people):
    torch.manual_seed(0)
    batch = tasks.batches(tok, people=people)[0]
    sae = S.SAE(model.cfg.dim, 4 * model.cfg.dim, k=8).eval().requires_grad_(False)
    measure = lambda logits, tap: tasks.logit_diff(logits, batch.right, batch.wrong)
    return batch, sae, measure


def test_the_dictionary_in_the_pass_changes_nothing(model, tok, people):
    batch, sae, measure = setup(model, tok, people)
    plain, _ = model(batch.clean)
    z, err, gz, gerr, value = attribute.through(model, sae, batch.clean, 2, measure)
    assert torch.allclose(value, tasks.logit_diff(plain, batch.right, batch.wrong), atol=1e-4)
    assert z.shape == (len(batch), batch.clean.shape[1], sae.width) and gz.shape == z.shape and gerr.shape == err.shape
    with Tap(model, edit={("resid", 2): attribute.rebuilt_with(sae, z, err)}, keep=()):
        again, _ = model(batch.clean)
    assert torch.allclose(again, plain, atol=1e-4)
    assert not any(p.requires_grad for p in model.parameters())  # the model is still frozen


def test_the_gradient_predicts_a_small_change_and_not_exactly_a_large_one(model, tok, people):
    batch, sae, measure = setup(model, tok, people)
    z, err, gz, gerr, value = attribute.through(model, sae, batch.clean, 2, measure)
    torch.manual_seed(1)
    with Tap(model, keep=("resid",)) as tap:
        model(batch.clean)
    step = torch.randn_like(err) * tap["resid", 2].std()  # a direction of the stream's own size at that depth

    def moved(size):
        with torch.no_grad(), Tap(model, edit={("resid", 2): attribute.rebuilt_with(sae, z, err + size * step)}, keep=()):
            logits, _ = model(batch.clean)
        return tasks.logit_diff(logits, batch.right, batch.wrong) - value

    predicted = (step * gerr).sum((1, 2))
    small, large = moved(1e-3) / 1e-3, moved(1.0)
    off_small, off_large = (small - predicted).abs().mean(), (large - predicted).abs().mean()
    assert off_small < 0.1 * predicted.abs().mean()  # first order: right for a small step
    assert off_large > 5 * off_small  # and off for a large one


def test_a_position_cannot_matter_to_an_earlier_one(model, tok, people):
    batch, sae, _ = setup(model, tok, people)
    early = lambda logits, tap: logits[:, 3, 0]  # a number read at position 3
    _, _, gz, gerr, _ = attribute.through(model, sae, batch.clean, 2, early)
    assert gerr[:, 4:].abs().max() == 0 and gz[:, 4:].abs().max() == 0
    assert gerr[:, :4].abs().max() > 0
