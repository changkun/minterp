import torch

from minterp import sae as S
from minterp import steer, tasks
from minterp.tap import Tap


def test_a_zero_dose_changes_nothing_and_a_dose_adds_exactly_that(model, tok, people):
    batch = tasks.batches(tok, people=people)[0]
    torch.manual_seed(0)
    direction = torch.randn(model.cfg.dim)
    plain, _ = model(batch.clean)
    assert torch.allclose(steer.steered(model, batch.clean, 2, direction, [5], 0.0), plain[:, -1], atol=1e-6)
    with Tap(model, keep=("resid",)) as before:
        model(batch.clean)
    with Tap(model, edit={("resid", 2): steer.add(direction, [5], 0.5)}, keep=("resid",)) as after:
        model(batch.clean)
    moved = after["resid", 2] - before["resid", 2]
    assert torch.allclose(moved[:, 5], 0.5 * direction.expand(len(batch), -1), atol=1e-6)
    assert moved[:, :5].abs().max() == 0 and moved[:, 6:].abs().max() == 0


def test_a_direction_per_prompt(model, tok, people):
    batch = tasks.batches(tok, people=people)[0]
    torch.manual_seed(0)
    directions = torch.randn(len(batch), model.cfg.dim)
    with Tap(model, keep=("resid",)) as before:
        model(batch.clean)
    with Tap(model, edit={("resid", 1): steer.add(directions, [3], 2.0)}, keep=("resid",)) as after:
        model(batch.clean)
    assert torch.allclose((after["resid", 1] - before["resid", 1])[:, 3], 2.0 * directions, atol=1e-5)


def test_the_control_has_the_same_length_and_another_direction():
    g = torch.Generator().manual_seed(0)
    d = torch.randn(7, 48)
    r = steer.like(d, g)
    assert torch.allclose(r.norm(dim=1), d.norm(dim=1), atol=1e-5)
    assert torch.nn.functional.cosine_similarity(d, r, dim=1).abs().max() < 0.6


def test_elsewhere_ignores_the_tokens_left_out():
    plain = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    changed = plain.clone()
    changed[0, 0] += 5.0  # only a token that is left out moves
    assert steer.elsewhere(plain, changed, torch.tensor([[0]])).abs().max() < 1e-6
    assert steer.elsewhere(plain, changed, torch.tensor([[3]])).item() > 0.5
    assert steer.elsewhere(plain, plain, torch.tensor([[0]])).abs().max() < 1e-6


def test_a_latent_writes_its_decoder_direction_at_the_streams_scale():
    torch.manual_seed(0)
    sae = S.SAE(dim=16, width=40, k=4)
    sae.scale.fill_(7.0)
    z = sae.encode(torch.randn(5, 16) * 7.0)
    more = z.clone()
    more[:, 11] += 2.5
    assert torch.allclose(sae.decode(more) - sae.decode(z), 2.5 * steer.writes(sae, 11).expand(5, -1), atol=1e-5)
