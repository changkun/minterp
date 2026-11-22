import torch

from minterp import match
from minterp import sae as S


def pair(width=50, dim=16, noise=0.0, seed=0):
    """A dictionary, and the same one with its latents in another order and some noise on every direction."""
    torch.manual_seed(seed)
    one, two = S.SAE(dim, width, 4), S.SAE(dim, width, 4)
    order = torch.randperm(width)
    with torch.no_grad():
        two.dec.copy_(match.unit(one.dec[order] + noise * torch.randn(width, dim)))
        two.enc.copy_(one.enc[:, order] + noise * torch.randn(dim, width))
    return one, two, order


def test_a_shuffled_copy_is_found_again():
    one, two, order = pair()
    values, where = match.best(one, two)
    assert torch.allclose(values, torch.ones(50), atol=1e-5) and torch.equal(order[where], torch.arange(50))
    assert torch.equal(order[match.one_to_one(match.cosines(one, two))], torch.arange(50))
    assert match.shared(one, two) == (1.0, 1.0)
    assert match.shares(values) == {"at a cosine of 0.9 or more": 1.0, "at 0.7 or more": 1.0, "at 0.5 or more": 1.0, "median cosine": values.median().item()}


def test_two_unrelated_dictionaries_share_nothing():
    torch.manual_seed(0)
    one, two = S.SAE(64, 200, 4), S.SAE(64, 200, 4)
    assert match.shared(one, two) == (0.0, 0.0)
    assert (match.one_to_one(match.cosines(one, two)) == -1).all()
    assert match.best(one, two)[0].max() < 0.7
    # The yardsticks agree with that: random directions in 64 dimensions do not come close to each other.
    assert match.by_chance(200, 200, 64).median() < 0.5 and match.inside(one).median() < 0.5


def test_one_to_one_uses_no_latent_twice_and_the_floor_changes_nobody_above_it():
    one, two, _ = pair(noise=0.35, seed=1)
    c = match.cosines(one, two)
    high, low = match.one_to_one(c, 0.7), match.one_to_one(c, 0.3)
    used = high[high >= 0]
    assert len(used) == len(set(used.tolist()))
    assert torch.equal(high[high >= 0], low[high >= 0])  # whoever is matched at 0.7 keeps that partner with a lower floor
    # A latent whose best counterpart is taken by a closer one stays unmatched, which the best single counterpart does not show.
    assert (high >= 0).sum() <= (match.best(one, two)[0] >= 0.7).sum()


def test_the_strict_measure_needs_the_encoder_to_agree():
    one, two, order = pair()
    with torch.no_grad():
        two.enc.copy_(two.enc[:, torch.randperm(50)])  # the decoder still matches, the encoder points elsewhere
    both, decoder_only = match.shared(one, two)
    assert decoder_only == 1.0 and both < 0.2


def test_the_nearest_other_latent_leaves_the_latent_itself_out():
    one, _, _ = pair()
    assert match.inside(one).max() < 0.999
