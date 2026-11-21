import torch

from minterp import sae as S


def planted(rows=20000, dim=24, directions=60, active=3, seed=0):
    """Vectors that really are sums of a few unit directions out of many more than there are dimensions."""
    g = torch.Generator().manual_seed(seed)
    D = torch.randn(directions, dim, generator=g)
    D = D / D.norm(dim=1, keepdim=True)
    z = torch.zeros(rows, directions)
    for r in range(active):
        idx = torch.randint(0, directions, (rows,), generator=g)
        z[torch.arange(rows), idx] = 1 + torch.rand(rows, generator=g)
    return z @ D + 5.0, D  # an offset, so that the bias has something to do


def test_at_most_k_latents_are_active_and_none_is_negative():
    torch.manual_seed(0)
    sae = S.SAE(dim=16, width=64, k=5)
    z = sae.encode(torch.randn(10, 16))
    assert z.shape == (10, 64) and (z >= 0).all() and ((z > 0).sum(1) <= 5).all()
    rebuilt, again = sae(torch.randn(10, 16))
    assert rebuilt.shape == (10, 16) and again.shape == (10, 64)


def test_it_finds_directions_that_were_planted():
    x, D = planted()
    sae = S.train(x[:18000], width=120, k=3, steps=1500, batch=1024, lr=2e-3)
    assert S.unexplained(sae, x[18000:]) < 0.1  # on vectors it did not train on
    assert torch.allclose(sae.dec.norm(dim=1), torch.ones(120), atol=1e-5)
    # Most planted directions have a dictionary direction close to them, though the dimension is 24 and they are 60.
    closest = (D @ sae.dec.T).max(1).values
    assert (closest > 0.9).float().mean() > 0.8


def test_the_scale_makes_it_blind_to_the_size_of_the_vectors():
    x, _ = planted(rows=6000)
    small = S.train(x, width=120, k=3, steps=300, batch=1024, lr=2e-3)
    large = S.train(60 * x, width=120, k=3, steps=300, batch=1024, lr=2e-3)
    assert abs(large.scale / small.scale - 60) < 1e-3
    assert abs(S.unexplained(small, x) - S.unexplained(large, 60 * x)) < 1e-3
    assert torch.allclose(small.encode(x[:50]), large.encode(60 * x[:50]), atol=1e-3)


def test_save_and_load(tmp_path):
    x, _ = planted(rows=3000)
    sae = S.train(x, width=120, k=3, steps=50, batch=512)
    S.save(sae, tmp_path / "sae.pt")
    again = S.load(tmp_path / "sae.pt")
    assert again.k == 3 and again.width == 120
    assert torch.equal(sae(x[:20])[0], again(x[:20])[0])


def test_two_runs_with_one_seed_agree():
    x, _ = planted(rows=3000)
    a = S.train(x, width=120, k=3, steps=40, batch=512, seed=3)
    b = S.train(x, width=120, k=3, steps=40, batch=512, seed=3)
    assert torch.equal(a.dec, b.dec)
