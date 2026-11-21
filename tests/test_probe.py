import torch

from minterp import probe


def toy(n=240, d=20, k=4, places=3, noise=0.3, seed=0):
    """Rows whose label is written along k fixed directions, at every place."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, k, (n,), generator=g)
    directions = torch.randn(places, k, d, generator=g, dtype=torch.float64)
    X = directions[:, labels] + noise * torch.randn(places, n, d, generator=g, dtype=torch.float64)
    return X, labels


def test_fit_solves_the_ridge_problem():
    X, labels = toy()
    (X,) = probe.standardize(X)
    Y = probe.onehot(labels[None], 4)
    W, b = probe.fit(X, Y, 0.1)
    # At the solution the gradient of |XW + b - Y|^2 + n * lam * |W|^2 is zero.
    n = X.shape[1]
    gradient = X.transpose(-1, -2) @ (X @ W + b - Y) + n * 0.1 * W
    assert gradient.abs().max() < 1e-8
    assert torch.allclose(b, Y.mean(0))
    # A stronger penalty gives smaller weights.
    assert probe.fit(X, Y, 10.0)[0].norm() < W.norm()


def test_a_written_label_is_read_and_a_random_one_is_not():
    X, labels = toy()
    g = torch.Generator().manual_seed(1)
    random = torch.randint(0, 4, labels.shape, generator=g)
    groups = torch.arange(len(labels)) % 4
    train, test = slice(0, 180), slice(180, 240)
    out, chosen = probe.read(X[:, train], torch.stack([labels, random])[:, train], groups[train], X[:, test], 4)
    assert out.shape == (3, 60, 2, 4) and chosen.shape == (3, 2)
    guess = out.argmax(-1)
    assert (guess[:, :, 0] == labels[test]).double().mean() > 0.95
    assert (guess[:, :, 1] == random[test]).double().mean() < 0.45  # chance is 0.25


def test_two_runs_give_the_same_numbers_and_scale_does_not_matter():
    X, labels = toy()
    groups = torch.arange(len(labels)) % 4
    args = (labels[None, :180], groups[:180])
    one, _ = probe.read(X[:, :180], *args, X[:, 180:], 4)
    two, _ = probe.read(X[:, :180], *args, X[:, 180:], 4)
    assert torch.equal(one, two)
    # Standardizing by the training rows makes the reader blind to the stream's size.
    big, _ = probe.read(60 * X[:, :180], *args, 60 * X[:, 180:], 4)
    assert torch.allclose(one, big, atol=1e-9)


def test_standardize_uses_the_training_rows_only():
    X, _ = toy()
    train, test = probe.standardize(X[:, :100], X[:, 100:])
    assert train.mean(1).abs().max() < 1e-12 and torch.allclose(train.std(1), torch.ones(3, 20, dtype=torch.float64))
    shifted = probe.standardize(X[:, :100], X[:, 100:] + 5.0)[1]
    assert torch.allclose(shifted - test, 5.0 / X[:, :100].std(1, keepdim=True).expand_as(test))


def test_onehot_puts_the_targets_side_by_side():
    y = probe.onehot(torch.tensor([[0, 2], [1, 1]]), 3)
    assert y.tolist() == [[1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0]]


def test_weights_are_the_reader_that_read_applies():
    X, labels = toy()
    groups = torch.arange(len(labels)) % 4
    out, _ = probe.read(X[:1, :180], labels[None, :180], groups[:180], X[:1, 180:], 4)
    W, b, mean, spread = probe.weights(X[0, :180], labels[:180], groups[:180], 4)
    again = ((X[0, 180:] - mean) / spread) @ W + b
    assert torch.allclose(again, out[0, :, 0], atol=1e-9)
    # The scores of two parts of an input add up to the score of the input.
    part = 0.3 * X[0, 180:]
    rest = X[0, 180:] - part
    assert torch.allclose(((part - mean) / spread) @ W + b + (rest / spread) @ W, again, atol=1e-9)
