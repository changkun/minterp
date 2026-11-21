"""
A probe: a small reader trained to tell a property from the stream.

The lens of step 1 reads the stream with the model's own output matrix. A
probe brings its own: for one place in the model (a depth and a position)
it fits a linear map from the stream there to a label, on some prompts, and
is judged on prompts it has not seen. If it gets them right, the property
can be read out linearly at that place.

The reader here is the simplest one that is exactly reproducible: a linear
map fitted by least squares to one-hot labels, with a ridge penalty. It has
a closed form, so there is no optimizer, no learning rate and no random
start, and two runs on the same data give the same weights.

    W = (X'X + n * lam * I)^-1 X'Y        scores = X W + mean(Y)

Three precautions, each because the alternative gives numbers that mean
nothing:

- The stream is standardized with the mean and spread of the training rows
  only. Its size grows about seventy-fold with depth (step 1), and one ridge
  penalty would otherwise mean something different at every depth.
- The penalty is picked by a rule that never sees the test rows: leave one
  group of training rows out in turn, and keep the strength that reads the
  left-out groups best.
- Everything is float64 on the CPU. The systems are small.

Every function takes a leading "places" axis, so that all depths and
positions are solved in one call.
"""

import torch

# From hardly any penalty to one that leaves hardly any weights. The first
# grid stopped at 10, and the rule below picked 10 nearly everywhere at the
# last position: a choice at the edge of a grid says the grid is too short.
LAMS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


def standardize(train, *others):
    """Center and scale every dimension by the training rows. Shapes (places, rows, dim)."""
    mean = train.mean(-2, keepdim=True)
    spread = train.std(-2, keepdim=True).clamp_min(1e-8)
    return tuple((x - mean) / spread for x in (train, *others))


def onehot(labels, k):
    """(targets, rows) integer labels -> (rows, targets * k), the targets side by side."""
    t, n = labels.shape
    y = torch.zeros(n, t, k, dtype=torch.float64)
    y.scatter_(2, labels.T[:, :, None], 1.0)
    return y.reshape(n, t * k)


def fit(X, Y, lam):
    """
    Ridge regression in closed form. X is (places, rows, dim), standardized;
    Y is (rows, outputs). Returns the weights (places, dim, outputs) and the
    intercept (outputs,), which is the mean of Y because X has mean zero.
    """
    n, d = X.shape[-2:]
    b = Y.mean(0)
    gram = X.transpose(-1, -2) @ X + n * lam * torch.eye(d, dtype=X.dtype)
    W = torch.linalg.solve(gram, X.transpose(-1, -2) @ (Y - b))
    return W, b


def scores(X, W, b, k):
    """(places, rows, dim) -> (places, rows, targets, k)."""
    out = X @ W + b
    return out.unflatten(-1, (-1, k))


def read(train, labels, groups, test, k, lams=LAMS):
    """
    Fit readers on the training rows and apply them to the test rows.

    train, test   (places, rows, dim), not yet standardized
    labels        (targets, training rows), integers below k
    groups        (training rows,), which group a row belongs to; the penalty
                  is chosen by leaving each group out in turn

    Returns the test scores (places, test rows, targets, k) under the chosen
    penalty, and the index of that penalty in `lams`, (places, targets).
    """
    train, test = train.double(), test.double()
    targets = labels.shape[0]
    inner = torch.zeros(len(lams), train.shape[0], targets, dtype=torch.float64)
    for g in groups.unique():
        held = groups == g
        a, v = standardize(train[:, ~held], train[:, held])
        y = onehot(labels[:, ~held], k)
        for i, lam in enumerate(lams):
            guess = scores(v, *fit(a, y, lam), k).argmax(-1)  # (places, held rows, targets)
            inner[i] += (guess == labels[:, held].T).double().mean(1)
    # The strongest penalty among the best: flip, so that argmax meets it first.
    chosen = len(lams) - 1 - inner.flip(0).argmax(0)  # (places, targets)

    a, t = standardize(train, test)
    y = onehot(labels, k)
    by_lam = torch.stack([scores(t, *fit(a, y, lam), k) for lam in lams])  # (lams, places, rows, targets, k)
    pick = chosen[None, :, None, :, None].expand(1, -1, by_lam.shape[2], -1, k)
    return by_lam.gather(0, pick)[0], chosen


def weights(train, labels, groups, k, lams=LAMS):
    """
    The reader itself, for one place and one property: its weights (dim, k),
    intercept (k,), and the mean and spread it standardizes with, at the
    penalty that read() would choose. With them a reader fitted on one input
    can be applied to parts of that input: its scores are linear, so the
    scores of two parts add up to the score of their sum.
    """
    train = train.double()[None]
    labels = labels[None]
    inner = torch.zeros(len(lams), dtype=torch.float64)
    for g in groups.unique():
        held = groups == g
        a, v = standardize(train[:, ~held], train[:, held])
        y = onehot(labels[:, ~held], k)
        for i, lam in enumerate(lams):
            guess = scores(v, *fit(a, y, lam), k).argmax(-1)[0, :, 0]
            inner[i] += (guess == labels[0, held]).double().mean()
    lam = lams[len(lams) - 1 - inner.flip(0).argmax().item()]
    mean, spread = train.mean(-2, keepdim=True), train.std(-2, keepdim=True).clamp_min(1e-8)
    W, b = fit((train - mean) / spread, onehot(labels, k), lam)
    return W[0], b, mean[0], spread[0]
