"""
Steering: write to the stream, and see whether the model does what the description said it would.

The steps so far read the model. Each ended with a description: this
position holds who acts, this latent stands for a kind of name. A
description that is right about what the model goes by makes a prediction:
change that thing, and the behavior changes in a known way. Steering tests
the prediction by adding a vector to the stream at one place while the
model runs,

    x  ->  x + dose * direction        at one depth, at chosen positions,

and looking at what the model says next. It is the patch of step 3 with a
vector one has built in place of a vector taken from another prompt.

Three things are worth knowing about any steering result, and the
functions here return them together: how the effect grows with the dose,
what a random direction of the same length does, and what else changed.
"""

import torch

from .tap import Tap


def add(direction, positions, dose=1.0):
    """An edit that adds dose * direction to the stream at some positions. direction is (dim,) or (prompts, dim)."""
    def edit(x):
        x = x.clone()
        x[:, positions] += (dose * direction).to(x.dtype).unsqueeze(-2) if direction.dim() == 2 else (dose * direction).to(x.dtype)
        return x
    return edit


@torch.no_grad()
def steered(model, tokens, block, direction, positions, dose):
    """The logits at the last position with the direction added after `block`."""
    with Tap(model, edit={("resid", block): add(direction, positions, dose)}, keep=()):
        logits, _ = model(tokens)
    return logits[:, -1].float()


def writes(sae, j):
    """What one unit of latent j adds to the stream: decode(z + e_j) - decode(z), whatever z is."""
    return sae.dec[j].detach() * sae.scale


def like(direction, generator):
    """A random direction of the same length as each given one: the control every steering result needs."""
    noise = torch.randn(direction.shape, generator=generator).to(direction.device, direction.dtype)
    return noise / noise.norm(dim=-1, keepdim=True) * direction.norm(dim=-1, keepdim=True)


def elsewhere(plain, changed, leave_out):
    """
    What else changed: the divergence between the two next-token
    distributions after the tokens in `leave_out` (prompts, some) are taken
    out of both, in nats per prompt. Zero means that apart from those tokens
    the model says what it said.
    """
    keep = torch.ones_like(plain, dtype=torch.bool).scatter_(1, leave_out, False)
    p = plain.masked_fill(~keep, float("-inf")).log_softmax(-1)
    q = changed.masked_fill(~keep, float("-inf")).log_softmax(-1)
    return (p.exp() * (p - q).masked_fill(~keep, 0)).sum(-1)


def against(names, target, pool):
    """A name's logit minus the mean over a pool of names, per prompt: names is (prompts, names), pool a mask of that shape."""
    rows = torch.arange(names.shape[0], device=names.device)
    return names[rows, target] - (names * pool).sum(1) / pool.sum(1)


def stand_ins(A, B, C, she):
    """
    For every prompt two names that are in none of its versions: D of B's
    kind and E of A's kind, taken in turn. A direction built for D and E in
    the way a direction for B and A is built is a control: it is about
    other people, of the same kinds.
    """
    k, D, E = len(she), [], []
    for i, (a, b, c) in enumerate(zip(A.tolist(), B.tolist(), C.tolist())):
        pool = [j for j in range(k) if she[j] == she[b] and j not in (a, b, c)]
        D.append(pool[i % len(pool)])
        pool = [j for j in range(k) if she[j] == she[a] and j not in (a, b, c, D[-1])]
        E.append(pool[i % len(pool)])
    return torch.tensor(D, device=A.device), torch.tensor(E, device=A.device)


def term_per_name(A, B, k):
    """
    The projection that keeps of one vector per ordered pair (A, B) only
    what a constant, a term for the first name and a term for the second
    can make, by least squares: (prompts, prompts). What it drops is what
    belongs to the two names together.
    """
    one_hot = torch.nn.functional.one_hot
    Z = torch.cat([torch.ones(len(A), 1), one_hot(A.cpu(), k), one_hot(B.cpu(), k)], dim=1).double()
    return (Z @ torch.linalg.pinv(Z)).float().to(A.device)


def kind_only(mean_when, A, B, she):
    """
    The mean over the "She" names minus the mean over the "He" names of a
    table (names, dim), without A and B, signed towards B's kind and zero
    where A and B are of one kind: (prompts, dim). It carries the kind of
    name from one to the other, and no name.
    """
    index = torch.arange(len(she), device=she.device)[None]
    others = (index != A[:, None]) & (index != B[:, None])
    she_, he_ = (others & she[None]).float(), (others & ~she[None]).float()
    toward = (she[B].float() - she[A].float())[:, None]
    return toward * (she_ @ mean_when / she_.sum(1, keepdim=True) - he_ @ mean_when / he_.sum(1, keepdim=True))
