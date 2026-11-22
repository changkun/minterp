"""
Which latents of one dictionary have a counterpart in another?

Two dictionaries for the same place are two lists of directions in no
common order. A latent of the first has a counterpart in the second when
some latent there points the same way, to within a chosen cosine between
their decoder directions. Step 8 uses three ways of counting, from lenient
to strict:

    best        the best single counterpart of every latent, whoever else
                has the same one
    one_to_one  a matching in which no latent of the second is used twice,
                made greedily from the highest cosine down
    shared      the measure of Paulo and Belrose (2025): the same partner
                in a one-to-one matching by the encoder and in one by the
                decoder, at a cosine of 0.7 or more in both

and two yardsticks for what a cosine means in this many dimensions: the
nearest other latent inside one dictionary, and two sets of random
directions.
"""

import torch


def unit(v):
    return v / v.norm(dim=1, keepdim=True)


def cosines(one, two):
    """(width of one, width of two): the cosine between every pair of decoder directions."""
    return unit(one.dec.detach().float().cpu()) @ unit(two.dec.detach().float().cpu()).T


def best(one, two):
    """For every latent of the first dictionary: the cosine of its best counterpart in the second, and which latent that is."""
    top = cosines(one, two).max(1)
    return top.values, top.indices


def one_to_one(c, floor=0.7):
    """
    A greedy one-to-one matching of rows to columns of a cosine table,
    highest first. Pairs under the floor are left unmatched (-1): they could
    not change who is matched above it, since a greedy pass takes the
    higher pairs first.
    """
    rows, cols = (c >= floor).nonzero(as_tuple=True)
    partner, taken = torch.full((c.shape[0],), -1), torch.zeros(c.shape[1], dtype=torch.bool)
    for k in c[rows, cols].argsort(descending=True).tolist():
        i, j = rows[k].item(), cols[k].item()
        if partner[i] < 0 and not taken[j]:
            partner[i], taken[j] = j, True
    return partner


def shared(one, two, floor=0.7):
    """The share of the first dictionary's latents with one partner by decoder and by encoder alike, and the share matched by the decoder alone."""
    by_dec = one_to_one(cosines(one, two), floor)
    by_enc = one_to_one(unit(one.enc.detach().float().cpu().T) @ unit(two.enc.detach().float().cpu().T).T, floor)
    return ((by_dec >= 0) & (by_dec == by_enc)).float().mean().item(), (by_dec >= 0).float().mean().item()


def shares(values):
    """How many of some best cosines reach 0.9, 0.7 and 0.5, and their median."""
    return {"at a cosine of 0.9 or more": (values >= 0.9).float().mean().item(), "at 0.7 or more": (values >= 0.7).float().mean().item(),
            "at 0.5 or more": (values >= 0.5).float().mean().item(), "median cosine": values.median().item()}


def inside(one):
    """For every latent: the cosine of the nearest other latent of the same dictionary."""
    c = cosines(one, one)
    c.fill_diagonal_(-1)
    return c.max(1).values


def by_chance(width_one, width_two, dim, seed=0):
    """The best cosine of each of some random directions with a second set of random directions: what counts as nothing."""
    g = torch.Generator().manual_seed(seed)
    return (unit(torch.randn(width_one, dim, generator=g)) @ unit(torch.randn(width_two, dim, generator=g)).T).max(1).values
