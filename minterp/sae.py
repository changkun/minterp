"""
A sparse autoencoder: a dictionary for the stream.

The stream at one place is a vector of a few hundred numbers. The idea
behind a dictionary is that this vector is a sum of a small number of
directions out of a large set, each direction standing for something the
model keeps track of, and that neither the directions nor their number can
be read off the model's weights. A sparse autoencoder looks for them. It
maps a stream vector to many more numbers than the vector has, the latents,
of which only a few may be non-zero, and maps those back to the vector:

    z = top_k(W_enc (x - b) + b_enc)         latents: all but the k largest set to zero
    x_hat = W_dec z + b                      the vector rebuilt from k directions

It is trained to make x_hat close to x, on stream vectors collected from
ordinary text. Each column of W_dec is one direction of the dictionary, and
a latent's value says how much of that direction a given vector contains.

Keeping the k largest latents is the variant of Gao et al. (2024). The
older way is a penalty on the size of the latents, which has to be tuned
against the reconstruction; with top-k the number of active latents is set
directly.

A newer variant, the batch top-k dictionary of Bussmann et al. (2024),
takes the largest values over a whole batch of vectors while it trains: of
n vectors, the n * k largest latents stay, wherever they are, so that one
vector may use more than k and another fewer. When it is used there is no
batch to rank against, so it keeps what exceeds one threshold, estimated
after training as the mean over many batches of the smallest value that
was kept. `across_batch=True` trains that variant.

Nothing here knows what the model is. The input is vectors, the output is
vectors, and whether a direction means anything is a question for the
experiments.
"""

import torch
from torch import nn


class SAE(nn.Module):
    def __init__(self, dim, width, k, across_batch=False):
        super().__init__()
        self.k, self.across_batch = k, across_batch
        if across_batch:  # at use, a latent is on where it exceeds this
            self.register_buffer("threshold", torch.zeros(()))
        directions = torch.randn(width, dim)
        directions = directions / directions.norm(dim=1, keepdim=True)
        self.dec = nn.Parameter(directions.clone())  # (width, dim): one unit direction per latent
        self.enc = nn.Parameter(directions.T.clone())  # (dim, width): starts as the transpose
        self.b_enc = nn.Parameter(torch.zeros(width))
        self.b = nn.Parameter(torch.zeros(dim))
        # The stream's size differs between depths and models. Vectors are divided by this
        # before they enter, so that one learning rate fits every place.
        self.register_buffer("scale", torch.ones(()))

    @property
    def width(self):
        return self.dec.shape[0]

    def encode(self, x):
        """
        Stream vectors (..., dim) -> latents (..., width). At most k of them
        are non-zero; in a batch top-k dictionary, those above its
        threshold, about k on average. A vector gets the same latents alone
        and beside others.
        """
        pre = (x / self.scale - self.b) @ self.enc + self.b_enc
        if self.across_batch:
            return torch.where(pre > self.threshold, pre, torch.zeros_like(pre))
        top = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, top.indices, top.values.relu())

    def across(self, x):
        """
        Training only, for a batch top-k dictionary: of a batch of n vectors
        the n * k largest latents stay, wherever they are. Returns them and
        the smallest value that stayed.
        """
        pre = ((x / self.scale - self.b) @ self.enc + self.b_enc).relu()
        cut = pre.flatten().topk(self.k * pre.shape[0]).values[-1]
        return torch.where(pre >= cut, pre, torch.zeros_like(pre)), cut

    def decode(self, z):
        return (z @ self.dec + self.b) * self.scale

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def train(vectors, width, k, steps=4000, batch=4096, lr=4e-4, seed=0, device="cpu", log=None, across_batch=False):
    """
    Fit a dictionary to stream vectors (rows, dim), which stay on the CPU
    while batches of them go to the device. Returns the autoencoder.
    """
    torch.manual_seed(seed)
    dim = vectors.shape[1]
    sae = SAE(dim, width, k, across_batch)
    sae.scale.fill_(vectors[:100_000].norm(dim=1).mean().item() / dim ** 0.5)
    sae.b.data = vectors[:100_000].mean(0) / sae.scale
    sae = sae.to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    pick = torch.Generator().manual_seed(seed)
    for step in range(steps):
        x = vectors[torch.randint(0, vectors.shape[0], (batch,), generator=pick)].to(device)
        rebuilt = sae.decode(sae.across(x)[0]) if across_batch else sae(x)[0]
        loss = ((rebuilt - x) / sae.scale).pow(2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():  # keep every direction at length one, so that a latent's value is a size
            sae.dec.data /= sae.dec.data.norm(dim=1, keepdim=True)
        if log and (step % 500 == 0 or step == steps - 1):
            log(step, loss.item())
    if across_batch:  # the threshold for use: the smallest value kept in a batch, averaged over a hundred batches
        with torch.no_grad():
            cuts = [sae.across(vectors[torch.randint(0, vectors.shape[0], (batch,), generator=pick)].to(device))[1] for _ in range(100)]
            sae.threshold.fill_(torch.stack(cuts).mean().item())
    return sae.eval().requires_grad_(False)


def path(model, block, tag=""):
    """Where a dictionary for one model and depth is kept. A tag names another dictionary for the same place."""
    return f"out/sae/{model}-b{block}{'-' + tag if tag else ''}.pt"


@torch.no_grad()
def unexplained(sae, vectors, batch=8192):
    """The share of the vectors' variance that the rebuilt vectors miss: 0 is perfect, 1 is no better than the mean."""
    device = sae.dec.device
    miss, total, mean = 0.0, 0.0, vectors.mean(0).to(device)
    for i in range(0, vectors.shape[0], batch):
        x = vectors[i:i + batch].to(device)
        miss += (sae(x)[0] - x).pow(2).sum().item()
        total += (x - mean).pow(2).sum().item()
    return miss / total


def save(sae, path):
    torch.save({"state": sae.state_dict(), "dim": sae.dec.shape[1], "width": sae.width, "k": sae.k, "across_batch": sae.across_batch}, path)


def load(path, device="cpu"):
    blob = torch.load(path, map_location=device, weights_only=True)
    sae = SAE(blob["dim"], blob["width"], blob["k"], blob.get("across_batch", False))  # files written before the variant existed lack the key
    sae.load_state_dict(blob["state"])
    return sae.to(device).eval().requires_grad_(False)
