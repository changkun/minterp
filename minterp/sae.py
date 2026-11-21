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

Nothing here knows what the model is. The input is vectors, the output is
vectors, and whether a direction means anything is a question for the
experiments.
"""

import torch
from torch import nn


class SAE(nn.Module):
    def __init__(self, dim, width, k):
        super().__init__()
        self.k = k
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
        """Stream vectors (..., dim) -> latents (..., width), at most k of them non-zero."""
        pre = (x / self.scale - self.b) @ self.enc + self.b_enc
        top = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, top.indices, top.values.relu())

    def decode(self, z):
        return (z @ self.dec + self.b) * self.scale

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def train(vectors, width, k, steps=4000, batch=4096, lr=4e-4, seed=0, device="cpu", log=None):
    """
    Fit a dictionary to stream vectors (rows, dim), which stay on the CPU
    while batches of them go to the device. Returns the autoencoder.
    """
    torch.manual_seed(seed)
    dim = vectors.shape[1]
    sae = SAE(dim, width, k)
    sae.scale.fill_(vectors[:100_000].norm(dim=1).mean().item() / dim ** 0.5)
    sae.b.data = vectors[:100_000].mean(0) / sae.scale
    sae = sae.to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    pick = torch.Generator().manual_seed(seed)
    for step in range(steps):
        x = vectors[torch.randint(0, vectors.shape[0], (batch,), generator=pick)].to(device)
        rebuilt, _ = sae(x)
        loss = ((rebuilt - x) / sae.scale).pow(2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():  # keep every direction at length one, so that a latent's value is a size
            sae.dec.data /= sae.dec.data.norm(dim=1, keepdim=True)
        if log and (step % 500 == 0 or step == steps - 1):
            log(step, loss.item())
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
    torch.save({"state": sae.state_dict(), "dim": sae.dec.shape[1], "width": sae.width, "k": sae.k}, path)


def load(path, device="cpu"):
    blob = torch.load(path, map_location=device, weights_only=True)
    sae = SAE(blob["dim"], blob["width"], blob["k"])
    sae.load_state_dict(blob["state"])
    return sae.to(device).eval().requires_grad_(False)
