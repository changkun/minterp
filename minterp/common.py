"""
Where the models, the tokenizer and the results are.

minterp trains no language model. It opens the story model of minllm's
stage 3, which lives in a checkout of minllm beside this one (or wherever
MINLLM_DIR points): the checkpoint, the tokenizer and the encoded stories.
"""

import json
import os
import re

import torch
from minllm.common import load_checkpoint, pick_device
from minllm.data import EOT
from minllm.tokenizer import Tokenizer

MINLLM = os.environ.get("MINLLM_DIR", "../minllm")

# The models the notes report. They differ only in the seed of their training
# run: `uv run scripts/train.py` and `uv run scripts/train.py --seed 1 --out
# out/base-seed1` in minllm.
MODELS = {"seed0": "out/base/ckpt.pt", "seed1": "out/base-seed1/ckpt.pt"}

# A published model of 0.5 billion parameters, for step 8. minllm's stage 5
# reads its weights and tokenizer into the same Transformer class, so every
# hook of minterp/tap.py finds the module it names: `uv run
# scripts/download_qwen.py` in minllm downloads it.
OPEN = {"qwen": ("data/qwen2.5-0.5b-instruct", "Qwen/Qwen2.5-0.5B-Instruct")}

# The published record. Scripts write to out/runs/ unless MINTERP_RUNS=runs,
# so that trying a script does not change the numbers the notes quote.
RESULTS = os.environ.get("MINTERP_RUNS", "out/runs")


def model_path(name):
    """A name from MODELS, or a path to any checkpoint of scripts/train.py."""
    return os.path.join(MINLLM, MODELS[name]) if name in MODELS else name


def has_model(name="seed0"):
    if name in OPEN:
        return os.path.exists(os.path.join(MINLLM, OPEN[name][0], "model.safetensors"))
    return os.path.exists(model_path(name)) and os.path.exists(os.path.join(MINLLM, "out/tokenizer.json"))


def load(name="seed0", device=None):
    """The model, frozen and in evaluation mode, with its tokenizer."""
    device = device or pick_device()
    if name in OPEN:
        from minllm.pretrained import load_model, load_tokenizer

        folder, repo = OPEN[name]
        model = load_model(os.path.join(MINLLM, folder), device=device).requires_grad_(False)
        cfg = model.cfg
        config = {"dim": cfg.dim, "n_layers": cfg.n_layers, "n_heads": cfg.n_heads, "n_kv_heads": cfg.n_kv_heads, "vocab_size": cfg.vocab_size}
        return model, load_tokenizer(os.path.join(MINLLM, folder)), {"name": name, "repo": repo, "config": config}
    model, ckpt = load_checkpoint(model_path(name), device)
    model.eval().requires_grad_(False)
    tok = Tokenizer.load(os.path.join(MINLLM, "out/tokenizer.json"))
    return model, tok, {"name": name, "step": ckpt["step"], "val_loss": ckpt["val_loss"], "config": ckpt["config"]}


def eot(tok):
    """The token that opens every story in the training data."""
    return tok.special_tokens[EOT]


def save(path, result):
    """Write a result under runs/ (see RESULTS), as JSON a figure script can read."""
    assert path.startswith("runs/")
    path = os.path.join(RESULTS, path[len("runs/"):])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = json.dumps(result, indent=1)
    # A list of numbers on one line, so that a table of them reads as a table.
    text = re.sub(r"\[[\s\d.,eE+-]+\]", lambda m: "[" + " ".join(m.group(0)[1:-1].split()) + "]", text)
    with open(path, "w") as f:
        f.write(text + "\n")
    print("wrote", path)
    return path


def rounded(t, digits=4):
    """A tensor as nested lists of rounded floats, for JSON."""
    return torch.as_tensor(t).detach().cpu().double().numpy().round(digits).tolist()  # to the CPU first: Apple's GPU has no float64
