"""
A model and a tokenizer small enough for every test to build its own.

The tests of the mechanics (hooks, the task's construction, the algebra of
attribution) run on a model with random weights: what they check holds for
any weights. Tests of what a note claims about the trained story model are
marked `needs_model` and are skipped where that model is absent.
"""

import re

import pytest
import torch
from minllm.data import EOT
from minllm.model import Config, Transformer

from minterp import common, tasks


class WordTokenizer:
    """One token per word or punctuation mark, a space counted with the word after it."""

    def __init__(self, texts):
        self.special_tokens = {EOT: 0}
        self.vocab = [EOT]
        for text in texts:
            for piece in self._split(text):
                if piece not in self.vocab:
                    self.vocab.append(piece)

    @staticmethod
    def _split(text):
        return re.findall(r" ?[A-Za-z]+|[^A-Za-z ]", text)

    def encode_ordinary(self, text):
        return [self.vocab.index(p) for p in self._split(text)]

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)


@pytest.fixture(scope="session")
def tok():
    return WordTokenizer([t.format(A=n, B=n, S=n) for t in tasks.TEMPLATES for n in tasks.CANDIDATES[:5]])


@pytest.fixture(scope="session")
def people():
    return tasks.CANDIDATES[:5]


@pytest.fixture(scope="session")
def model(tok):
    torch.manual_seed(0)
    cfg = Config(vocab_size=len(tok.vocab), max_seq_len=32, dim=48, n_layers=3, n_heads=4, n_kv_heads=2, mlp_dim=96)
    return Transformer(cfg).eval().requires_grad_(False)


needs_model = pytest.mark.skipif(not common.has_model("seed0"), reason="train the story model in minllm: uv run scripts/train.py")
needs_open = pytest.mark.skipif(not common.has_model("qwen"), reason="download the open model in minllm: uv run scripts/download_qwen.py")
