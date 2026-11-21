"""
Step 8 runs the same code on a published model. Before any measurement on
it, the identities that every earlier step rests on are checked there: a
model read from published weights has attention biases and grouped keys
that the story model lacks.
"""

import pytest
import torch

from conftest import needs_open
from minterp import attention, lens, tasks
from minterp.common import load
from minterp.tap import Tap, head_writes


@pytest.fixture(scope="module")
def recorded():
    model, tok, _ = load("qwen")
    batch = tasks.batches(tok, next(model.parameters()).device, people=tasks.names(tok))[0]
    with torch.no_grad(), Tap(model) as tap:
        logits, _ = model(batch.clean)
    return model, batch, tap, logits


@needs_open
def test_the_names_are_single_tokens_there_too(recorded):
    _, batch, _, _ = recorded
    assert len(batch) == 132


@needs_open
def test_the_recorded_pieces_add_up_to_the_stream(recorded):
    model, _, tap, _ = recorded
    n = model.cfg.n_layers
    total = tap["resid", 0] + sum(tap["attn", i] + tap["mlp", i] for i in range(1, n + 1))
    size = tap["resid", n].norm(dim=-1).mean()
    assert (total - tap["resid", n]).abs().max() < 1e-5 * size
    for i in range(1, n + 1):
        assert (head_writes(model, tap, i).sum(2) - tap["attn", i]).abs().max() < 1e-5 * size


@needs_open
def test_the_attribution_adds_up_to_the_logit_difference(recorded):
    model, batch, tap, logits = recorded
    parts = lens.attribute(model, tap, batch.right, batch.wrong)
    difference = tasks.logit_diff(logits[:, -1].float(), batch.right, batch.wrong)
    assert (parts["embed"] + parts["heads"].sum((1, 2)) + parts["mlp"].sum(1) - difference).abs().max() < 1e-3


@needs_open
def test_recomputed_attention_is_the_models_own(recorded):
    model, _, tap, _ = recorded
    weights = attention.patterns(model, tap)  # (batch, blocks, heads, seq, seq)
    assert (weights.sum(-1) - 1).abs().max() < 1e-5
    for i in (1, 12, 24):
        block = model.layers[i - 1]
        values = attention.values(model, block.attn, block.attn_norm(tap["resid", i - 1]))
        again = torch.einsum("bhts,bhsd->bthd", weights[:, i - 1], values)
        assert (again - tap["heads", i]).abs().max() < 1e-4
