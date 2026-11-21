"""
What docs/01-read.md claims, checked against the record under runs/read/
and, where the story model is present, against the model itself. If a run
is repeated and a number moves, these fail and the note has to follow.
"""

import json

import numpy as np
import pytest
import torch

from conftest import needs_model
from minterp import lens, tasks
from minterp.common import eot, load
from minterp.tap import Tap

VERSIONS = ("clean", "flipped", "neutral")


def pooled(seed):
    with open(f"runs/read/behavior-{seed}.json") as f:
        run = json.load(f)
    return {v: np.concatenate([np.array(t[v]["logit_diff"]) for t in run["templates"]]) for v in VERSIONS}, run


def attribution(seed):
    with open(f"runs/read/lens-{seed}.json") as f:
        return json.load(f)


@pytest.mark.parametrize("seed, clean, flipped, medians, neutral, swing", [
    ("seed0", 99.8, 84.5, (7.42, -4.03), (1.84, 59.4), (11.84, 99.8)),
    ("seed1", 95.5, 75.8, (5.36, -2.79), (1.28, 62.7), (7.49, 99.2)),
])
def test_both_models_do_the_task(seed, clean, flipped, medians, neutral, swing):
    ld, run = pooled(seed)
    assert len(ld["clean"]) == 660 and len(run["names"]) == 12
    assert round(100 * (ld["clean"] > 0).mean(), 1) == clean
    assert round(100 * (ld["flipped"] < 0).mean(), 1) == flipped
    assert (round(np.median(ld["clean"]), 2), round(np.median(ld["flipped"]), 2)) == medians
    # A lean to the second name when a third one acts.
    assert (round(ld["neutral"].mean(), 2), round(100 * (ld["neutral"] > 0).mean(), 1)) == neutral
    paired = ld["clean"] - ld["flipped"]
    assert (round(paired.mean(), 2), round(100 * (paired > 0).mean(), 1)) == swing


@pytest.mark.parametrize("seed, clean, flipped", [("seed0", (99, 100), (82, 88)), ("seed1", (86, 99), (67, 84))])
def test_every_pattern_is_done_in_most_prompts(seed, clean, flipped):
    _, run = pooled(seed)
    c = [round(100 * (np.array(t["clean"]["logit_diff"]) > 0).mean()) for t in run["templates"]]
    f = [round(100 * (np.array(t["flipped"]["logit_diff"]) < 0).mean()) for t in run["templates"]]
    assert (min(c), max(c)) == clean and (min(f), max(f)) == flipped


def test_a_logit_difference_is_not_the_top_token():
    _, run = pooled("seed0")
    entry = next(t for t in run["templates"] if t["template"].endswith("showed the map to"))
    assert round(100 * (np.array(entry["clean"]["logit_diff"]) > 0).mean()) == 99
    assert round(100 * entry["clean"]["top_is_B"]) == 17


def blocks(seed, version):
    """What each block writes into B minus A, mean over the prompts: (blocks,)."""
    a = attribution(seed)["attribution"][version]
    return np.array(a["heads"]["mean"]).sum(1) + np.array(a["mlp"]["mean"])


@pytest.mark.parametrize("seed, total, late, early", [("seed0", 7.90, 7.42, 0.48), ("seed1", 5.25, 4.74, 0.51)])
def test_the_last_two_blocks_write_most_of_the_clean_difference(seed, total, late, early):
    b = blocks(seed, "clean")
    embed = attribution(seed)["attribution"]["clean"]["embed"]["mean"]
    assert b.sum() + embed == pytest.approx(total, abs=0.011)
    assert b[6:].sum() == pytest.approx(late, abs=0.011) and b[:6].sum() == pytest.approx(early, abs=0.011)


def test_the_models_divide_the_work_differently():
    by = lambda seed, block: [round(float(blocks(seed, v)[block - 1]), 2) for v in VERSIONS]
    # seed1: blocks 2 and 7 lean to B whoever acts, block 8 follows the acting name.
    assert by("seed1", 2) == [0.45, 0.52, 0.50] and by("seed1", 7) == [1.45, 1.08, 1.17]
    assert by("seed1", 8) == [3.29, -3.75, -0.37]
    # seed0: block 7 follows the acting name and is silent for a third; block 8 does both.
    assert by("seed0", 7) == [2.42, -2.36, -0.0] and by("seed0", 8) == [5.00, -1.69, 1.53]
    assert by("seed0", 3) == [0.49, 0.04, 0.28]


def test_single_parts_in_the_three_versions():
    a0, a1 = attribution("seed0")["attribution"], attribution("seed1")["attribution"]
    head = lambda a, block, h: [round(a[v]["heads"]["mean"][block - 1][h], 2) for v in VERSIONS]
    mlp = lambda a, block: [round(a[v]["mlp"]["mean"][block - 1], 2) for v in VERSIONS]
    assert head(a0, 7, 0) == [1.79, -2.08, -0.12] and head(a1, 8, 0) == [1.85, -1.95, -0.10]
    assert head(a0, 8, 4) == [2.57, -1.23, 0.56] and head(a0, 8, 3) == [1.09, 0.06, 0.57]
    assert mlp(a0, 8) == [1.18, -0.51, 0.32] and mlp(a1, 8) == [1.45, -0.52, 0.37]
    assert head(a1, 7, 1) == [1.07, 0.87, 0.91] and head(a1, 2, 0) == [0.37, 0.39, 0.39]
    assert head(a1, 8, 5) == [-0.53, -0.56, -0.53]
    q = a1["clean"]["heads"]
    assert (round(q["q25"][7][5], 2), round(q["q75"][7][5], 2)) == (-1.12, 0.13)  # the same on average, not in every prompt


def lens_of(seed, version):
    run = attribution(seed)
    return np.concatenate([np.array(t["lens"][version]) for t in run["templates"]], axis=1)  # (depth, prompts)


def test_the_first_step_differs_between_the_models():
    med = lambda seed, version, depth: round(float(np.median(lens_of(seed, version)[depth])), 2)
    assert [med("seed0", v, 3) for v in VERSIONS] == [1.38, 0.12, 0.79]
    assert [med("seed1", v, 2) for v in VERSIONS] == [1.58, 1.91, 1.77]
    assert [med("seed0", v, 8) for v in VERSIONS] == [7.42, -4.03, 1.33]
    assert [med("seed1", v, 7) for v in VERSIONS] == [2.79, 2.03, 2.33] and med("seed1", "flipped", 8) == -2.79


def test_early_on_the_stream_leans_to_the_name_that_acts():
    above = lambda seed, depth: round(100 * float((lens_of(seed, "clean")[depth] > lens_of(seed, "flipped")[depth]).mean()))
    assert [above("seed1", d) for d in (2, 3, 4, 5)] == [12, 1, 2, 17]
    assert above("seed0", 2) == 7 and above("seed0", 3) == 92


def test_the_stream_grows_with_depth():
    size = np.mean([t["size"] for t in attribution("seed0")["templates"]], axis=0)
    assert [round(float(size[d]), 2) for d in (0, 3, 8)] == [0.05, 1.10, 3.46]


@needs_model
def test_on_the_story_model_the_split_is_exact_and_the_task_is_done():
    model, tok, _ = load("seed0", device="cpu")
    batch = tasks.batches(tok)[0]
    with Tap(model) as tap:
        logits, _ = model(batch.clean)
    final = tasks.logit_diff(logits, batch.right, batch.wrong)
    parts = lens.attribute(model, tap, batch.right, batch.wrong)
    total = parts["embed"] + parts["heads"].sum((1, 2)) + parts["mlp"].sum(1)
    assert torch.allclose(total, final, atol=1e-3)
    assert (final > 0).float().mean() > 0.9
    assert len(tasks.names(tok)) == 12
    # The prompt of the README and the note: " Tom" is the likeliest next token.
    ids = torch.tensor([[eot(tok)] + tok.encode_ordinary("One day, Lily and Tom were playing in the garden. Lily said hello to")])
    probs = model(ids)[0][0, -1].softmax(-1)
    assert tok.decode([probs.argmax().item()]) == " Tom" and round(probs.max().item(), 2) == 0.99
