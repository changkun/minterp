"""
What docs/02-probe.md claims, checked against the record under runs/probe/.
The readers have a closed form and run in float64 on the CPU, so a repeated
run gives the same file; if a number here moves, the note has to follow.
"""

import json

import numpy as np
import pytest
import torch

from conftest import needs_model

ROLES, TARGETS = ("A", "B", "S", "S+1", "end"), ("due", "acts", "first", "second", "control")


def run(seed):
    with open(f"runs/probe/probe-{seed}.json") as f:
        return json.load(f)


def score(seed, rule, target, role):
    """Percent right, mean over the held-out patterns: (versions, depth)."""
    a = np.array(run(seed)[rule]["accuracy"])[:, :, :, ROLES.index(role), TARGETS.index(target)]
    return np.round(100 * a.mean(0)).astype(int)


def test_the_design():
    for seed in ("seed0", "seed1"):
        r = run(seed)
        assert r["pairs"] == {"held out": 44, "trained": 88}  # ordered pairs: 22 and 44 of the 66
        assert (r["new"]["train rows"], r["new"]["test rows"]) == (704, 88)
        assert tuple(r["roles"]) == ROLES and tuple(r["targets"]) == TARGETS and len(r["names"]) == 12
        assert r["penalties"] == [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


@pytest.mark.parametrize("seed", ["seed0", "seed1"])
def test_the_two_free_checks(seed):
    # After the embedding the last position is " to" in every prompt: nothing can be read.
    for target in TARGETS[:4]:
        assert score(seed, "new", target, "end")[:, 0].tolist() == [7, 7]
    # Before the acting name the two versions are the same tokens: at most one in two.
    a = np.array(run(seed)["new"]["accuracy"])[..., TARGETS.index("due")].mean(0)  # (versions, depth, roles)
    for role in ("A", "B"):
        assert (a[:, :, ROLES.index(role)].mean(0) <= 0.5 + 1e-9).all()
    assert a[0, 1, 0] == 0.0 and a[1, 1, 0] == 1.0  # at the first name: wrong on clean, right on flipped


def test_where_the_due_name_can_be_read():
    assert score("seed0", "new", "due", "end").tolist() == [[7, 8, 12, 32, 27, 33, 23, 31, 95], [7, 7, 10, 15, 15, 18, 17, 28, 73]]
    assert score("seed0", "new", "due", "S").tolist() == [[0, 6, 11, 53, 95, 80, 68, 60, 40], [0, 6, 11, 58, 90, 74, 65, 60, 36]]
    assert score("seed1", "new", "due", "end").tolist() == [[7, 7, 70, 44, 39, 27, 21, 83, 87], [7, 7, 10, 10, 9, 11, 14, 13, 65]]
    assert score("seed1", "new", "due", "S").tolist() == [[0, 31, 60, 76, 80, 69, 67, 66, 51], [0, 36, 45, 62, 71, 68, 61, 60, 46]]
    # The spread over the held-out patterns at the acting name after block 4.
    for seed, low, high in (("seed0", 77, 100), ("seed1", 48, 95)):
        folds = 100 * np.array(run(seed)["new"]["accuracy"])[:, :, 4, ROLES.index("S"), 0]
        assert (round(folds.min()), round(folds.max())) == (low, high)


def test_what_the_last_position_holds_before_the_final_block():
    assert score("seed0", "new", "acts", "end").tolist() == [[7, 9, 8, 65, 52, 48, 28, 57, 50], [7, 9, 7, 70, 59, 50, 27, 55, 60]]
    assert score("seed1", "new", "second", "end").tolist() == [[7, 8, 69, 47, 38, 33, 23, 94, 87], [7, 8, 74, 51, 42, 35, 31, 89, 81]]


def test_on_new_pairs_the_reader_finds_no_more_than_the_lens():
    lens = lambda seed: np.round(100 * np.array(run(seed)["new"]["lens"]).mean(0)).astype(int)
    assert lens("seed0").tolist() == [[9, 7, 11, 39, 37, 32, 30, 29, 96], [9, 7, 9, 20, 19, 19, 19, 19, 75]]
    assert lens("seed1").tolist() == [[9, 9, 62, 49, 49, 47, 47, 66, 88], [9, 9, 7, 9, 11, 12, 10, 8, 70]]

    def against(seed, rule, depths):
        reader = np.array(run(seed)[rule]["accuracy"])[:, :, depths, ROLES.index("end"), 0].mean(0)
        other = np.array(run(seed)[rule]["lens"]).mean(0)[:, depths]
        return int((reader > other + 1e-9).sum()), int((reader < other - 1e-9).sum()), 100 * (reader - other).mean()

    before = slice(1, 8)  # the seven depths before the last block
    new = [against(seed, "new", before) for seed in ("seed0", "seed1")]
    assert (new[0][0] + new[1][0], new[0][1] + new[1][1]) == (13, 15)  # of 28 comparisons
    assert [round(n[2], 1) for n in new] == [-1.4, -2.2]
    # After the final block, where the answer is there, the reader is still behind in all four rows.
    assert [against(seed, "new", slice(8, 9))[1] for seed in ("seed0", "seed1")] == [2, 2]
    # On pairs it trained on, the same reader is ahead in 20 of the 28.
    seen = [against(seed, "seen", before) for seed in ("seed0", "seed1")]
    assert (seen[0][0] + seen[1][0], seen[0][1] + seen[1][1]) == (20, 7)


def test_a_reader_can_do_the_work():
    for seed in ("seed0", "seed1"):
        tokens = run(seed)["tokens only"]
        assert tokens["first, second and acts"] == {"clean": 1.0, "flipped": 1.0, "both": 1.0, "tied": False}
        # Without the acting name the two versions are one input: right on one of them, and the two names tie.
        assert tokens["first and second"]["both"] == 0.5 and tokens["first and second"]["tied"]


def test_a_reader_can_remember():
    made_up = lambda seed, rule: 100 * np.array(run(seed)[rule]["accuracy"])[..., 4].mean((0, 1))  # (depth, roles)
    b, end = ROLES.index("B"), ROLES.index("end")
    for seed, block1, later, last, anywhere in (("seed0", 67, (37, 59), (14, 24), (5, 15)), ("seed1", 88, (37, 63), (14, 26), (5, 14))):
        seen, new = made_up(seed, "seen"), made_up(seed, "new")
        assert round(seen[0, end]) == 15 and round(new[0, end]) == 11  # what always naming the commonest label gives
        assert round(seen[1, b]) == block1 and (round(seen[2:, b].min()), round(seen[2:, b].max())) == later
        assert (round(seen[:, end].min()), round(seen[:, end].max())) == last
        assert (round(new.min()), round(new.max())) == anywhere
    # The due name at the acting name, pairs it trained on against new pairs.
    at = lambda seed, rule, depth: score(seed, rule, "due", "S")[:, depth].tolist()
    assert at("seed0", "seen", 2) == [53, 60] and at("seed0", "new", 2) == [11, 11]
    assert at("seed1", "seen", 2) == [79, 69] and at("seed1", "new", 2) == [60, 45]
    assert at("seed0", "seen", 4) == [95, 92] and at("seed0", "new", 4) == [95, 90]
    assert (run("seed0")["seen"]["train rows"], run("seed0")["new"]["train rows"]) == (1056, 704)


@needs_model
def test_the_script_reproduces_the_record(tmp_path, monkeypatch):
    """Two runs give the same file: nothing in the fit depends on chance or on the machine's GPU."""
    import os
    import subprocess
    import sys

    env = dict(os.environ, MINTERP_RUNS=str(tmp_path))
    subprocess.run([sys.executable, "scripts/probe.py"], check=True, env=env, capture_output=True)
    again = json.load(open(tmp_path / "probe" / "probe-seed0.json"))
    assert torch.allclose(torch.tensor(again["new"]["accuracy"]), torch.tensor(run("seed0")["new"]["accuracy"]))
