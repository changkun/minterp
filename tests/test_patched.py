"""
What docs/03-patch.md claims, checked against the record under runs/patch/.
"""

import json

import numpy as np
import pytest

TRAVEL = ("before", "A", "B", "S", "S+1", "between", "end", "all")
NAME = ("B", "B to S", "S", "S+1", "between", "end")
CHOICES = ("flipped", "neutral", "mean", "zero")


def run(seed):
    with open(f"runs/patch/patch-{seed}.json") as f:
        return json.load(f)


def mean(seed, pick):
    return np.mean([pick(t) for t in run(seed)["templates"]], axis=0)


def two(x):
    return [round(float(v) + 0.0, 2) for v in x]


@pytest.mark.parametrize("seed", ["seed0", "seed1"])
def test_the_two_free_checks(seed):
    carry = np.array([t["travel"]["carry"] for t in run(seed)["templates"]])  # (patterns, depth, groups)
    assert tuple(run(seed)["travel groups"]) == TRAVEL and tuple(run(seed)["name groups"]) == NAME
    for group in ("before", "A", "B"):  # the two versions are the same tokens there
        assert np.abs(carry[:, :, TRAVEL.index(group)]).max() < 5e-4
    assert np.abs(carry[:, :, TRAVEL.index("all")] - 1).max() < 5e-4  # the whole stream is the whole answer
    for group in ("S+1", "between"):
        assert np.abs(carry[:, :, TRAVEL.index(group)].mean(0)).max() < 0.01


def test_where_who_acts_is():
    at = lambda seed, kind, group: two(mean(seed, lambda t: t["travel"][kind])[:, TRAVEL.index(group)])
    assert at("seed0", "carry", "S") == [1.0, 1.0, 1.0, 0.97, 0.97, 0.94, 0.94, -0.05, 0.0]
    assert at("seed0", "carry", "end") == [0.0, 0.0, 0.0, 0.04, 0.04, 0.06, 0.05, 1.02, 1.0]
    assert at("seed1", "carry", "S") == [1.0, 1.0, 1.01, 1.03, 1.04, 1.01, 0.9, 0.92, 0.0]
    assert at("seed1", "carry", "end") == [0.0, 0.0, -0.01, -0.03, -0.04, -0.02, 0.08, 0.07, 1.0]
    for seed in ("seed0", "seed1"):
        for group in ("S", "end"):
            gap = np.abs(np.array(at(seed, "carry", group)) - np.array(at(seed, "break", group)))
            assert gap.max() <= 0.03 + 1e-9


def test_where_the_second_name_is():
    at = lambda seed, group: two(mean(seed, lambda t: t["name"])[:, NAME.index(group)])
    assert at("seed0", "B") == [1.0, 1.0, 0.97, 0.81, 0.7, 0.7, 0.7, 0.67, 0.0]
    assert at("seed0", "B to S") == [0.0, 0.0, 0.03, 0.09, 0.1, 0.09, 0.08, 0.09, 0.0]
    assert at("seed0", "S") == [0.0, 0.0, 0.0, 0.01, 0.1, 0.1, 0.1, 0.0, 0.0]
    assert at("seed0", "end") == [0.0, 0.0, 0.01, 0.08, 0.09, 0.1, 0.1, 0.23, 1.0]
    assert at("seed1", "B") == [1.0, 1.0, 0.5, 0.41, 0.33, 0.33, 0.34, -0.01, 0.0]
    assert at("seed1", "B to S") == [0.0, 0.0, 0.37, 0.42, 0.42, 0.41, 0.41, 0.41, 0.0]
    assert at("seed1", "S") == [0.0, 0.0, 0.01, 0.03, 0.1, 0.1, 0.1, 0.11, 0.0]
    assert at("seed1", "end") == [0.0, 0.0, 0.12, 0.13, 0.13, 0.14, 0.15, 0.48, 1.0]
    measure = lambda seed, v: round(float(np.mean([t["name measure"][v] for t in run(seed)["templates"]])), 2)
    assert (measure("seed0", "clean"), measure("seed0", "other")) == (-7.99, 7.97)
    assert (measure("seed1", "clean"), measure("seed1", "other")) == (-7.24, 7.26)


def test_where_the_first_name_is():
    first = ("A", "A to S", "S", "S+1", "between", "end")
    assert tuple(run("seed0")["first name groups"]) == first
    at = lambda seed, group: two(mean(seed, lambda t: t["first name"])[:, first.index(group)])
    assert at("seed0", "A") == [1.0, 0.99, 0.9, 0.61, 0.45, 0.45, 0.44, 0.42, 0.0]
    assert at("seed0", "A to S") == [0.0, 0.01, 0.1, 0.3, 0.31, 0.3, 0.29, 0.29, 0.0]
    assert at("seed0", "S") == [0.0, 0.0, 0.0, 0.03, 0.14, 0.15, 0.15, 0.0, 0.0]
    assert at("seed0", "end") == [0.0, 0.0, 0.01, 0.05, 0.07, 0.08, 0.09, 0.27, 1.0]
    assert at("seed1", "A") == [1.0, 0.98, 0.42, 0.3, 0.2, 0.19, 0.19, 0.0, 0.0]
    assert at("seed1", "A to S") == [0.0, 0.02, 0.52, 0.61, 0.61, 0.6, 0.59, 0.66, 0.0]
    assert at("seed1", "S") == [0.0, 0.0, 0.01, 0.03, 0.12, 0.13, 0.12, 0.12, 0.0]
    assert at("seed1", "end") == [0.0, 0.0, 0.05, 0.06, 0.06, 0.08, 0.09, 0.21, 1.0]
    measure = lambda seed, v: round(float(np.mean([t["first name measure"][v] for t in run(seed)["templates"]])), 2)
    assert (measure("seed0", "flipped"), measure("seed0", "other")) == (-5.8, 5.69)
    # In the name experiments the groups' shares add up to one, so a share reads as a fraction.
    for seed in ("seed0", "seed1"):
        for key in ("name", "first name"):
            total = mean(seed, lambda t, key=key: t[key]).sum(1)
            assert round(float(total.min()), 2) >= 0.98 and round(float(total.max()), 2) <= 1.0


def test_who_acts_and_the_names_at_the_acting_name():
    # Who acts is taken from this position; a name mostly is not.
    for seed, who, low, high in (("seed0", 0.94, 0.90, 0.97), ("seed1", 0.90, 0.84, 0.99)):
        s = np.array([t["travel"]["carry"] for t in run(seed)["templates"]])[:, 6, TRAVEL.index("S")]
        assert (round(float(s.mean()), 2), round(float(s.min()), 2), round(float(s.max()), 2)) == (who, low, high)
    with open("runs/probe/probe-seed0.json") as f:
        acc = np.array(json.load(f)["new"]["accuracy"])  # (folds, versions, depth, roles, targets)
    assert [round(100 * float(acc[:, v, 3, 4, 1].mean())) for v in (0, 1)] == [65, 70]  # step 2 read who acts at the last position
    assert two(mean("seed0", lambda t: t["travel"]["carry"])[3:7, TRAVEL.index("end")]) == [0.04, 0.04, 0.06, 0.05]  # where it carries this


def part(seed, block, head, kind="carry"):
    """carry at all positions, at the last one, and at the acting name, for one part (head 6 is the MLP)."""
    return two(mean(seed, lambda t: t["parts"][kind][s])[block - 1, head] for s in ("all", "last", "S"))


def written(seed, block, head):
    """What a part writes directly on clean minus on flipped prompts (step 1), as a share of the gap."""
    with open(f"runs/read/lens-{seed}.json") as f:
        a = json.load(f)["attribution"]
    with open(f"runs/read/behavior-{seed}.json") as f:
        b = json.load(f)["templates"]
    gap = np.mean(np.concatenate([np.array(t["clean"]["logit_diff"]) - np.array(t["flipped"]["logit_diff"]) for t in b]))
    of = lambda v: a[v]["heads"]["mean"][block - 1][head] if head < 6 else a[v]["mlp"]["mean"][block - 1]
    return round(float((of("clean") - of("flipped")) / gap) + 0.0, 2)


def test_which_parts_carry_it():
    assert tuple(run("seed0")["scopes"]) == ("all", "last", "not last", "S")
    assert part("seed0", 1, 6) == [1.0, 0.0, 1.0] and part("seed1", 1, 6) == [0.99, 0.0, 0.99]
    assert part("seed0", 7, 0) == [0.87, 0.86, 0.0] and part("seed0", 4, 2) == [0.52, 0.0, 0.52]
    assert part("seed0", 8, 4) == [0.41, 0.41, 0.0] and part("seed0", 8, 3) == [0.11, 0.11, 0.0]
    assert part("seed1", 8, 0) == [0.69, 0.69, 0.0] and part("seed1", 8, 6) == [0.26, 0.26, 0.0]
    assert part("seed1", 4, 4) == [0.21, 0.0, 0.21]
    assert [written("seed0", b, h) for b, h in ((1, 6), (7, 0), (4, 2), (8, 4), (8, 3))] == [0.0, 0.33, 0.0, 0.32, 0.09]
    assert [written("seed1", b, h) for b, h in ((1, 6), (8, 0), (8, 6), (4, 4))] == [0.0, 0.51, 0.26, 0.0]
    spread = lambda seed, block, head: two(f(np.array([t["parts"]["carry"]["all"] for t in run(seed)["templates"]])[:, block - 1, head]) for f in (np.min, np.max))
    assert spread("seed0", 7, 0) == [0.83, 0.95] and spread("seed1", 8, 0) == [0.6, 0.85]
    assert two(mean("seed0", lambda t: t["parts"][k]["all"])[3, 6] for k in ("carry", "break")) == [0.15, 0.08]


def test_two_parts_moved_together():
    pairs = lambda seed: {tuple(p["parts"]): (p["together"], sum(p["each"])) for p in run(seed)["pairs"]}
    close = lambda got, want: all(abs(g - w) < 0.006 for g, w in zip(got, want))
    # The telling pairs are those whose sum is below one.
    assert close(pairs("seed0")["block 4 head 2", "block 8 head 4"], (0.61, 0.93))
    assert close(pairs("seed1")["block 8 head 0", "block 8 MLP"], (0.78, 0.95))
    assert close(pairs("seed0")["block 7 head 0", "block 4 head 2"], (0.97, 1.39))  # more than the larger share, 0.87
    for seed, low, high in (("seed0", 0.65, 0.89), ("seed1", 0.81, 1.05)):
        ratio = [together / total for together, total in pairs(seed).values()]
        assert len(ratio) == 6 and (round(min(ratio), 2), round(max(ratio), 2)) == (low, high)


def test_the_first_mlp_makes_the_token():
    for seed, emb, after, mlp in (("seed0", 0.05, 0.85, 0.83), ("seed1", 0.05, 0.79, 0.78)):
        size = {k: round(float(np.mean([t["size at S"][k] for t in run(seed)["templates"]])), 2) for k in ("embedding", "after block 1", "MLP 1", "attention 1")}
        assert (size["embedding"], size["after block 1"], size["MLP 1"]) == (emb, after, mlp) and size["attention 1"] < 0.05
    for seed, total, apart in (("seed0", 3.85, 0.07), ("seed1", 3.05, 0.05)):
        carry, broken = (mean(seed, lambda t, k=k: t["parts"][k]["all"]) for k in ("carry", "break"))
        assert carry.size == 56 and round(float(carry.sum()), 2) == total  # the shares do not add up to one
        assert round(float(np.abs(carry - broken).max()), 2) == apart


def test_a_head_that_writes_nothing_directly_and_matters():
    for seed, block, head in (("seed0", 4, 2), ("seed1", 4, 4)):
        with open(f"runs/read/lens-{seed}.json") as f:
            direct = json.load(f)["attribution"]["clean"]["heads"]["mean"][block - 1][head]
        assert abs(direct) < 0.005 and part(seed, block, head)[0] > 0.2


def test_what_stands_in_for_a_part_changes_the_answer():
    change = lambda seed, block, head: two(mean(seed, lambda t, c=c: t["choice"][c])[block - 1, head] for c in CHOICES)
    assert tuple(run("seed0")["choices"]) == CHOICES
    assert change("seed0", 7, 0) == [-10.8, -5.27, -4.5, -5.22]
    assert change("seed0", 7, 6) == [-1.71, -0.9, -0.79, 0.66]  # the sign turns when nothing stands in
    assert change("seed0", 6, 6) == [-0.66, -0.29, -0.19, -2.31]
    assert change("seed0", 3, 4) == [0.01, 0.0, 0.01, -1.8]  # the same in every prompt: only "nothing" shows it
    assert change("seed1", 8, 0) == [-5.34, -2.85, -2.62, -2.64]
    assert change("seed1", 7, 1) == [-0.24, -0.17, -1.31, -1.28]  # the same whoever acts
    assert change("seed1", 2, 0) == [0.01, 0.01, -0.6, -0.47]
    assert change("seed1", 1, 1) == [-0.01, -0.01, -0.03, -1.09]
