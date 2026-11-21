"""
What docs/05-features.md claims, checked against the record under runs/sae/.
The dictionaries were trained on a GPU with a fixed seed. A new run on
another machine gives slightly different numbers and other latents, so
these tests hold for the committed record.
"""

import json

import numpy as np
import pytest


def run(seed):
    with open(f"runs/sae/sae-{seed}.json") as f:
        return json.load(f)


def pct(x):
    return round(100 * x)


@pytest.mark.parametrize("seed, missed, fixed, median, losses, recovered, rises", [
    ("seed0", 7.8, 57, 0.40, (1.489, 1.537, 7.646), 99.2, 3.2), ("seed1", 8.4, 56, 0.39, (1.490, 1.548, 7.437), 99.0, 3.9),
])
def test_judged_the_usual_way(seed, missed, fixed, median, losses, recovered, rises):
    r = run(seed)
    assert (r["training vectors"], r["width"], r["k"], r["block"]) == (1_024_000, 4096, 32, 4)
    b, c = r["rebuilds"], r["costs"]
    assert round(100 * b["unexplained variance"], 1) == missed and b["latents that never fire"] == 0
    assert pct(b["unexplained variance with the same number of fixed directions"]) == fixed
    assert round(100 * b["fires on share of tokens"]["median"], 2) == median and round(b["active per token"]) == 32
    assert tuple(round(c[k], 3) for k in ("loss", "loss with the rebuilt stream", "loss with the mean stream")) == losses
    assert round(100 * c["share of the loss recovered"], 1) == recovered and round(100 * c["loss rises by"], 1) == rises
    assert pct(r["one token"]["latents whose 30 strongest are one token in 80% or more"]) == 35
    assert len(r["fires on"]) == 16 and sum(b["log10 of how often each latent fires"]) == 4096


def test_the_sample_of_latents():
    sample = {x["latent"]: x for x in run("seed0")["fires on"]}
    shown = "share of all its firings on the token of its strongest example"
    for latent, share, token, of_all in ((1526, 1.81, " She", 15), (138, 1.08, " day", 63), (2028, 0.66, " upon", 44), (1878, 0.38, " day", 31), (2389, 0.17, " sunny", 18)):
        assert round(100 * sample[latent]["fires on share of tokens"], 2) == share and sample[latent]["examples"][0][1] == token
        assert pct(sample[latent][shown]) == of_all  # what the strongest examples can hide
    one_word = minority = 0
    for seed, one, mixed in (("seed0", 7, 6), ("seed1", 5, 6)):
        strongest = [x["share of its 30 strongest on one token"] for x in run(seed)["fires on"]]
        assert (sum(s >= 0.8 for s in strongest), sum(s < 0.4 for s in strongest)) == (one, mixed)
        one_word += one
        minority += sum(x[shown] < 0.5 for x in run(seed)["fires on"] if x["share of its 30 strongest on one token"] >= 0.8)
    assert (one_word, minority) == (12, 8)


@pytest.mark.parametrize("seed, acts, other, shared, by_firing, by_size, token_share, one_token", [
    ("seed0", ((97, 100), (0, 18)), ((47, 100), (5, 62)), 7, [], ["Tim", "Max", "Jack"], (1, 60), ["Lily", "Sue"]),
    ("seed1", ((93, 100), (0, 27)), ((46, 100), (7, 75)), 8, ["Max"], ["Max", "Jack"], (2, 42), ["Tim", "Mia", "Lucy"]),
])
def test_a_latent_for_the_name_that_acts_and_for_few_names_as_the_other(seed, acts, other, shared, by_firing, by_size, token_share, one_token):
    names = run(seed)["task"]["names"]
    span = lambda rows, k: (pct(min(x[k] for x in rows)), pct(max(x[k] for x in rows)))
    a, o = names["acts"], names["other"]
    assert (span(a, "fires when it is"), span(a, "fires when it is not")) == acts
    assert (span(o, "fires when it is"), span(o, "fires when it is not")) == other
    assert len({x["latent"] for x in a}) == 12 and len({x["latent"] for x in o}) == shared
    assert all(x["best separation by size"] >= 0.99 for x in a)
    assert [x["name"] for x in o if x["a latent that fires on 80% with and under 20% without"]] == by_firing
    assert [x["name"] for x in o if x["best separation by size"] >= 0.9] == by_size
    # On stories the latents for the acting name are not latents for the token alone.
    assert span(a, "of its story firings, on the name's token") == token_share
    assert [x["name"] for x in a if x["share of its 30 strongest on the name's token"] >= 0.8] == one_token


def test_one_latent_stands_for_three_names_as_the_other():
    rows = {x["name"]: x for x in run("seed0")["task"]["names"]["other"]}
    assert rows["Lily"]["latent"] == rows["Mia"]["latent"] == rows["Lucy"]["latent"]
    assert pct(rows["Lily"]["fires when it is not"]) == 62 and round(100 * rows["Lily"]["fires on share of story tokens"], 1) == 4.1


@pytest.mark.parametrize("seed, name, kind, variance, size, margin", [
    ("seed0", {"stream": (95, 90), "rebuilt": (42, 33), "missed": (92, 85), "averaged": (0, 0)},
     {"stream": (98, 96), "rebuilt": (87, 86), "missed": (97, 94), "averaged": (51, 49)}, (11, 29), 32, (61, 58)),
    ("seed1", {"stream": (80, 71), "rebuilt": (37, 32), "missed": (81, 69), "averaged": (0, 0)},
     {"stream": (90, 89), "rebuilt": (81, 83), "missed": (95, 92), "averaged": (47, 49)}, (12, 30), 32, (60, 54)),
])
def test_the_dictionary_keeps_the_kind_of_name_and_loses_much_of_which_name(seed, name, kind, variance, size, margin):
    t = run(seed)["task"]
    for k in ("stream", "rebuilt", "missed", "averaged"):
        assert (pct(t["reader"][k]["clean"]), pct(t["reader"][k]["flipped"])) == name[k]
        assert tuple(pct(v) for v in t["reader"][k]["a name of the right kind"]) == kind[k]
        assert np.array(t["reader"][k]["by pattern"]).shape == (5, 2)
    assert t["names that go on with She"] == ["Lily", "Mia", "Lucy", "Anna", "Sue", "Sara"]
    m = t["misses at the acting name"]
    assert (pct(m["of the variance around the stories' mean"]), pct(m["of the variance among these prompts"])) == variance and pct(m["of the stream's size"]) == size
    # The reader fitted on the stream: more than half of its margin for the right name comes from the missed part.
    assert tuple(pct(v) for v in t["reader fitted on the stream"]["from the missed part: share of the margin"]) == margin
    # The same reader and split as step 2, so the stream's score is step 2's.
    with open(f"runs/probe/probe-{seed}.json") as f:
        step2 = np.array(json.load(f)["new"]["accuracy"])[:, :, 4, 2, 0].mean(0)  # after block 4, at the acting name, the due name
    assert np.allclose(step2, [t["reader"]["stream"]["clean"], t["reader"]["stream"]["flipped"]], atol=0.02)


@pytest.mark.parametrize("seed, head, plain, rebuilt, out, moves, adds, largest, everywhere, right", [
    ("seed0", "block 7 head 0", (7.90, -3.93), (7.96, -4.01), (7.35, -3.21), ((0.37, 0.38), (0.65, 1.22)),
     ((0.42, 0.52), (0.35, 0.42), (-0.07, 0.0)), ((16, 16), (13, 13), (8, 9)), (7.31, -5.51), ((100, 85), (98, 92))),
    ("seed1", "block 8 head 0", (5.25, -2.24), (5.19, -2.25), (4.64, -1.60), ((0.24, 0.24), (0.61, 0.69)),
     ((0.39, 0.44), (0.31, 0.35), (0.03, 0.06)), ((17, 16), (15, 13), (10, 10)), (5.38, -2.59), ((95, 76), (96, 78))),
])
def test_the_model_loses_less_than_the_reader(seed, head, plain, rebuilt, out, moves, adds, largest, everywhere, right):
    t = run(seed)["task"]
    assert t["veto head"] == head
    ways = ("plain", "rebuilt at the acting name", "the other name averaged out at the acting name")
    get = lambda where, k, digits=2: tuple(round(t["effects"][where][v][k]["mean"] + 0.0, digits) for v in ("clean", "flipped"))
    assert tuple(get(w, "logit difference") for w in ways) == (plain, rebuilt, out)
    assert tuple(get(w, "change per prompt") for w in ways[1:]) == moves
    assert tuple(get(w, "veto head adds to the other name") for w in ways) == adds
    share = lambda w: tuple(pct(t["effects"][w][v]["other name largest in what the veto head writes"]["mean"]) for v in ("clean", "flipped"))
    assert tuple(share(w) for w in ways) == largest  # the head is not a reader of the name
    kept = [b / a for a, b in zip(adds[0], adds[1])]
    assert 0.78 <= min(kept) and max(kept) <= 0.84  # four fifths of what it added
    assert all(abs(a - b) < 0.1 for a, b in zip(plain, rebuilt)) and all(0.5 < abs(a - b) < 0.75 for a, b in zip(plain, out))
    assert get("rebuilt everywhere", "logit difference") == everywhere
    side = lambda w: tuple(pct(t["effects"][w][v]["right side"]["mean"]) for v in ("clean", "flipped"))
    assert (side("plain"), side("rebuilt everywhere")) == right
    lean = lambda w: sum(t["effects"][w][v]["logit difference"]["mean"] for v in ("clean", "flipped")) / 2
    assert (round(lean("plain"), 2), round(lean("rebuilt everywhere"), 2)) == {"seed0": (1.98, 0.90), "seed1": (1.50, 1.40)}[seed]
    assert round(lean("plain") - lean("rebuilt everywhere"), 2) == {"seed0": 1.09, "seed1": 0.11}[seed]
