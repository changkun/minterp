"""
What docs/06-feature-circuit.md claims, checked against the record under
runs/features/. The record rests on the dictionaries of step 5, so it holds
for those: another training run gives other latents.
"""

import json

import numpy as np
import pytest

SHE = ["Lily", "Mia", "Lucy", "Anna", "Sue", "Sara"]


def run(seed):
    with open(f"runs/features/features-{seed}.json") as f:
        return json.load(f)


def mean(seed, pick):
    return float(np.mean([pick(t) for t in run(seed)["templates"]]))


def two(x, digits=2):
    return round(float(x) + 0.0, digits)


@pytest.mark.parametrize("seed, who, second, first, estimated, spread, differ", [
    ("seed0", (0.94, 0.06, 0.97), (0.075, 0.025, 0.099), (0.114, 0.030, 0.144), (0.57, 0.04), (0.90, 0.97), (48, 16, 16, 16)),
    ("seed1", (1.00, 0.03, 1.04), (0.071, 0.027, 0.097), (0.085, 0.033, 0.118), (0.82, 0.03), (0.98, 1.02), (47, 15, 15, 17)),
])
def test_what_the_dictionary_carries(seed, who, second, first, estimated, spread, differ):
    parts = ("rebuilt", "missed", "both")
    assert tuple(two(mean(seed, lambda t, k=k: t["parts"][k])) for k in parts) == who
    assert tuple(two(mean(seed, lambda t, k=k: t["name parts"]["second name"][k]), 3) for k in parts) == second
    assert tuple(two(mean(seed, lambda t, k=k: t["name parts"]["first name"][k]), 3) for k in parts) == first
    assert tuple(two(mean(seed, lambda t, k=k: t["estimate"][k])) for k in ("rebuilt", "missed")) == estimated
    per_pattern = [t["parts"]["rebuilt"] for t in run(seed)["templates"]]
    assert (two(min(per_pattern)), two(max(per_pattern))) == spread
    keys = ("latents that differ", "on in the clean prompt only", "on in the flipped prompt only", "on in both")
    assert tuple(round(mean(seed, lambda t, k=k: t["estimate"][k])) for k in keys) == differ
    # "Both" is the stream, so it is step 3's share at this place: who acts, and each name.
    with open(f"runs/patch/patch-{seed}.json") as f:
        step3 = json.load(f)["templates"]
    for mine, theirs in ((lambda t: t["parts"]["both"], lambda t: t["travel"]["carry"][4][3]),
                         (lambda t: t["name parts"]["second name"]["both"], lambda t: t["name"][4][2]),
                         (lambda t: t["name parts"]["first name"]["both"], lambda t: t["first name"][4][2])):
        assert abs(np.mean([theirs(t) for t in step3]) - mean(seed, mine)) < 0.01
    # About three quarters of a name's share here is in what the dictionary rebuilds.
    for label in ("second name", "first name"):
        share = mean(seed, lambda t: t["name parts"][label]["rebuilt"]) / mean(seed, lambda t: t["name parts"][label]["both"])
        assert 0.7 <= share <= 0.8


@pytest.mark.parametrize("seed, real, estimated, corr, sign, first", [
    ("seed0", [0.17, 0.30, 0.48, 0.69, 0.84, 0.91], [0.13, 0.23, 0.36, 0.48, 0.54, 0.57], 0.95, 99.9, (2.03, 1.59)),
    ("seed1", [0.27, 0.46, 0.68, 0.85, 0.95, 0.98], [0.25, 0.40, 0.58, 0.72, 0.79, 0.82], 0.99, 100.0, (2.02, 1.81)),
])
def test_not_a_few_latents_and_an_estimate_that_ranks_well_and_counts_low(seed, real, estimated, corr, sign, first):
    assert run(seed)["counts"] == [1, 2, 4, 8, 16, 32]
    assert [two(mean(seed, lambda t, i=i: t["by count"]["real"][i])) for i in range(6)] == real
    assert [two(mean(seed, lambda t, i=i: t["by count"]["estimated"][i])) for i in range(6)] == estimated
    assert two(mean(seed, lambda t: t["one latent"]["correlation"])) == corr
    assert round(100 * mean(seed, lambda t: t["one latent"]["same sign"]), 1) == sign
    assert (two(mean(seed, lambda t: t["one latent"]["real"][0])), two(mean(seed, lambda t: t["one latent"]["estimated"][0]))) == first


@pytest.mark.parametrize("seed, step5, head, gaps, alone, unstable", [
    ("seed0", (0.05, 0.03, 0.09, 0.05), (0.23, -0.01, 0.20, 0.27), (4.95, 9.94, 15.49), ["Tom"], {"Ben": 60, "Jack": 50}),
    ("seed1", (0.19, 0.10, 0.40, 0.14), (0.43, 0.02, 0.41, 0.45), (1.05, 6.26, 10.69), ["Tim", "Sam", "Max"], {"Max": 69, "Jack": 55}),
])
def test_a_latent_for_the_name_twice(seed, step5, head, gaps, alone, unstable):
    kinds = ("both She", "both He", "one of each")

    def shares(choice):
        return (two(mean(seed, lambda t: t["names"][choice]["moved for real"])),) + tuple(two(mean(seed, lambda t, k=k: t["names"][choice]["by pair"][k]["moved for real"])) for k in kinds)

    assert (shares("step 5's latents"), shares("the head's latents")) == (step5, head)
    assert tuple(two(mean(seed, lambda t, k=k: t["pairs"][k]["gap"])) for k in kinds) == gaps
    assert [run(seed)["templates"][0]["pairs"][k]["prompts"] for k in kinds] == [30, 30, 72]
    names = {x["name"]: x for x in run(seed)["names"]}
    assert run(seed)["names that go on with She"] == SHE
    # One latent for the six names the model follows with "She", and it fires for all six.
    assert len({names[n]["the head's latent"] for n in SHE}) == 1 == len(run(seed)["one latent for the six"])
    assert names["Lily"]["names for which the head's latent fires on 90% or more"] == SHE
    # Six different latents for the other six, few of them for one name alone.
    he = [n for n in names if n not in SHE]
    assert len({names[n]["the head's latent"] for n in he}) == 6
    assert [n for n in he if names[n]["names for which the head's latent fires on 90% or more"] == [n]] == alone
    # Found on four patterns, it holds on the fifth as often as overall.
    left_out = {n: round(100 * x["on a pattern left out"]) for n, x in names.items()}
    assert {n: v for n, v in left_out.items() if v < 86} == unstable and all(v >= 86 for n, v in left_out.items() if n not in unstable)
    assert all(abs(x["on a pattern left out"] - x["the head rests on it most in"]) < 0.01 for x in names.values())


@pytest.mark.parametrize("seed, she, he, off", [
    ("seed0", (-1.01, -0.26, -1.40), (-1.75, -1.12, -1.19), (0.68, 0.59, -0.03, 1.30)),
    ("seed1", (-1.20, -0.23, -1.80), (-1.72, -1.16, -1.07), (0.77, 0.72, -0.02, 1.44)),
])
def test_name_or_kind(seed, she, he, off):
    keys = ("against the nine", "against its own kind", "its own kind against the other")
    get = lambda kind: tuple(two(mean(seed, lambda t, k=k: t["head"]["by kind of the acting name"][kind][k])) for k in keys)
    assert (get("She"), get("He")) == (she, he)
    block = "the shared latent switched off, where a She name acts"
    keys = ("the acting name against the nine", "estimated", "the acting name against its own kind", "its own kind against the other")
    assert tuple(two(mean(seed, lambda t, k=k: t["head"][block][k])) for k in keys) == off


@pytest.mark.parametrize("seed, head, acting, other, own, net", [
    ("seed0", "block 7 head 0", (-1.38, -0.72, -1.25, -0.63, -0.74), (0.42, 0.16, 0.74, -0.30, 1.02), 91, (-1.50, 0.78)),
    ("seed1", "block 8 head 0", (-1.46, -1.02, -1.20, -0.76, -0.81), (0.39, 0.18, 0.88, -0.43, 1.07), 94, (-1.55, 0.54)),
])
def test_what_the_veto_head_rests_on(seed, head, acting, other, own, net):
    assert run(seed)["veto head"] == head
    h = lambda *keys: two(mean(seed, lambda t: __import__("functools").reduce(lambda d, k: d[k], keys, t["head"])))
    assert (h("adds to the acting name"), h("estimated to rest on", "the latents at the acting name"), h("really lost when switched off", "all latents at the acting name"),
            h("estimated to rest on", "its highest latent"), h("really lost when switched off", "its highest latent")) == acting
    o = "the other name"
    assert (h(o, "adds"), h(o, "estimated to rest on the latents at the acting name"), h(o, "really lost when all latents there are switched off"),
            h(o, "adds, same kind"), h(o, "adds, other kind")) == other
    assert round(100 * h("its highest latent is the head's latent for the name")) == own
    lost = mean(seed, lambda t: t["head"]["really lost when switched off"]["all latents at the acting name"]) / mean(seed, lambda t: t["head"]["adds to the acting name"])
    assert round(100 * lost) == {"seed0": 91, "seed1": 83}[seed]  # most of what the head adds to the acting name
    assert (h("estimated to rest on", "the latents below zero"), h("estimated to rest on", "the latents above zero")) == net
