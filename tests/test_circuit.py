"""
What docs/04-circuit.md claims, checked against the record under runs/circuit/.
"""

import json

import numpy as np
import pytest

VERSIONS = ("clean", "flipped", "neutral")


def run(seed):
    with open(f"runs/circuit/circuit-{seed}.json") as f:
        return json.load(f)


def avg(seed, pick):
    return float(np.mean([pick(t) for t in run(seed)["templates"]]))


def two(values):
    return [round(v + 0.0, 2) for v in values]


def head(name):
    block, h = name.split(".")
    return f"block {block} head {h}"


def reads(seed, name, version, places=("A", "B", "S")):
    return two(avg(seed, lambda t, k=k: t["attention"][head(name)][version][k]) for k in places)


def source(seed, name, version, place):
    """What a head adds to A and to B from what it read at one place."""
    return two(avg(seed, lambda t, m=m: t["sources"][head(name)][version][place][m]) for m in ("A", "B"))


def writes(seed, name, version):
    return two(avg(seed, lambda t, k=k: t["writes"][head(name)][version][k]) for k in ("A", "B", "C"))


def test_the_heads_come_from_the_earlier_steps():
    chosen = lambda seed: [(f"{x['block']}.{x['head']}", x["acts at"]) for x in run(seed)["heads"]]
    assert chosen("seed0") == [("3.0", "end"), ("3.2", "end"), ("4.2", "S"), ("7.0", "end"), ("7.1", "end"), ("8.3", "end"), ("8.4", "end")]
    assert chosen("seed1") == [("2.0", "end"), ("4.4", "S"), ("6.0", "end"), ("7.1", "end"), ("8.0", "end"), ("8.1", "end"), ("8.5", "end")]
    for seed in ("seed0", "seed1"):
        with open(f"runs/patch/patch-{seed}.json") as f:
            carry = np.mean([t["parts"]["carry"]["all"] for t in json.load(f)["templates"]], axis=0)[:, :6]
        with open(f"runs/read/lens-{seed}.json") as f:
            a = json.load(f)["attribution"]
        wrote = np.max([np.abs(a[v]["heads"]["mean"]) for v in VERSIONS], axis=0)
        rule = {(b + 1, h) for b in range(8) for h in range(6) if carry[b, h] >= 0.05 or wrote[b, h] >= 0.3}
        assert rule == {(x["block"], x["head"]) for x in run(seed)["heads"]}


def test_where_the_heads_read_from():
    # at the acting name: the other name
    assert reads("seed0", "4.2", "clean")[:2] == [0.01, 0.91] and reads("seed0", "4.2", "flipped")[:2] == [0.78, 0.07]
    assert reads("seed1", "4.4", "clean")[:2] == [0.05, 0.85] and reads("seed1", "4.4", "flipped")[:2] == [0.79, 0.1]
    # at the last position: the acting name's position
    assert reads("seed0", "7.0", "clean") == [0.02, 0.08, 0.79] and reads("seed0", "7.0", "flipped") == [0.02, 0.06, 0.83]
    assert reads("seed0", "7.1", "clean")[2] == 0.86 and reads("seed0", "3.2", "clean") == [0.13, 0.1, 0.36]
    assert reads("seed1", "8.0", "clean") == [0.01, 0.05, 0.93] and reads("seed1", "8.0", "flipped") == [0.01, 0.03, 0.96]
    assert reads("seed1", "8.1", "clean")[2] == 0.91 and reads("seed1", "6.0", "clean")[2] == 0.6
    # at the last position: a name
    assert reads("seed0", "8.4", "clean") == [0.04, 0.63, 0.01] and reads("seed0", "8.4", "flipped") == [0.31, 0.13, 0.01]
    assert reads("seed0", "8.3", "clean") == [0.03, 0.67, 0.02] and reads("seed0", "8.3", "flipped") == [0.22, 0.23, 0.06]
    assert reads("seed0", "3.0", "clean") == [0.16, 0.29, 0.1]
    assert reads("seed1", "7.1", "clean") == [0.04, 0.86, 0.01] and reads("seed1", "7.1", "flipped") == [0.09, 0.74, 0.02]
    assert reads("seed1", "2.0", "clean") == [0.08, 0.32, 0.02]
    assert reads("seed1", "8.5", "clean") == [0.0, 0.0, 0.0] and reads("seed1", "8.5", "flipped") == [0.0, 0.0, 0.0]
    # two heads read the full stop that ends the first sentence
    stop = lambda seed, name: [reads(seed, name, v, ("stop",))[0] for v in ("clean", "flipped")]
    assert stop("seed1", "8.5") == [0.89, 0.92] and stop("seed0", "8.4") == [0.21, 0.39]
    assert all(stop(seed, name)[0] <= 0.09 for seed, name in (("seed0", "8.3"), ("seed0", "7.0"), ("seed1", "7.1"), ("seed1", "2.0")))
    # on neutral prompts the fetching heads split their attention between the two names
    assert reads("seed0", "4.2", "neutral")[:2] == [0.38, 0.58] and reads("seed1", "4.4", "neutral")[:2] == [0.52, 0.43]


def test_what_the_heads_write():
    # a head that reads the acting name's position lowers whichever name acts
    assert writes("seed0", "7.0", "clean")[:2] == [-1.38, 0.42] and writes("seed0", "7.0", "flipped")[:2] == [0.52, -1.56]
    assert writes("seed0", "7.0", "neutral")[2] == -1.7 and writes("seed0", "3.2", "clean")[:2] == [-0.5, -0.1]
    assert writes("seed1", "8.0", "clean")[:2] == [-1.46, 0.39] and writes("seed1", "8.0", "flipped")[:2] == [0.44, -1.51]
    assert writes("seed1", "8.0", "neutral")[2] == -1.65
    # a head that reads a name raises it
    assert writes("seed0", "8.4", "clean")[:2] == [0.97, 3.54] and writes("seed0", "8.4", "flipped")[:2] == [2.8, 1.57]
    assert writes("seed0", "8.3", "clean")[:2] == [0.12, 1.21] and writes("seed0", "8.3", "flipped")[:2] == [0.45, 0.51]
    assert writes("seed1", "7.1", "clean")[:2] == [0.07, 1.14] and writes("seed1", "7.1", "flipped")[:2] == [0.16, 1.03]
    assert writes("seed1", "2.0", "clean")[:2] == [0.17, 0.55]
    # one head raises both names of the pair, whoever acts, and all of it comes from the full stop
    assert writes("seed1", "8.5", "clean")[:2] == [2.54, 2.01] and writes("seed1", "8.5", "flipped")[:2] == [2.65, 2.09]
    assert source("seed1", "8.5", "clean", "stop") == [2.55, 2.03]
    # the weaker heads that read the acting name's position lower the acting name too
    for seed, name in (("seed0", "3.2"), ("seed0", "7.1"), ("seed1", "8.1"), ("seed1", "6.0")):
        acting = [writes(seed, name, "clean")[0], writes(seed, name, "flipped")[1], writes(seed, name, "neutral")[2]]
        assert all(-0.5 <= v < 0 for v in acting)
    assert writes("seed0", "3.0", "clean")[:2] == [0.46, 0.53]
    # where a copying head's write comes from
    assert source("seed0", "8.4", "clean", "B")[1] == 3.01
    assert source("seed0", "8.4", "flipped", "A")[0] == 1.48 and source("seed0", "8.4", "flipped", "stop") == [1.25, 0.91]
    assert source("seed1", "7.1", "clean", "B")[1] == 1.14 and source("seed1", "7.1", "flipped", "B")[1] == 1.0
    # what a veto head adds to the other name is read at the acting name's position
    assert source("seed0", "7.0", "clean", "S") == [-1.34, 0.52] and source("seed1", "8.0", "clean", "S") == [-1.43, 0.44]


def hold(seed, sender, receiver):
    """An earlier head's share alone, with a later head held, and relayed by the later head alone."""
    T = run(seed)["templates"]
    i = next(i for i, x in enumerate(T[0]["holds"]) if (x["sender"], x["receiver"]) == (head(sender), head(receiver)))
    return two(np.mean([t["holds"][i][k] for t in T]) for k in ("alone", "held", "relayed"))


def steer(seed, reader, sender, which, places=("A", "B")):
    T = run(seed)["templates"]
    i = next(i for i, x in enumerate(T[0]["steers"]) if (x["reader"], x["sender"]) == (head(reader), head(sender)))
    return two(np.mean([t["steers"][i][which][k] for t in T]) for k in places)


def test_which_works_through_which():
    assert hold("seed0", "4.2", "7.0") == [0.52, 0.06, 0.42] and hold("seed0", "4.2", "8.4") == [0.52, 0.22, 0.32]
    assert hold("seed0", "7.0", "8.4") == [0.87, 0.57, 0.31] and hold("seed0", "7.0", "8.3") == [0.87, 0.77, 0.1]
    assert hold("seed1", "4.4", "8.0") == [0.21, 0.08, 0.12] and hold("seed0", "4.2", "8.3") == [0.52, 0.42, 0.11]
    changed = 0
    for seed in ("seed0", "seed1"):
        T = run(seed)["templates"]
        changed += sum(abs(np.mean([t["holds"][i]["alone"] - t["holds"][i]["held"] for t in T])) >= 0.03 for i in range(len(T[0]["holds"])))
    assert changed == 11
    lost = lambda receiver: round(float(np.mean([next(x["alone"] - x["held"] for x in t["holds"] if (x["sender"], x["receiver"]) == (head("4.2"), head(receiver))) for t in run("seed0")["templates"]])), 2)
    assert (lost("7.0"), lost("8.4")) == (0.47, 0.31)  # more than 0.52 together: one route runs through both
    # what passes from the fetching head to the veto head: the veto stays, the other name's share goes
    assert steer("seed0", "7.0", "4.2", "flipped", ("S",)) == [0.83] and steer("seed0", "7.0", "4.2", "patched", ("S",)) == [0.81]
    assert writes("seed0", "7.0", "flipped")[:2] == [0.52, -1.56] and steer("seed0", "7.0", "4.2", "patched writes") == [0.09, -1.04]
    assert writes("seed1", "8.0", "flipped")[:2] == [0.44, -1.51] and steer("seed1", "8.0", "4.4", "patched writes") == [0.1, -1.18]
    # the veto head's output moves where the copying heads look
    three = ("A", "B", "stop")
    assert steer("seed0", "8.4", "7.0", "flipped", three) == [0.31, 0.13, 0.39] and steer("seed0", "8.4", "7.0", "patched", three) == [0.05, 0.46, 0.33]
    assert steer("seed0", "8.4", "4.2", "patched", three) == [0.08, 0.51, 0.29] and steer("seed0", "8.4", "7.0", "clean", three) == [0.04, 0.63, 0.21]
    assert steer("seed0", "8.3", "7.0", "flipped") == [0.22, 0.23] and steer("seed0", "8.3", "7.0", "patched") == [0.03, 0.54]
    assert steer("seed0", "8.3", "4.2", "patched") == [0.05, 0.63] and steer("seed0", "8.3", "7.0", "clean") == [0.03, 0.67]
    assert steer("seed0", "8.4", "7.0", "patched writes")[1] == 3.08
    # in seed1 nothing is moved much
    T = run("seed1")["templates"]
    shifts = [abs(np.mean([t["steers"][i]["patched"][k] - t["steers"][i]["flipped"][k] for t in T])) for i in range(len(T[0]["steers"])) for k in ("A", "B", "stop", "S")]
    assert round(max(shifts), 2) == 0.12
    assert steer("seed1", "7.1", "4.4", "flipped")[1] == 0.74 and steer("seed1", "7.1", "4.4", "patched")[1] == 0.86


@pytest.mark.parametrize("seed, mine, others, median, reach, low, high", [
    ("seed0", 1.07, -0.03, 0.13, 65, 1.03, 1.10), ("seed1", 0.97, 0.11, 0.07, 23, 0.94, 1.04),
])
def test_the_set_against_random_sets(seed, mine, others, median, reach, low, high):
    T = run(seed)["templates"]
    assert len(run(seed)["heads"]) == 7
    assert all(abs(t["sets"]["all heads"] - 1) < 5e-4 for t in T)  # only attention moves information between positions
    assert round(avg(seed, lambda t: t["sets"]["the heads"]), 2) == mine
    assert round(avg(seed, lambda t: t["sets"]["all other heads"]) + 0.0, 2) == others
    per_pattern = [t["sets"]["the heads"] for t in T]
    assert (round(min(per_pattern), 2), round(max(per_pattern), 2)) == (low, high)
    random = np.mean([t["sets"]["random"] for t in T], axis=0)
    assert len(random) == 200 and round(float(np.median(random)), 2) == median and int((random >= 0.5).sum()) == reach
    # Which random sets reach half the gap: those that happen to hold a head that carries a great deal alone.
    drawn = run(seed)["random sets"]
    large = {"seed0": ("7.0", "4.2", "8.4"), "seed1": ("8.0",)}[seed]
    holds_one = [any(head(h) in s for h in large) for s in drawn]
    assert all(h for h, v in zip(holds_one, random) if v >= 0.5)
    if seed == "seed1":
        assert all(v >= 0.5 for h, v in zip(holds_one, random) if h)  # exactly the sets with head 8.0
    mine = {f"block {x['block']} head {x['head']}" for x in run(seed)["heads"]}
    assert round(max(v for s, v in zip(drawn, random) if not mine & set(s)), 2) == {"seed0": 0.02, "seed1": 0.08}[seed]
    assert round(avg(seed, lambda t: t["sets"]["the heads that carry"]), 2) == {"seed0": 1.09, "seed1": 0.98}[seed]
