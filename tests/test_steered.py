"""
What docs/07-steer.md claims, checked against the record under runs/steer/.
The record rests on the dictionaries of step 5 and the latents of step 6.
"""

import json

import numpy as np
import pytest

KINDS = ("both She", "both He", "one of each")
OUTSIDE = "to the flipped prompt, outside the two names"
BACK = "the shared latent as the dictionary reads it, against its usual value"
NARROW, WIDE = "the two pronouns and the names", "every form of the pronouns and the names"


def run(seed):
    with open(f"runs/steer/steer-{seed}.json") as f:
        return json.load(f)


def who(seed, pick):
    return float(np.mean([pick(t) for t in run(seed)["who acts"]["templates"]]))


def kind(seed, pick):
    return float(np.mean([pick(t) for t in run(seed)["kind"]["templates"]]))


def at(seed, dose):
    return run(seed)["doses"].index(dose)


def two(x, digits=2):
    return round(float(x) + 0.0, digits)


def steered(seed, name, measure, dose=1.0):
    i = at(seed, dose)
    return who(seed, lambda t: t["steered"][name][measure][i])


def above(seed, name, dose=1.0):
    """The share of prompts with A above B, in percent."""
    return round(100 * steered(seed, name, "turned", dose))


def logits_by_pair(seed, name, group):
    i = at(seed, 1.0)
    return who(seed, lambda t: t["steered"][name]["moved, in logits, by pair"][group][i])


def by_pair(seed, name, group):
    return logits_by_pair(seed, name, group) / who(seed, lambda t: t["pairs"][group]["gap"])


def length(seed, name):
    return who(seed, lambda t: t["length"][name]["all"])


@pytest.mark.parametrize("seed, size, long, wipe, third, flipped, clean", [
    ("seed0", 29.4, (0.23, 0.54, 0.54), 0.38, 0.51, 85, 1),
    ("seed1", 31.9, (0.14, 0.45, 0.45), 0.50, 0.53, 76, 30),
])
def test_what_damage_alone_does(seed, size, long, wipe, third, flipped, clean):
    assert run(seed)["doses"] == [-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    assert two(who(seed, lambda t: t["stream size at the acting name"]), 1) == size
    # The main directions are 20 to 25 long, and the long random control has the length of the pair direction.
    assert all(19.5 <= length(seed, name) <= 25.5 for name in ("own", "pair", "names", "means + order", "means"))
    assert (two(length(seed, "order"), 1), two(length(seed, "latents"), 1)) == {"seed0": (11.1, 10.2), "seed1": (8.0, 11.1)}[seed]
    assert above(seed, "wipe") == {"seed0": 4, "seed1": 23}[seed]
    assert abs(length(seed, "random, long") - length(seed, "pair")) < 1e-3 and abs(length(seed, "random") - length(seed, "means")) < 1e-3
    assert tuple(two(steered(seed, "random, long", "share", d)) for d in (1.0, 2.0, -2.0)) == long
    assert two(steered(seed, "wipe", "share")) == wipe
    assert two(who(seed, lambda t: t["plain"]["a third name acts, share"])) == third
    assert round(100 * who(seed, lambda t: t["plain"]["the flipped prompt turned"])) == flipped
    assert sum(round(132 * t["plain"]["the clean prompt turned"]) for t in run(seed)["who acts"]["templates"]) == clean
    # A dose of nothing changes nothing.
    for name in run(seed)["who acts"]["directions"]:
        assert abs(steered(seed, name, "share", 0.0)) < 1e-6


@pytest.mark.parametrize("seed, table", [
    ("seed0", {"own": (0.96, 81), "pair": (0.95, 81), "names": (0.96, 86), "means + order": (0.81, 78), "means": (0.38, 9), "order": (0.40, 18), "reader": (0.33, 4),
               "latents": (0.19, 0), "kind": (0.24, 3), "other names": (0.31, 5), "random": (0.15, 1), "random, long": (0.23, 3), "means + random": (0.41, 9)}),
    ("seed1", {"own": (1.04, 78), "pair": (1.04, 78), "names": (1.04, 78), "means + order": (0.85, 67), "means": (0.81, 64), "order": (0.07, 5), "reader": (0.68, 50),
               "latents": (0.43, 19), "kind": (0.44, 25), "other names": (0.48, 30), "random": (0.10, 5), "random, long": (0.14, 5), "means + random": (0.81, 64)}),
])
def test_who_acts_at_dose_one(seed, table):
    assert list(run(seed)["who acts"]["directions"]) == list(table) + ["wipe"]
    assert {name: (two(steered(seed, name, "share")), above(seed, name)) for name in table} == table
    # A patch of step 3: the prompt's own difference at dose 1 is the stream swapped, which step 6 recorded as "both" for the mirror case.
    with open(f"runs/features/features-{seed}.json") as f:
        swapped = np.mean([t["parts"]["both"] for t in json.load(f)["templates"]])
    assert abs(steered(seed, "own", "share") - swapped) < 0.02


@pytest.mark.parametrize("seed, outside, apart, curve, turned, far, back, long", [
    ("seed0", (0.02, 1.55), (0.25, 0.73, 2.44), (0.14, 0.42, 0.74), (9, 65), 1.13, (-0.04, 0.08, 0.40), 0.54),
    ("seed1", (0.01, 1.64), (0.10, 0.58, 2.73), (0.22, 0.51, 0.80), (23, 63), 1.43, (-0.21, -0.15, 0.25), 0.45),
])
def test_the_pair_direction_and_its_dose(seed, outside, apart, curve, turned, far, back, long):
    assert (two(steered(seed, "pair", OUTSIDE)), two(steered(seed, "pair", OUTSIDE, 0.0))) == outside
    assert two(who(seed, lambda t: t["plain"]["from the clean prompt to the flipped, outside the two names"])) == outside[1]
    assert tuple(two(who(seed, lambda t, g=g: t["plain"]["the same, by pair"][g])) for g in KINDS) == apart
    assert 0.10 <= min(apart[:2]) and max(apart[:2]) <= 0.73
    assert tuple(two(steered(seed, "pair", "share", d)) for d in (0.25, 0.5, 0.75)) == curve
    assert (above(seed, "pair", 0.5), above(seed, "pair", 0.75)) == turned
    # The count jumps where the answer passes half-way: the largest step is the one from half to three quarters.
    doses = run(seed)["doses"]
    steps = [steered(seed, "pair", "turned", b) - steered(seed, "pair", "turned", a) for a, b in zip(doses, doses[1:])]
    assert int(np.argmax(steps)) == doses.index(0.5)
    assert two(steered(seed, "pair", "share", 2.0)) == far
    assert tuple(two(steered(seed, "pair", "share", d)) for d in (-0.5, -1.0, -2.0)) == back
    # At dose -2 the pair direction moves the answer less than a random direction of its length.
    assert two(steered(seed, "random, long", "share", -2.0)) == long > back[2]


@pytest.mark.parametrize("seed, names, order, logits, summed, noise, beyond, reader", [
    ("seed0", (0.96, 86, 0.04), (11.1, 0.40), (4.4, 5.4), (0.81, 78), 0.41, 0.15, 0.91),
    ("seed1", (1.04, 78, 0.02), (8.0, 0.07), (0.1, 0.7), (0.85, 67), 0.81, 0.19, 0.90),
])
def test_what_the_pair_direction_is_made_of(seed, names, order, logits, summed, noise, beyond, reader):
    assert (two(steered(seed, "names", "share")), above(seed, "names"), two(steered(seed, "names", OUTSIDE))) == names
    assert (two(length(seed, "order"), 1), two(steered(seed, "order", "share"))) == order
    moved = [logits_by_pair(seed, "order", g) for g in KINDS]
    assert (two(min(moved), 1), two(max(moved), 1)) == logits
    assert (two(steered(seed, "means + order", "share")), above(seed, "means + order")) == summed
    assert two(steered(seed, "means + random", "share")) == noise
    assert two(two(steered(seed, "names", "share")) - two(steered(seed, "means + order", "share"))) == beyond
    # In seed0 the names' means alone take the answer as far as the wipe does, though not in the same pairs.
    if seed == "seed0":
        assert two(steered(seed, "means", "share")) == two(steered(seed, "wipe", "share")) == 0.38
        assert (two(by_pair(seed, "wipe", "both She")), two(by_pair(seed, "means", "both She"))) == (0.42, 0.11)
    # A random vector of order's length adds 0.02 and nothing.
    assert two(steered(seed, "means + random", "share") - steered(seed, "means", "share")) == {"seed0": 0.02, "seed1": 0.0}[seed]
    assert two(who(seed, lambda t: t["cosine with means"]["reader"])) == reader
    # The reader's direction has the length of means and does a little less.
    assert abs(length(seed, "reader") - length(seed, "means")) < 1e-3
    assert steered(seed, "reader", "share") < steered(seed, "means", "share") and steered(seed, "reader", "turned") < steered(seed, "means", "turned")


@pytest.mark.parametrize("seed, pairs, own", [
    ("seed0", {"pair": (0.87, 0.96, 0.96), "means": (0.11, 0.31, 0.44), "order": (1.09, 0.47, 0.28), "other names": (0.14, 0.13, 0.36), "kind": (0.0, 0.0, 0.33), "random": (0.19, 0.11, 0.13)},
     {"pair": (0.94, 0.94), "names": (0.94, 0.94), "means + order": (0.83, 0.79), "means": (0.32, 0.25), "order": (0.45, 0.51), "latents": (0.16, 0.09), "kind": (0.08, 0.06),
      "other names": (0.16, 0.21), "random": (0.13, 0.17), "wipe": (0.48, 0.23)}),
    ("seed1", {"pair": (1.15, 1.03, 1.03), "means": (0.90, 0.73, 0.82), "order": (0.13, 0.07, 0.07), "other names": (0.18, 0.13, 0.59), "kind": (0.0, 0.0, 0.57), "random": (0.09, 0.11, 0.10)},
     {"pair": (1.04, 1.04), "names": (1.04, 1.04), "means + order": (0.89, 0.78), "means": (0.74, 0.79), "order": (0.14, -0.02), "latents": (0.39, 0.35), "kind": (0.08, -0.02),
      "other names": (0.17, 0.20), "random": (0.11, 0.11), "wipe": (0.80, 0.12)}),
])
def test_the_controls_find_the_kind(seed, pairs, own):
    assert run(seed)["who acts"]["pairs"] == {"both She": 30, "both He": 30, "one of each": 72}
    assert {name: tuple(two(by_pair(seed, name, g)) for g in KINDS) for name in pairs} == pairs
    assert {name: tuple(two(steered(seed, name, f"{x} against its own kind")) for x in "AB") for name in own} == own
    # Where the two names are of one kind, the control built from two other names is about at random's level.
    for g in KINDS[:2]:
        assert 0.13 <= two(by_pair(seed, "other names", g)) <= 0.18 and 0.09 <= two(by_pair(seed, "random", g)) <= 0.19
    # On the 72 pairs of two kinds, where the kind direction is not zero: the kind direction and the random one.
    mixed = lambda name: tuple(two(who(seed, lambda t, x=x: t["steered"][name][f"{x} against its own kind, by pair"]["one of each"][at(seed, 1.0)])) for x in "AB")
    assert (mixed("kind"), mixed("random")) == {"seed0": ((0.13, 0.09), (0.13, 0.17)), "seed1": ((0.13, -0.04), (0.11, 0.11))}[seed]
    assert who(seed, lambda t: t["length"]["kind"]["both She"]) == 0 == who(seed, lambda t: t["length"]["kind"]["both He"])
    # On B, the wipe and the three controls of the table reach at most 0.23 in seed0 and 0.20 in seed1.
    assert max(own[name][1] for name in ("kind", "other names", "random", "wipe")) == {"seed0": 0.23, "seed1": 0.20}[seed]


def test_seed1_has_one_logit_between_two_she_names():
    assert two(who("seed1", lambda t: t["pairs"]["both She"]["gap"])) == 1.05


@pytest.mark.parametrize("seed, here, step6, count", [
    ("seed0", (0.19, -0.01, 0.19, 0.21), (0.23, -0.01, 0.20, 0.27), 3),
    ("seed1", (0.43, 0.02, 0.42, 0.45), (0.43, 0.02, 0.41, 0.45), 127),
])
def test_the_latents_as_predicted(seed, here, step6, count):
    assert (two(steered(seed, "latents", "share")),) + tuple(two(by_pair(seed, "latents", g)) for g in KINDS) == here
    with open(f"runs/features/features-{seed}.json") as f:
        T = json.load(f)["templates"]
    names = lambda t: t["names"]["the head's latents"]
    then = (np.mean([names(t)["moved for real"] for t in T]),) + tuple(np.mean([names(t)["by pair"][g]["moved for real"] for t in T]) for g in KINDS)
    assert tuple(two(x) for x in then) == step6
    i = at(seed, 1.0)
    assert sum(round(132 * t["steered"]["latents"]["turned"][i]) for t in run(seed)["who acts"]["templates"]) == count  # of 660, with those before steering
    # The mirror of step 6's move: nearly the same vector.
    mirror = "the latents' direction against the mirror of step 6's move"
    assert two(who(seed, lambda t: t[mirror]["cosine"])) == 0.99
    assert round(100 * abs(who(seed, lambda t: t[mirror]["length ratio"]) - 1), 1) == {"seed0": 0.4, "seed1": 1.0}[seed]
    # Nothing to write where the two names share a latent.
    assert who(seed, lambda t: t["length"]["latents"]["both She"]) < 0.2 * who(seed, lambda t: t["length"]["latents"]["both He"])


def gaps(seed, name, group, doses=(0.0, 0.5, 0.75, 1.0, 1.5, 2.0)):
    return [two(kind(seed, lambda t, d=d: t["steered"][name]["gap"][group][at(seed, d)])) for d in doses]


def turned(seed, name, group, doses=(0.0, 0.5, 0.75, 1.0, 1.5, 2.0)):
    return [sum(t["steered"][name]["turned"][group][at(seed, d)] for t in run(seed)["kind"]["templates"]) for d in doses]


@pytest.mark.parametrize("seed, plain, means, latent, random, counts", [
    ("seed0", (-7.31, 7.79), [-7.31, -0.38, 3.00, 5.56, 8.43, 9.54], [-7.31, -4.24, -2.45, -0.72, 2.21, 4.47], [-7.31, -7.19, -7.05, -6.85, -6.30, -5.62],
     ([0, 14, 35, 36, 36, 36], [0, 0, 4, 12, 35, 36])),
    ("seed1", (-7.11, 8.74), [-7.11, -0.83, 2.19, 4.94, 9.37, 12.01], [-7.11, -4.32, -2.93, -1.59, 0.93, 3.19], [-7.11, -6.94, -6.79, -6.59, -6.06, -5.42],
     ([0, 11, 31, 36, 36, 36], [0, 0, 4, 9, 24, 36])),
])
def test_kind_by_dose(seed, plain, means, latent, random, counts):
    K = run(seed)["kind"]
    assert len(K["sentences"]) == 6 and K["names per kind"] == 6 and "One day, {N} went to the park." not in K["sentences"]
    assert (two(kind(seed, lambda t: t["plain"]["He names"])), two(kind(seed, lambda t: t["plain"]["She names"]))) == plain
    assert kind(seed, lambda t: t["plain"]["names on the side step 5 sorted them to"]) == 1.0
    assert (gaps(seed, "means", "He names"), gaps(seed, "latent", "He names"), gaps(seed, "random", "He names")) == (means, latent, random)
    assert (turned(seed, "means", "He names"), turned(seed, "latent", "He names")) == counts
    # At dose 1 the difference of means turns every pronoun, and a random direction none.
    assert turned(seed, "means", "She names", (1.0,)) == [36] and turned(seed, "random", "He names", (1.0,)) == [0] == turned(seed, "random", "She names", (1.0,))


@pytest.mark.parametrize("seed, narrow, possessive, wide, reversed_", [
    ("seed0", (0.11, 1.81, 0.14), 0.15, (0.12, 0.14), (3.1, -10.4, -10.3)),
    ("seed1", (0.16, 1.18, 0.21), 0.21, (0.17, 0.20), (6.4, -13.5, -12.3)),
])
def test_kind_is_its_pronouns_here(seed, narrow, possessive, wide, reversed_):
    apart = lambda leave: (two(kind(seed, lambda t: t["plain"]["between two names of one kind"][leave])), two(kind(seed, lambda t: t["plain"]["between two names of two kinds"][leave])))
    far = np.mean([kind(seed, lambda t, g=g: t["steered"]["means"]["to a name of the other kind"][NARROW][g][at(seed, 1.0)]) for g in ("He names", "She names")])
    assert apart(NARROW) + (two(far),) == narrow
    assert apart("the two pronouns, His and Her, and the names")[1] == possessive
    assert apart(WIDE) == wide and len(run(seed)["kind"]["pronoun forms left out"]) == 17
    low = kind(seed, lambda t: t["steered"]["means"]["gap"]["He names"][at(seed, -1.0)])
    lower = kind(seed, lambda t: t["steered"]["means"]["gap"]["He names"][at(seed, -2.0)])
    assert (two(kind(seed, lambda t: t["plain"]["He names"]) - low, 1), two(low, 1), two(lower, 1)) == reversed_
    assert lower > low  # no further at dose -2


@pytest.mark.parametrize("seed, usual, share, who_turns, she_turns, lengths, cosine", [
    ("seed0", (-0.72, 12, 2.12, 3), (0.51, 0.43), {"Sam": 6, "Max": 4, "Tom": 2}, {"Sue": 2, "Mia": 1}, (9.3, 18.2), 0.67),
    ("seed1", (-1.59, 9, 3.63, 0), (0.46, 0.42), {"Sam": 6, "Max": 3}, {}, (10.2, 21.0), 0.59),
])
def test_the_shared_latent_at_its_mean_value(seed, usual, share, who_turns, she_turns, lengths, cosine):
    assert (gaps(seed, "latent", "He names", (1.0,))[0], turned(seed, "latent", "He names", (1.0,))[0],
            gaps(seed, "latent", "She names", (1.0,))[0], turned(seed, "latent", "She names", (1.0,))[0]) == usual
    # What the latent moves at its mean value, as a share of what the difference of means moves.
    moved = lambda name, g: abs(gaps(seed, name, g, (1.0,))[0] - gaps(seed, name, g, (0.0,))[0])
    assert tuple(two(moved("latent", g) / moved("means", g)) for g in ("He names", "She names")) == share  # for "He" names, for "She" names
    # Which names turn.
    K = run(seed)["kind"]
    each = np.array([t["steered"]["latent"]["gap by name"] for t in K["templates"]])[:, :, at(seed, 1.0)]  # (sentences, names)
    she = run(seed)["names that go on with She"]
    counts = {p: int(((each[:, j] < 0) if p in she else (each[:, j] > 0)).sum()) for j, p in enumerate(K["order"])}
    assert {p: c for p, c in counts.items() if c and p not in she} == who_turns and {p: c for p, c in counts.items() if c and p in she} == she_turns
    assert tuple(two(kind(seed, lambda t, n=n: t["length"][n]), 1) for n in ("latent", "means")) == lengths
    assert two(kind(seed, lambda t: t["cosine with means"]["latent"])) == cosine


def test_at_twice_its_mean_value_the_latent_turns_all_but_one():
    total = sum(turned(seed, "latent", g, (2.0,))[0] for seed in ("seed0", "seed1") for g in ("He names", "She names"))
    assert total == 143


@pytest.mark.parametrize("seed, table, differ, mean, largest, halves", [
    ("seed0", {"means": (5.56, -5.45, 36, 36), "rebuilt": (5.37, -5.12, 36, 36), "missed": (-6.94, 7.64, 0, 0),
               "rebuilt, without the shared latent": (-0.58, 1.51, 13, 6), "latent": (-0.72, 2.12, 12, 3)}, 30, 6.1, 1.5, (4.1, 2.1)),
    ("seed1", {"means": (4.94, -3.53, 36, 36), "rebuilt": (5.03, -3.70, 36, 36), "missed": (-7.29, 8.81, 0, 0),
               "rebuilt, without the shared latent": (0.18, 1.65, 17, 5), "latent": (-1.59, 3.63, 9, 0)}, 32, 6.0, 1.7, (4.3, 2.9)),
])
def test_the_rest_is_in_other_latents(seed, table, differ, mean, largest, halves):
    row = lambda name: (gaps(seed, name, "He names", (1.0,))[0], gaps(seed, name, "She names", (1.0,))[0], turned(seed, name, "He names", (1.0,))[0], turned(seed, name, "She names", (1.0,))[0])
    assert {name: row(name) for name in table} == table
    # The missed part moves the measure by less than 0.4 logits.
    plain = (kind(seed, lambda t: t["plain"]["He names"]), kind(seed, lambda t: t["plain"]["She names"]))
    assert all(abs(table["missed"][i] - plain[i]) < 0.4 for i in range(2))
    K = run(seed)["kind"]
    assert round(kind(seed, lambda t: t["latents that differ by more than half a unit between the kinds"])) == differ
    assert all(t["the latent that differs most between the kinds"] == K["the shared latent"] for t in K["templates"])
    # The rest split by sign: what each half moves a "He" name, in logits, used alone.
    move = lambda name: two(kind(seed, lambda t: t["steered"][name]["gap"]["He names"][at(seed, 1.0)]) - plain[0], 1)
    assert (move("latents higher on He names"), move("latents higher on She names, without the shared one")) == halves
    # The dose's unit against the largest value the latent takes in these sentences.
    on = K["the latent on She names"]
    assert (two(on["mean"], 1), two(2 * on["mean"] / on["largest"], 1)) == (mean, largest)


@pytest.mark.parametrize("seed, back, unit, start", [("seed0", (0.71, 0.02, 0.15), 0.88, -0.17), ("seed1", (0.71, 0.07, 0.09), 0.89, -0.18)])
def test_the_dictionary_does_not_read_back_what_was_written(seed, back, unit, start):
    read = lambda name, g, d: two(kind(seed, lambda t: t["steered"][name][BACK][g][at(seed, d)]))
    assert (read("latent", "He names", 1.0), read("latent", "He names", 0.25), read("latent", "She names", 1.0)) == back
    assert read("means", "He names", 1.0) == 1.00
    K = run(seed)["kind"]
    assert (two(K["the encoder's reading of one unit of the latent's direction"]), two(K["the reading before the cut on He names, against the usual value"])) == (unit, start)
    assert K["the latent fires on"] == {"She names": 1.0, "He names": 0.0}
    # Taken away from a "She" name, what remains is what the encoder does not read of its own direction.
    assert abs((1 - K["the encoder's reading of one unit of the latent's direction"]) - back[2]) < 0.04
