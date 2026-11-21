"""
What docs/08-limits.md claims, checked against runs/limits/limits.json and
the records of the open model. The tables of that note are the tables of
the record, so the tests pin the record and a few of its sources.
"""

import json

import numpy as np
import pytest

from conftest import needs_model, needs_open


def limits():
    with open("runs/limits/limits.json") as f:
        return json.load(f)


def read(path):
    with open(f"runs/{path}.json") as f:
        return json.load(f)


def shown(v):
    return v if v is None or isinstance(v, (str, int)) else round(v + 0.0, 2)


BOTH, S0, S1, NEITHER = [True, True], [True, False], [False, True], [False, False]
MODELS = [  # step, seed0, seed1, the open model, the seeds agree, which seeds the open model agrees with
    (1, 1.00, 0.95, 1.00, True, BOTH), (1, 0.85, 0.76, 1.00, True, S0), (1, 0.94, 0.90, 0.92, True, BOTH), (2, 0.95, 0.80, None, True, None),
    (2, 0.90, 0.71, None, False, None), (3, 0.75, 0.88, 0.62, True, S0), (4, "7.0", "8.0", None, False, None), (4, 1.07, 0.97, None, True, None),
    (5, 0.08, 0.08, None, True, None), (5, 0.03, 0.04, None, True, None), (6, 0.94, 1.00, None, True, None), (6, 0.17, 0.27, None, False, None),
    (6, 6, 6, None, True, None), (6, -0.26, -0.23, None, True, None), (7, 0.95, 1.04, 1.00, True, BOTH), (7, 0.96, 1.04, 1.00, True, BOTH),
    (7, 0.81, 0.85, 1.00, True, BOTH), (7, 0.38, 0.81, 0.80, False, S1), (7, 0.40, 0.07, 0.09, False, S1), (7, 0.23, 0.14, 0.07, False, NEITHER),
    (7, 0.33, 0.57, 0.10, False, NEITHER), (7, 0.32, 0.10, 1.01, False, NEITHER), (7, 72, 72, None, True, None), (7, 15, 9, None, False, None),
]
DICTIONARIES = [  # seed0 first and second, seed1 first and second, and whether the two agree in each model
    (0.08, 0.08, 0.08, 0.08, True, True), (0.03, 0.03, 0.04, 0.04, True, True), (0.94, 0.91, 1.00, 0.99, True, True), (0.17, 0.21, 0.27, 0.28, True, True),
    (0.69, 0.71, 0.85, 0.86, True, True), (0.91, 0.89, 0.98, 0.98, True, True), (0.95, 0.91, 0.99, 0.98, True, True), (0.23, 0.28, 0.43, 0.45, True, True),
    (6, 6, 6, 5, True, False), (6, 5, 6, 4, False, False), (0.68, 0.76, 0.77, 0.67, True, True), (-0.03, -0.02, -0.02, -0.02, True, True),
    (0.19, 0.22, 0.43, 0.45, True, True), (6.59, 6.80, 5.52, 6.13, True, True), (15, 18, 9, 14, False, False), (12, 12, 9, 10, True, False),
    (3, 6, 0, 4, False, False), (72, 72, 71, 72, True, False), (72, 72, 72, 72, True, True), (0, 0, 0, 0, True, True),
    (0.38, 0.51, 0.19, 0.07, False, False), (0.71, 0.89, 0.71, 0.74, True, True), (0.88, 0.87, 0.89, 0.91, True, True), (-0.17, 0.02, -0.18, -0.17, False, True),
    (29.5, 28.67, 31.83, 31.0, True, True),
]


def test_the_table_of_the_models():
    m = limits()["models"]
    assert m["columns"] == ["seed0", "seed1", "qwen"]
    assert [(r["step"], *(shown(v) for v in r["values"]), r["the two story models agree"], r["the open model agrees with"]) for r in m["rows"]] == MODELS
    # Fifteen of the 24 are the same in the two story models; fourteen with the second dictionaries.
    assert m["rows the two story models agree on"] == {"with the first dictionaries": 15, "with the second dictionaries": 14, "of": 24}
    assert len([r for r in m["rows"] if r["values"][2] is not None]) == 12


def test_the_depth_row_depends_on_the_line_in_the_open_model_only():
    lines = {k: [round(x, 2) for x in v] for k, v in limits()["models"]["the depth row with another line"].items()}
    assert lines == {"0.9": [0.75, 0.88, 0.58], "0.5": [0.75, 0.88, 0.62], "0.25": [0.75, 0.88, 0.83], "0.1": [0.75, 0.88, 0.92]}


def test_the_shared_vector_by_depth():
    d = limits()["models"]["by depth"]
    at = lambda name, key, blocks: [round(d[name][key][d[name]["depths"].index(b)] + 0.0, 2) for b in blocks]
    table = lambda name, blocks: [at(name, key, blocks) for key in ("means", "order", "means + order")]
    assert table("seed0", (2, 4, 6)) == [[1.01, 0.38, 0.37], [0.00, 0.40, 0.38], [1.01, 0.81, 0.78]]
    assert table("seed1", (2, 4, 6)) == [[0.99, 0.81, 0.75], [0.00, 0.07, 0.00], [0.99, 0.85, 0.74]]
    assert table("qwen", (10, 12, 14)) == [[1.04, 0.80, 0.44], [-0.01, 0.09, 0.54], [1.00, 1.00, 1.02]]
    # At block 4 the story models' columns are step 7's numbers.
    for seed in ("seed0", "seed1"):
        step7 = read(f"steer/steer-{seed}")
        i = step7["doses"].index(1.0)
        for key in ("means", "order", "means + order"):
            assert abs(d[seed][key][d[seed]["depths"].index(4)] - np.mean([x["steered"][key]["share"][i] for x in step7["who acts"]["templates"]])) < 1e-3
    # In the open model the vector alone does nothing from block 16 on, where the pair direction carries 0.46 and the means 0.43.
    late = [d["qwen"]["depths"].index(b) for b in (16, 18, 20, 22)]
    assert (round(min(d["qwen"]["order"][i] for i in late), 2), round(max(d["qwen"]["order"][i] for i in late) + 0.0, 2)) == (-0.04, 0.00)
    assert (round(d["qwen"]["pair"][late[0]], 2), round(d["qwen"]["means"][late[0]], 2)) == (0.46, 0.43)
    # At block 14 the vector alone is more than a fifth above seed0's: four patterns give 0.45 to 0.50 and the fifth 0.79.
    r = read("open/steer-qwen")
    each = sorted(round(x["depth"]["order"]["share"][r["depths"].index(14)], 2) for x in r["templates"])
    assert (each[0], each[3], each[4]) == (0.45, 0.50, 0.79)
    there, seed0 = d["qwen"]["order"][d["qwen"]["depths"].index(14)], d["seed0"]["order"][d["seed0"]["depths"].index(4)]
    assert there - seed0 > 0.2 * there


def test_the_table_of_the_dictionaries():
    rows = limits()["dictionaries"]["rows"]
    assert [(*(shown(v) for v in r["values"]), *r["the two dictionaries agree"]) for r in rows] == DICTIONARIES
    # The cell on the line: the same on the values themselves, different on the printed ones.
    back = next(r for r in rows if r["what"].startswith("the dictionary reads the latent back"))["values"]
    assert abs(back[0] - back[1]) <= 0.2 * back[1] < round(back[1], 2) - round(back[0], 2)


def test_the_rule():
    assert "a fifth" in limits()["rule"] and "0.05" in limits()["rule"]
    same = lambda a, b: abs(a - b) <= max(0.2 * max(abs(a), abs(b)), 0.05)
    assert same(0.85, 0.76) and not same(0.38, 0.81) and same(-0.03, -0.02) and not same(0.62, 0.88)
    # 1.00 against 0.95 is 1 against 30 wrong prompts of 660.
    wrong = lambda seed: sum(round(132 * (1 - (np.array(t["clean"]["logit_diff"]) > 0).mean())) for t in read(f"read/behavior-{seed}")["templates"])
    assert (wrong("seed0"), wrong("seed1")) == (1, 30)


@pytest.mark.parametrize("seed, share, inside, firing, strict, shared, by_head, by_firing", [
    ("seed0", (3, 9, 30, 0.41), 0.24, ((161, 1), (3272, 4), (663, 33)), 5, (0.83, 0.97, 0.72), (7, 5, 4), (12, 5, 3)),
    ("seed1", (3, 13, 39, 0.46), 0.25, ((175, 1), (3220, 7), (701, 48)), 8, (0.77, 0.98, 0.29), (7, 5, 4), (12, 3, 3)),
])
def test_which_latents_have_a_counterpart(seed, share, inside, firing, strict, shared, by_head, by_firing):
    m = limits()["matches"][seed]
    assert (m["latents of the first dictionary"], m["latents that never fire on the stories"], m["story tokens"]) == (4096, 0, 76800)
    pct = lambda d: (round(100 * d["at a cosine of 0.9 or more"]), round(100 * d["at 0.7 or more"]), round(100 * d["at 0.5 or more"]), round(d["median cosine"], 2))
    assert pct(m["with a counterpart in the second dictionary"]) == share
    assert round(m["for comparison"]["the nearest other latent of the same dictionary"]["median cosine"], 2) == inside
    assert round(m["for comparison"]["two sets of random directions"]["median cosine"], 2) == 0.18
    assert tuple((d["latents"], round(100 * d["at 0.7 or more"])) for d in m["by how often a latent fires"].values()) == firing
    assert round(100 * m["one to one"]["the same partner by encoder and decoder, 0.7 or more in both"]) == strict
    s = m["the shared latent"]
    assert (round(s["cosine of the best counterpart"], 2), round(s["correlation of the two over all prompts"], 2), round(s["among the prompts where a She name acts"], 2)) == shared
    assert s["the best counterpart is the second dictionary's shared latent"] and s["prompts"] == 1320
    # Both fire wherever a "She" name acts, and on 1% to 20% of the rest.
    assert s["share of the She prompts on which each fires"] == [1.0, 1.0]
    assert all(0.005 <= x <= 0.205 for x in s["share of the other prompts on which each fires"])
    hit = "the counterpart is the second dictionary's latent for most of these names"
    count = lambda table: (len(m[table]), sum(x[hit] for x in m[table]), sum(x["cosine of the best counterpart"] >= 0.7 for x in m[table]))
    assert (count("the veto head's latents"), count("step 5's latents")) == (by_head, by_firing)


def test_what_the_second_dictionary_changes_in_steps_6_and_7():
    names = lambda seed: {x["name"]: x for x in read(f"features/features-{seed}-s1")["names"]}
    latent = lambda seed, name: names(seed)[name]["the head's latent"]
    she = read("features/features-seed1-s1")["names that go on with She"]
    # In seed1 the head rests most on another latent when Anna acts, one that fires for Anna and Sara; the shared one still fires for Anna.
    assert len({latent("seed1", p) for p in she if p != "Anna"}) == 1 and latent("seed1", "Anna") != latent("seed1", "Lily")
    assert names("seed1")["Anna"]["names for which the head's latent fires on 90% or more"] == ["Anna", "Sara"]
    assert "Anna" in names("seed1")["Lily"]["names for which the head's latent fires on 90% or more"]
    assert read("features/features-seed1-s1")["the shared latent"] == latent("seed1", "Lily") and len({latent("seed0", p) for p in she}) == 1
    # Ben and Jack share a latent in seed0; Tom, Ben and Jack in seed1, where it fires for all six "He" names.
    assert latent("seed0", "Ben") == latent("seed0", "Jack") and len({latent("seed0", p) for p in ("Tom", "Tim", "Sam", "Max", "Ben")}) == 5
    assert latent("seed1", "Tom") == latent("seed1", "Ben") == latent("seed1", "Jack")
    assert names("seed1")["Tom"]["names for which the head's latent fires on 90% or more"] == ["Tom", "Tim", "Ben", "Sam", "Max", "Jack"]
    # The second dictionaries are trained from another seed, with everything else as it was.
    for seed in ("seed0", "seed1"):
        one, two = read(f"sae/sae-{seed}"), read(f"sae/sae-{seed}-s1")
        assert (two["seed"], two["width"], two["k"], two["steps"], two["block"]) == (1, one["width"], one["k"], one["steps"], one["block"]) == (1, 4096, 32, 4000, 4)
    # Step 7's read-back: in seed0's second dictionary the reading on a "He" name starts just above zero.
    k = read("steer/steer-seed0-s1")["kind"]
    assert round(k["the reading before the cut on He names, against the usual value"], 2) == 0.02


def test_the_open_model_by_depth():
    r = read("patch/patch-qwen")
    at = lambda group: np.mean([t["travel"]["carry"] for t in r["templates"]], 0)[:, r["travel groups"].index(group)]
    assert len(at("S")) == 25 and round(at("S")[14], 2) == 0.98
    assert [round(v, 2) for v in at("end")[[14, 15, 16, 20, 21, 23]]] == [-0.03, 0.25, 0.50, 0.54, 0.76, 1.00]
    assert "parts" not in r["templates"][0]  # the heads were not looked at there
    behavior = read("read/behavior-qwen")
    assert behavior["model"]["config"]["n_layers"] == 24 and behavior["model"]["config"]["n_heads"] == 14 and behavior["model"]["config"]["dim"] == 896
    assert len(behavior["names"]) == 12


def test_the_open_model_steered():
    r = read("open/steer-qwen")
    T, doses = r["templates"], r["doses"]
    mean = lambda pick: float(np.mean([pick(t) for t in T]))
    at = lambda name, measure, dose=1.0: round(mean(lambda t: t["steered"][name][measure][doses.index(dose)]), 2)
    assert r["block"] == 12 and r["names that go on with She"] == read("steer/steer-seed0")["names that go on with She"]
    assert (at("pair", "share"), at("pair", "turned"), at("names", "share"), at("means + order", "share")) == (1.00, 1.00, 1.00, 1.00)
    # After every second block the pair direction carries what the patch carries: within 0.01 on the mean, 0.04 in any pattern.
    depth = np.array([t["depth"]["pair"]["share"] for t in T])
    travel = read("patch/patch-qwen")
    patch = np.array([[t["travel"]["break"][d][travel["travel groups"].index("S")] for d in r["depths"]] for t in travel["templates"]])
    assert r["depths"] == list(range(0, 25, 2)) and np.abs(depth.mean(0) - patch.mean(0)).max() <= 0.0105 and np.abs(depth - patch).max() <= 0.04
    # The gaps do not depend on the kind.
    gaps = [round(mean(lambda t, g=g: t["pairs"][g]["gap"]), 1) for g in ("both She", "both He", "one of each")]
    assert gaps == [16.0, 14.5, 15.8]
    # The dose: a step, and no overshoot.
    assert [at("pair", "share", d) for d in (0.25, 0.5, 0.75, 1.5, 2.0)] == [0.06, 0.59, 0.99, 0.93, 0.86] and at("pair", "turned", 0.5) == 0.72
    assert (at("random, long", "share"), at("wipe", "share"), at("means + random", "share"), at("means", "share")) == (0.07, 0.49, 0.75, 0.80)
    # The pair direction against the stream's size, there and in the story models.
    size = lambda rec, templates: round(float(np.mean([t["length"]["pair"]["all"] for t in templates]) / np.mean([t["stream size at the acting name"] for t in templates])), 2)
    story = [size(s, s["who acts"]["templates"]) for s in (read("steer/steer-seed0"), read("steer/steer-seed1"))]
    assert (size(r, T), *story) == (0.60, 0.85, 0.78)


@needs_model
@needs_open
def test_the_open_model_is_35_times_the_story_model():
    from minterp.common import load

    count = lambda name: sum(p.numel() for p in {id(p): p for p in load(name)[0].parameters()}.values())  # a tied matrix once
    story, published = count("seed0"), count("qwen")
    assert round(story / 1e6) == 14 and round(published / 1e6) == 494 and round(published / story) == 35
