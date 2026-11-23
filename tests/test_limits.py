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


def test_the_depth_row_depends_on_the_line_in_the_open_model_and_in_the_third_story_model():
    lines = {k: [round(x, 2) for x in v] for k, v in limits()["models"]["the depth row with another line"].items()}
    assert lines == {"0.9": [0.75, 0.88, 0.58], "0.5": [0.75, 0.88, 0.62], "0.25": [0.75, 0.88, 0.83], "0.1": [0.75, 0.88, 0.92]}
    # Seed0 and seed1 move who acts off the acting name in one block; the third story model in two, so its row moves with the line.
    t = limits()["models"]["a third seed"]
    assert {k: round(v, 3) for k, v in t["the depth row with another line"].items()} == {"0.9": 0.625, "0.5": 0.75, "0.25": 0.75, "0.1": 0.75}
    by_block = {s: [round(x, 2) for x in v] for s, v in t["looks after the check"]["the acting name's position moved alone, by block"].items()}
    assert by_block["seed2"][5:8] == [1.04, 0.56, -0.02] and by_block["seed0"][6:8] == [0.94, -0.05] and by_block["seed1"][7:] == [0.92, 0.0]
    same = lambda a, b: abs(a - b) <= max(0.2 * max(abs(a), abs(b)), 0.05)
    assert same(0.75, 0.625) and not same(0.875, 0.625)


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


ALL = True
THIRD_MODEL = [  # seed2's value, and what it is the same as: all three, or which of seed0 and seed1
    (0.99, ALL), (0.84, ALL), (0.74, S1), (0.88, ALL), (0.82, BOTH), (0.75, ALL), ("7.3", NEITHER), (1.08, ALL),
    (0.08, ALL), (0.04, ALL), (1.02, ALL), (0.28, S1), (2, NEITHER), (-0.76, NEITHER), (1.06, ALL), (1.07, ALL),
    (0.85, ALL), (0.79, S1), (0.06, S1), (0.31, NEITHER), (0.44, NEITHER), (0.30, S0), (72, ALL), (27, NEITHER),
]
THIRD_DICTIONARY = [  # the third dictionary's value in seed0 and in seed1, and whether each is the same as the first and as the second
    (0.08, 0.08, ALL, ALL), (0.03, 0.04, ALL, ALL), (0.88, 1.00, ALL, ALL), (0.26, 0.31, S1, ALL), (0.75, 0.83, ALL, ALL), (0.86, 0.98, ALL, ALL),
    (0.94, 0.98, ALL, ALL), (0.31, 0.43, S1, ALL), (6, 6, ALL, S0), (3, 6, NEITHER, S0), (0.84, 0.90, ALL, S0), (-0.05, -0.03, ALL, ALL),
    (0.26, 0.43, S1, ALL), (7.53, 7.19, ALL, S1), (32, 22, NEITHER, NEITHER), (18, 16, NEITHER, NEITHER), (14, 6, NEITHER, NEITHER), (72, 72, ALL, S1),
    (72, 72, ALL, ALL), (0, 0, ALL, ALL), (0.29, 0.16, NEITHER, S0), (0.84, 0.84, ALL, ALL), (0.87, 0.94, ALL, ALL), (-0.03, -0.09, NEITHER, NEITHER),
    (29.17, 25.17, ALL, S1),
]
verdict = lambda v: ALL if v["the same in all three"] else v["agrees with"]


def test_the_script_states_the_rule_for_three():
    """The script says how three columns are read, and where the third model's dictionary goes, in the words the note uses.
    That it said so before the runs is a matter of commit and file times, which no test here can check."""
    with open("scripts/limits.py") as f:
        doc = f.read().split('"""')[1]
    for words in ("applied to each of the three pairs", "written here before either had been run", "largest share of the gap moved at that position alone",
                  "no row is run again", "One look was added after the rows had been read"):
        assert words in " ".join(doc.split())
    same = lambda a, b: abs(a - b) <= max(0.2 * max(abs(a), abs(b)), 0.05)
    # The rule does not carry over: 0.82 is the same as 0.90 and as 0.71, which differ.
    assert same(0.90, 0.82) and same(0.82, 0.71) and not same(0.90, 0.71)


def test_a_third_model():
    m = limits()["models"]
    t = m["a third seed"]
    assert [(shown(r["value"]), verdict(r)) for r in t["rows"]] == THIRD_MODEL
    assert [(r["step"], r["what"]) for r in t["rows"]] == [(r["step"], r["what"]) for r in m["rows"]]
    # Twelve of the 24 in all three; the pairs agree on 15, 14 and 17; of the 15 seen twice, 12 a third time.
    assert t["tally"] == {
        "rows": 24, "the same in all three": 12, "by pair": {"the first and the second": 15, "the first and the third": 14, "the second and the third": 17},
        "of the rows the first two agreed on": {"rows": 15, "the third is the same as both": 12},
        "of the rows they differed on": {"rows": 9, "the first only": 1, "the second only": 3, "both": 1, "neither": 4},
    }
    # The same names, seven heads, and the dictionary where the rule puts it: no record of the third model at another block.
    block = t["the block of the head with the largest share at the acting name"]
    assert (block, t["the same block for the first two"], t["heads of the circuit"]) == (4, [4, 4], [7, 7, 7])
    assert all(read(f"{step}/{step}-seed2")["block"] == block for step in ("sae", "features", "steer")) and read("open/steer-seed2")["block"] == block
    assert t["names that go on with She"] == read("features/features-seed0")["names that go on with She"] == read("features/features-seed1")["names that go on with She"]
    assert [round(read(f"read/behavior-{s}")["model"]["val_loss"], 3) for s in ("seed0", "seed1", "seed2")] == [1.465, 1.461, 1.460]
    one, three = read("read/behavior-seed1")["model"], read("read/behavior-seed2")["model"]
    assert one["config"] == three["config"] and one["step"] == three["step"] == 6000
    # A value per sentence pattern for the rows that are means over the patterns: the dots of the figure.
    assert sum(r["by pattern"][2] is not None for r in t["rows"]) == 14 and all(len(x) == 5 for r in t["rows"] for x in r["by pattern"] if x is not None)
    assert sum(r["kind"] == "number" for r in t["rows"]) == 20  # the rows of the figure
    assert all(abs(np.mean(r["by pattern"][2]) - r["value"]) < 1e-9 for r in t["rows"] if r["by pattern"][2] is not None)


def test_what_did_not_hold_in_the_third_model():
    # Where the answer is written: head 5 of block 6 writes 0.16 of the clean logit difference; in the first two the largest head of block 6 writes 0.01.
    def block_six(seed):
        a = read(f"read/lens-{seed}")["attribution"]["clean"]
        heads = np.array(a["heads"]["mean"])
        return heads[5] / (heads.sum() + np.sum(a["mlp"]["mean"]) + a["embed"]["mean"])
    assert round(block_six("seed2")[5], 2) == 0.16 and block_six("seed2").argmax() == 5
    assert [round(float(np.abs(block_six(s)).max()), 2) for s in ("seed0", "seed1")] == [0.01, 0.01]
    # That row sits on the line: 0.94 against 0.74 misses by 0.196 where the rule allows 0.188.
    rows = {r["what"]: r for r in limits()["models"]["rows"]}
    third = {r["what"]: r for r in limits()["models"]["a third seed"]["rows"]}
    late = "share of the clean logit difference written by the last quarter of the blocks"
    assert (round(rows[late]["values"][0] - third[late]["value"], 3), round(0.2 * rows[late]["values"][0], 3)) == (0.196, 0.188)
    # The rule ignores the spread over the patterns: a random direction, and the first probe row.
    spread = lambda what, i: (round(min(third[what]["by pattern"][i]), 2), round(max(third[what]["by pattern"][i]), 2))
    random = "a random direction as long as the pair direction"
    assert (spread(random, 0), spread(random, 2)) == ((0.14, 0.35), (0.22, 0.41)) and third[random]["agrees with"] == NEITHER and not rows[random]["the two story models agree"]
    probe = "the due name read at the acting name at half depth, on new pairs, clean prompts, share right"
    assert spread(probe, 1)[0] == spread(probe, 2)[0] == 0.48 and third[probe]["the same in all three"]
    # Step 6's head, 7.3: one latent for Mia and Lucy, four others for the other four names.
    f, k = read("features/features-seed2"), read("steer/steer-seed2")["kind"]
    latent = {x["name"]: x["the head's latent"] for x in f["names"]}
    she = f["names that go on with She"]
    assert f["veto head"] == "block 7 head 3" and latent["Mia"] == latent["Lucy"] == f["the shared latent"] and len({latent[p] for p in she}) == 5
    # The dictionary has the latent: it fires in all 36 sentences with a "She" name and in 2 of the 36 with a "He" name.
    assert k["names that share it"] == ["Mia", "Lucy"] and k["the latent fires on"]["She names"] == 1.0 and round(36 * k["the latent fires on"]["He names"]) == 2
    assert len(k["sentences"]) * k["names per kind"] == 36
    # An acting name against its own kind: "He" 1.12, 1.16 and 1.29; "She" 0.26, 0.23 and 0.76.
    def own(seed, kind):
        return round(float(np.mean([t["head"]["by kind of the acting name"][kind]["against its own kind"] for t in read(f"features/features-{seed}")["templates"]])), 2)
    assert [own(s, "He") for s in ("seed0", "seed1", "seed2")] == [-1.12, -1.16, -1.29] and [own(s, "She") for s in ("seed0", "seed1", "seed2")] == [-0.26, -0.23, -0.76]
    assert 0.195 < own("seed1", "She") / own("seed1", "He") < own("seed0", "She") / own("seed0", "He") < 0.25


def test_the_look_added_after_the_rows():
    t = limits()["models"]["a third seed"]["the heads at the last position, by kind"]
    by = lambda seed: {h["head"]: h for h in t[seed]}
    share = lambda seed, head: round(by(seed)[head]["carry"], 2)
    she = lambda seed, head: round(by(seed)[head]["adds to an acting name against its own kind"]["She"], 2)
    # Two heads that read the acting name and lower it in seed2, one far above the next in the first two.
    assert (share("seed2", "7.3"), share("seed2", "6.5")) == (0.62, 0.45)
    assert (share("seed0", "7.0"), share("seed0", "7.1"), share("seed1", "8.0"), share("seed1", "8.1")) == (0.87, 0.13, 0.69, 0.14)
    assert all(by(s)[h]["reads the acting name with"] > 0.65 for s, h in (("seed2", "7.3"), ("seed2", "6.5"), ("seed0", "7.0"), ("seed0", "7.1"), ("seed1", "8.0"), ("seed1", "8.1")))
    writes = read("circuit/circuit-seed2")["templates"]
    for head in ("block 7 head 3", "block 6 head 5"):
        assert all(t_["writes"][head]["clean"]["A"] < -0.4 and t_["writes"][head]["flipped"]["B"] < -0.4 for t_ in writes)
    # The smaller adds -0.26 to an acting "She" name against its own kind, the same as -0.26 and -0.23; the larger -0.76.
    assert (she("seed2", "6.5"), she("seed2", "7.3"), she("seed0", "7.0"), she("seed1", "8.0")) == (-0.26, -0.76, -0.26, -0.23)
    beside = "an acting She name, beside the first two models' veto heads"
    assert by("seed2")["6.5"][beside]["the same in all three"] and by("seed2")["7.3"][beside]["agrees with"] == NEITHER
    # The heads script gives step 6's number for the head step 6 studies.
    step6 = np.mean([x["head"]["by kind of the acting name"]["She"]["against its own kind"] for x in read("features/features-seed2")["templates"]])
    assert abs(by("seed2")["7.3"]["adds to an acting name against its own kind"]["She"] - step6) < 2e-4


def test_the_looks_made_after_the_check():
    a = limits()["models"]["a third seed"]["looks after the check"]
    kinds = lambda h: [round(h["adds where a name of each kind acts"][k][m], 2) for k in ("She", "He") for m in ("against the nine", "against its own kind", "its own kind against the other")]
    first = a["by dictionary"]["first"]
    # The table of the note: the first two models' veto heads and the third model's two.
    assert [h["head"] for h in a["the first two models' veto heads"].values()] == ["block 7 head 0", "block 8 head 0"]
    assert [kinds(h) for h in a["the first two models' veto heads"].values()] == [[-1.01, -0.26, -1.40, -1.75, -1.12, -1.19], [-1.20, -0.23, -1.80, -1.72, -1.16, -1.07]]
    assert (first["the head step 6 picks"]["head"], kinds(first["the head step 6 picks"])) == ("block 7 head 3", [-1.18, -0.76, -0.79, -1.60, -1.29, -0.57])
    assert (first["the second head"]["head"], kinds(first["the second head"])) == ("block 6 head 5", [-0.67, -0.26, -0.76, -0.61, -0.51, -0.20])
    # About half the first two models' effect on the kind; on a "He" name against its own kind not theirs by the rule.
    same = lambda x, y: abs(x - y) <= max(0.2 * max(abs(x), abs(y)), 0.05)
    assert 0.4 < 0.76 / 1.80 < 0.76 / 1.40 < 0.56 and not same(-0.51, -1.12) and not same(-0.51, -1.16)
    # A write does not pass through a dictionary: the same numbers with each of the three.
    assert all(kinds(d[which]) == kinds(first[which]) for d in a["by dictionary"].values() for which in ("the head step 6 picks", "the second head"))
    # The count does: two, five and five of the six "She" names for the head step 6 picks, six in each for the second head.
    count = lambda which: [d[which]["She names that share its first choice"] for d in a["by dictionary"].values()]
    assert list(a["by dictionary"]) == ["first", "second", "third"] and count("the head step 6 picks") == [2, 5, 5] and count("the second head") == [6, 6, 6]
    for tag in ("s1", "s2"):  # Anna on a latent of her own, as in seed1's second dictionary
        f = read(f"features/features-seed2-{tag}")
        latent = {x["name"]: x["the head's latent"] for x in f["names"]}
        assert {p for p in f["names that go on with She"] if latent[p] != f["the shared latent"]} == {"Anna"}
        one, other = read("sae/sae-seed2"), read(f"sae/sae-seed2-{tag}")
        assert (other["seed"], other["width"], other["k"], other["steps"], other["block"]) == (int(tag[1]), one["width"], one["k"], one["steps"], one["block"])
    f = read("features/features-seed1-s1")
    assert {x["name"] for x in f["names"] if x["name"] in f["names that go on with She"] and x["the head's latent"] != f["the shared latent"]} == {"Anna"}
    # The two largest heads that read the acting name, together, against its own kind.
    two = a["the two largest heads that read the acting name, together, against its own kind"]
    assert {s: (d["heads"], round(d["She"], 2), round(d["He"], 2)) for s, d in two.items()} == {
        "seed0": (["7.0", "7.1"], -0.29, -1.27), "seed1": (["8.0", "8.1"], -0.28, -1.39), "seed2": (["7.3", "6.5"], -1.02, -1.80)}


def test_the_tables_of_the_third_section_are_the_record():
    """The cells of the note's three tables on a third model and a third dictionary, read from the note and set against the record."""
    with open("docs/08-limits.md") as f:
        note = f.read()
    def table(header):
        lines = note[note.index(header):].split("\n\n")[0].split("\n")[2:]
        return [[c.strip().replace("**", "").replace("−", "-") for c in line.strip("|").split("|")] for line in lines]
    cell = lambda v: str(v) if isinstance(v, int) else v if isinstance(v, str) else f"{v:.1f}" if abs(v) > 20 else f"{v:.2f}"
    m, t = limits()["models"], limits()["models"]["a third seed"]
    says = lambda r: "both" if r["the same in all three"] else {(True, True): "each, though they differ", (True, False): "seed0 only", (False, True): "seed1 only", (False, False): "neither"}[tuple(r["agrees with"])]
    rows = table("| Step | Number | seed0 | seed1 | seed2 | seed2 is the same as |")
    assert len(rows) == 24
    for cells, old, new in zip(rows, m["rows"], t["rows"]):
        want = [cell(v) for v in (*old["values"][:2], new["value"])]
        if old["kind"] == "name":
            want = [f"block {v.split('.')[0]}, head {v.split('.')[1]}" for v in want]
        assert [cells[0], *cells[2:]] == [str(old["step"]), *want, says(new)], cells
    d, third = limits()["dictionaries"], limits()["dictionaries"]["a third dictionary"]
    with open("docs/08-limits.md") as f:
        bold = {line.split("|")[2].strip(): [i for i, c in enumerate(line.strip().strip("|").split("|")) if "**" in c] for line in f if line.startswith("| ")}
    rows = table("| Step | With the dictionary | seed0: first | second | third | seed1: first | second | third |")
    shown_rows = [(old, new) for old, new in zip(d["rows"], third["rows"]) if not all(v["agrees with"] == BOTH for v in new["by model"])]
    assert len(rows) == len(shown_rows) == 14
    for cells, (old, new) in zip(rows, shown_rows):
        v = old["values"]
        assert cells[2:] == [cell(x) for x in (v[0], v[1], new["values"][0], v[2], v[3], new["values"][1])], cells
        # Bold marks a third value that differs from the first's or the second's.
        assert bold[cells[1]] == [i for i, x in zip((4, 7), new["by model"]) if x["agrees with"] != BOTH], cells
    a = t["looks after the check"]
    heads = [*a["the first two models' veto heads"].values(), a["by dictionary"]["first"]["the head step 6 picks"], a["by dictionary"]["first"]["the second head"]]
    rows = table('| Head | "She": against the nine | its own kind | kind against the other | "He": against the nine | its own kind | kind against the other |')
    assert len(rows) == 4
    for cells, seed, h in zip(rows, ("seed0", "seed1", "seed2", "seed2"), heads):
        block, head = h["head"].split()[1], h["head"].split()[3]
        assert cells[0] == f"{seed}, block {block}, head {head}"
        assert cells[1:] == [cell(h["adds where a name of each kind acts"][k][x]) for k in ("She", "He") for x in ("against the nine", "against its own kind", "its own kind against the other")]


def test_what_the_three_circuits_share():
    for seed, head, clean, flipped in (("seed0", "block 4 head 2", 0.91, 0.78), ("seed1", "block 4 head 4", 0.85, 0.79), ("seed2", "block 4 head 2", 0.81, 0.70)):
        c = read(f"circuit/circuit-{seed}")
        at = [h for h in c["heads"] if h["acts at"] == "S"]
        assert len(at) == 1 and f"block {at[0]['block']} head {at[0]['head']}" == head
        # At the acting name it reads the other name: the second on clean prompts, the first on flipped ones.
        assert round(float(np.mean([t["attention"][head]["clean"]["B"] for t in c["templates"]])), 2) == clean
        assert round(float(np.mean([t["attention"][head]["flipped"]["A"] for t in c["templates"]])), 2) == flipped
    # The pair direction added at the acting name after block 6.
    d = limits()["models"]["by depth"]
    assert [round(d[s]["pair"][d[s]["depths"].index(6)], 2) for s in ("seed0", "seed1", "seed2")] == [0.95, 0.92, 0.59]
    # Step 4's hold in the third model: of the fetching head's 0.26, 0.12 and 0.13 come along with one or the other veto head held.
    T = read("circuit/circuit-seed2")["templates"]
    hold = lambda receiver, key: round(float(np.mean([next(h for h in t["holds"] if (h["sender"], h["receiver"]) == ("block 4 head 2", receiver))[key] for t in T])), 2)
    assert [(hold(r, "alone"), hold(r, "held")) for r in ("block 6 head 5", "block 7 head 3")] == [(0.26, 0.12), (0.26, 0.13)]


def test_a_third_dictionary():
    d = limits()["dictionaries"]
    t = d["a third dictionary"]
    assert [(*(shown(v) for v in r["values"]), *(verdict(v) for v in r["by model"])) for r in t["rows"]] == THIRD_DICTIONARY
    # 16 and 14 of the 25 in all three; of the 20 and 18 the first two agreed on, the third joins 16 and 14.
    tally = lambda seed: (t["tally"][seed]["the same in all three"], t["tally"][seed]["of the rows the first two agreed on"], t["tally"][seed]["by pair"])
    assert tally("seed0") == (16, {"rows": 20, "the third is the same as both": 16}, {"the first and the second": 20, "the first and the third": 16, "the second and the third": 19})
    assert tally("seed1") == (14, {"rows": 18, "the third is the same as both": 14}, {"the first and the second": 18, "the first and the third": 18, "the second and the third": 17})
    # Fourteen rows have a third value that differs from the first's or the second's in one model; the note's table holds those.
    differs = [r["what"] for r in t["rows"] if not all(v["agrees with"] == BOTH for v in r["by model"])]
    assert len(differs) == 14 and not any(w.startswith(("variance", "the loss", "who acts", "by the eight", "by the 32", "the gradient")) for w in differs)
    # Trained as the first, from seed 2.
    for seed in ("seed0", "seed1"):
        one, three = read(f"sae/sae-{seed}"), read(f"sae/sae-{seed}-s2")
        assert (three["seed"], three["width"], three["k"], three["steps"], three["block"]) == (2, one["width"], one["k"], one["steps"], one["block"])
    # Pronouns the shared latent turns: between 9 and 59 of 72 over nine dictionaries, three per model.
    turned = [r for r in d["rows"] if r["what"] == "pronouns it turns, of 72"][0]["values"] + [r for r in t["rows"] if r["what"] == "pronouns it turns, of 72"][0]["values"]
    seed2 = [x["pronouns the shared latent turns, of 72"] for x in limits()["models"]["a third seed"]["looks after the check"]["by dictionary"].values()]
    assert seed2 == [27, 52, 59] and seed2[0] == [r for r in limits()["models"]["a third seed"]["rows"] if r["what"].startswith("pronouns turned by the shared latent")][0]["value"]
    assert sorted(turned + seed2) == [9, 14, 15, 18, 22, 27, 32, 52, 59]
    # Two aggregates the second dictionaries kept and a third did not, in one model each.
    row = lambda what, seed: [shown(v) for v in next(r for r in d["rows"] if r["what"].startswith(what))["values"][2 * seed:2 * seed + 2]] + [shown(next(r for r in t["rows"] if r["what"].startswith(what))["values"][seed])]
    assert row("carried by the highest-ranked latent", 0) == [0.17, 0.21, 0.26] and row("the shared latent at its mean value moves", 1) == [5.52, 6.13, 7.19]


@pytest.mark.parametrize("seed, in_third, in_both, shared, he", [
    ("seed0", (3, 9, 31, 0.41), (6, 28, 0.74), (0.78, 0.80), 3),
    ("seed1", (4, 14, 40, 0.45), (9, 39, 0.69), (0.91, 0.76), 6),
])
def test_which_latents_come_back_a_third_time(seed, in_third, in_both, shared, he):
    m = limits()["matches"][seed]
    a = m["a third dictionary"]
    pct = lambda d: (round(100 * d["at a cosine of 0.9 or more"]), round(100 * d["at 0.7 or more"]), round(100 * d["at 0.5 or more"]), round(d["median cosine"], 2))
    assert pct(a["the first dictionary's latents with a counterpart in the third"]) == in_third
    both = a["latents of the first with a counterpart at 0.7 or more in the second and in the third"]
    assert (round(100 * both), round(100 * a["of those that fire on more than 1% of story tokens"]), round(both / m["with a counterpart in the second dictionary"]["at 0.7 or more"], 2)) == in_both
    s = a["the shared latent"]
    assert (round(s["cosine of the first's best counterpart"], 2), round(s["cosine of the second's best counterpart"], 2)) == shared
    assert s["which is the third's shared latent"] and s["which is the third's shared latent, from the second"]
    # The veto head's first choice in the third dictionary: one latent for the six "She" names, and so many for the six "He" names.
    f = read(f"features/features-{seed}-s2")
    latent = {x["name"]: x["the head's latent"] for x in f["names"]}
    she = f["names that go on with She"]
    assert {latent[p] for p in she} == {f["the shared latent"]} == {s["third"]} and len({latent[p] for p in latent if p not in she}) == he
    if seed == "seed0":
        assert latent["Tom"] == latent["Tim"] == latent["Ben"] == latent["Jack"]
    # The reading on a "He" name, and what the dictionary reads back, over the three dictionaries.
    start = [round(read(f"steer/steer-{seed}{tag}")["kind"]["the reading before the cut on He names, against the usual value"] + 0.0, 2) for tag in ("", "-s1", "-s2")]
    back = [shown(r["values"][2 * ("seed0", "seed1").index(seed) + i]) for r in limits()["dictionaries"]["rows"] if r["what"].startswith("the dictionary reads the latent back") for i in (0, 1)]
    back += [shown(r["values"][("seed0", "seed1").index(seed)]) for r in limits()["dictionaries"]["a third dictionary"]["rows"] if r["what"].startswith("the dictionary reads the latent back")]
    assert (start, back) == {"seed0": ([-0.17, 0.02, -0.03], [0.71, 0.89, 0.84]), "seed1": ([-0.18, -0.17, -0.09], [0.71, 0.74, 0.84])}[seed]


def units():
    with open("runs/limits/units.json") as f:
        return json.load(f)


UNITS = {  # per model and kind: latents, with a counterpart at 0.9 / 0.7 / 0.5 in % and the median, the stricter match in %, latents firing on more than 1% and of those at 0.7 in %, never firing, "She" names sharing a first choice
    "seed0": {"plain": (4096, (3, 9, 30, 0.41), 5.3, (663, 33), [0, 0], [6, 6]), "batch": (4096, (2, 9, 26, 0.35), 5.1, (644, 32), [0, 0], [3, 6]), "wide": (8192, (2, 8, 22, 0.30), 4.7, (530, 52), [6, 6], [6, 6])},
    "seed1": {"plain": (4096, (3, 13, 39, 0.46), 7.7, (701, 48), [0, 0], [6, 5]), "batch": (4096, (3, 13, 32, 0.36), 8.0, (682, 44), [0, 0], [6, 4]), "wide": (8192, (2, 11, 27, 0.32), 6.2, (568, 56), [5, 12], [6, 6])},
}
STRICT = "the same partner by encoder and decoder, 0.7 or more in both"
pct = lambda d: (round(100 * d["at a cosine of 0.9 or more"]), round(100 * d["at 0.7 or more"]), round(100 * d["at 0.5 or more"]), round(d["median cosine"], 2))


def test_a_newer_kind_of_dictionary():
    u = units()
    assert u["kinds"] == {"plain": ["", "s1"], "batch": ["b0", "b1"], "wide": ["w0", "w1"]} and "a fifth" in u["rule"]
    for seed, kinds in UNITS.items():
        for kind, (width, share, strict, often, dead, she) in kinds.items():
            e = u["models"][seed][kind]
            first = e["among latents that fire on more than 1% of story tokens"]
            assert (e["width"], e["batch top-k"], pct(e["with a counterpart in the other seed"]), round(100 * e["one to one"][STRICT], 1)) == (width, kind != "plain", share, strict), (seed, kind)
            assert ((first["latents"], round(100 * first["at 0.7 or more"])), e["latents that never fire on the stories"], e["step 6"]["She names that share the veto head's first choice"]) == (often, dead, she), (seed, kind)
            assert e["story tokens"] == 76800 and all(31.9 < x < 32.5 for x in e["latents on per vector, mean"])
            # The shared latent comes back in every pair: its best counterpart is the other dictionary's own.
            assert e["step 6"]["which is the second's shared latent"] and 0.645 <= e["step 6"]["cosine of the first's best counterpart"] <= 0.865
            assert sum(e["best cosines, counts in 40 bins from 0 to 1"]["the pair"]) == width
            # Trained alike but for the kind, the width and the seed.
            for i, tag in enumerate(u["kinds"][kind]):
                r = read(f"sae/sae-{seed}{'-' + tag if tag else ''}")
                # (the first dictionaries' records were written before a record named its seed, which was 0)
                assert (r["width"], r["k"], r["steps"], r["block"], r.get("seed", 0), r.get("across batch", False)) == (width, 32, 4000, 4, i, kind != "plain")
        # The question, by the rule: the same as the plain pair on both measures, at both widths.
        for kind in ("batch", "wide"):
            assert u["models"][seed][kind]["against the plain dictionaries"] == {"a counterpart at 0.7 or more": "the same", "the stricter one-to-one match": "the same"}
        # The plain pair is the first and the second dictionary of the section on a second dictionary.
        plain, before = u["models"][seed]["plain"], limits()["matches"][seed]
        assert plain["with a counterpart in the other seed"] == before["with a counterpart in the second dictionary"] and plain["one to one"][STRICT] == before["one to one"][STRICT]
    # What a dictionary costs: variance missed and the rise of the loss, for the two dictionaries of a pair.
    cost = lambda seed, kind, key, digits: [round(x, digits) for x in u["models"][seed][kind][key]]
    assert [cost(s, k, "variance missed", 3) for s in UNITS for k in ("plain", "batch", "wide")] == [[0.078, 0.078], [0.082, 0.083], [0.072, 0.073], [0.084, 0.084], [0.090, 0.091], [0.079, 0.079]]
    assert [cost(s, k, "the loss rises by", 2) for s in UNITS for k in ("plain", "batch", "wide")] == [[0.03, 0.03], [0.04, 0.03], [0.03, 0.03], [0.04, 0.04], [0.04, 0.04], [0.04, 0.04]]
    # Looked at after the result: the plain dictionary beside the batch top-k one from the same seed, and from the other.
    look = lambda seed, which: u["models"][seed]["plain"]["set beside another kind, looked at after the result"][f"the batch dictionary from {which} seed"]
    assert [round(100 * look(s, "the same")["at 0.7 or more"]) for s in UNITS] == [16, 17] and [round(100 * look(s, "the other")["at 0.7 or more"]) for s in UNITS] == [8, 11]
    assert [round(100 * look(s, "the same")["the best counterpart has the same number"]) for s in UNITS] == [56, 44] and all(look(s, "the other")["the best counterpart has the same number"] < 0.001 for s in UNITS)


def test_the_newer_kind_after_the_check():
    u = units()
    m, after = u["models"], u["looked at after the check"]
    share = lambda s, k: m[s][k]["with a counterpart in the other seed"]["at 0.7 or more"]
    strict = lambda s, k: m[s][k]["one to one"][STRICT]
    # All eight comparisons of the question are within a fifth of the larger; seed1's wide pair at 0.7 exactly on the line.
    for s in UNITS:
        for k in ("batch", "wide"):
            assert after[s][k]["the question's two comparisons, within a fifth of the larger alone"] == {
                "a counterpart at 0.7 or more": True, "the stricter one-to-one match": True, "by the rule, as above": m[s][k]["against the plain dictionaries"]}
    assert round(share("seed1", "plain") - share("seed1", "wide"), 9) == round(0.2 * share("seed1", "plain"), 9)
    # At twice the width all four shares are lower, by a twentieth to a fifth; in latents they are more.
    lower = [1 - f(s, "wide") / f(s, "plain") for s in UNITS for f in (share, strict)]
    assert all(0.05 <= x <= 0.2 for x in lower) and round(min(lower), 3) == 0.051 and round(max(lower), 3) == 0.200
    assert [round(share(s, k) * m[s][k]["width"]) for s in UNITS for k in ("plain", "wide")] == [353, 670, 545, 872]
    # A fifth to a third of the latents of the batch top-k dictionaries stayed near their first direction; 1% to 2% of the plain ones.
    near = lambda s, k: [round(100 * x[1]) for x in after[s][k]["share of the latents at a cosine to their first direction of 0.9, 0.7 and 0.5 or more, each dictionary"]]
    assert [near(s, k) for s in UNITS for k in ("plain", "batch", "wide")] == [[2, 1], [21, 22], [34, 35], [1, 1], [27, 29], [35, 36]]
    for s in UNITS:
        assert all(1.3 < 100 * x[1] < 1.7 for x in after[s]["plain"]["share of the latents at a cosine to their first direction of 0.9, 0.7 and 0.5 or more, each dictionary"])
        for k in ("plain", "batch", "wide"):
            e = after[s][k]
            assert e["latents near their first direction with a counterpart at 0.7 or more"] == 0
            assert sum(e["best cosines of the latents that left, counts in 40 bins from 0 to 1"]) == e["among the latents that left their first direction"]["latents"]
            assert e["among the latents that left their first direction"]["latents"] + e["latents at 0.7 or more to their first direction, each dictionary"][0] == m[s][k]["width"]
    # They fire rarely, and their best cosines sit where random directions have theirs.
    rate = lambda s, k: [round(100 * x, 2) for x in after[s][k]["first dictionary, how often a latent fires on story tokens, median: near its first direction, the others"]]
    assert [rate(s, "batch") for s in UNITS] == [[0.14, 0.48], [0.08, 0.51]] and all(rate(s, "wide")[0] < rate(s, "batch")[0] for s in UNITS)
    assert [round(after[s][k]["median best cosine of the latents near their first direction"], 2) for s in UNITS for k in ("batch", "wide")] == [0.21, 0.21, 0.20, 0.20]
    edges = np.linspace(0, 1, 41)
    for s in UNITS:
        for k in ("plain", "wide"):
            chance = np.cumsum(m[s][k]["best cosines, counts in 40 bins from 0 to 1"]["random directions"])
            chance = chance / chance[-1]
            assert [round(edges[i], 3) for i in (np.searchsorted(chance, 0.005), np.searchsorted(chance, 0.995) + 1)] == [0.15, 0.25]
    # They make the peak near 0.2: the latents whose best counterpart lies from 0.15 to 0.25, and of them those near their start.
    peak = [after[s][k]["best cosines from 0.15 to 0.25: latents, and of them near their first direction"] for s in UNITS for k in ("batch", "wide")]
    assert peak == [[1150, 771], [3062, 2445], [1259, 1050], [3067, 2626]]
    # Among the latents that left their start: a little higher in all four, the same by the rule, two by its 0.05 alone.
    left = lambda s, k: after[s][k]["among the latents that left their first direction"]
    assert [round(100 * left(s, k)["at 0.7 or more"]) for s in UNITS for k in ("plain", "batch", "wide")] == [9, 11, 12, 14, 18, 16]
    assert [round(left(s, k)["median cosine"], 2) for s in UNITS for k in ("plain", "batch", "wide")] == [0.42, 0.40, 0.40, 0.46, 0.46, 0.45]
    assert [after[s][k]["among the latents that left, against the plain dictionaries"] for s in UNITS for k in ("batch", "wide")] == [
        {"by the rule": "the same", "within a fifth of the larger": True}, {"by the rule": "the same", "within a fifth of the larger": False},
        {"by the rule": "the same", "within a fifth of the larger": False}, {"by the rule": "the same", "within a fifth of the larger": True}]
    # At the same width the batch top-k dictionaries miss 5% to 8% more of the variance, and raise the loss slightly more.
    ratio = [x / m[s]["plain"]["variance missed"][0] for s in UNITS for x in m[s]["batch"]["variance missed"]]
    assert all(1.05 <= x <= 1.08 for x in ratio) and all(x < 1 for s in UNITS for x in m[s]["wide"]["variance missed"])
    loss = lambda s, k: sorted(round(x, 3) for x in m[s][k]["the loss rises by"])
    assert [loss(s, k) for s in UNITS for k in ("plain", "batch")] == [[0.032, 0.033], [0.034, 0.036], [0.039, 0.039], [0.041, 0.041]]
    # After a second check: the verdict among the latents that left depends on the line to the start and on how often they fire.
    under = lambda s, k, line: after[s][k]["among the latents under another line to their first direction, with a counterpart at 0.7 or more"][line]
    assert [after[s][k]["under another line, against the plain dictionaries by the rule"] for s in UNITS for k in ("batch", "wide")] == [
        {"0.9": "the same", "0.5": "the same"}, {"0.9": "the same", "0.5": "higher"}, {"0.9": "the same", "0.5": "higher"}, {"0.9": "the same", "0.5": "higher"}]
    assert [round(100 * under(s, k, "0.5")["share"], 1) for s in UNITS for k in ("plain", "batch", "wide")] == [8.9, 12.4, 16.5, 13.8, 18.9, 20.5]
    fire = lambda s, k: [round(100 * x, 2) for x in after[s][k]["the latents that left, how often they fire: the median, and the share on more than 1% of story tokens"]][0]
    assert [fire(s, k) for s in UNITS for k in ("plain", "batch")] == [0.40, 0.48, 0.39, 0.51]
    assert [round(100 * after[s]["batch"]["the latents that left, with a counterpart at 0.7 or more, weighted to the plain ones' firing rates"]) for s in UNITS] == [10, 15]
    # A latent that fires often comes back more often, in every kind: the 500 that fire most often of every 4,096.
    assert [round(100 * after[s][k]["the latents of the first dictionary that fire most often, 500 of every 4,096, with a counterpart at 0.7 or more"]) for s in UNITS for k in ("plain", "batch", "wide")] == [39, 36, 40, 54, 52, 47]
    assert all(m[s][k]["among latents that fire on more than 1% of story tokens"]["at 0.7 or more"] > 2 * share(s, k) for s in UNITS for k in ("plain", "batch", "wide"))
    # The shared latent, seen from the second dictionary: the first's shared latent in five pairs of six.
    back = {(s, k): after[s][k]["the second's shared latent, its best counterpart in the first"] for s in UNITS for k in ("plain", "batch", "wide")}
    assert [b["which is the first's shared latent"] for b in back.values()] == [True, True, False, True, True, True]
    assert (round(back["seed0", "wide"]["cosine"], 2), round(back["seed0", "wide"]["cosine to the first's shared latent"], 2)) == (0.90, 0.74)
    assert [round(m[s]["batch"]["step 6"]["cosine of the first's best counterpart"], 3) for s in UNITS] == [0.647, 0.700] and m["seed1"]["batch"]["step 6"]["cosine of the first's best counterpart"] >= 0.7
    # Kind against seed: another kind from the same seed, the same kind from another seed, both.
    ks = [after[s]["the plain first dictionary, a counterpart at 0.7 or more"] for s in UNITS]
    assert [[round(100 * k[w]) for w in ("another kind from the same seed", "the same kind from another seed", "another kind from another seed")] for k in ks] == [[16, 9, 8], [17, 13, 11]]
    assert [k["the first two by the rule"] for k in ks] == ["differ", "the same"]


def test_the_table_of_the_newer_kind_is_the_record():
    with open("docs/08-limits.md") as f:
        note = f.read()
    header = "| Between two seeds of a dictionary | seed0: plain | batch | batch, wide | seed1: plain | batch | batch, wide |"
    rows = {cells[0]: cells[1:] for cells in ([c.strip() for c in line.strip("|").split("|")] for line in note[note.index(header):].split("\n\n")[0].split("\n")[2:])}
    u = units()["models"]
    cells = lambda pick: [pick(u[s][k]) for s in ("seed0", "seed1") for k in ("plain", "batch", "wide")]
    two = lambda values, digits: ", ".join(f"{v:.{digits}f}" if digits else str(v) for v in values)
    share = lambda key: lambda e: f"{100 * e['with a counterpart in the other seed'][key]:.0f}%"
    often = "among latents that fire on more than 1% of story tokens"
    assert rows == {
        "latents": cells(lambda e: f"{e['width']:,}"),
        "with a counterpart at a cosine of 0.7 or more": cells(share("at 0.7 or more")),
        "one to one, the same partner by encoder and decoder": cells(lambda e: f"{100 * e['one to one'][STRICT]:.1f}%"),
        "read with the result: with a counterpart at 0.9 or more": cells(share("at a cosine of 0.9 or more")),
        "read with the result: at 0.5 or more": cells(share("at 0.5 or more")),
        "read with the result: median cosine of the best counterpart": cells(lambda e: f"{e['with a counterpart in the other seed']['median cosine']:.2f}"),
        "latents on per vector, mean": cells(lambda e: two(e["latents on per vector, mean"], 2)),
        "latents of the first dictionary that fire on more than 1% of story tokens": cells(lambda e: str(e[often]["latents"])),
        "of those, with a counterpart at 0.7 or more": cells(lambda e: f"{100 * e[often]['at 0.7 or more']:.0f}%"),
        "variance missed on stories, the two dictionaries": cells(lambda e: two(e["variance missed"], 2)),
        "the loss rises by, as a share of the loss": cells(lambda e: two(e["the loss rises by"], 2)),
        "latents that never fire on 76,800 story tokens": cells(lambda e: two(e["latents that never fire on the stories"], 0)),
        '"She" names that share the veto head\'s first choice': cells(lambda e: two(e["step 6"]["She names that share the veto head's first choice"], 0)),
    }


def test_the_open_model_by_depth():
    r = read("patch/patch-qwen")
    at = lambda group: np.mean([t["travel"]["carry"] for t in r["templates"]], 0)[:, r["travel groups"].index(group)]
    assert len(at("S")) == 25 and round(at("S")[14], 2) == 0.98
    assert [round(v, 2) for v in at("end")[[14, 15, 16, 20, 21, 23]]] == [-0.03, 0.25, 0.50, 0.54, 0.76, 1.00]
    behavior = read("read/behavior-qwen")
    assert behavior["model"]["config"]["n_layers"] == 24 and behavior["model"]["config"]["n_heads"] == 14 and behavior["model"]["config"]["dim"] == 896
    assert len(behavior["names"]) == 12


def parts(model, kind="carry", scope="all"):
    """Each part's share in each pattern, (patterns, blocks, heads + 1), the MLP last."""
    return np.array([t["parts"][kind][scope] for t in read(f"patch/patch-{model}")["templates"]])


def test_the_open_model_part_by_part():
    two = lambda v: round(float(v) + 0.0, 2)
    r = read("patch/patch-qwen")
    carry, broken, last, at_name = parts("qwen"), parts("qwen", "break"), parts("qwen", scope="last"), parts("qwen", scope="S")
    assert carry.shape == (5, 24, 15)  # 336 heads and 24 MLPs
    mlp = 14
    # The table: carry, its range over the patterns, at the last position, at the acting name, break.
    row = lambda b, h: [two(carry[:, b - 1, h].mean()), two(carry[:, b - 1, h].min()), two(carry[:, b - 1, h].max()),
                        two(last[:, b - 1, h].mean()), two(at_name[:, b - 1, h].mean()), two(broken[:, b - 1, h].mean())]
    assert row(1, mlp) == [1.00, 1.00, 1.00, 0.00, 1.00, 1.00]
    assert row(21, 9) == [0.22, 0.16, 0.29, 0.21, 0.00, 0.23]
    assert row(14, 2) == [0.18, 0.11, 0.30, 0.00, 0.18, 0.14]
    assert row(17, mlp) == [0.17, 0.13, 0.22, 0.11, 0.04, 0.15]
    assert row(15, 3) == [0.16, 0.09, 0.21, 0.15, 0.00, 0.15]
    assert row(23, mlp) == [0.16, 0.11, 0.18, 0.16, 0.00, 0.16]
    assert row(16, mlp) == [0.14, 0.10, 0.20, 0.13, 0.00, 0.25]
    assert row(24, 12) == [-0.12, -0.14, -0.09, -0.12, 0.00, -0.12]
    assert row(20, 6) == [-0.12, -0.14, -0.07, -0.12, 0.00, -0.13]
    mean = carry.mean(0)
    rest = np.abs(mean).copy()
    rest[0, mlp] = 0
    largest = [(int(b) + 1, int(h)) for b, h in np.dstack(np.unravel_index(np.argsort(-rest, axis=None), rest.shape))[0][:8]]
    assert largest == [(21, 9), (14, 2), (17, mlp), (15, 3), (23, mlp), (16, mlp), (24, 12), (20, 6)]  # the rows after the first
    # The first MLP, and how small the embedding is beside what it writes, in all three models.
    size = lambda model: {k: float(np.mean([t["size at S"][k] for t in read(f"patch/patch-{model}")["templates"]])) for k in ("embedding", "after block 1")}
    assert [two(parts(m)[:, 0, -1].mean()) for m in ("seed0", "seed1")] == [1.00, 0.99]
    assert (round(size("qwen")["embedding"], 3), two(size("qwen")["after block 1"])) == (0.013, 0.25)
    assert [(two(size(m)["embedding"]), two(size(m)["after block 1"])) for m in ("seed0", "seed1")] == [(0.05, 0.85), (0.05, 0.79)]
    assert [round(size(m)["embedding"] / size(m)["after block 1"], 3) for m in ("qwen", "seed0", "seed1")] == [0.052, 0.056, 0.060]
    # No head carries most of it.
    heads = lambda m: parts(m).mean(0)[:, :-1]
    assert [two(np.abs(heads(m)).max()) for m in ("qwen", "seed0", "seed1")] == [0.22, 0.87, 0.69]
    count = lambda m: (int((np.abs(heads(m)) >= 0.05).sum()), int((heads(m) >= 0.05).sum()), heads(m).size)
    assert [count(m) for m in ("qwen", "seed0", "seed1")] == [(14, 9, 336), (7, 6, 48), (4, 4, 48)]
    assert np.abs(heads("qwen")[:13]).max() < 0.03
    assert [two(parts("qwen", scope=s)[:, 14, 0].mean()) for s in ("all", "last", "not last")] == [0.11, 0.05, 0.02]
    # One head acts at the acting name in each model.
    there = lambda m: [(int(b) + 1, int(h), two(parts(m, scope="S").mean(0)[b, h])) for b, h in zip(*np.where(np.abs(parts(m, scope="S").mean(0)[:, :-1]) >= 0.05))]
    assert [there(m) for m in ("qwen", "seed0", "seed1")] == [[(14, 2, 0.18)], [(4, 2, 0.52)], [(4, 4, 0.21)]]
    stream = np.mean([t["travel"]["carry"] for t in r["templates"]], 0)[:, r["travel groups"].index("S")]
    assert [two(stream[14]), two(stream[15])] == [0.98, 0.55]
    # The parts at the last position, at two depths, against the stream.
    at_last = last.mean(0)
    where = [int(b) + 1 for b, h in zip(*np.where(np.abs(at_last) >= 0.05))]
    assert len(where) == 18 and sum(15 <= b <= 17 for b in where) == 3 and sum(20 <= b <= 24 for b in where) == 15 and sum(b in (17, 20, 24) for b in where) == 8
    assert np.abs(mean[17:19]).max() < 0.03 and np.abs(at_last[15, :mlp]).max() < 0.05
    assert [two(at_last[b - 1, mlp]) for b in (16, 17)] == [0.13, 0.11]
    end = np.mean([t["travel"]["carry"] for t in r["templates"]], 0)[:, r["travel groups"].index("end")]
    rise = lambda b: two(end[b] - end[b - 1])
    assert [rise(b) for b in (15, 16, 21, 22, 23)] == [0.28, 0.26, 0.23, 0.09, 0.15] and all(abs(rise(b)) <= 0.03 for b in (17, 20, 24))
    # Five heads have a negative share.
    against = [(int(b) + 1, int(h)) for b, h in zip(*np.where(heads("qwen") <= -0.05))]
    assert against == [(20, 6), (20, 8), (23, 6), (24, 2), (24, 12)]
    assert [two(mean[b - 1, h]) for b, h in against] == [-0.12, -0.09, -0.06, -0.07, -0.12]
    choice = lambda b, h, c: np.array([t["choice"][c][b - 1][h] for t in r["templates"]])
    stand_in = lambda b, h: [two(choice(b, h, c).mean()) for c in ("flipped", "neutral", "mean", "zero")]
    assert [two(choice(b, h, "zero").mean()) for b, h in against] == [0.90, 0.14, 1.16, 0.27, 0.88]
    assert (two(choice(20, 8, "zero").min()), two(choice(20, 8, "zero").max())) == (-0.29, 0.37)
    assert [two(v) for v in heads("seed0")[heads("seed0") <= -0.05]] == [-0.07] and not (heads("seed1") <= -0.05).any()
    # The four largest parts, two at a time, against the sum of their shares.
    ratios = lambda m: [p["together"] / sum(p["each"]) for p in read(f"patch/patch-{m}")["pairs"]]
    assert [(two(min(ratios(m))), two(max(ratios(m)))) for m in ("qwen", "seed0", "seed1")] == [(1.01, 1.19), (0.65, 0.89), (0.81, 1.05)]
    first = r["pairs"][0]
    assert first["parts"] == ["block 21 head 9", "block 14 head 2"] and (two(first["together"]), two(sum(first["each"]))) == (0.47, 0.39)
    assert len(r["pairs"]) == 6 and len({part for p in r["pairs"] for part in p["parts"]}) == 4
    assert [sum(x > 1 for x in ratios(m)) for m in ("qwen", "seed0", "seed1")] == [6, 0, 1]
    rest_of = lambda m: np.delete(parts(m).mean(0).ravel(), parts(m).shape[2] - 1)  # every part but the first MLP
    signed = lambda x: (two(x[x > 0].sum()), two(x[x < 0].sum()))
    assert [signed(rest_of(m)) for m in ("qwen", "seed0", "seed1")] == [(2.76, -0.95), (2.99, -0.13), (2.11, -0.05)]
    small = lambda m: rest_of(m)[np.abs(rest_of(m)) < 0.05]
    assert small("qwen").size == 338 and [signed(small(m)) for m in ("qwen", "seed0", "seed1")] == [(0.95, -0.50), (0.07, -0.06), (0.31, -0.05)]
    # Carry and break.
    apart = lambda m: np.abs(parts(m).mean(0) - parts(m, "break").mean(0))
    assert [two(apart(m).max()) for m in ("qwen", "seed0", "seed1")] == [0.11, 0.07, 0.05] and two(np.sort(apart("qwen").ravel())[-2]) == 0.06
    large = lambda m: np.delete(np.abs(parts(m).mean(0)).ravel() >= 0.05, parts(m).shape[2] - 1)
    assert [round(float(np.median(np.delete(apart(m).ravel(), parts(m).shape[2] - 1)[large(m)])), 3) for m in ("qwen", "seed0", "seed1")] == [0.008, 0.025, 0.013]
    assert [(two(broken[:, b - 1, h].min()), two(broken[:, b - 1, h].max())) for b, h in ((15, 3), (14, 2))] == [(0.04, 0.33), (0.05, 0.26)]  # the last bullet under "does not show"
    assert (two(carry[4, 15, mlp]), two(broken[4, 15, mlp]), two(carry[1, 13, 2]), two(broken[1, 13, 2])) == (0.11, 0.38, 0.30, 0.10)
    within = lambda m: np.delete(np.abs(parts(m) - parts(m, "break")).reshape(5, -1), parts(m).shape[2] - 1, axis=1).max()
    assert two(within("qwen")) == 0.26 and two(within("seed0")) == 0.14 and 0.09 < within("seed1") < 0.10
    kept = np.abs(mean) >= 0.05
    kept[0, mlp] = False
    assert int(kept.sum()) == 21 and int(kept[:, mlp].sum()) == 7
    assert int((np.abs(broken[:, kept]).argmax(0) == 4).sum()) == 12 and int((np.abs(carry[:, kept]).argmax(0) == 1).sum()) == 11
    # What stood in.
    assert two(np.mean([t["logit_diff"]["clean"] for t in r["templates"]])) == 7.07 and stand_in(21, 9) == [-3.67, -2.27, -2.12, -2.53]
    assert round(3.67 / 2.27, 1) == 1.6 and stand_in(20, mlp) == [-0.95, -1.28, -0.32, 2.01] and stand_in(21, 12) == [-1.34, -1.04, 0.10, 0.12]
    assert (choice(20, mlp, "zero") > 0).all() and all((choice(21, 12, c) > 0).any() and (choice(21, 12, c) < 0).any() for c in ("mean", "zero"))


def looks(model):
    with open(f"runs/open/heads-{model}.json") as f:
        return json.load(f)


def test_what_the_open_models_heads_read_and_write():
    two = lambda v: round(float(v) + 0.0, 2)
    r = looks("qwen")
    T = r["templates"]
    name = lambda b, h: f"block {b} head {h}"
    reads = lambda b, h, version, place, record=T: np.array([t["attention"][name(b, h)][version][place] for t in record])
    adds = lambda b, h, version, to, record=T: two(np.mean([t["writes"][name(b, h)][version][to] for t in record]))
    assert len(r["heads"]) == 14 and [x["acts at"] for x in r["heads"]].count("S") == 1
    # The head at the acting name reads that name's first mention, and the story models' read the other name.
    assert (two(reads(14, 2, "clean", "A").mean()), two(reads(14, 2, "flipped", "B").mean())) == (0.56, 0.75)
    assert [two(f(reads(14, 2, v, k))) for v, k in (("clean", "A"), ("flipped", "B")) for f in (np.min, np.max)] == [0.51, 0.64, 0.71, 0.81]
    assert (two(reads(14, 2, "clean", "B").mean()), two(reads(14, 2, "flipped", "A").mean()), two(reads(14, 2, "neutral", "first token").mean())) == (0.02, 0.01, 0.96)
    story = {m: looks(m)["templates"] for m in ("seed0", "seed1")}
    assert [two(reads(4, h, v, k, story[m]).mean()) for m, h in (("seed0", 2), ("seed1", 4)) for v, k in (("clean", "B"), ("flipped", "A"))] == [0.91, 0.78, 0.85, 0.79]
    # Step 4's hold: the head at the acting name works through the two heads of block 15, the fetching heads through a veto head.
    held = lambda record: {h["receiver"]: (two(np.mean([t["holds"][i]["alone"] for t in record])), two(np.mean([t["holds"][i]["held"] for t in record]))) for i, h in enumerate(record[0]["holds"])}
    left = held(T)
    assert left[name(15, 3)] == (0.18, 0.02) and left[name(15, 0)] == (0.18, 0.04)
    later = [v[1] for k, v in left.items() if not k.startswith("block 15 ")]
    assert len(left) == 13 and len(later) == 11 and (min(later), max(later)) == (0.12, 0.30)
    exact = {h["receiver"]: float(np.mean([t["holds"][i]["held"] for t in T])) for i, h in enumerate(T[0]["holds"])}
    alone = float(np.mean([t["holds"][0]["alone"] for t in T]))
    assert sum(v > alone for k, v in exact.items() if not k.startswith("block 15 ")) == 7
    second = {h["receiver"]: (two(h["alone"]), two(h["held"])) for h in T[1]["holds"]}
    assert second[name(15, 3)] == (0.30, 0.17)
    assert held(story["seed0"])[name(7, 0)] == (0.52, 0.06) and held(story["seed1"])[name(8, 0)] == (0.21, 0.08)
    # The two heads of block 15 read the acting name's position and add nothing to the names directly.
    assert [two(reads(15, h, v, "S").mean()) for h in (3, 0) for v in ("clean", "flipped")] == [0.35, 0.39, 0.51, 0.56]
    assert all(reads(15, 0, v, "S")[i] == max(t["attention"][name(15, 0)][v][k] for k in r["places"]) for v in ("clean", "flipped") for i, t in enumerate(T))
    assert [two(reads(15, 3, v, "B").mean()) for v in ("clean", "flipped")] == [0.24, 0.23]
    assert [int((reads(15, 3, v, "B") > reads(15, 3, v, "S")).sum()) for v in ("clean", "flipped", "neutral")] == [2, 2, 2]
    assert all(abs(adds(15, h, v, k)) < 0.015 for h in (3, 0) for v in ("clean", "flipped") for k in ("A", "B", "C"))
    # Outside the seven places.
    other = lambda b, h, v: np.array([t["elsewhere"][name(b, h)][v]["weight"] for t in T])
    assert [two(other(20, 11, v).mean()) for v in ("clean", "flipped")] == [0.26, 0.28] and other(20, 11, "clean")[2] > reads(20, 11, "clean", "end")[2]
    assert int((other(24, 12, "flipped") > reads(24, 12, "flipped", "A")).sum()) == 1 and not (other(24, 12, "clean") > reads(24, 12, "clean", "B")).any()
    # The table: what a head reads most, and what it adds to the name that acts and to the name that is due, clean then flipped.
    acts, due = {"clean": "A", "flipped": "B"}, {"clean": "B", "flipped": "A"}
    row = lambda b, h, place: ([two(reads(b, h, v, place(v)).mean()) for v in ("clean", "flipped")],
                               [adds(b, h, v, acts[v]) for v in ("clean", "flipped")], [adds(b, h, v, due[v]) for v in ("clean", "flipped")])
    at_name, due_name, own = (lambda v: "S"), (lambda v: due[v]), (lambda v: "end")
    assert row(21, 9, at_name) == ([0.42, 0.47], [-2.01, -2.22], [-0.27, -0.27])
    assert row(23, 0, at_name) == ([0.27, 0.29], [-1.04, -1.18], [-0.59, -0.42])
    assert row(22, 0, at_name) == ([0.35, 0.38], [-0.67, -0.74], [-0.16, -0.12])
    assert row(21, 12, due_name) == ([0.53, 0.61], [0.67, 0.34], [1.23, 1.67])
    assert row(22, 11, due_name) == ([0.45, 0.40], [0.46, 0.37], [0.83, 0.85])
    assert row(24, 12, due_name) == ([0.42, 0.34], [-0.28, -0.14], [-1.11, -0.96])
    assert row(24, 2, due_name) == ([0.33, 0.36], [-0.28, -0.17], [-0.63, -0.73])
    assert row(20, 6, due_name) == ([0.60, 0.42], [0.00, 0.00], [0.00, 0.00])
    assert row(20, 8, own) == ([0.53, 0.55], [0.00, 0.00], [-0.01, -0.01])
    assert row(20, 11, own) == ([0.36, 0.35], [0.00, 0.00], [0.01, 0.01])
    assert row(23, 6, due_name)[1:] == ([0.97, 0.95], [0.48, 0.60])
    assert two(max(reads(23, 6, v, k).mean() for v in ("clean", "flipped") for k in r["places"])) == 0.19
    assert {(x["block"], x["head"]): two(x["carry"]) for x in r["heads"]}[(21, 9)] in (0.21, 0.22)  # 0.215 in the record of three decimals
    # Seed0's veto head beside them, and no head of the story models that reads the due name and lowers it.
    seven = lambda place: row_of("seed0", 7, 0, place)
    row_of = lambda m, b, h, place: ([two(reads(b, h, v, place(v), story[m]).mean()) for v in ("clean", "flipped")],
                                     [adds(b, h, v, acts[v], story[m]) for v in ("clean", "flipped")], [adds(b, h, v, due[v], story[m]) for v in ("clean", "flipped")])
    assert seven(at_name) == ([0.79, 0.83], [-1.38, -1.56], [0.42, 0.52])
    most = lambda record, b, h, v: max(r["places"], key=lambda k: np.mean([t["attention"][name(b, h)][v][k] for t in record]))
    for m in ("seed0", "seed1"):
        for x in looks(m)["heads"]:
            if x["acts at"] == "end":
                _, _, to_due = row_of(m, x["block"], x["head"], due_name)
                reads_due_most = all(most(story[m], x["block"], x["head"], v) == due[v] for v in ("clean", "flipped"))
                assert not (reads_due_most and max(to_due) < 0)
    assert all(most(T, b, h, v) == due[v] for b, h in ((24, 12), (24, 2), (21, 12), (22, 11)) for v in ("clean", "flipped"))
    # The veto heads: the range of the strongest, the edge case, and the neutral prompts.
    both = np.concatenate([reads(21, 9, v, "S") for v in ("clean", "flipped")])
    assert (two(both.min()), two(both.max())) == (0.25, 0.62)
    assert [two(reads(3, 2, v, "S", story["seed0"]).mean()) for v in ("clean", "flipped")] == [0.36, 0.36]  # the weakest story-model head that meets step 4's definition
    assert [two(reads(15, h, "neutral", "S").mean()) for h in (0, 3)] == [0.46, 0.29]
    assert all(0.33 < x < 0.6 for x in (0.59 / 1.04, 0.42 / 1.18))
    assert [two(reads(b, h, "neutral", "S").mean()) for b, h in ((21, 9), (22, 0), (23, 0))] == [0.26, 0.23, 0.17]
    assert [two(reads(b, h, "neutral", "S", story[m]).mean()) for m, b, h in (("seed0", 7, 0), ("seed1", 8, 0))] == [0.83, 0.95]
    assert [adds(b, h, "neutral", "C") for b, h in ((21, 9), (22, 0), (23, 0))] == [-1.31, -0.47, -0.69]
    # Step 6's measure: against the uninvolved names of the name's own kind.
    own = lambda b, h, role, kind, record=T: [two(np.mean([t["by kind"][name(b, h)][v][role][kind] for t in record])) for v in ("clean", "flipped")]
    assert r["names that go on with She"] == looks("seed0")["names that go on with She"] == read("steer/steer-seed0")["names that go on with She"]
    assert (own(7, 0, "acts", "She", story["seed0"]), own(7, 0, "acts", "He", story["seed0"])) == ([-0.26, -0.31], [-1.12, -1.24])
    assert (own(21, 9, "acts", "She"), own(21, 9, "acts", "He")) == ([-1.59, -1.82], [-1.98, -2.18])
    assert (own(22, 0, "acts", "She"), own(22, 0, "acts", "He")) == ([-0.90, -1.00], [-0.36, -0.38])
    assert (own(22, 11, "due", "She"), own(22, 11, "due", "He")) == ([1.20, 1.22], [0.31, 0.31])
    # Only those two differ between the kinds by a factor of more than two, in the role where the head writes most.
    factor = lambda b, h, role: [max(abs(s), abs(m)) / min(abs(s), abs(m)) for s, m in zip(own(b, h, role, "She"), own(b, h, role, "He"))]
    assert all(f > 2 for f in factor(22, 0, "acts") + factor(22, 11, "due"))
    assert all(f < 2 for b, h, role in ((21, 9, "acts"), (23, 0, "acts"), (21, 12, "due"), (24, 12, "due"), (24, 2, "due")) for f in factor(b, h, role))


def test_the_heads_script_gives_step_4s_numbers_on_the_story_models():
    """scripts/heads.py repeats two measures of scripts/circuit.py on another choice of heads."""
    for model in ("seed0", "seed1", "seed2"):
        mine = looks(model)["templates"]
        with open(f"runs/circuit/circuit-{model}.json") as f:
            theirs = json.load(f)["templates"]
        shared = 0
        for a, b in zip(mine, theirs):
            assert a["template"] == b["template"]
            for kind in ("attention", "writes"):
                for head in set(a[kind]) & set(b[kind]):
                    for version in a[kind][head]:
                        for key in set(a[kind][head][version]) & set(b[kind][head][version]):
                            assert abs(a[kind][head][version][key] - b[kind][head][version][key]) < 2e-4
                            shared += 1
            theirs_held = {(h["sender"], h["receiver"]): h for h in b["holds"]}
            for h in a["holds"]:
                if (h["sender"], h["receiver"]) in theirs_held:
                    assert all(abs(h[k] - theirs_held[h["sender"], h["receiver"]][k]) < 2e-4 for k in ("alone", "held"))
                    shared += 1
        assert shared > 300


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
