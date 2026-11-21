"""
What docs/09-stack.md claims about the one library that was run, checked
against runs/stack/. The record was made with the library laid over the
project (`uv run --with transformer-lens==4.4.0 --with transformers==5.24.0
scripts/stack.py`), so these tests need neither the library nor the open
model. What the note says of documentation cannot be tested here.
"""

import json


def run():
    with open("runs/stack/stack-qwen.json") as f:
        return json.load(f)


def test_the_versions_and_the_prompts_the_note_names():
    r = run()
    assert r["versions"] == {"transformer-lens": "4.4.0", "transformers": "5.24.0", "torch": "2.15.0"}
    assert (r["prompts"]["count"], r["prompts"]["of"], r["prompts"]["first names"]) == (16, 132, {"Lily": 11, "Tom": 5})
    assert r["the tokenizers decode the prompt alike"] and r["a string gets no first token from the library"]
    # The open model's blocks, as transformers builds it: what a wrapper without names of its own exposes.
    assert r["module paths of the transformers model"] == {"model.layers": 24, "lm_head": True, "transformer.h": False}


def test_three_comparisons_of_logits_and_what_wrapping_changes():
    r = run()
    logits = r["logits"]
    assert round(logits["largest entry"], 1) == 21.4
    three = [logits[k] for k in ("minllm's loader against transformers, before wrapping", "the library against minterp", "the library against transformers, before wrapping")]
    assert all(round(v, 5) == 0.00008 for v in three) and len(set(three)) == 3  # three measurements, not one
    # Wrapped, the library computes the attention itself; in compatibility mode the MLPs as well.
    assert r["calls of transformers' own modules in one pass"] == {"before wrapping": {"attention": 24, "MLP": 24}, "wrapped": {"attention": 0, "MLP": 24},
                                                                  "wrapped, in compatibility mode": {"attention": 0, "MLP": 0}}


def test_the_same_pieces_under_other_names():
    r = run()
    pieces = r["pieces"]
    largest = lambda kind, key: max(v[key] for k, v in pieces.items() if k.startswith(f'("{kind}"'))
    assert pieces['("resid", 0) against hook_embed'] == {"difference": 0, "largest entry": pieces['("resid", 0) against hook_embed']["largest entry"]}
    assert round(pieces['("resid", 0) against hook_embed']["largest entry"], 2) == 0.07
    # The stream: the largest difference is after block 24, the largest entry after block 12.
    assert round(pieces['("resid", 24) against blocks.23.hook_resid_post']["difference"], 4) == 0.0002 == round(largest("resid", "difference"), 4)
    assert (round(pieces['("resid", 24) against blocks.23.hook_resid_post']["largest entry"]), round(pieces['("resid", 12) against blocks.11.hook_resid_post']["largest entry"])) == (40, 1394)
    assert (round(r["sizes"]["the stream after the middle block, without the first position"]), round(r["sizes"]["attention weights"], 2)) == (12, 1.0)
    assert round(max(largest("attn", "difference"), largest("mlp", "difference")), 4) == 0.0001 and round(max(largest("attn", "largest entry"), largest("mlp", "largest entry"))) == 63
    assert round(largest("heads", "difference"), 5) == 0.00002 and round(largest("heads", "largest entry")) == 6
    assert round(max(r["pattern"]), 6) == 0.000007 and len(pieces) == 13
    assert r["names"] == {'("resid", 0)': "hook_embed", '("resid", i)': "blocks.i-1.hook_resid_post", '("attn", i)': "blocks.i-1.hook_attn_out",
                          '("mlp", i)': "blocks.i-1.hook_mlp_out", '("heads", i)': "blocks.i-1.attn.hook_z", "attention.patterns(model, tap)[:, i-1]": "blocks.i-1.attn.hook_pattern"}
    # The names of the table are aliases: the library's own names for the same hooks, read from the wrapped model.
    assert r["canonical names"] == {"hook_embed": "embed.hook_out", "blocks.i.hook_resid_post": "blocks.i.hook_out", "blocks.i.hook_attn_out": "blocks.i.attn.hook_out",
                                    "blocks.i.hook_mlp_out": "blocks.i.mlp.hook_out", "blocks.i.attn.hook_z": "blocks.i.attn.o.hook_in"}
    # The name a hook function is given: canonical on a model as wrapped; in the mode, the alias for two of the four tried.
    seen = r["the name a hook function sees"]
    asked = ("hook_embed", "blocks.11.hook_resid_post", "blocks.11.attn.hook_z", "blocks.11.attn.hook_pattern")
    assert [seen[f"as wrapped: {name}"] for name in asked] == ["embed.hook_out", "blocks.11.hook_out", "blocks.11.attn.o.hook_in", "blocks.11.attn.hook_pattern"]
    assert sum(seen[f"in compatibility mode: {name}"] == name and seen[f"as wrapped: {name}"] != name for name in asked) == 2


def test_two_interventions_both_ways():
    r = run()
    p, s = r["patch"], r["steer"]
    assert p["where"] == "after block 12, at the acting name"
    assert (round(p["plain"], 3), round(p["by minterp"], 3), round(p["by the library"], 3)) == (6.552, -7.431, -7.431)
    assert (round(s["by minterp"], 3), round(s["by the library"], 3)) == (0.087, 0.087)
    assert p["logits, largest difference"] < 1e-3 and s["logits, largest difference"] < 1e-3


def test_the_helper_that_needed_care():
    a = run()["attribute"]
    assert (a["parts"], a["parts of lens.attribute before the heads are summed"]) == (49, 361)
    assert (round(a["the logit difference"], 3), round(a["minterp's parts sum to"], 3)) == (6.552, 6.552)
    raw = a["the library's helper on the published weights"]
    assert (round(raw["sums to"], 3), round(raw["largest difference to minterp, per part"], 2)) == (0.793, 5.04)
    gain = a["the final norm's gain"]
    assert (round(gain["smallest"], 2), round(gain["largest"], 1)) == (-0.01, 16.8)
    by_hand, in_mode = a["with the stack multiplied by the gain first"], a["in compatibility mode"]
    assert (round(by_hand["sums to"], 3), round(in_mode["sums to"], 3)) == (6.552, 6.552)
    assert 0.00001 < by_hand["largest difference to minterp, per part"] < 0.00002 and 0.00001 < in_mode["largest difference to minterp, per part"] < 0.00002
    # For one token's logit the helper's parts sum to the centered logit.
    assert (round(in_mode["for one token's logit it sums to"], 2), round(in_mode["that token's published logit"], 2)) == (22.48, 18.91)
    assert abs(in_mode["for one token's logit it sums to"] - in_mode["the published logit less its mean over the vocabulary"]) < 1e-3


def test_what_else_the_mode_changes():
    m = run()["mode"]
    assert round(m["logits, largest difference to the published model's"], 2) == 5.33 and m["log-probabilities, largest difference"] < 0.0001
    # One constant over the vocabulary at each prompt and position: the mean of the published logits there.
    assert m["the shift within one prompt and position, largest spread over the vocabulary"] < 0.0002
    assert m["the shift against the mean of the published logits over the vocabulary"] < 0.0001
    # The record's shift is the published logits less the mode's, which is that mean: it is negative everywhere.
    whole, last = m["the shift over prompts and positions"], m["at the last position"]
    assert (round(whole["smallest"], 2), round(whole["largest"], 2)) == (-5.33, -0.25) and (round(last["smallest"], 2), round(last["largest"], 2)) == (-4.19, -3.06)
    # The heads record something else in the mode; what the attention and the MLP write does not change.
    pieces = m["pieces after block 12 against minterp's, on the published weights"]
    assert (round(pieces["heads"], 1), round(m["largest entry of the heads there"], 1)) == (1.3, 2.9)
    assert pieces["attn"] < 0.0001 and pieces["mlp"] < 0.0001
    # The heads' change is one vector, the same at every prompt and position, as large as the block's value bias.
    change = m["the heads' change"]
    assert change["largest spread over prompts and positions"] < 0.0001 and abs(change["largest entry"] - change["largest entry of that block's value bias"]) < 0.0001
