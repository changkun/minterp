import torch

from minterp import patch, tasks
from minterp.tap import head_writes, Tap


def setup(model, tok, people):
    batch = tasks.batches(tok, people=people)[0]
    clean, clean_logits = patch.record(model, batch.clean)
    flipped, flipped_logits = patch.record(model, batch.flipped)
    return batch, clean, clean_logits, flipped, flipped_logits


def test_the_whole_stream_carries_the_whole_answer(model, tok, people):
    batch, clean, clean_logits, _, _ = setup(model, tok, people)
    for depth in range(model.cfg.n_layers + 1):
        out = patch.run(model, batch.flipped, {("resid", depth): patch.swap(clean["resid", depth])})
        assert torch.allclose(out[:, -1], clean_logits[:, -1], atol=1e-5)


def test_a_patch_where_the_runs_agree_changes_nothing(model, tok, people):
    batch, clean, _, _, flipped_logits = setup(model, tok, people)
    before = list(range(batch.pos["S"]))  # the two versions are the same tokens up to the acting name
    for depth in range(model.cfg.n_layers + 1):
        out = patch.run(model, batch.flipped, {("resid", depth): patch.swap(clean["resid", depth], positions=before)})
        assert torch.allclose(out, flipped_logits, atol=1e-5)


def test_a_patch_reaches_no_earlier_position(model, tok, people):
    batch, clean, _, _, flipped_logits = setup(model, tok, people)
    end = batch.pos["end"]
    out = patch.run(model, batch.flipped, {("resid", 1): patch.swap(clean["resid", 1], positions=[end])})
    assert torch.allclose(out[:, :end], flipped_logits[:, :end], atol=1e-5)
    assert not torch.allclose(out[:, end], flipped_logits[:, end], atol=1e-4)


def test_swapping_one_head_moves_only_that_heads_write(model, tok, people):
    batch, clean, _, flipped, _ = setup(model, tok, people)
    with Tap(model, edit={("heads", 2): patch.swap(clean["heads", 2], head=1)}) as tap:
        model(batch.flipped)
    writes = head_writes(model, tap, 2)
    theirs = head_writes(model, type("S", (), {"__getitem__": lambda self, k: clean[k]})(), 2)
    ours = head_writes(model, type("S", (), {"__getitem__": lambda self, k: flipped[k]})(), 2)
    assert torch.allclose(writes[:, :, 1], theirs[:, :, 1], atol=1e-5)
    for h in (0, 2, 3):
        assert torch.allclose(writes[:, :, h], ours[:, :, h], atol=1e-5)


def test_mean_zero_and_share():
    a, b = {"k": torch.ones(2, 3, 4)}, {"k": 3 * torch.ones(2, 3, 4)}
    assert patch.mean([a, b], "k").shape == (1, 3, 4) and (patch.mean([a, b], "k") == 2).all()
    assert (patch.zero(a["k"]) == 0).all()
    edit = patch.swap(patch.mean([a, b], "k"), positions=[1])
    out = edit(a["k"])
    assert (out[:, 1] == 2).all() and (out[:, 0] == 1).all() and (a["k"] == 1).all()  # the input is left alone
    t, s = torch.tensor([0.0, 2.0]), torch.tensor([4.0, 6.0])
    assert patch.share(t, t, s) == 0.0 and patch.share(s, t, s) == 1.0 and patch.share((t + s) / 2, t, s) == 0.5
