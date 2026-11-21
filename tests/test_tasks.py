import torch

from minterp import tasks
from minterp.common import eot


def test_names_keeps_single_tokens(tok, people):
    assert tasks.names(tok, people) == people

    class Splits(type(tok)):
        def encode_ordinary(self, text):
            return [1, 2] if text == " " + people[0] else super().encode_ordinary(text)

    two = Splits.__new__(Splits)
    two.__dict__ = tok.__dict__
    assert tasks.names(two, people) == people[1:]


def test_the_three_versions_differ_in_one_token(tok, people):
    for batch in tasks.batches(tok, people=people):
        n = len(people) * (len(people) - 1)
        assert len(batch) == n and batch.clean.shape == batch.flipped.shape == batch.neutral.shape
        assert (batch.clean[:, 0] == eot(tok)).all()
        s = batch.pos["S"]
        for other in (batch.flipped, batch.neutral):
            differs = other != batch.clean
            assert differs[:, s].all() and differs.sum() == n
        assert batch.pos["A"] < batch.pos["B"] < s < batch.pos["end"] == batch.clean.shape[1] - 1
        assert (batch.clean[:, batch.pos["A"]] == batch.wrong).all()
        assert (batch.clean[:, batch.pos["B"]] == batch.right).all()
        assert (batch.clean[:, s] == batch.wrong).all()  # A acts, so B is the answer
        assert (batch.flipped[:, s] == batch.right).all()
        for (a, b, c), row in zip(batch.names, batch.neutral):
            assert len({a, b, c}) == 3 and tok.decode([row[s].item()]) == " " + c
        # The third name is not tied to the second: it stands at every distance from it in the list.
        offsets = {(people.index(c) - people.index(b)) % len(people) for _, b, c in batch.names}
        assert offsets == set(range(1, len(people)))
        counts = [sum(c == n for _, _, c in batch.names) for n in people]
        assert max(counts) - min(counts) <= 2
        # The fourth version differs from the clean one at the second name only, and has C there.
        differs = batch.other != batch.clean
        assert differs[:, batch.pos["B"]].all() and differs.sum() == n
        assert (batch.other[:, batch.pos["B"]] == batch.third).all()
        differs = batch.other_first != batch.flipped
        assert differs[:, batch.pos["A"]].all() and differs.sum() == n
        assert (batch.other_first[:, batch.pos["A"]] == batch.third).all()
        assert [batch.labels[batch.pos[k]] for k in ("A", "B", "S")] == ["A", "B", "S"]
        assert len(batch.labels) == batch.clean.shape[1]


def test_logit_diff_reads_the_last_position():
    logits = torch.zeros(2, 3, 5)
    logits[0, -1, 4], logits[0, -1, 1] = 3.0, 1.0
    logits[1, 0, 4] = 9.0  # not the last position
    right, wrong = torch.tensor([4, 4]), torch.tensor([1, 1])
    assert tasks.logit_diff(logits, right, wrong).tolist() == [2.0, 0.0]
    assert tasks.logit_diff(logits[:, -1], right, wrong).tolist() == [2.0, 0.0]
