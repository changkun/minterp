"""
The behavior that every step explains.

    One day, Lily and Tom were playing in the garden. Lily said hello to

Two names are introduced, one of them acts, and the text stops where a
receiver is due. A reader expects the other name, and so does the model: it
continues with " Tom". The task is small enough to take apart and has a
clean measure, the logit of the right name minus the logit of the wrong one
at the last position.

Each prompt comes in three versions that differ in one token, the name that
acts (S):

    clean     S = A, the first name      the answer is B
    flipped   S = B, the second name     the answer is A
    neutral   S = C, a third name        neither A nor B is due more

Each prompt begins with the token that opens every story in the training
data, as the stories the model saw did.

Later steps move activations from one version into another. That works
position by position only because every name is one token, so the three
versions have the same length and the same token at every other position.
"""

import itertools
from dataclasses import dataclass

import torch

from .common import eot

# Names that TinyStories uses often. Only those that are a single token
# after a space are kept, which depends on the tokenizer (see names()).
CANDIDATES = ["Lily", "Tom", "Tim", "Ben", "Mia", "Sam", "Lucy", "Max", "Anna", "Sue", "Jack", "Sara"]

# How these came about: a first check on one model tried the first two and
# another wording of the third. The third was reworded so that every name
# follows a space, which makes it the same token wherever it stands, and the
# last two were added before the full run. None was dropped or changed for
# its result. The names were fixed in that first check.
TEMPLATES = [
    "Once upon a time, {A} and {B} went to the park. {S} gave a toy to",
    "One day, {A} and {B} were playing in the garden. {S} said hello to",
    "Once there were two friends, {A} and {B}. One day, {S} gave a big hug to",
    "One sunny day, {A} and {B} had a red ball. {S} threw the ball to",
    "One morning, {A} and {B} went to the zoo. {S} showed the map to",
]


@dataclass
class Batch:
    template: str
    names: list  # (A, B, C) of every prompt
    clean: torch.Tensor  # (prompts, seq) token ids
    flipped: torch.Tensor
    neutral: torch.Tensor
    right: torch.Tensor  # (prompts,) the id of B, the answer to the clean prompt
    wrong: torch.Tensor  # (prompts,) the id of A
    pos: dict  # where A, B and S stand, and "end", the last position
    labels: list  # one string per position: the token, or A, B, S

    def __len__(self):
        return len(self.names)


def names(tok, candidates=CANDIDATES):
    """The candidates that are one token after a space."""
    return [n for n in candidates if len(tok.encode_ordinary(" " + n)) == 1]


def build(tok, template, people, device="cpu"):
    """All ordered pairs of names in one template, in its three versions."""
    start = eot(tok)  # every story of the training data opens with it
    ident = {n: tok.encode_ordinary(" " + n)[0] for n in people}

    def encode(a, b, s):
        return [start] + tok.encode_ordinary(template.format(A=a, B=b, S=s))

    triples, clean, flipped, neutral = [], [], [], []
    for k, (a, b) in enumerate(itertools.permutations(people, 2)):
        # A third name, taken in turn from the other names, so that which
        # name it is does not depend on A or on B.
        others = [n for n in people if n not in (a, b)]
        c = others[k % len(others)]
        triples.append((a, b, c))
        clean.append(encode(a, b, a))
        flipped.append(encode(a, b, b))
        neutral.append(encode(a, b, c))
    assert len({len(p) for p in clean + flipped + neutral}) == 1, "a name is more than one token"

    a, b, _ = triples[0]
    first = clean[0]
    at_a = [i for i, t in enumerate(first) if t == ident[a]]
    at_b = [i for i, t in enumerate(first) if t == ident[b]]
    assert len(at_a) == 2 and len(at_b) == 1, "the template must name A twice and B once"
    pos = {"A": at_a[0], "B": at_b[0], "S": at_a[1], "end": len(first) - 1}
    labels = [tok.decode([t]).strip() or "·" for t in first]
    for key in ("A", "B", "S"):
        labels[pos[key]] = key

    ids = lambda rows: torch.tensor(rows, device=device)
    return Batch(
        template, triples, ids(clean), ids(flipped), ids(neutral),
        right=ids([ident[b] for _, b, _ in triples]), wrong=ids([ident[a] for a, _, _ in triples]),
        pos=pos, labels=labels,
    )


def batches(tok, device="cpu", templates=TEMPLATES, people=None):
    people = people or names(tok)
    return [build(tok, t, people, device) for t in templates]


def logit_diff(logits, right, wrong):
    """
    The logit of the right name minus the logit of the wrong one, per prompt.
    logits is (prompts, seq, vocab), read at the last position, or already
    (prompts, vocab). A difference of 2.3 means the model holds the right
    name ten times as likely as the wrong one, whatever else it considers.
    """
    last = logits[:, -1] if logits.dim() == 3 else logits
    return last.gather(1, right[:, None])[:, 0] - last.gather(1, wrong[:, None])[:, 0]
