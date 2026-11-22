# 8. Limits

Scripts: [`scripts/limits.py`](../scripts/limits.py),
[`scripts/open.py`](../scripts/open.py),
[`scripts/heads.py`](../scripts/heads.py) ·
Tests: [`tests/test_limits.py`](../tests/test_limits.py),
[`tests/test_open.py`](../tests/test_open.py) ·
Results: [`runs/limits/`](../runs/limits/), [`runs/open/`](../runs/open/)

Every note so far ended with a list of what it does not show. Since step 5
the lists say that there is one dictionary per model. Two more limits were
never on a list: both models come from one recipe and differ only in the
seed of their training, and nobody else uses them. This step moves each of
the three a little.

- **Two seeds.** What did the two story models agree on, by one rule
  applied to 24 numbers picked from the seven notes?
- **A second dictionary.** Steps 5 to 7 again with a dictionary trained in
  the same way from another seed.
- **Another model.** Parts of steps 1, 3, 4 and 7 on a published model of 0.5
  billion parameters, 35 times the story model.

A later section sets a third model and a third dictionary beside the first
two of each, read by a rule for three that was fixed before they were run.

The rule: two numbers are called the same when they differ by at most a
fifth of the larger, or by 0.05 in the row's own unit. A count or a name
is the same when it is equal. A fifth is arbitrary. The rule is lenient
near a ceiling and strict on counts: 1.00 against 0.95 below is 1 against
30 wrong prompts of 660, while 71 against 72 pronouns counts as a
difference. The tables give the numbers, so another threshold can be
applied.

```sh
# in minllm: uv run scripts/download_qwen.py       the open model, 1 GB
export MINTERP_RUNS=runs                        # so that each script finds the last one's record

uv run scripts/sae.py --seed 1 --dict s1        # a second dictionary; also with --model seed1
uv run scripts/features.py --dict s1            # step 6 on it
uv run scripts/steer.py --dict s1               # step 7 on it

uv run scripts/behavior.py --model qwen         # step 1 on the open model
uv run scripts/lens.py --model qwen
uv run scripts/patch.py --model qwen --stream-only   # step 3's first half: minutes
uv run scripts/patch.py --model qwen           # and with every part moved: hours; it writes the same file
uv run scripts/heads.py                         # where the heads it finds look, and what they write
uv run scripts/heads.py --model seed0           # the same on the story models, to set beside step 4
uv run scripts/heads.py --model seed1
uv run scripts/open.py                          # step 7's first experiment there
uv run scripts/open.py --model seed0 --block 4  # and on the story models, by depth
uv run scripts/open.py --model seed1 --block 4

uv run scripts/limits.py                        # the tables of this note
```

## Two seeds

| Step | Number | seed0 | seed1 | |
|---|---|---|---|---|
| 1 | clean prompts with B above A, share | 1.00 | 0.95 | same |
| 1 | flipped prompts with A above B, share | 0.85 | 0.76 | same |
| 1 | share of the clean logit difference written by the last quarter of the blocks | 0.94 | 0.90 | same |
| 2 | the due name read at the acting name at half depth, on new pairs, clean prompts | 0.95 | 0.80 | same |
| 2 | the same on flipped prompts | 0.90 | 0.71 | differs |
| 3 | the last depth at which the acting name alone carries half of the gap, as a share of the depth | 0.75 | 0.88 | same |
| 4 | the head that carries most at the last position | block 7, head 0 | block 8, head 0 | differs |
| 4 | the seven heads of the circuit together carry, share of the gap | 1.07 | 0.97 | same |
| 5 | variance the dictionary misses on stories, share | 0.08 | 0.08 | same |
| 5 | the loss rises by, as a share of the loss | 0.03 | 0.04 | same |
| 6 | who acts carried by what the dictionary rebuilds, share of the gap | 0.94 | 1.00 | same |
| 6 | carried by the highest-ranked latent | 0.17 | 0.27 | differs |
| 6 | of the six "She" names, how many share the veto head's latent | 6 | 6 | same |
| 6 | the veto head's write to an acting "She" name against its own kind, logits | −0.26 | −0.23 | same |
| 7 | the pair direction at dose 1, share of the way | 0.95 | 1.04 | same |
| 7 | its part with a term per name | 0.96 | 1.04 | same |
| 7 | the names' means with the vector every flipped prompt shares | 0.81 | 0.85 | same |
| 7 | the names' means alone | 0.38 | 0.81 | differs |
| 7 | the shared vector alone | 0.40 | 0.07 | differs |
| 7 | a random direction as long as the pair direction | 0.23 | 0.14 | differs |
| 7 | the kind alone, on pairs of two kinds, share of their gap | 0.33 | 0.57 | differs |
| 7 | the gap between two "She" names against the gap between names of two kinds | 0.32 | 0.10 | differs |
| 7 | pronouns turned by the difference of means at dose 1, of 72 | 72 | 72 | same |
| 7 | pronouns turned by the shared latent at its mean value, of 72 | 15 | 9 | differs |

Fifteen of the 24 are the same by the rule, most of them about where
something is or how much of it a method captures. The nine that differ:
which head does the veto; how much one latent does (0.17 or 0.27 of the
gap, 15 or 9 pronouns); how well the due name is read on flipped prompts;
what a random direction costs; three rows on how the steering divides
among the names' means, the shared vector and the kind; and how far apart
the models hold two "She" names (0.32 or 0.10 of the gap between names of
two kinds). The split is not clean. The models agree on the means with the
vector, and on which names share a latent.

The rule does not always agree with the earlier notes' own lists. It calls
the same what steps 3 and 7 listed as different: the block at which who
acts leaves the acting name, which is one block apart, and 1 against 30 of
660 clean prompts with A above B. It calls different what step 7 listed as
the same, a random direction's cost of 0.23 and 0.14. And the head row and
the depth row describe one fact, the block that moves who acts to the last
position, with opposite verdicts.

Chughtai et al. (2023) report a split of this kind with far more to stand
on. For small networks trained on group operations they could characterize
the family of circuits, and across seeds the precise circuits a given
network learned were arbitrary.

Two models are not a distribution. A row that is the same here has been
seen twice. [A third model](#a-third-model-a-third-dictionary) below joins
the two on twelve of the fifteen. One it does not join them on is how the
veto treats the six "She" names.

## A second dictionary

The second dictionary is trained with another seed, which sets its initial
directions and how the training vectors are ordered. Width (4,096), the
number of active latents (32), the place, the training vectors and the
number of steps are the same.

| Step | With the dictionary | seed0: first | second | seed1: first | second |
|---|---|---|---|---|---|
| 5 | variance it misses on stories, share | 0.08 | 0.08 | 0.08 | 0.08 |
| 5 | the loss rises by, as a share of the loss | 0.03 | 0.03 | 0.04 | 0.04 |
| 6 | who acts carried by what it rebuilds | 0.94 | 0.91 | 1.00 | 0.99 |
| 6 | carried by the highest-ranked latent | 0.17 | 0.21 | 0.27 | 0.28 |
| 6 | by the eight highest | 0.69 | 0.71 | 0.85 | 0.86 |
| 6 | by the 32 highest | 0.91 | 0.89 | 0.98 | 0.98 |
| 6 | the gradient's estimate against the real effect, correlation | 0.95 | 0.91 | 0.99 | 0.98 |
| 6 | the two names' latents moved for real | 0.23 | 0.28 | 0.43 | 0.45 |
| 6 | of the six "She" names, how many share the veto head's latent | 6 | 6 | 6 | **5** |
| 6 | different latents among the six "He" names | 6 | **5** | 6 | **4** |
| 6 | the shared latent off: the acting "She" name against the nine, logits | 0.68 | 0.76 | 0.77 | 0.67 |
| 6 | the same against its own kind | −0.03 | −0.02 | −0.02 | −0.02 |
| 7 | the two names' latents steered | 0.19 | 0.22 | 0.43 | 0.45 |
| 7 | the shared latent at its mean value moves the pronoun measure after a "He" name by, logits | 6.59 | 6.80 | 5.52 | 6.13 |
| 7 | pronouns it turns, of 72 | 15 | **18** | 9 | **14** |
| 7 | of them after the 36 "He" names | 12 | 12 | 9 | **10** |
| 7 | and after the 36 "She" names | 3 | **6** | 0 | **4** |
| 7 | at twice that value, of 72 | 72 | 72 | 71 | **72** |
| 7 | the rebuilt part of the difference of means turns, of 72 | 72 | 72 | 72 | 72 |
| 7 | the missed part turns, of 72 | 0 | 0 | 0 | 0 |
| 7 | the missed part moves the measure by, logits (the larger of the two groups' means) | 0.38 | **0.51** | 0.19 | **0.07** |
| 7 | the dictionary reads the latent back at, share of what was written | 0.71 | 0.89 | 0.71 | 0.74 |
| 7 | its encoder's reading of one unit of the latent's direction | 0.88 | 0.87 | 0.89 | 0.91 |
| 7 | where that reading starts on a "He" name, share of the latent's mean value | −0.17 | **0.02** | −0.18 | −0.17 |
| 7 | latents that differ by more than 0.5 between the kinds, mean of six sentences | 29.5 | 28.7 | 31.8 | 31.0 |

Bold marks a value that differs from the first dictionary's by the rule.
One cell sits on the line: 0.71 against 0.89 passes on the unrounded
values and would fail on the printed ones.

![Two results with the first dictionary and with a second one trained from another seed, for both models: how much of the gap the highest-ranked latents carry when moved for real, by the number of latents, with a dot per sentence pattern and a line for the mean; and the logit of She minus the logit of He as the shared latent is added at a He name, by dose, with a thin line per name and sentence and a thick line for the mean](figures/limits-dictionaries.svg)

**What a dictionary says in aggregate held.** How much it misses, how much
of who acts it carries, how spread out that is, how well the gradient
ranks latents, and by how many logits the shared latent moves the pronoun
measure are the same in the second dictionary, in both models. How many
pronouns that turns is not, by the rule: 18 and 14 against 15 and 9, most
of the difference after "She" names. The missed part moves the measure by
0.51 and 0.07 where it was 0.38 and 0.19, and turns no pronoun in any of
the four dictionaries.

**Which latent is which mostly did not.** A latent of the first dictionary
has a counterpart in the second when some latent there points the same
way, to within a chosen cosine between their decoder directions:

| Latents of the first dictionary whose best counterpart in the second has | seed0 | seed1 |
|---|---|---|
| a cosine of 0.9 or more | 3% | 3% |
| 0.7 or more | 9% | 13% |
| 0.5 or more | 30% | 39% |
| median cosine | 0.41 | 0.46 |
| for comparison, the nearest other latent of the same dictionary: median | 0.24 | 0.25 |
| for comparison, two sets of random directions: median | 0.18 | 0.18 |

No latent is dead: each of the 4,096 fires somewhere in 76,800 tokens of
held-out stories. The share rises steeply with how often a latent fires:

| A counterpart at a cosine of 0.7 or more, among latents that fire on | seed0 | seed1 |
|---|---|---|
| up to 0.1% of story tokens (161 and 175 latents) | 1% | 1% |
| 0.1% to 1% (3,272 and 3,220) | 4% | 7% |
| more than 1% (663 and 701) | 33% | 48% |

Paulo and Belrose (2025) trained dictionaries on the same model and data
with different seeds and found that they identify different sets of
latents, and that top-k dictionaries depend on the seed more than those
trained with an L1 penalty. They kept the data in one fixed order, and
here it changes with the seed. Their measure is stricter than the one
above: a one-to-one matching in which a latent has the same partner by its
encoder and by its decoder direction, at a cosine above 0.7 in both. On it
they report 30% shared latents in their largest case, and more in smaller
models and dictionaries. By that definition, with a greedy matching, the
dictionaries here share 5% and 8%. That is below their figures although
the model and the dictionaries are far smaller. They also found less
overlap with shorter training, and these were trained for 4,000 steps. Why
the share is low here was not tested.

**The shared latent came back; the names' latents in part.** The shared
latent of step 6 has a counterpart at a cosine of 0.83 and 0.77, and that
counterpart is the second dictionary's own shared latent. At the acting
name of the 1,320 clean and flipped prompts, both fire wherever a "She"
name acts and on 1% to 20% of the rest. Their values correlate at 0.97 and
0.98 over all prompts, which mostly says that both mark the kind, and at
0.72 and 0.29 among the 660 where a "She" name acts.

For the names, counted by latent: the veto head rests on seven different
latents over the twelve names in each first dictionary. For five of the
seven, in both models, the best counterpart is the latent the second
dictionary's veto head rests on for most of the same names. For the twelve
latents step 5 found by how they fire, the best counterpart is the latent
the second dictionary's step 5 finds for the same name for five and three. With
a cosine of 0.7 or more as the test, it is four of seven against three of
twelve in both models. These are too few to rank the two ways of finding a
latent, and the count depends on what the second dictionary's own
selection picks.

The second dictionary changes two statements of step 6 and one of step 7.

- In seed1 the head rests most on another latent when Anna acts, one that
  fires for Anna and Sara. So the shared latent is its first choice for
  five names. It still fires wherever Anna acts.
- The six "He" names do not have six different latents. Ben and Jack
  share one in seed0, and Tom, Ben and Jack one in seed1, where it fires
  for all six "He" names. "One latent for the six, six for the other six"
  described the first dictionaries.
- Step 7 explained what the dictionary reads back by two things, one of
  them that the reading on a "He" name starts below zero. In seed0's
  second dictionary it starts at +0.02 of the latent's mean value, and the
  dictionary reads back 0.89 of what was written.

## A third model, a third dictionary

The two sections above saw each thing twice. This one adds a third of
each: a third story model, seed2, trained by the same command with seed 2,
and a third dictionary for each of the first two models, also with seed 2.

How three columns are read was written into `scripts/limits.py` and
published before any script here had run on the third model and before a
third dictionary was trained. The rule for two is applied to each of
the three pairs, because it does not carry over: a can be the same as b
and b as c without a being the same as c. A row is the same in all three
story models when all three pairs are. The third model's dictionary is
trained after the block of its head with the largest share at the acting
name. That reading of "its head at the acting name" was written down when
steps 1 to 3 had run on the third model and step 4 had not. Nothing hung
on it: each model has one head with 0.05 or more there, in block 4.

```sh
# in minllm: uv run scripts/train.py --seed 2 --out out/base-seed2
export MINTERP_RUNS=runs                        # as above
for s in behavior lens probe patch circuit sae features steer; do
  uv run scripts/$s.py --model seed2            # steps 1 to 7 on the third model
done
uv run scripts/open.py --model seed2 --block 4

for m in seed0 seed1; do                        # a third dictionary for each of the first two
  uv run scripts/sae.py --model $m --seed 2 --dict s2
  uv run scripts/features.py --model $m --dict s2
  uv run scripts/steer.py --model $m --dict s2
done

uv run scripts/heads.py --model seed2           # for the looks added afterwards
uv run scripts/features.py --model seed2 --head 6.5
for n in 1 2; do                                # two more dictionaries for the third model
  uv run scripts/sae.py --model seed2 --seed $n --dict s$n
  uv run scripts/features.py --model seed2 --dict s$n
  uv run scripts/steer.py --model seed2 --dict s$n
  uv run scripts/features.py --model seed2 --dict s$n --head 6.5
done

uv run scripts/limits.py
```

The third model ends its training at a loss of 1.460 on held-out stories,
against 1.465 and 1.461. It sorts the twelve names into the same six and
six, and step 4's selection gives it seven heads, as it gave the first
two.

| Step | Number | seed0 | seed1 | seed2 | seed2 is the same as |
|---|---|---|---|---|---|
| 1 | clean prompts with B above A, share | 1.00 | 0.95 | 0.99 | both |
| 1 | flipped prompts with A above B, share | 0.85 | 0.76 | 0.84 | both |
| 1 | share of the clean logit difference written by the last quarter of the blocks | 0.94 | 0.90 | 0.74 | seed1 only |
| 2 | the due name read at the acting name at half depth, on new pairs, clean prompts | 0.95 | 0.80 | 0.88 | both |
| 2 | the same on flipped prompts | 0.90 | 0.71 | 0.82 | each, though they differ |
| 3 | the last depth at which the acting name alone carries half of the gap, as a share of the depth | 0.75 | 0.88 | 0.75 | both |
| 4 | the head that carries most at the last position | block 7, head 0 | block 8, head 0 | block 7, head 3 | neither |
| 4 | the seven heads of the circuit together carry, share of the gap | 1.07 | 0.97 | 1.08 | both |
| 5 | variance the dictionary misses on stories, share | 0.08 | 0.08 | 0.08 | both |
| 5 | the loss rises by, as a share of the loss | 0.03 | 0.04 | 0.04 | both |
| 6 | who acts carried by what the dictionary rebuilds, share of the gap | 0.94 | 1.00 | 1.02 | both |
| 6 | carried by the highest-ranked latent | 0.17 | 0.27 | 0.28 | seed1 only |
| 6 | of the six "She" names, how many share the veto head's latent | 6 | 6 | 2 | neither |
| 6 | the veto head's write to an acting "She" name against its own kind, logits | −0.26 | −0.23 | −0.76 | neither |
| 7 | the pair direction at dose 1, share of the way | 0.95 | 1.04 | 1.06 | both |
| 7 | its part with a term per name | 0.96 | 1.04 | 1.07 | both |
| 7 | the names' means with the vector every flipped prompt shares | 0.81 | 0.85 | 0.85 | both |
| 7 | the names' means alone | 0.38 | 0.81 | 0.79 | seed1 only |
| 7 | the shared vector alone | 0.40 | 0.07 | 0.06 | seed1 only |
| 7 | a random direction as long as the pair direction | 0.23 | 0.14 | 0.31 | neither |
| 7 | the kind alone, on pairs of two kinds, share of their gap | 0.33 | 0.57 | 0.44 | neither |
| 7 | the gap between two "She" names against the gap between names of two kinds | 0.32 | 0.10 | 0.30 | seed0 only |
| 7 | pronouns turned by the difference of means at dose 1, of 72 | 72 | 72 | 72 | both |
| 7 | pronouns turned by the shared latent at its mean value, of 72 | 15 | 9 | 27 | neither |

Where seed2 is the same as both, the first two are the same as each other,
so the row is the same in all three. The one exception is marked: 0.82
passes the rule against 0.90 and against 0.71, which do not pass against
each other.

![The twenty rows of the table that are numbers, for the three story models: a tick at each model's value and, where the row is a mean over the five sentence patterns, a dot per pattern; beside each row, which of the first two models the third is the same as by the rule](figures/limits-seeds.svg)

**Twelve of the 24 rows are the same in all three story models.** The
first two agreed on 15. Seed2 agrees with seed0 on 14 and with seed1 on
17. Of the 15 rows seen twice, 12 were seen a third time. Of the nine rows
the first two differed on, seed2 is the same as seed1 on three (what the
highest-ranked latent carries, and how the steering divides between the
names' means and the shared vector), as seed0 on one (how far apart two
"She" names are held), as each on one, and as neither on four.

The rule compares one number per model and ignores the spread over the
sentence patterns, which the figure shows. A random direction's cost
counts as different in all three pairs, with seed0's patterns between 0.14
and 0.35 and seed2's between 0.22 and 0.41. The first probe row counts as
the same in all three, with one pattern at 0.48 in seed1 and in seed2.

**Three rows that had been the same did not hold.** The first is where the
answer is written, and it sits on the rule's line. The last quarter of the
blocks writes 0.74 of the clean logit difference in seed2, which passes
against seed1's 0.90 and differs from seed0's 0.94 by 0.196 where the rule
allows 0.188. Head 5 of block 6 writes 0.16 of it there. In the first
two models no head of block 6 writes more than 0.01 of it in either
direction, to two decimals.

One row that held would not with another line. Seed2 takes who acts off
the acting name in two blocks: moving that position alone carries 1.04 of
the gap after block 5, 0.56 after block 6 and nothing after block 7, where
seed0 and seed1 lose it in one block. At a line of nine tenths in place of
a half, seed2's depth row reads 0.62, which passes against seed0's 0.75
and not against seed1's 0.88.

The other two are step 6's, and they are not equally firm.

The write needs no dictionary. Seed2's head with the largest share
at the last position, head 3 of block 7, lowers an acting "She" name by
0.76 logits against the other names of its kind, where the first two
models' heads lower it by 0.26 and 0.23. Step 6 found a veto head that
lowers a "He" name against its own kind by 1.12 and 1.16 logits and a
"She" name by a fifth to a quarter of that. In seed2 the two figures are
1.29 and 0.76.

The count is the dictionary's as much as the model's. In seed2's first
dictionary that head has one latent as its first choice for two of the six
"She" names, Mia and Lucy, and four other latents for the other four. The
dictionary does have a latent for the kind: the one Mia and Lucy share
fires in all 36 of step 7's sentences with a "She" name and in two of the
36 with a "He" name.

**Three looks were added after the rows had been read**, and none is a
row. Two are about a second head. Step 4's record for seed2 has two heads at
the last position that read the acting name and lower it: head 3 of block
7 with a share of 0.62, and head 5 of block 6 with 0.45. In the first two
models one such head stands far above the next: 0.87 against 0.13, and
0.69 against 0.14. Step 6 studies the largest, so in seed2 its two rows
are about one of two heads.

The first look had its comparison written into `scripts/limits.py` before
`scripts/heads.py` was run on seed2: what head 5 of block 6 adds to an
acting "She" name against its own kind, beside −0.26 and −0.23, by the
rule. It adds −0.26, which is the same.

The other two come from an independent check of this section and had no
comparison written first. One runs step 6 on that second head
(`--head 6.5`). The table gives what a head adds where a name of each kind
acts, in logits: to the acting name against the nine uninvolved names, to
it against the uninvolved names of its own kind, and to its kind against
the other kind.

| Head | "She": against the nine | its own kind | kind against the other | "He": against the nine | its own kind | kind against the other |
|---|---|---|---|---|---|---|
| seed0, block 7, head 0 | −1.01 | −0.26 | −1.40 | −1.75 | −1.12 | −1.19 |
| seed1, block 8, head 0 | −1.20 | −0.23 | −1.80 | −1.72 | −1.16 | −1.07 |
| seed2, block 7, head 3 | −1.18 | −0.76 | −0.79 | −1.60 | −1.29 | −0.57 |
| seed2, block 6, head 5 | −0.67 | −0.26 | −0.76 | −0.61 | −0.51 | −0.20 |

Where a "She" name acts, head 5 of block 6 does what the first two models'
heads do, with about half their effect on the kind: it lowers the kind by
0.76 and the name within it by 0.26, and its first choice is one latent
for all six "She" names. Where a "He"
name acts it does not: 0.51 against its own kind is not 1.12 or 1.16 by
the rule. Head 3 of block 7 lowers both the name and the kind where a
"She" name acts. The direct writes of two heads add. Together seed2's two
lower an acting "She" name by 1.02 logits against the names of its kind
and a "He" name by 1.80. The two largest such heads of seed0 lower them by
0.29 and 1.27, and those of seed1 by 0.28 and 1.39. So the row's −0.76 is
not made by which head the script picks. By direct write, in logits,
seed2's veto tells the "She" names apart, and that of the first two models
mostly does not.

The last look trains a second and a third dictionary for seed2 and runs
steps 6 and 7 on them. Head 3 of block 7 then has one latent as its first
choice for five of the six "She" names in both, with Anna on a latent of
her own, as in seed1's second dictionary. Head 5 of block 6 has one latent
for all six in each of the three. By the rule five is not six, so against
the first two models' first dictionaries the row would still read
"neither". But "two of six" described one
dictionary, and the difference between the models that this row was meant
to show is in the write.

**What the three circuits share is the roles of their heads.** The head
row cannot be the same unless two models put the head at the same place.
In all three, a head of block 4 at the acting name reads the other name,
with 0.91 and 0.78 of its attention on clean and flipped prompts in seed0,
0.85 and 0.79 in seed1, and 0.81 and 0.70 in seed2. In all three, heads at
the last position read the acting name and lower it, other heads read a
name and raise it, and seven heads together carry the gap: 1.07, 0.97 and
1.08. Where the veto sits differs each time: block 7, block 8, and blocks
6 and 7. Step 4's hold shows the two in seed2 as two routes: of the 0.26
its fetching head carries, 0.12 comes along with head 5 of block 6 held at
the flipped run's output, and 0.13 with head 3 of block 7 held. And a
direction added after block 6 comes too late for a head of block 6: step
7's pair direction, added there at the acting name, moves 0.59 of the way
in seed2, against 0.95 and 0.92.

The third dictionaries, with the rows on which one of them differs from
the first or the second of its model:

| Step | With the dictionary | seed0: first | second | third | seed1: first | second | third |
|---|---|---|---|---|---|---|---|
| 6 | carried by the highest-ranked latent | 0.17 | 0.21 | **0.26** | 0.27 | 0.28 | 0.31 |
| 6 | the two names' latents moved for real | 0.23 | 0.28 | **0.31** | 0.43 | 0.45 | 0.43 |
| 6 | of the six "She" names, how many share the veto head's latent | 6 | 6 | 6 | 6 | 5 | **6** |
| 6 | different latents among the six "He" names | 6 | 5 | **3** | 6 | 4 | **6** |
| 6 | the shared latent off: the acting "She" name against the nine, logits | 0.68 | 0.76 | 0.84 | 0.77 | 0.67 | **0.90** |
| 7 | the two names' latents steered | 0.19 | 0.22 | **0.26** | 0.43 | 0.45 | 0.43 |
| 7 | the shared latent at its mean value moves the pronoun measure after a "He" name by, logits | 6.59 | 6.80 | 7.53 | 5.52 | 6.13 | **7.19** |
| 7 | pronouns it turns, of 72 | 15 | 18 | **32** | 9 | 14 | **22** |
| 7 | of them after the 36 "He" names | 12 | 12 | **18** | 9 | 10 | **16** |
| 7 | and after the 36 "She" names | 3 | 6 | **14** | 0 | 4 | **6** |
| 7 | at twice that value, of 72 | 72 | 72 | 72 | 71 | 72 | **72** |
| 7 | the missed part moves the measure by, logits (the larger of the two groups' means) | 0.38 | 0.51 | **0.29** | 0.19 | 0.07 | **0.16** |
| 7 | where that reading starts on a "He" name, share of the latent's mean value | −0.17 | 0.02 | **−0.03** | −0.18 | −0.17 | **−0.09** |
| 7 | latents that differ by more than 0.5 between the kinds, mean of six sentences | 29.5 | 28.7 | 29.2 | 31.8 | 31.0 | **25.2** |

Bold marks a third value that differs by the rule from the first
dictionary's or from the second's. The other eleven rows of the section
above are the same in all three dictionaries, in both models. In seed0 16
of the 25 rows are the same in all three, and in seed1 14. The first two
dictionaries agreed on 20 and 18, and the third joins them on 16 and 14 of
those.

**Most of the aggregate numbers held a third time. The pronoun count did
not.** How much a dictionary misses, how much of who acts it rebuilds,
what the eight and the 32 highest latents carry and how well the gradient
ranks latents are among the eleven. Two that the section above counted as
holding did not hold in one model each: the highest-ranked latent carries
0.17, 0.21 and 0.26 in seed0's three dictionaries, and the shared latent
moves the pronoun measure by 5.52, 6.13 and 7.19 logits in seed1's. The
shared latent at its mean value turns 15, 18 and 32 of 72 pronouns in
seed0's three dictionaries, 9, 14 and 22 in seed1's, and 27, 52 and 59 in
seed2's. Nine dictionaries give between 9 and 59. Step 7 gave this count
for one dictionary per model.

**Which latent is which, as before.** Of the first dictionary's latents,
3%, 9% and 31% have a counterpart in the third at a cosine of 0.9, 0.7 and
0.5 or more in seed0, and 4%, 14% and 40% in seed1. The second gave 3%,
9% and 30%, and 3%, 13% and 39%. A counterpart at 0.7 or more in both
other dictionaries is found for 6% and 9% of the latents, and for 28% and
39% of those that fire on more than 1% of story tokens. So about seven in
ten of the latents with a counterpart in the second dictionary have one in
the third as well.

**The shared latent came back a third time.** Its best counterpart in the
third dictionary has a cosine of 0.78 and 0.91 and is that dictionary's
own shared latent, and the same holds starting from the second
dictionary, at 0.80 and 0.76. Of the three statements the second
dictionary changed, none is settled by a third. In seed1 the head rests on
the shared latent for all six "She" names again. The six "He" names have
6, 5 and 3 different latents in seed0's three dictionaries, where Tom,
Tim, Ben and Jack share one in the third, and 6, 4 and 6 in seed1's. The
reading on a "He" name starts at −0.17, +0.02 and −0.03 of the latent's
mean value in seed0 and at −0.18, −0.17 and −0.09 in seed1, and the
dictionary reads back 0.71, 0.89 and 0.84 of what was written, and 0.71,
0.74 and 0.84.

Three are not a distribution either. What the third of each adds is a
count. Of 15 rows the first two models agreed on, three did not hold in
the third. One of them sits on the rule's line. The other two are the two
halves of what step 6 said about the "She" names: the write did not hold
in the third model, and the count of names depends on the dictionary as
well as on the model. Of the 20 and 18 rows two dictionaries agreed on,
four and four did not hold in a third.

## Another model

The open model is Qwen2.5-0.5B-Instruct: 24 blocks, 14 heads per block,
width 896. minllm reads its published weights into the same class as the
story model, so the hooks of step 1 find the modules they name.
[`tests/test_open.py`](../tests/test_open.py) checks, on the 132 clean
prompts of the first sentence pattern, the identities the earlier steps
rest on: the recorded pieces add up to the stream, the attribution adds up
to the logit difference, and the recomputed attention is the model's own.
All twelve names are single tokens in its vocabulary, and it follows the
same six with " She".

The scripts of steps 1 and 3 run on it as they are. Step 3's has two
halves: the stream moved by position and depth, and every part moved one
at a time, which has a section of its own below. `scripts/open.py` repeats
step 7's first experiment at the acting name after block 12, the middle of
the model. Block 4 of the story model was chosen as the block of the
fetching heads. Block 12 was chosen before the open model's parts had been
moved, so the two places match in depth only. The one head found acting at
the acting name sits two blocks later.

| Step | Number | seed0 | seed1 | open | The open model is the same as |
|---|---|---|---|---|---|
| 1 | clean prompts with B above A | 1.00 | 0.95 | 1.00 | both |
| 1 | flipped prompts with A above B | 0.85 | 0.76 | 1.00 | seed0 |
| 1 | share of the clean logit difference written by the last quarter of the blocks | 0.94 | 0.90 | 0.92 | both |
| 3 | the last depth at which the acting name alone carries half of the gap, as a share of the depth | 0.75 | 0.88 | 0.62 | seed0 |
| 7 | the pair direction at dose 1 | 0.95 | 1.04 | 1.00 | both |
| 7 | its part with a term per name | 0.96 | 1.04 | 1.00 | both |
| 7 | the names' means with the vector every flipped prompt shares | 0.81 | 0.85 | 1.00 | both |
| 7 | the names' means alone | 0.38 | 0.81 | 0.80 | seed1 |
| 7 | the shared vector alone | 0.40 | 0.07 | 0.09 | seed1 |
| 7 | a random direction as long as the pair direction | 0.23 | 0.14 | 0.07 | neither |
| 7 | the kind alone, on pairs of two kinds | 0.33 | 0.57 | 0.10 | neither |
| 7 | the gap between two "She" names against the gap between names of two kinds | 0.32 | 0.10 | 1.01 | neither |

![Who acts moved at one position, by depth as a share of the model's blocks, at the acting name and at the last position; and the pair direction added at the acting name at half depth, by dose, as the share of the way to the flipped answer and the share of prompts with A above B: for the two story models and the open model, means over the five sentence patterns with their range](figures/limits-models.svg)

**The answer is written late and who acts sits at the acting name, as in
the story models. How it leaves differs.** The open model does the task in
every prompt, in both directions. Its last quarter of blocks writes 0.92
of the clean logit difference. Moving the acting name's position alone
still carries 0.98 of the gap after block 14. Then who acts goes to the
last position in two stages: about half of it at blocks 15 and 16, on
average over the patterns and more in one of the five, and the rest at
blocks 21 to 23. Seed0 and seed1 move it in one block, seed1 in its last. So the
depth row depends on the line drawn in the open model and not in those two:
0.58 of the depth for nine tenths of the gap, 0.62 for half, 0.83 for a
quarter. The third story model moves it in two neighboring blocks, and its row
depends on the line too, as the section above says.

**The pair direction works there as it did.** Built on four sentence
patterns and added in the fifth, it takes the answer 1.00 of the way and
puts A above B in every prompt. After every second block its mean over the
five patterns is within 0.01 of what step 3's patch of the flipped stream
into the clean run carries there, and within 0.04 in any single pattern.
As in step 7 this is a low bar: the position is the acting name's own
token. At block 12 its part with a term per name also goes 1.00 of the
way.

**The shared vector appears part of the way down, in two of the three
models.** Step 7
found that seed0's answer rests in part on a vector that every flipped
prompt shares, and seed1's hardly. By depth, at dose 1:

| Share of the way | seed0: after block 2 | 4 | 6 | seed1: 2 | 4 | 6 | open: 10 | 12 | 14 |
|---|---|---|---|---|---|---|---|---|---|
| the names' means alone | 1.01 | 0.38 | 0.37 | 0.99 | 0.81 | 0.75 | 1.04 | 0.80 | 0.44 |
| the shared vector alone | 0.00 | 0.40 | 0.38 | 0.00 | 0.07 | 0.00 | −0.01 | 0.09 | 0.54 |
| the two together | 1.01 | 0.81 | 0.78 | 0.99 | 0.85 | 0.74 | 1.00 | 1.00 | 1.02 |

At block 4 the story models' columns are step 7's numbers, computed again
by `scripts/open.py`. Early on, the names' means do everything in all
three. Later, seed0 and the open model come to rest on the shared vector
as well, the open model at blocks 12 and 14, where the two together still
do everything. From block 16 on, where the pair direction itself carries
0.46 of the gap, the vector alone does nothing there (−0.04 to 0.00) and
the means alone 0.43. Seed1 does without the vector at every depth tried.
In the open model at block 12 a random vector of the shared vector's
length adds nothing to the means (0.75 against 0.80).

So the rows "the names' means alone" and "the shared vector alone" of the
table above say seed1 at block 12. Two blocks later the first would say
seed0 (0.44 against 0.38). The second is 0.54 there, more than a fifth
above seed0's 0.40: four patterns give 0.45 to 0.50 and the fifth 0.79.

**The kind does little in the open model at this place.** In the story
models a direction that carries only the kind moves pairs of two kinds by
0.33 and 0.57 of their gap. In the open model it moves them by 0.10, at
block 12 and dose 1, where a quarter of the pair direction itself moves
the answer 0.06. The open model's gaps depend little on the kind: 16.0
logits between two "She" names, 14.5 between two "He" names, 15.8 between
names of two kinds. The story models' gap between two "She" names is a
third and a tenth of their own gap between names of two kinds. That fits a model that tells two names of
one kind apart. It was not tested as the reason. Whether the open model
uses the kind elsewhere, and whether size, data or training makes the
difference, was not looked at.

**The dose behaves differently at this depth.** In the open model the
answer hardly moves at a quarter of the pair direction (0.06), is past
half-way at half of it (0.59, with A above B in 72%), and is there at
three quarters (0.99). Past 1 it does not overshoot: it falls back to 0.93
and 0.86 at doses 1.5 and 2. A random direction as long as the pair
direction costs 0.07 at dose 1, against 0.23 and 0.14 in the story models.
The pair direction is 0.60 of the stream's size there, against 0.85 and
0.78. The mean stream in place of the prompt's takes the answer 0.49 of
the way (0.38 and 0.50 in the story models).

## The open model, part by part

Step 3 moved the output of every head and MLP from the clean prompt into
the flipped run, one part at a time. `scripts/patch.py --model qwen` does
that to the open model's 336 heads and 24 MLPs: at all positions, at the
last position only, at all positions before it and at the acting name
only, in both directions. It also replaces each part on the clean run by
the four stand-ins of step 3, and moves the four largest parts besides the
first MLP two at a time. The stream half of its record is the record the
section above used, value for value. `scripts/heads.py` then records where
the heads found this way look, what they write, and which later heads the
one at the acting name works through, by measures of steps 4 and 6.

![Left: every head and MLP of the open model moved from the clean prompt into the flipped run at all positions, the share of the gap that comes along as a mean over the five sentence patterns, shaded up to 0.25. Right: the parts whose mean share is 0.05 or more in size, the first MLP aside, with that share and the share in the other direction in each of the five patterns; the ticks are the means](figures/limits-parts.svg)

The nine largest parts. Carry is the clean part moved into the flipped
run, break the flipped part moved into the clean run, both as a share of
the gap between the two runs and as a mean over the five patterns.

| Part | carry | over the five patterns | at the last position | at the acting name | break |
|---|---|---|---|---|---|
| block 1, MLP | 1.00 | 1.00 to 1.00 | 0.00 | 1.00 | 1.00 |
| block 21, head 9 | 0.22 | 0.16 to 0.29 | 0.21 | 0.00 | 0.23 |
| block 14, head 2 | 0.18 | 0.11 to 0.30 | 0.00 | 0.18 | 0.14 |
| block 17, MLP | 0.17 | 0.13 to 0.22 | 0.11 | 0.04 | 0.15 |
| block 15, head 3 | 0.16 | 0.09 to 0.21 | 0.15 | 0.00 | 0.15 |
| block 23, MLP | 0.16 | 0.11 to 0.18 | 0.16 | 0.00 | 0.16 |
| block 16, MLP | 0.14 | 0.10 to 0.20 | 0.13 | 0.00 | 0.25 |
| block 24, head 12 | −0.12 | −0.14 to −0.09 | −0.12 | 0.00 | −0.12 |
| block 20, head 6 | −0.12 | −0.14 to −0.07 | −0.12 | 0.00 | −0.13 |

**The first MLP carries all of it, as in the story models.** Moved at the
acting name it carries 1.00 of the gap in every pattern, against 1.00 and
0.99. The embedding of the acting name is 0.013 in root mean square and
the stream after block 1 is 0.25, nearly all of it what the first MLP
adds. The story models have 0.05 against 0.85 and 0.79. In all three the
embedding is about a twentieth of the stream after block 1: 0.052, 0.056
and 0.060. Step 3's limit holds here too: patching does not tell a part
that works something out from a part that hands its input on.

**No head carries most of it.** The largest head carries 0.22 of the gap,
between 0.16 and 0.29 over the patterns. The largest heads of the story
models carry 0.87 and 0.69. Of the open model's 336 heads, 14 have a share
of 0.05 or more in either direction, nine of them positive. The story
models have 7 and 4 such heads of 48. No head of the first thirteen blocks
has a share above 0.03. One of the nine has its share at no single place:
head 0 of block 15 carries 0.11 moved at all positions, 0.045 at the last
position alone and 0.02 at the positions before it.

**One head acts at the acting name in each model. The open model's reads
something else there.** Head 2 of block 14, of 24, carries 0.18, all of it
at the acting name's position, and is the only head with 0.05 or more
there. The story models have one such head each, both in block 4 of 8:
0.52 in seed0 and 0.21 in seed1. Step 4 called those two fetching heads,
because from the acting name they read the other name: with 0.91 and 0.78
of their attention in clean and flipped prompts in seed0, and 0.85 and
0.79 in seed1. Head 2 of block 14 reads the acting name's own first
mention: 0.56 of its attention is on the first name in clean prompts,
where the first name acts, and 0.75 on the second name in flipped ones
(0.51 to 0.64 and 0.71 to 0.81 over the patterns), with 0.02 and 0.01 on
the other name. In neutral prompts, where the acting name has not been
mentioned before, 0.96 is on the first token. Wang et al. (2023) call a
head that reads the first mention of a repeated name a duplicate token
head, and step 4 named this as what separates their heads from the
fetching heads. What head 2 of block 14 writes at the acting name was not
decoded. It sits in the last block after which that position alone still
carries the gap: 0.98 after block 14, 0.55 after block 15.

Step 4's hold says which later heads it works through. Its clean output
alone carries 0.18. With head 3 of block 15 held at the flipped run's own
output 0.02 comes along, and with head 0 of block 15 held, 0.04. These are
means over the patterns: in the second, 0.17 of 0.30 still comes along
with head 3 held. With any one of the eleven later heads held, between
0.12 and 0.30 comes along, for seven of them more than the 0.18. In the
story models the hold points at the strong veto head: of seed0's 0.52,
0.06 comes along with head 0 of block 7 held, and of seed1's 0.21, 0.08
with head 0 of block 8 held. The two heads of block 15 add nothing to a
name directly, as the next paragraph shows, so in the open model something
must follow them. Which later head does, no single hold shows.

**The parts at the last position lie at about the two depths where the
stream moved.** Of the 18 parts with 0.05 or more at the last position,
three lie in blocks 15 to 17 and fifteen in blocks 20 to 24. No part of
blocks 18 and 19 has a share above 0.03. The stream's share at the last
position rises by 0.28 and 0.26 at blocks 15 and 16 and by 0.23, 0.09 and
0.15 at blocks 21 to 23. At blocks 17, 20 and 24, where eight of the 18
lie, it changes by 0.03 or less. So the parts and the stream agree on two
depths and not block by block. At the first depth, the two heads of block
15 that the head at the acting name works through read that name's
position from the last one. Head 0 puts 0.51 and 0.56 of its attention
there in clean and flipped prompts, the most in every pattern. Head 3 puts
0.35 and 0.39 there and 0.24 and 0.23 on the second name, whoever acts,
and in two of the five patterns more on the second name. Both add 0.00
logits to any name directly. Wang et al.'s S-inhibition heads are active
at the last token, attend to the second mention of the subject and write
into the query of the heads that copy. Whether the two heads of block 15
write that way was not tested. On neutral prompts, where the acting name
is no second mention, they still put 0.46 and 0.29 on its position. No
head of block 16 carries 0.05 or more. Its part is the MLP, with 0.13 at
the last position, as is block 17's, with 0.11. An MLP moves nothing
between positions.

**At the second depth heads veto and copy, as in the story models, and
two heads lower the due name.** For each head of 0.05 or more in either
direction, `scripts/heads.py` records what it reads from the position
where it acts and, for a head at the last position, what it adds there to
the logit of each name, against the mean of the nine names that are in no
version of the prompt. Each cell gives clean prompts, then flipped ones.

| Head | share | reads most, of the seven places recorded | adds to the name that acts | adds to the name that is due |
|---|---|---|---|---|
| block 21, head 9 | 0.22 | the acting name's position: 0.42, 0.47 | −2.01, −2.22 | −0.27, −0.27 |
| block 23, head 0 | 0.09 | the acting name's position: 0.27, 0.29 | −1.04, −1.18 | −0.59, −0.42 |
| block 22, head 0 | 0.07 | the acting name's position: 0.35, 0.38 | −0.67, −0.74 | −0.16, −0.12 |
| block 21, head 12 | 0.09 | the name that is due: 0.53, 0.61 | +0.67, +0.34 | +1.23, +1.67 |
| block 22, head 11 | 0.06 | the name that is due: 0.45, 0.40 | +0.46, +0.37 | +0.83, +0.85 |
| block 24, head 12 | −0.12 | the name that is due: 0.42, 0.34 | −0.28, −0.14 | −1.11, −0.96 |
| block 24, head 2 | −0.07 | the name that is due: 0.33, 0.36 | −0.28, −0.17 | −0.63, −0.73 |
| block 23, head 6 | −0.06 | no one place: at most 0.19 | +0.97, +0.95 | +0.48, +0.60 |
| block 20, head 6 | −0.12 | the name that is due: 0.60, 0.42 | 0.00, 0.00 | 0.00, 0.00 |
| block 20, head 8 | −0.09 | its own position: 0.53, 0.55 | 0.00, 0.00 | −0.01, −0.01 |
| block 20, head 11 | 0.10 | its own position: 0.36, 0.35 | 0.00, 0.00 | 0.01, 0.01 |

The places recorded are seven, and the table reads from means. In single
patterns a head can read more from a token outside them: head 11 of block
20 puts 0.26 and 0.28 on the token before the last one, more than on its
own position in the third pattern, and in one pattern head 12 of block 24
reads a comma a little more than the due name.

Head 9 of block 21 vetoes, in step 4's word: it reads the acting name's
position, with 0.25 to 0.62 of its attention over the patterns and the two
versions, and lowers the name that acts by 2.01 and 2.22 logits. Seed0's
head 0 of block 7 reads that position with 0.79 and 0.83 and lowers the
acting name by 1.38 and 1.56. It raises the due name by 0.42 and 0.52,
where head 9 of block 21 lowers it a little. Heads 0 of blocks 22 and 23
do the same more weakly, and head 0 of block 23 is at the edge of the
word. It puts 0.27 and 0.29 on the acting name's position, where the
weakest story-model head that meets step 4's definition puts 0.36, and it
lowers the due name by 0.59 and 0.42, a third to somewhat over a half of
what it takes off the acting name. On neutral prompts the three read the
acting name's position less, with 0.26, 0.23 and 0.17, where the story
models' strong veto heads keep 0.83 and 0.95. They still lower the name
that acts, by 1.31, 0.47 and 0.69.

Step 6 found, on clean prompts, that where a "She" name acts the story
models' veto heads lower the kind more than the name: against the
uninvolved names of her own kind, seed0's head 0 of block 7 lowers an
acting "She" name by 0.26 and an acting "He" name by 1.12. On flipped
prompts `scripts/heads.py` gets 0.31 and 1.24. Head 9 of block 21 lowers
both against their own kind: "She" names by 1.59 on clean prompts and 1.82
on flipped ones, "He" names by 1.98 and 2.18. Two of the open model's
heads depend on the kind by a factor of more than two. Head 0 of block 22
lowers an acting "She" name by 0.90 and 1.00 and a "He" name by 0.36 and
0.38, and head 11 of block 22 raises a due "She" name by 1.20 and 1.22 and
a "He" name by 0.31 and 0.31.

Two heads copy: they read the due name and raise it more than the other.
Two heads of the last block read the due name and lower it, which is what
Wang et al. describe for their negative name movers. Step 4 cited that
class for a head that lowers the name it reads. There the name is the
acting one. Here it is the due one, as in Wang et al. Among the story
models' heads of 0.05 or more at the last position, none reads the due
name most and lowers it.

So the five heads with a negative share are not of one kind. Two lower the
due name directly. Head 6 of block 20 reads the due name and adds nothing
to any name directly, head 8 of block 20 reads its own position, and head
6 of block 23 raises both names of a clean or flipped prompt, the acting
one more. The veto heads lower a name too, and their shares are positive.
With nothing in its place, the clean logit difference rises by 0.90, 1.16,
0.27 and 0.88 logits for four of the five, and by 0.14 for head 8 of block
20, between −0.29 and +0.37 over the patterns. Seed0 has one head with a
share below −0.05, at −0.07, and seed1 has none.

**The four largest parts add up, and a little more.** Moved two at a
time, the four largest parts besides the first MLP carry between 1.01 and
1.19 of the sum of their two shares: 0.47 against 0.39 for head 9 of block
21 with head 2 of block 14. In the story models the same test gave 0.65 to
0.89 in seed0 and 0.81 to 1.05 in seed1, where the largest parts overlap.
These are six pairs among four parts. Together exceeds the sum in all six
pairs of the open model and in one of the twelve of the story models. Over
all parts but the first MLP, the positive shares sum to 2.76 and the
negative ones to −0.95. In the story models they sum to 2.99 and −0.13,
and to 2.11 and −0.05.

**Carry and break: the largest disagreement is larger, the usual one is
not.** In the story models the two shares of a part differ by at most 0.07
and 0.05. In the open model the largest difference is 0.11, for the MLP of
block 16, which carries 0.14 and breaks 0.25, and the next is 0.06. Over
the parts of 0.05 or more besides the first MLP, the median difference is
0.008, against 0.025 and 0.013 in the story models. Within one pattern the
two shares can be further apart: the MLP of block 16 carries 0.11 and
breaks 0.38 in the fifth pattern, and head 2 of block 14 carries 0.30 and
breaks 0.10 in the second. The largest such difference is 0.26, against
0.14 and just under 0.10 in the story models. The break share is largest
in size in the fifth pattern for 12 of the 21 parts of 0.05 or more
besides the first MLP, and the carry share in the second pattern for 11.
Why was not looked at.

**What stood in can turn the sign, as in step 3.** Replacing head 9 of
block 21 on the clean run lowers a logit difference of 7.07 by 3.67 logits
with the flipped prompt's output, 2.27 with the neutral prompt's, 2.12
with the average and 2.53 with nothing: one sign, and the flipped output
about 1.6 times the neutral. For the MLP of block 20 the four give −0.95,
−1.28, −0.32 and +2.01 logits, and the last is positive in every pattern.
For head 12 of block 21, which copies, the first two give −1.34 and −1.04
and the last two leave nothing: +0.10 and +0.12, of either sign by
pattern.

## What this does not show

- **A distribution.** Three seeds of the model, three seeds of the
  dictionary for each, and one other model. "The same" means seen two or
  three times, and three of fifteen is a count from one draw, not a rate.
- **The third model's rows with another dictionary.** Six of seed2's 24
  rows pass through its first dictionary. Its second and third were read
  for two numbers only: the names that share a first choice, and the
  pronouns turned.
- **The third model's two veto heads as one mechanism.** Their direct
  writes were added. What each does through later heads was not measured
  beyond step 4's hold.
- **Looks that were planned.** Of the three looks added to the third
  model, one had its comparison written first. The other two were chosen
  by a checker who had read the section.
- **That the scorecard is neutral.** The 24 rows were picked after the
  results were known, hold only what can be put as one number, and are not
  independent. With the second dictionaries in place of the first,
  fourteen are the same and not fifteen.
- **That the rule is the right one.** A fifth, or 0.05. With another
  threshold some rows change sides.
- **A circuit in the open model.** Its parts were moved one at a time, and
  four of them two at a time. No set larger than a pair was moved
  together, so nothing here says how much of the gap the heads above carry
  between them, as step 4 said of seven heads in each story model. Step
  4's hold was run for the head at the acting name only, and its other
  tests not at all: which of the later heads works through which is not
  known. What head 2 of block 14 writes at the acting name was not
  decoded. Readers and a dictionary were not looked at, and the pronoun
  task of step 7 was not run.
- **The MLPs.** Seven of the 21 parts of 0.05 or more are MLPs. Each was
  moved as a whole and not opened.
- **The parts by kind of pair.** A share is taken on means over the 132
  prompts of a pattern, and the record holds one per part and pattern.
  Step 6 found that a measure on names can be about the kind of name.
  Whether these shares are cannot be read from such a record, here or in
  the story models'.
- **A share as a property of a part.** The largest head's share runs from
  0.16 to 0.29 over five sentence patterns, and one head's share in the
  other direction from 0.04 to 0.33.
- **Shares of parts as one quantity across models.** In the open model two
  parts moved together carried more than the sum of their shares in all
  six pairs, in the story models less in eleven of twelve. Its 338 parts
  under 0.05 sum to +0.95 and −0.50, the story models' to +0.07 and −0.06
  and to +0.31 and −0.05. So 0.22 against 0.87, and 14 heads against 7,
  set numbers side by side that do not add up in the same way.
- **The open model on its own ground.** It is a chat model, used here on
  raw story text. The twelve names and five sentence patterns were chosen
  for the story model, the kinds come from one sentence, and the prompts
  open with a token the story models were trained to start with.
- **Shares at dose 1 as one quantity across models.** The open model's
  answer moves in a step between a quarter and three quarters of the dose,
  so a partial direction's share means something else there. The controls
  and the dose curve were run at one depth.
- **Other dictionaries.** The second and the third differ from the first
  in their seed. Another width,
  another number of active latents, longer training or another place were
  not tried, and the share of latents with a counterpart may depend on all
  of them.
- **That a latent without a counterpart is noise.** A direction can be
  split over two latents in one dictionary and held by one in the other.
  The best single counterpart misses that.
- **How the shared latent is picked.** It is the latent most "She" names
  have as the veto head's first choice, chosen on all five patterns.
- **Another task.** One behavior, five sentence patterns, twelve names.

Step 9 maps each piece of this repository to a library that exists for it.

## Exercises

1. `runs/features/features-seed2-h6.5.json` is step 6's record for the
   third model's second veto head, and holds more than this note reports.
   Read "the shared latent switched off, where a She name acts" in it and
   in `features-seed0.json`. With the latent off, does the head's effect
   on the kind go and its effect on the name stay, in both?
2. Train a dictionary twice as wide, under a name of its own so that it
   does not replace the first (`--width 8192 --dict wide`). Does the
   shared latent split, and by what?
3. `scripts/heads.py` runs step 4's hold for the head at the acting name
   only. Lift that limit for the two heads of block 15, without
   `MINTERP_RUNS=runs` so that the record of this note is not replaced: is
   what each of them carries changed when head 9 of block 21 is held at
   the flipped run's own output, or when a copying head is?
4. Write the pronoun task of step 7 for the open model with
   `steer.steered` and the difference of means: `scripts/steer.py` needs a
   dictionary and cannot run there. Does one direction turn the pronoun?
5. Apply the rule with a tenth in place of a fifth. Which rows of the
   first table change sides?

## References

- Chughtai, Chan and Nanda (2023), [A Toy Model of Universality: Reverse Engineering How Networks Learn Group Operations](https://arxiv.org/abs/2302.03025). Mixed evidence for universality: the family of circuits and features that networks learn on a task could be characterized, and for a given network the precise circuits learned were arbitrary.
- Wang, Variengien, Conmy, Shlegeris and Steinhardt (2023), [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593), ICLR 2023. Their duplicate token heads are active at the second mention of the subject and attend to the first; their S-inhibition heads are active at the last token and attend to the second mention; their negative name movers attend as the name movers do and write in the opposite direction of the names they attend to.
- Paulo and Belrose (2025), [Sparse Autoencoders Trained on the Same Data Learn Different Features](https://arxiv.org/abs/2501.16615). Dictionaries trained on the same model and data that differ only in their seed identify different sets of features; in the largest case 30% were shared, and top-k dictionaries depended on the seed more than those trained with an L1 penalty.
