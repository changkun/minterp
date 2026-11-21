# 8. Limits

Scripts: [`scripts/limits.py`](../scripts/limits.py),
[`scripts/open.py`](../scripts/open.py) ·
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
- **Another model.** Parts of steps 1, 3 and 7 on a published model of 0.5
  billion parameters, 35 times the story model.

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
uv run scripts/patch.py --model qwen --stream-only
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
seen twice.

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

The scripts of steps 1 and 3 run on it as they are, the latter on the
stream only. `scripts/open.py` repeats step 7's first experiment at the
acting name after block 12, the middle of the model. Block 4 of the story
model was chosen as the block of the fetching heads. Where the open model
fetches is not known, so the two places match in depth only.

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
blocks 21 to 23. The story models move it in one block, seed1 in its last. So the
depth row depends on the line drawn in the open model and not in the story
models: 0.58 of the depth for nine tenths of the gap, 0.62 for half, 0.83
for a quarter.

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

## What this does not show

- **A distribution.** Two seeds of the model, two seeds of the dictionary
  and one other model. "The same" means seen two or three times.
- **That the scorecard is neutral.** The 24 rows were picked after the
  results were known, hold only what can be put as one number, and are not
  independent. With the second dictionaries in place of the first,
  fourteen are the same and not fifteen.
- **That the rule is the right one.** A fifth, or 0.05. With another
  threshold some rows change sides.
- **Anything inside the open model.** Its heads, a circuit, readers and a
  dictionary were not looked at. What was run is the behavior, the stream
  moved by position and depth, and the first experiment of step 7. The
  pronoun task of step 7 was not run there.
- **The open model on its own ground.** It is a chat model, used here on
  raw story text. The twelve names and five sentence patterns were chosen
  for the story model, the kinds come from one sentence, and the prompts
  open with a token the story models were trained to start with.
- **Shares at dose 1 as one quantity across models.** The open model's
  answer moves in a step between a quarter and three quarters of the dose,
  so a partial direction's share means something else there. The controls
  and the dose curve were run at one depth.
- **Other dictionaries.** The second differs in its seed. Another width,
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

1. Train a third dictionary for both models (`--seed 2 --dict s2`, and
   again with `--model seed1`; with `MINTERP_RUNS=runs` so that the next
   script finds its record) and run steps 6 and 7 on it.
   Does the shared latent have a counterpart there as well, and do the
   three agree on the "He" names? `scripts/limits.py` compares the first
   with `s1`; change `TAG` to compare another.
2. Train a dictionary twice as wide, under a name of its own so that it
   does not replace the first (`--width 8192 --dict wide`). Does the
   shared latent split, and by what?
3. Run `scripts/patch.py --model qwen` without `--stream-only`. It makes
   about eight times as many forward passes as the stream-only run, so
   allow some hours. Which heads carry who acts, and at which position do
   they act?
4. Write the pronoun task of step 7 for the open model with
   `steer.steered` and the difference of means: `scripts/steer.py` needs a
   dictionary and cannot run there. Does one direction turn the pronoun?
5. Apply the rule with a tenth in place of a fifth. Which rows of the
   first table change sides?

## References

- Chughtai, Chan and Nanda (2023), [A Toy Model of Universality: Reverse Engineering How Networks Learn Group Operations](https://arxiv.org/abs/2302.03025). Mixed evidence for universality: the family of circuits and features that networks learn on a task could be characterized, and for a given network the precise circuits learned were arbitrary.
- Paulo and Belrose (2025), [Sparse Autoencoders Trained on the Same Data Learn Different Features](https://arxiv.org/abs/2501.16615). Dictionaries trained on the same model and data that differ only in their seed identify different sets of features; in the largest case 30% were shared, and top-k dictionaries depended on the seed more than those trained with an L1 penalty.
