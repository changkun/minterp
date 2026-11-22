# 7. Steer

Code: [`minterp/steer.py`](../minterp/steer.py) ·
Script: [`scripts/steer.py`](../scripts/steer.py) ·
Tests: [`tests/test_steer.py`](../tests/test_steer.py),
[`tests/test_steered.py`](../tests/test_steered.py) ·
Results: [`runs/steer/`](../runs/steer/)

The steps so far read the model, and each ended with a description. A
description that is right about what the model goes by makes a prediction
about writing: change that thing, and the behavior changes in a known way.
This step tests three such predictions.

1. [Step 3](03-patch.md): the answer's dependence on who acts sits at the
   acting name's position after block 4. So some vector, added there,
   should turn the answer of a clean prompt into that of its flipped twin.
   Which vector, and built from what?
2. [Step 6](06-feature-circuit.md): the two names' latents carry 0.23 and
   0.43 of the gap. Adding their decoder directions should carry as much.
3. Step 6: one latent is shared by the six names the model follows with
   "She". If that latent is the model's kind of name, adding it to a "He"
   name should make the model go on with "She".

## Adding a vector

Steering adds a vector to the stream at one place while the model runs:

    x  ->  x + dose * direction        after block 4, at one position

[`steer.py`](../minterp/steer.py) does this with the edit of step 1's Tap.
A patch of step 3 is the special case where the direction is the other
run's stream minus this run's and the dose is 1. Adding the difference of
two prompts' activations is the activation addition of Turner et al.
(2023); averaging such differences over many pairs is the contrastive
activation addition of Rimsky et al. (2024).

An added vector can change the answer without standing for anything: it
can damage what the position held. The stream at the acting name has a
size of 29.4 in seed0 and 31.9 in seed1, and the main directions below are
20 to 25 long (`order` is 11.1 and 8.0, the two latents' direction 10.2
and 11.1). A random direction as long as the pair direction moves the
answer 0.23 of the way from clean to flipped in seed0 and 0.14 in seed1 at
dose 1, and 0.54 and 0.45 at dose 2. Two more reference points:

- With the stream at this position replaced by its mean over all prompts
  of the sentence pattern (`wipe`), the answer goes 0.38 and 0.50 of the
  way.
- A prompt in which a third name acts sits 0.51 and 0.53 of the way. That
  is an average over third names of both kinds, and not a measurement of
  who acts being removed.

So a direction that takes the answer a third or half of the way may have
done no more than remove what was there. Every result below therefore
comes with three things:

- **the dose**, from −2 to 2, on the assumption that damage does not
  depend on the sign of a vector while its own effect does. A random
  direction cannot test that assumption: reversed, it is another random
  direction.
- **controls**: random directions, a direction built in the same way for
  two names that are not in the prompt, and one that carries only the kind
  of name.
- **what else changed**, against a yardstick: how far the next-token
  distribution is from the one the steering aims at, outside the tokens
  that were meant to move. It is a KL divergence in nats, with the
  aimed-at distribution as the reference, over the vocabulary without
  those tokens.

Besides the share of the way, the script counts the prompts where A is
above B. That is so in 85% and 76% of the flipped prompts themselves. It
is also so in 1 of the 660 clean prompts in seed0 and in 30 in seed1,
before any steering, so seed1's column starts at 5%.

```sh
uv run scripts/steer.py            # needs the records of step 6
uv run scripts/steer.py --model seed1
```

## Who acts

A clean prompt has A acting and B due. Unless the table says otherwise, a
direction is built from the streams at the acting name in four sentence
patterns and added in the fifth.

| Direction, at dose 1 | seed0: share | A above B | seed1: share | A above B |
|---|---|---|---|---|
| **own**: the prompt's own flipped stream minus its clean one, a patch of step 3 | 0.96 | 81% | 1.04 | 78% |
| **pair**: the same difference for the same two names, mean of the other four patterns | 0.95 | 81% | 1.04 | 78% |
| **names**: `pair`, with a term for each name and none for the two together | 0.96 | 86% | 1.04 | 78% |
| **means + order** | 0.81 | 78% | 0.85 | 67% |
| **means**: the mean stream when B acts minus the mean when A acts | 0.38 | 9% | 0.81 | 64% |
| **order**: the mean of `pair` over all pairs, one vector for every prompt | 0.40 | 18% | 0.07 | 5% |
| **reader**: where step 2's kind of reader raises B over A, at the length of `means` | 0.33 | 4% | 0.68 | 50% |
| **latents**: step 6's latent for B at its usual value, and A's taken away | 0.19 | 0% | 0.43 | 19% |
| control: **kind**, the "She" names minus the "He" names, towards B's kind | 0.24 | 3% | 0.44 | 25% |
| control: **other names**, `means` for two names not in the prompt, of the kinds of B and A | 0.31 | 5% | 0.48 | 30% |
| control: a random direction as long as `means` | 0.15 | 1% | 0.10 | 5% |
| control: a random direction as long as `pair` | 0.23 | 3% | 0.14 | 5% |
| control: `means` plus a random direction as long as `order` | 0.41 | 9% | 0.81 | 64% |
| reference: `wipe`, the pattern's mean stream in place of the prompt's | 0.38 | 4% | 0.50 | 23% |
| no steering | 0 | 0% | 0 | 5% |

![How far a direction added at the acting name takes a clean prompt towards its flipped twin, by dose from −2 to 2: the share of the way, with a line where a prompt with a third name acting sits, and the share of prompts where A is now above B, with a line for the flipped prompt itself; for the pair direction, its part with a term per name, the means with and without the order vector, the order vector, and two controls; means over the five sentence patterns with their range, in both models](figures/steer-who.svg)

**The pair direction does what the patch does.** Built on other sentences,
it takes the answer 0.95 and 1.04 of the way and puts A above B as often
as the patch. This is a low bar. The position is the acting name's own
token, and the two prompts differ there in the token itself. What it shows
is that the mean difference from four sentence patterns works in the
fifth, and that it brings the flipped prompt's whole next-token
distribution: outside the two names the steered prompt is 0.02 and 0.01
nats from the flipped one, where the clean prompt is 1.55 and 1.64 nats
away. That yardstick is mostly the kind of the acting name. The clean
prompt is 2.44 and 2.73 nats from the flipped one where the two names are
of different kinds, and 0.10 to 0.73 where they are of one.

**The answer moves steadily with the dose.** At a quarter of the pair
direction it has moved 0.14 and 0.22 of the way, at half 0.42 and 0.51, at
three quarters 0.74 and 0.80: close to a straight line in seed1, with a
slower start in seed0. The count of prompts with A above B jumps where the
answer passes half-way, from 9% and 23% at half the dose to 65% and 63% at
three quarters. Past 1 the answer overshoots, to 1.13 and 1.43 of the way
at dose 2.

With the sign reversed, the answer in seed1 first moves away from the
flipped one, by 0.21 of the gap at dose −0.5 and 0.15 at −1. In seed0 it
hardly moves (0.04 away, then 0.08 towards). At dose −2 it has moved 0.40
and 0.25 towards the flipped answer, less than a random direction of the
same length does at that dose (0.54 and 0.45). Rimsky et al. report that
larger multipliers degraded their generated text, and limited them.

## What the pair direction is made of

Does the direction need the pair? `names` keeps of the pair direction only
what can be written as a term for the first-mentioned name plus a term for
the second-mentioned one plus a constant, fitted over the 132 pairs. It
does all that the pair direction does: 0.96 and 1.04 of the way, with A
above B in 86% and 78%, and 0.04 and 0.02 nats from the flipped prompt. So
nothing that belongs to the two names together is needed.

The simplest direction one would hope for is `means`: twelve vectors, one
per name, and the direction from A to B. In seed1 it does most of the
work, 0.81 of the way with A above B in 64%. In seed0 it does not: 0.38
and 9%. The `wipe` gives 0.38 as well, though not in the same pairs: 0.42
of the gap where both names are "She" names, where `means` gives 0.11.

What `means` leaves out is this. Each name's mean is taken over both
versions of the prompt, over the clean prompts where the name is
mentioned first and the flipped ones where it is mentioned second. So
whatever every flipped prompt has that its clean twin lacks, whoever the
names are, cancels. `order` is that vector: the mean of the pair direction
over all pairs. It is 11.1 long in seed0 and 8.0 in seed1.

- In seed0 `order` alone moves the answer 0.40 of the way: by 4.4 to 5.4
  logits in each of the three groups of pairs. In seed1 it moves it 0.07
  of the way, 0.1 to 0.7 logits.
- Added to `means` it gives 0.81 and 0.85 of the way, with A above B in
  78% and 67%. A random vector of its length adds 0.02 and nothing (0.41
  and 0.81).
- `names` goes 0.15 and 0.19 of the way further. What it has beyond
  `means + order` is its term for the other name, together with whatever
  of a name's term depends on being mentioned first or second.

By construction `order` separates the prompts in which the second-mentioned
name acts from those in which the first does. So in seed0 the answer rests
in part on which mention acts, in a form that is the same for all names,
and in seed1 it hardly does. What the model does with that vector was not
looked at.

The reader's direction is not an independent one: its cosine with `means`
is 0.91 and 0.90. At the same length it does a little less than `means` in
both models. Marks and Tegmark (2023) report the same order: in their
larger models a difference of means classified about as well as logistic
regression, and it changed the output more in seven of eight conditions,
each direction scaled by its own probe and not to one length.

## The controls find the kind again

The control built from two other names should do nothing, and it takes the
answer 0.31 and 0.48 of the way. By group of pairs, as the group's mean
move over the group's mean gap:

| Dose 1 | seed0: both "She" | both "He" | one of each | seed1: both "She" | both "He" | one of each |
|---|---|---|---|---|---|---|
| pair | 0.87 | 0.96 | 0.96 | 1.15 | 1.03 | 1.03 |
| means | 0.11 | 0.31 | 0.44 | 0.90 | 0.73 | 0.82 |
| order | 1.09 | 0.47 | 0.28 | 0.13 | 0.07 | 0.07 |
| control: other names | 0.14 | 0.13 | 0.36 | 0.18 | 0.13 | 0.59 |
| control: kind | 0 | 0 | 0.33 | 0 | 0 | 0.57 |
| control: random, as long as `means` | 0.19 | 0.11 | 0.13 | 0.09 | 0.11 | 0.10 |

In seed1 the gap of the pairs with two "She" names is 1.05 logits, so the
first seed1 column is a ratio over about one logit.

The two other names are of the kinds of B and A. Where B and A are of one
kind, the control does about what a random direction does (0.13 to 0.18
against 0.09 to 0.19). Where they are of two kinds it moves the answer by
0.36 and 0.59 of that group's gap. A direction that carries only the kind,
the mean over the "She" names minus the mean over the "He" names without A
and B, moves it by 0.33 and 0.57. It is the finding of step 6 from the
writing side.

So the script also measures each name against the names of its own kind
that are in no version of the prompt and are not the control's two names:

| Share of the way, dose 1 | seed0: A | B | seed1: A | B |
|---|---|---|---|---|
| pair | 0.94 | 0.94 | 1.04 | 1.04 |
| names | 0.94 | 0.94 | 1.04 | 1.04 |
| means + order | 0.83 | 0.79 | 0.89 | 0.78 |
| means | 0.32 | 0.25 | 0.74 | 0.79 |
| order | 0.45 | 0.51 | 0.14 | −0.02 |
| latents | 0.16 | 0.09 | 0.39 | 0.35 |
| control: kind | 0.08 | 0.06 | 0.08 | −0.02 |
| control: other names | 0.16 | 0.21 | 0.17 | 0.20 |
| control: random, as long as `means` | 0.13 | 0.17 | 0.11 | 0.11 |
| `wipe` | 0.48 | 0.23 | 0.80 | 0.12 |

The kind direction's row is low partly because the direction is zero for
the 60 pairs of one kind. On the 72 pairs of two kinds it moves A 0.13 of
the way in both models, about what the random direction moves it there
(0.13 and 0.11), and B 0.09 and −0.04 (random: 0.17 and 0.11). The control
built from two other names gives about what the random direction gives in
seed0, and more than it in seed1.

Neither column says by itself that a direction is about the names. A rises
0.48 and 0.80 of the way with the `wipe`, which holds no name. B falls
0.51 of the way in seed0 with `order`, which is the same vector for every
pair (−0.02 in seed1). What the B column does separate is the `wipe` and
the three controls of the table, which reach at most 0.23 in seed0 and
0.20 in seed1, from seed1's `means` at 0.79. In seed0 `means` reaches
0.25, `order` 0.51 and the two together 0.79.

## The latents, as predicted

Step 6 set the two names' latents in the flipped run to their clean values
and found 0.23 and 0.43 of the gap. Here what the same two latents write
is added to the clean run, at their mean values:

| Share of that group's gap | seed0: steered | step 6 | seed1: steered | step 6 |
|---|---|---|---|---|
| all pairs | 0.19 | 0.23 | 0.43 | 0.43 |
| both names followed by "She" | −0.01 | −0.01 | 0.02 | 0.02 |
| both by "He" | 0.19 | 0.20 | 0.42 | 0.41 |
| one of each | 0.21 | 0.27 | 0.45 | 0.45 |

The prediction holds in seed1 and nearly in seed0. It is not a strong
test: the two operations are mirror images of nearly the same vector. The
cosine between this step's direction and the reverse of step 6's move is
0.99, and their lengths differ by 0.4% and 1.0%. It checks the bookkeeping between a latent and a direction in the stream, and
it gives the same small part. In seed0, 3 of the 660 prompts have A above
B, one of them before any steering. In seed1, 19% have, and 5% had before.

## Kind: one direction for the pronoun

The third prediction needs another task. After a sentence about one name,
such as "Yesterday, Tom played in the garden.", the model goes on with
" He" or " She". Six such sentences and twelve names give 36 prompts with
a "He" name and 36 with a "She" name. The measure is the logit of " She"
minus the logit of " He" for the next token: −7.31 and −7.11 on average
for the "He" names, and +7.79 and +8.74 for the "She" names. A direction
is added at a "He" name and taken away at a "She" name.

- **means**: the mean stream at the name over "She" names minus the mean
  over "He" names, from the other five sentences and without the name it
  is used on.
- **latent**: the shared latent of step 6 at its mean value on "She"
  names in the other five sentences, through the decoder.
- **random**: a random direction of the length of `means`.

![The logit of She minus the logit of He after a sentence about one name, as a direction is added at a He name or taken away at a She name, by dose from −2 to 2, for the difference of means, the shared latent and a random direction: one thin line per name and sentence and the mean of each group, in both models](figures/steer-kind.svg)

| "He" names, by dose | 0 | 0.5 | 0.75 | 1 | 1.5 | 2 |
|---|---|---|---|---|---|---|
| seed0, means | −7.31 | −0.38 | +3.00 | +5.56 | +8.43 | +9.54 |
| turned, of 36 | 0 | 14 | 35 | 36 | 36 | 36 |
| seed0, latent | −7.31 | −4.24 | −2.45 | −0.72 | +2.21 | +4.47 |
| turned, of 36 | 0 | 0 | 4 | 12 | 35 | 36 |
| seed0, random | −7.31 | −7.19 | −7.05 | −6.85 | −6.30 | −5.62 |
| seed1, means | −7.11 | −0.83 | +2.19 | +4.94 | +9.37 | +12.01 |
| turned, of 36 | 0 | 11 | 31 | 36 | 36 | 36 |
| seed1, latent | −7.11 | −4.32 | −2.93 | −1.59 | +0.93 | +3.19 |
| turned, of 36 | 0 | 0 | 4 | 9 | 24 | 36 |
| seed1, random | −7.11 | −6.94 | −6.79 | −6.59 | −6.06 | −5.42 |

**One direction turns the pronoun.** At dose 1 the difference of means
turns the pronoun after all 36 "He" names and all 36 "She" names, in both
models. The random direction turns none at that dose.

The possessive goes with it. Outside " She", " He" and the twelve names,
the next-token distributions of two names of one kind are 0.11 and 0.16
nats apart, and those of two names of different kinds 1.81 and 1.18. A
steered name is 0.14 and 0.21 nats from a name of the other kind. Nearly
all of the distance between the kinds is " His" against " Her": with those
two left out as well, names of different kinds are 0.15 and 0.21 apart.

These sentences show almost nothing of a kind beyond its pronouns. With
every form of the two pronouns left out, seventeen tokens, names of
different kinds are hardly further apart than names of one kind: 0.14
against 0.12, and 0.20 against 0.17.

With the sign reversed the measure moves by 3.1 and 6.4 logits at dose −1,
to −10.4 and −13.5, and no further at dose −2.

**At its mean value the shared latent does about half of that.** It takes
a "He" name from −7.31 to −0.72 in seed0 and from −7.11 to −1.59 in seed1,
which is 0.51 and 0.46 of what the difference of means moves it at dose 1.
It turns 12 and 9 of the 36: Sam in every sentence, Max in four and three,
and in seed0 Tom in two. Taken away from a "She" name it leaves +2.12 and
+3.63, which is 0.43 and 0.42 of what the difference of means moves, and
turns 3 and none. Its
direction is half as long as the difference of means (9.3 against 18.2,
and 10.2 against 21.0), with a cosine of 0.67 and 0.59. At twice its mean
value it turns all but one of the 144 cases.

**The rest is in other latents.** The difference of means splits into the
part the dictionary rebuilds and the part it misses, and each can be added
alone:

| Dose 1: "He" names, "She" names, turned of 36 and 36 | seed0 | seed1 |
|---|---|---|
| no steering | −7.31, +7.79 | −7.11, +8.74 |
| means | +5.56, −5.45, 36 and 36 | +4.94, −3.53, 36 and 36 |
| the part the dictionary rebuilds | +5.37, −5.12, 36 and 36 | +5.03, −3.70, 36 and 36 |
| the part it misses | −6.94, +7.64, none | −7.29, +8.81, none |
| the rebuilt part without the shared latent | −0.58, +1.51, 13 and 6 | +0.18, +1.65, 17 and 5 |
| the shared latent | −0.72, +2.12, 12 and 3 | −1.59, +3.63, 9 and none |

The rebuilt part does everything and the missed part moves the measure by
less than 0.4 logits. About 30 latents (30 and 32 on average) differ by
more than 0.5 between the two kinds, on the scale on which the shared one
differs by 6.1 and 6.0, and the shared one differs most. Without it the
rest does about as much as it does alone in seed0, and somewhat more in
seed1. Split by sign, the latents that are higher on "He" names move a
"He" name by 4.1 and 4.3 logits towards "She" when used alone, and those
that are higher on "She" names, the shared one left out, by 2.1 and 2.9.

Templeton et al. (2024) steer a large model by clamping single latents to
multiples of their largest value over the dictionary's training data: 5 or
10 times in most of their examples, and 2 and 20 in others. They report
that interesting results typically needed values outside the range seen in
training, and suspect the reason is that one latent is changed while
others that usually fire with it are not. Here the other latents that
differ between the kinds matter as much as the shared one. The dose in
this note is in units of the latent's mean on "She" names; twice that is
1.5 and 1.7 times the largest value it takes in these sentences.

**The dictionary does not read back what was written.** After the latent's
direction is added to a "He" name at its mean value, the dictionary reads
that latent at 0.71 of that value in both models, and at a quarter of the
dose at 0.02 and 0.07. Two things in the dictionary account for it. Its
encoder reads one unit of the latent's own direction as 0.88 and 0.89 of a
unit. And on a "He" name the reading starts below zero, at −0.17 and −0.18
of the mean value, before the cut that keeps the 32 largest. Taken away
from a "She" name, 0.15 and 0.09 remain, which is the first of the two.
The difference of means at dose 1, by contrast, is read back as 1.00 of
the latent's mean value in both models. Step 6 left this open.
[Step 8](08-limits.md) finds the second of the two things in five of six
dictionaries, in two of them by less (−0.03 and −0.09). In one, the
reading on a "He" name starts just above zero.

## The same in both models, and not

The same: the pair direction does what the patch does (0.95 and 1.04), and
so does its part with a term per name and none for the pair (0.96 and
1.04); the answer moves steadily with the dose and overshoots past 1; a
random direction as long as the pair direction costs 0.23 and 0.14 at dose
1 and 0.54 and 0.45 at dose 2; what a direction from two other names does
on pairs of two kinds is what the kind alone does (0.36 and 0.33, 0.59 and
0.57); the difference of means turns every pronoun at dose 1, its rebuilt
part does so alone, and its missed part does nothing; and the shared
latent at its mean value turns a minority and at twice that nearly all.

Not the same: the vector that every flipped prompt shares moves the answer
0.40 of the way in seed0 and 0.07 in seed1, so the names' means alone
reach 0.38 in seed0 and 0.81 in seed1; the pair direction reversed moves
seed1's answer away from the flipped one by a fifth of the gap and seed0's
hardly; and 5% of seed1's clean prompts have A above B before any
steering.

## What this does not show

- **That these directions are the model's own variables.** A direction
  that turns the answer is enough to turn it here. The pair direction is
  an average of four patches for the same two names.
- **The other way.** Clean prompts were steered towards flipped ones, and
  never flipped ones towards clean.
- **Other sentences and names.** Five patterns of one shape for who acts.
  For the kind, six sentences of one shape and the twelve names the
  direction was built from, one left out at a time.
- **Text.** Everything is measured on the next token. Whether a story
  written after a steered name keeps the pronoun was not looked at.
- **Other places.** One depth and one position.
- **What `order` is to the model**, and why seed0 uses it and seed1 hardly
  does.
- **The spread over prompts.** Shares are ratios of means over the prompts
  of a pattern. The first figure shows the range over patterns and not
  over prompts.
- **An independent choice of latents.** The two names' latents and the
  shared one were chosen by step 6 on all five patterns.
- **That the latent is half of the kind in general.** One dictionary. A
  wider one could split the kind over more latents or gather it in one.

Step 8 asks how much of the seven steps holds for a second dictionary and
for another model.

## Exercises

1. Steer with the pair direction after blocks 2, 6 and 7, calling
   `steer.steered` with streams recorded at that depth. Step 3 found that
   who acts leaves this position after block 6 or 7. Where does the
   direction stop working?
2. Build the pair direction from one other sentence pattern instead of
   four, as Turner et al. do with a single pair of prompts. How much is
   lost?
3. Arditi et al. (2024) also remove a direction, by projecting it out of
   the stream at every layer and position. Write that edit for the Tap and
   project the difference of means for the kind out. Which pronoun
   follows?
4. Generate twenty tokens after a steered name. Does the story keep the
   pronoun, and does the name's own token come back?
5. Steer the flipped prompts towards the clean ones with the pair
   direction reversed. Is the result the mirror image of this note's?

## References

- Turner et al. (2023), [Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248). Activation addition: the difference of a model's activations on a contrast pair of prompts, added to the forward pass at inference time.
- Rimsky et al. (2024), [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681). Steering vectors as the mean difference of residual stream activations over many pairs of examples, added with a positive or negative multiplier; larger multipliers degraded open-ended text.
- Marks and Tegmark (2023), [The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets](https://arxiv.org/abs/2310.06824). Difference-of-means directions that classify about as well as logistic regression in the larger models and are more causally implicated in the output when the model is shifted along them.
- Templeton et al. (2024), [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html). Feature steering: single dictionary features clamped to artificially high or low values during the forward pass.
- Arditi et al. (2024), [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717). A difference-of-means direction that induces a behavior when added and removes it when projected out of the stream.
