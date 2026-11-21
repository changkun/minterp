# 6. Feature circuit

Code: [`minterp/attribute.py`](../minterp/attribute.py) ·
Script: [`scripts/features.py`](../scripts/features.py) ·
Tests: [`tests/test_attribute.py`](../tests/test_attribute.py),
[`tests/test_feature_circuit.py`](../tests/test_feature_circuit.py) ·
Results: [`runs/features/`](../runs/features/)

[Step 4](04-circuit.md) described the mechanism in heads, and
[step 5](05-features.md) trained a dictionary whose latents might be better
units. This step puts the two together at one place: the acting name's
position after block 4, where [step 3](03-patch.md) found that the answer's
dependence on who acts sits, and which step 4's veto head reads two or
three blocks later. Which latents does the answer depend on there? Is it a
few, and are they latents one can name?

It ends by correcting step 4. Asked in latents, the question "what does the
veto head lower" has a different answer for half of the names.

## Thousands of latents, one gradient

Step 3 moved a part from one run into another and ran the model again, one
run per part. The dictionary has 4,096 latents at every position, and one
run per latent is too many. A gradient gives an estimate for all of them at
once.

[`attribute.py`](../minterp/attribute.py) puts the dictionary into the
forward pass without changing it. The stream after block 4 is written as
what the dictionary rebuilds plus what it misses, `x = decode(z) + err`, so
that the latents `z` and the error `err` are things one can change. For a
number `m` computed from the logits, the gradient of `m` with respect to a
latent says how `m` moves when the latent moves a little. Taking a latent
from its value in one run to its value in another is not little, but to
first order

    m(with latent j from the other run) - m  ≈  (z_other[j] - z[j]) * dm/dz[j]

for every latent from one backward pass. This linear approximation of a
patch is known as attribution patching; Syed et al. (2023) use it to rank
the connections of a model, and Marks et al. (2024) apply it to the
features of a dictionary together with its error term, as here. Marks et
al. report that it underestimates effects early in their model, and
average the gradient over several points between the two runs instead.

So the estimate is used to choose what to look at, and what it chooses is
then moved for real.

```sh
uv run scripts/features.py            # needs the dictionary of step 5
uv run scripts/features.py --model seed1
```

## What the dictionary carries

First the two parts whole, moved for real at the acting name's position,
as in step 3: for who acts, from the clean prompt into the flipped run, and
for a name, from the prompt with another name in its place.

| Moved at the acting name | seed0: rebuilt | missed | both | seed1: rebuilt | missed | both |
|---|---|---|---|---|---|---|
| who acts, share of the gap | 0.94 | 0.06 | 0.97 | 1.00 | 0.03 | 1.04 |
| the second name, share of the way | 0.075 | 0.025 | 0.099 | 0.071 | 0.027 | 0.097 |
| the first name, share of the way | 0.114 | 0.030 | 0.144 | 0.085 | 0.033 | 0.118 |

"Both" is the stream itself, and matches step 3. Who acts is in what the
dictionary rebuilds, almost all of it, in both models and in every sentence
pattern (0.90 to 0.97 in seed0, 0.98 to 1.02 in seed1). Of the little that
this position carries of a name, about three quarters is in what the
dictionary rebuilds and a quarter in what it misses.

Step 5 found more than half of a reader's margin for the other name in the
part the dictionary misses. For the answer it is a quarter. The reader and
the model weigh the two parts differently, which is what step 5 concluded
from the other side.

The gradient's estimate has the order of the parts right and the size
wrong: for who acts it gives 0.57 for the rebuilt part in seed0 where the
real share is 0.94, and 0.82 against 1.00 in seed1.

## Not a few latents

Between a clean prompt and its flipped twin, 48 latents differ at this
position in seed0 and 47 in seed1, on average: about sixteen that are on in
one prompt only, sixteen in the other only, and sixteen that are on in both
with different values. The script ranks them by the size of their estimate,
prompt by prompt, and moves the highest-ranked ones for real.

![How much of the gap the highest-ranked latents carry at the acting name, estimated from the gradient and moved for real, by the number of latents moved: mean of the five sentence patterns with one dot per pattern, and the share of everything the dictionary rebuilds as a dashed line, for both models](figures/features-count.svg)

| Latents moved, highest-ranked first | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| seed0, moved for real | 0.17 | 0.30 | 0.48 | 0.69 | 0.84 | 0.91 |
| seed0, estimated | 0.13 | 0.23 | 0.36 | 0.48 | 0.54 | 0.57 |
| seed1, moved for real | 0.27 | 0.46 | 0.68 | 0.85 | 0.95 | 0.98 |
| seed1, estimated | 0.25 | 0.40 | 0.58 | 0.72 | 0.79 | 0.82 |

The dependence on who acts is spread out. The single highest-ranked latent
of a prompt carries 0.17 of the gap in seed0 and 0.27 in seed1, and it
takes eight to carry 0.69 and 0.85. The eight are chosen prompt by prompt
and are not the same latents everywhere.

How compact is that? Step 3's parts at this same position are the nearest
comparison: the MLP of block 1 carried 1.00 and 0.99 there, and the
fetching heads 0.52 and 0.21. A single latent carries less than the MLP in
both models, and less than the fetching head in seed0. In seed1 it carries
about as much as the head, 0.27 against 0.21. The veto heads,
with 0.87 and 0.69, act later and at another position, so they are not a
fair comparison for a latent after block 4.

**Here the estimate ranks well and counts low.** For each prompt's five
highest-ranked latents, moved one at a time, the estimate and the real
effect have the same sign in 99.9% of cases in seed0 and in all of them in
seed1, and correlate at 0.95 and 0.99. The real effect is larger: for the
highest-ranked latent, +2.03 logits against an estimated +1.59 in seed0,
and +2.02 against +1.81 in seed1. That is these two models, this place and
this measure. Syed et al. report a much weaker agreement between the two in one
of their cases.

## A latent for the name, twice

Step 5 found, for each name, a latent that fires when that name acts. If
those were the units of the mechanism, moving the two names' latents, one
off and one on, should carry the answer.

The veto head offers a second way to pick a latent for each name. What that
head adds to the acting name's logit is itself a number with a gradient. So
for every prompt, clean and flipped, the script finds the latent at the
acting name's position on which the head's write rests most, by size, and
for each name takes the latent that comes out most often.

| The two names' latents, moved for real | seed0: step 5's | the head's | seed1: step 5's | the head's |
|---|---|---|---|---|
| all pairs | 0.05 | 0.23 | 0.19 | 0.43 |
| both names followed by "She" (30 pairs) | 0.03 | −0.01 | 0.10 | 0.02 |
| both by "He" (30 pairs) | 0.09 | 0.20 | 0.40 | 0.41 |
| one of each (72 pairs) | 0.05 | 0.27 | 0.14 | 0.45 |

Neither choice comes near the 0.94 and 1.00 of the whole rebuilt stream.
Step 5's latents carry 0.05 of the gap in seed0. In seed1 they carry 0.40
where both names are followed by "He", and little elsewhere.

The head's latents are the same latent for six of the twelve names:

| The latent the veto head rests on most | seed0 | seed1 |
|---|---|---|
| when Lily, Mia, Lucy, Anna, Sue or Sara acts | one latent for all six | one latent for all six |
| when Tom, Tim, Ben, Sam, Max or Jack acts | six different latents | six different latents |

The six that share a latent are the six after which the model goes on with
"She" (step 5). The shared latent fires on at least 90% of the prompts in
which any of them acts. So where both names of a pair are of that kind, the
head's "two latents" are one, and moving it carries nothing.

The other six latents are different latents, and in seed0 still mostly not
latents for one name. By the test of firing on 90% of a name's prompts, seed0's
latent for Tim also fires for Ben and Jack, the one for Max also for Jack,
and the one for Jack for Tom and Ben. Only Tom's fires for Tom alone. In
seed1 three do: Tim's, Sam's and Max's.

How stable is the choice? A latent picked on four sentence patterns is the
one the head rests on most in the fifth as often as in the four. For ten
names in each model that is in 86% to 100% of the name's prompts in seed0
and 97% to 100% in seed1, and for Ben and Jack in seed0 and Max and Jack in
seed1 in 50% to 69%.

## Name or kind?

The shared latent raises a question about step 4. There, the veto head
"lowers the acting name" by 1.4 to 1.7 logits, measured against the mean of
nine uninvolved names. Of those nine, three to five are of the acting
name's kind, 4.1 on average, and the other 4.9 are of the other kind. So a
head that lowered every name of that kind alike would score more than half
as well on that measure as one that lowered the acting name alone.

So the script measures the head's write again on the clean prompts, against
the uninvolved names of the acting name's own kind:

| What the veto head adds | seed0: a "She" name acts | a "He" name acts | seed1: "She" | "He" |
|---|---|---|---|---|
| to the acting name, against the nine | −1.01 | −1.75 | −1.20 | −1.72 |
| to the acting name, against its own kind | −0.26 | −1.12 | −0.23 | −1.16 |
| to its own kind, against the other kind | −1.40 | −1.19 | −1.80 | −1.07 |

The first row is the second plus 0.55 times the third, since 4.9 of the
nine are of the other kind.

**Where one of the six "She" names acts, the head mostly lowers the kind.**
Against other names of its kind the acting name goes down by 0.26 and 0.23
logits. The rest of the −1.01 and −1.20 is all names of that kind going
down against the other kind. Where a "He" name acts, the head lowers the
kind as well, and lowers the name within its kind by 1.12 and 1.16.

Switching the shared latent off for real, at the acting name's position on
the prompts where a "She" name acts, shows the same thing from the
dictionary's side:

| The shared latent switched off | seed0 | seed1 |
|---|---|---|
| the acting name against the nine | +0.68 | +0.77 |
| the same, estimated | +0.59 | +0.72 |
| the acting name against its own kind | −0.03 | −0.02 |
| its own kind against the other | +1.30 | +1.44 |

That latent is a latent for the kind, as far as the head is concerned: off,
it takes away most of the kind's lowering and leaves the name where it was
among its kind.

Step 4's sentence stands as a measurement and was too narrow as a
description. On a prompt with one name of each kind, lowering the kind of
the name that acts is enough to decide between the two. Those are 72 of the
132 pairs. For the other 60 it decides nothing, and the gap between the
clean and the flipped prompt shows where the model can still tell the names
apart:

| The gap, in logits | seed0 | seed1 |
|---|---|---|
| one name of each kind (72 pairs) | 15.49 | 10.69 |
| both followed by "He" (30) | 9.94 | 6.26 |
| both followed by "She" (30) | 4.95 | 1.05 |

Among the "He" names the models tell the two apart within the kind. Among
the "She" names seed0 does so less, and seed1 hardly at all, which fits a
head that there lowers the kind and not the name.

## What the veto head rests on

What the head adds to the acting name, against the nine as in step 4, what
the gradient says it rests on at the acting name's position, and what is
really lost when latents there are switched off. In logits, on clean
prompts.

| | seed0, head 7.0 | seed1, head 8.0 |
|---|---|---|
| adds to the acting name | −1.38 | −1.46 |
| estimated to rest on all latents at the acting name | −0.72 | −1.02 |
| really lost with all of them off | −1.25 | −1.20 |
| estimated to rest on its highest latent | −0.63 | −0.76 |
| really lost with that latent off | −0.74 | −0.81 |
| adds to the other name | +0.42 | +0.39 |
| estimated to rest on all latents at the acting name | +0.16 | +0.18 |
| really lost with all of them off | +0.74 | +0.88 |

Most of what the head adds to the acting name is lost when the latents at
that position are off, 91% of it in seed0 and 83% in seed1, and more than
half of it with one latent, which the estimate has about right. For all
latents together the estimate is low again, and for the other name it is
far off, +0.16 against a real +0.74. That last figure is more than the head
adds: with all latents off, what it writes for the other name turns
negative.

The −0.72 is a net figure. In seed0 the latents that the estimate counts
against the acting name sum to −1.50 and those it counts for it to +0.78. The
gradient here holds the size of the final stream fixed, so that it is the
gradient of the head's write alone.

The other name's +0.42 is itself an average of two cases. Where the two
names are of different kinds the head adds +1.02 to the other name, and
where they are of the same kind it takes 0.30 off (+1.07 and −0.43 in
seed1). The split is the kind effect seen from the other name. The average
is not: a head that lowered only the kind of the acting name would add
nothing to the other name on average, so about 0.4 logits are added beyond
that.

## The same in both models, and not

The same: who acts is carried by what the dictionary rebuilds (0.94 and
1.00); it is spread over many latents, eight per prompt for 0.69 and 0.85;
here the estimate ranks latents well (0.95 and 0.99) and counts low; the
veto head rests on one shared latent for the six names the model follows
with "She"; and where one of those acts the head lowers the kind and hardly
the name (−0.26 and −0.23 against its own kind). Not the same: one latent
carries 0.17 in seed0 and 0.27 in seed1; step 5's latents carry 0.09 and
0.40 where both names are followed by "He"; and the estimate of the rebuilt
part is further off in seed0 (0.57 of 0.94) than in seed1 (0.82 of 1.00).

## What this does not show

- **A circuit in latents.** This step looked at one place. A feature
  circuit would follow the latents from this position through the veto
  head to the answer, with a dictionary at each stage. Here only the first
  stage was opened, and it was not compact.
- **That a better dictionary would not be compact.** One dictionary of
  4,096 latents with at most 32 active. Step 5's exercises on width apply.
- **What the many small latents are.** Eight latents carry 0.69 and only
  the highest was looked at.
- **How the dictionary would read a moved latent.** Moving a latent
  changes the stream. Encoding the changed stream again need not give the
  latents that were set, so "this latent, moved" is a statement about the
  vector and not about the dictionary's own reading of it.
- **How much of steps 3 and 4 is kind and how much name.** The kind
  measure was applied to the veto head's write on clean prompts only.

Step 7 turns from reading to writing. If a direction found in the earlier
steps is what the model goes by, adding it to the stream should change the
answer in a way that can be predicted.

## Exercises

1. Rank the latents by the estimate taken in the clean run instead of the
   flipped one, or by the mean of the two. Does the real share of the top
   eight change?
2. Marks et al. average the gradient over several points between the two
   runs. Do that with ten points. Does the estimate of the rebuilt part
   come closer to the real 0.94?
3. Take one fixed set of eight latents for all prompts of a pattern, the
   eight with the largest mean estimate. How much do they carry, against
   the 0.69 of eight chosen per prompt?
4. Apply the kind measure to step 3: split the pairs by kind and repeat the
   share of the veto head for each group.
5. Apply it to step 4's copying heads. Do they raise a name, or a kind?

## References

- Syed, Rager and Conmy (2023), [Attribution Patching Outperforms Automated Circuit Discovery](https://arxiv.org/abs/2310.10348). A linear approximation to activation patching from two forward passes and a backward pass, used to estimate the importance of each connection in a model.
- Marks et al. (2024), [Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models](https://arxiv.org/abs/2403.19647). A model's activations written as a sparse sum of dictionary features plus an error term, with effects attributed to both by a first-order approximation, and by gradients averaged between the two runs where that underestimates.
