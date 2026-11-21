# minterp

A minimal walk through interpretability, on a language model small enough to
train on a laptop and written to be read. It takes one thing the model does
and follows it through the methods people use to look inside a network:
reading activations, probes, patching, circuits, sparse autoencoders and
steering.

The model is the story model that [minllm](https://github.com/changkun/minllm)
trains in twenty minutes: 8 blocks, 6 heads in each, 14 million parameters.
Given

    One day, Lily and Tom were playing in the garden. Lily said hello to

it holds " Tom" by far the likeliest next word. Nobody wrote a rule for
that. The question of this
repository is how the model does it, and the steps ask it in an order: each
one can support a stronger claim than the one before, and needs what the one
before found.

The only machine learning dependency is PyTorch. Hooks, probes, patching
and the autoencoder are written here, a few hundred lines per step.
minllm supplies the model code and the tokenizer, and nothing in it is
changed: every look inside goes through hooks.

## Steps

Each step has a note that explains it, code that implements it, tests that
check what the note claims, and saved results that every number in the note
is read from. Steps without a link are not written yet.

| | Step | The claim it can support | What gets built |
|---|---|---|---|
| 1 | [Read](docs/01-read.md) | The answer is visible here | The task and its measure, hooks that record the stream, the stream decoded after every block, and the final answer split into what each head and MLP wrote |
| 2 | [Probe](docs/02-probe.md) | The answer can be read out here | Linear readers trained for every depth at the positions of the names and the last one, and controls that show what a reader's score does not mean |
| 3 | Patch | The model needs this part | The activations of one run placed into another: blocks and positions, then heads, and what the choice of the other run changes |
| 4 | Circuit | These parts do it, in this way | What the heads found in step 3 attend to and write, the paths between them, and two tests: the circuit alone and the model without it, against random sets of heads |
| 5 | Features | These are the units | A sparse autoencoder on the stream: what it reconstructs, what it costs the model, and what its latents fire on |
| 6 | Feature circuit | The mechanism in those units | The answer attributed to latents by gradients, checked by ablation, and set beside the heads of step 4 |
| 7 | Steer | The description is good enough to change the behavior | Directions from steps 2, 3 and 5 added while the model writes: the dose, the effect, and the damage elsewhere |
| 8 | Limits | This much holds for another model | What held in both models through the first seven steps, a second dictionary, and the same task in an open model of 0.5 billion parameters |
| 9 | The practitioner's stack | | Each piece mapped to the library people use for it |

Every note ends with what its finding does not show, and the next step
starts there.

## Your model is not mine

The notes report two models that differ only in the seed of their training
run. Yours will differ again: which head does what is not fixed by the
recipe. So no script has a head or a position written into it. Each one
finds them, and the notes say which findings were the same in both models
and which were not.

## Where the ideas are explained

The notes say what a method does and what it found here. Where a method
comes from, what the papers behind it measured, and how far it carries for
a claim about safety is the subject of Part VII of
[AI Alignment](https://alignment.latere.site), which the steps follow:

| Steps | Chapter |
|---|---|
| 1 to 3 | [Questions and methods](https://alignment.latere.site/en/inside/01-questions-and-methods.html) |
| 5 | [Superposition and features](https://alignment.latere.site/en/inside/02-superposition-and-features.html) |
| 4 and 6 | [Circuits](https://alignment.latere.site/en/inside/03-circuits.html) |
| 2 and 7 | [Representations and steering](https://alignment.latere.site/en/inside/04-representations-and-steering.html) |
| 8 | [Limits of inspection](https://alignment.latere.site/en/inside/05-limits-of-inspection.html) |

## Quick start

The model comes from minllm, in a checkout beside this one (or wherever
`MINLLM_DIR` points).

```sh
git clone https://github.com/changkun/minllm && cd minllm
uv sync
uv run scripts/download_data.py      # TinyStories
uv run scripts/train_tokenizer.py
uv run scripts/encode_data.py
uv run scripts/train.py                                  # the story model, 20 minutes
uv run scripts/train.py --seed 1 --out out/base-seed1    # a second one, for the comparisons
cd ../minterp

uv sync
uv run pytest
```

Scripts write their results to `out/runs/`. The files under `runs/` are the
record the notes quote; `MINTERP_RUNS=runs` writes there.

## License

MIT, see [LICENSE](LICENSE).
