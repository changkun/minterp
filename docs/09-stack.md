# 9. The practitioner's stack

Script: [`scripts/stack.py`](../scripts/stack.py) ·
Tests: [`tests/test_stack.py`](../tests/test_stack.py) ·
Results: [`runs/stack/`](../runs/stack/)

This repository wrote its own hooks, lens, patches, dictionary and steering
so that nothing is hidden. Libraries exist for each of these pieces. This
step maps the pieces to them, and runs one library on the open
model of [step 8](08-limits.md) to see that it gives the same numbers.

Two rules for this note. Every call below was read on 2026-11-21 in the
library's documentation or, for `decompose_resid`, `logit_attrs`,
`logit_lens` and the `fwd_hooks` argument with its hook function, in the
docstrings and signatures of the installed package.
`AutoModelForCausalLM.from_pretrained` was run and not looked up. And each
row says whether it was run here or only read. Versions are given where a
library states one. circuit-tracer was read at its main branch (its latest
tag is v0.5.3), and the documentation sites of TransformerLens and nnsight
do not say which version they describe.

The first rule earned its place while this note was written. Tutorials for
TransformerLens written before version 3 load a model with
`HookedTransformer.from_pretrained`. Its README says that this "was
removed in TransformerLens 4.0", and the migration guide that importing
the class raises an error. The replacement is `TransformerBridge`.

## The map

| In this repository | A library | The call, as documented | |
|---|---|---|---|
| `Tap`: record a module's output, or replace it | PyTorch 2.15 | `module.register_forward_hook(hook)`: the hook receives `(module, args, output)` and may return another output. `register_forward_pre_hook(hook)`: `(module, args)`, the positional arguments, and may return others. Both return a handle with `remove()`. | run: it is what `Tap` uses |
| `with Tap(model) as tap: model(tokens)` | TransformerLens 4.4.0 | `bridge = TransformerBridge.boot_transformers(name)`, then `logits, cache = bridge.run_with_cache(tokens)` | run, with `hf_model=` and `tokenizer=` from the files on disk |
| `tap["resid", i]` | | `cache["blocks.{i-1}.hook_resid_post"]`, and `cache["hook_embed"]` for `tap["resid", 0]` | run |
| `tap["attn", i]`, `tap["mlp", i]` | | `cache["blocks.{i-1}.hook_attn_out"]`, `cache["blocks.{i-1}.hook_mlp_out"]` | run |
| `tap["heads", i]` | | `cache["blocks.{i-1}.attn.hook_z"]`: per head, the input of the output projection | run |
| `attention.patterns` | | `cache["blocks.{i-1}.attn.hook_pattern"]` | run |
| `lens.decode` | | `cache.logit_lens()`, over `cache.accumulated_resid(...)` | read |
| `lens.attribute`, summed over the heads of a block | | `cache.decompose_resid(pos_slice=-1, return_labels=True)` and `cache.logit_attrs(stack, tokens=..., incorrect_tokens=..., pos_slice=-1)` | run, with a caveat below |
| `Tap(model, edit=...)`, `patch.run`, `steer.steered` | | `bridge.run_with_hooks(tokens, fwd_hooks=[(name, fn)])`, where `fn(activation, hook)` returns the activation and its second parameter must be called `hook` | run, on the stream at one place |
| the same, without names of its own | nnsight 0.8.3 | `model = TransformersModel(name, task='text-generation', ...)`; inside `with model.trace(prompt):` read `model.transformer.h[-1].output.save()`, write `model.transformer.h[0].output[:] = 0`, and take `model.lm_head.output.save()`. These are GPT-2's paths, the quick start's example. | read |
| `probe.fit` | none named here | a ridge or logistic regression on recorded activations | neither |
| `sae.SAE`, `encode`, `decode` | SAELens 6.56.1 | `sae = SAE.from_pretrained(release=..., sae_id=...)`, `sae.encode(activations)`, `sae.decode(features)` | read |
| `sae.train` | | `LanguageModelSAETrainingRunner(LanguageModelSAERunnerConfig(..., sae=TopKTrainingSAEConfig(k=..., d_in=..., d_sae=...))).run()` | read |
| `attribute.through` | circuit-tracer | `circuit-tracer attribute --prompt ... --transcoder_set ...` with `--graph_output_path` or `--slug` and `--graph_file_dir`: a graph of the direct effects among the features of transcoders, error nodes, tokens and logits | read |
| `minllm.pretrained.load_model` | transformers 5.24.0 | `AutoModelForCausalLM.from_pretrained(folder)` | run |

Five of the six TransformerLens names in the table are its older ones,
which it keeps as aliases; `attn.hook_pattern` is already the canonical
one. Its Model Structure page asks new code to use the canonical names:
`embed.hook_out`, `blocks.{i}.hook_out`, `blocks.{i}.attn.hook_out`,
`blocks.{i}.mlp.hook_out` and `blocks.{i}.attn.o.hook_in`. On a model as
wrapped, a hook function is given the canonical name in `hook.name`. In
compatibility mode it was given the alias for two of the four names tried
here.

What the documentation adds to these calls:

- **TransformerLens** says it loads "15,000+ open source language models
  across 140+ architecture families". Its registry file lists 15,753
  models, 1,204 of them verified, and among those
  `Qwen/Qwen2.5-0.5B-Instruct`. A wrapped model keeps the published
  weights. `bridge.enable_compatibility_mode()` folds the norms into the
  weights, centers them and moves the attention's value biases into its
  output bias, as the removed class did by default.
- **nnsight** wraps a model without renaming anything, so the paths depend
  on the model. The open model, as transformers builds it, has its blocks
  under `model.layers` and no `transformer.h`. A value read inside a trace
  must be saved with `.save()` to exist afterwards. Its quick start warns
  that with transformers 5 a block returns a bare tensor, while an
  attention module still returns a tuple, so that an index `[0]` copied
  from older code silently takes the first prompt of the batch.
- **SAELens** describes its dictionaries as standard PyTorch modules that
  work with activations from any source. It lists BatchTopK, JumpReLU and
  Matryoshka BatchTopK as the state of the art, and the plain top-k
  dictionary of step 5 among the legacy ones.
- **circuit-tracer** implements the attribution graphs of Ameisen et al.
  (2025) with transcoders, which stand in for the MLPs, one per layer or
  across layers. Step 6 used a dictionary on the stream and a gradient
  between two runs. Its README lists transcoders for certain sizes of
  Gemma 2 and 3, Llama 3.1 and 3.2, Qwen 3 and GPT-OSS. The open model of
  step 8 is not among them. It pins transformers below 5 and by default
  builds on the removed `HookedTransformer`, so it needs an environment of
  its own, apart from the versions in this table.

## One of them, run

```sh
uv run --with transformer-lens==4.4.0 --with transformers==5.24.0 scripts/stack.py
```

The library is not a dependency of this repository. `--with` lays it over
the project for one command, here at the versions the record was made
with. The script loads the open model from the same files by minllm's
loader and by transformers, compares the two, and then wraps the
transformers model in a `TransformerBridge`. Both get the same token ids:
the first sixteen of the 132 clean prompts of the first sentence pattern,
in which the first name is Lily in eleven and Tom in five.

| Compared | Largest absolute difference, any entry | Largest absolute entry |
|---|---|---|
| logits: minllm's loader and transformers, before wrapping | 0.00008 | 21.4 |
| logits: the library and minterp | 0.00008 | 21.4 |
| logits: the library and transformers before wrapping | 0.00008 | 21.4 |
| the embedding | 0 | 0.07 |
| the stream after blocks 1, 12 and 24 | 0.0002, after block 24 | 40 after block 24; 1,394 after block 12, at the first token (12 elsewhere) |
| what the attention and the MLP of those blocks write | 0.0001 | 63 |
| the heads of those blocks, at the input of the output projection | 0.00002 | 6 |
| the attention weights of those blocks | 0.000007 | 1 |

Wrapping changes the transformers model in place. The library puts its own
attention in place of transformers', computed from transformers'
projection layers, so that it can expose the attention weights: its code
says it "reimplements attention step-by-step". One pass of the unwrapped
model calls transformers' 24 attention modules 24 times, and one pass of
the wrapped model calls them 0 times. The MLPs that run are still
transformers' own, 24 calls. The logits move by 0.00008. So what agrees
here is minllm's forward pass with transformers' modules and the library's
attention, and the mapping of the names.

Two interventions give the same answer both ways, as a mean over the
sixteen prompts. With the flipped run's stream put into the clean run at
the acting name after block 12, the logit difference goes from +6.552 to
−7.431 by `patch.run` and to −7.431 by `run_with_hooks`. With half of the
difference between the two runs' streams at that place added there
instead, one vector per prompt, it is +0.087 by `steer.steered` and +0.087
by the library.

**One helper needed care.** `cache.logit_attrs` splits a logit difference
over the embedding and every block's attention and MLP: 49 parts here.
Step 1's `lens.attribute` splits each attention further, by head, 361
parts in all, and the script sums its heads within a block to compare. Those 49
sums add up to the logit difference, +6.552. The library's 49 parts sum to
+0.793 on the weights as published.

The helper divides each part by the final norm's scale and leaves out the
norm's learned gain. That gain runs from −0.01 to 16.8 over the
dimensions, so the result is not a rescaling of the right one: single
parts are off by up to 5.04 logits. With the stack multiplied by the gain
before the helper is called, its parts sum to +6.552 and agree with
minterp's part by part, within 0.00002. In compatibility mode, where the
gain is folded into the unembedding, the helper is right as it stands for
a difference of two logits: +6.552, and within 0.00002 part by part.

So the helper assumes folded weights, and the wrapped model does not fold
them unless asked. The library documents this, though not on the helper.
Its compatibility-mode page lists direct logit attribution among the
analyses that need the mode: "raw HF weights produce different (wrong)
attributions". The helper's own docstring is silent, and nothing is
printed when it runs on unfolded weights. The check that caught it here is
the one step 1 used on its own code: the parts must add up to the whole.

**The mode changes other numbers.** It also centers the unembedding. The
logits then differ from the published model's by up to 5.33. At each
position of each prompt the difference is one constant over the whole
vocabulary: the mode subtracts the mean of the published logits there.
That mean is negative, between −5.33 and −0.25 here and −4.19 to −3.06 at
the last position, so the mode's logits are higher by as much. The
log-probabilities agree within 0.0001, and a difference of two logits is
unchanged. For one token's logit the helper's parts sum to the centered
logit: for the right name, as a mean over the sixteen prompts, 22.48 where
the published logit is 18.91.

And the names of the table were compared on the published weights. In the
mode `hook_z` of block 12 differs from `tap["heads", 12]` by up to 1.3,
where its largest entry is 2.9. The difference is the same vector at every
prompt and position, and as large as that block's value bias, which the
mode has moved into the output bias. What the attention and the MLP write
agrees as before. In the mode the library computes the MLPs itself as
well: transformers' own are called 0 times.

## What this does not show

- **That the other libraries give these numbers.** Only TransformerLens
  was run. The rows for nnsight, SAELens and circuit-tracer are what their
  documentation says.
- **A fully independent implementation.** As wrapped, the library runs
  transformers' own blocks, norms and MLPs, and computes the attention
  itself from transformers' weights. The agreement is between minllm's
  forward pass and that mixture, plus the mapping of names.
- **Which library is used most**, or which is best for what.
- **Tomorrow's versions.** The calls were read on 2026-11-21. Three
  versions are in the record; the others are in the table. Two
  documentation sites state no version, and the command above is pinned
  because the newest transformers was published on the day of the record.
- **More than a corner of the model.** One model, the first sixteen
  prompts of one sentence pattern with two first names, three of 24
  blocks, the CPU in 32-bit floats. Only the stream at one place was
  edited through the library: no attention, MLP or head, and no
  attribution per head.
- **Compatibility mode beyond three checks.** The attribution sum, the
  logits and the pieces after block 12 were looked at there.
- **The whole of step 8 through a library.** It trains no dictionary and
  attributes nothing through latents.
- **Text passed as a string.** The script passes token ids. For this
  model the library's tokenizer adds no first token to a string, while
  this repository opens every prompt with `<|endoftext|>`.

## Exercises

1. Redo the position-by-depth patch of step 8 on the open model with
   `run_with_hooks`, on the same prompts as `runs/patch/patch-qwen.json`,
   and compare. Depth 0 is `hook_embed`.
2. Write the same patch with nnsight, in a file or a notebook cell and not
   piped to the interpreter, since a trace reads its own source. The open
   model's blocks are `model.model.layers[i]`.
3. Train the dictionary of step 5 with SAELens on the story model's stream
   after block 4, once as top-k and once as BatchTopK. The runner in the
   table takes a model that TransformerLens or transformers can load, and
   the story model is neither: SAELens's documentation (Extending SAELens)
   points to `SAETrainer` with your own source of activations. Does the
   shared latent of step 6 appear in both? Step 8 trains batch top-k
   dictionaries with this repository's own code, which leaves out the
   auxiliary loss of Bussmann et al. (2024), so its answer need not be the
   library's.
4. Split the logit difference by head through the library:
   `bridge.set_use_attn_result(True)`, then
   `cache.get_full_resid_decomposition(expand_neurons=False, pos_slice=-1, return_labels=True)`
   and `cache.logit_attrs` on that stack multiplied by the gain. Compare
   its head parts with `lens.attribute`'s.
5. In an environment of its own, from a clone of circuit-tracer, pick a
   model it has transcoders for and trace the prompt of step 1. Some of
   those models need an access token. Is there a feature for the kind of
   name?

## References

Documentation, as read on 2026-11-21:

- PyTorch 2.15, [torch.nn.Module](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Module.html): `register_forward_hook` and `register_forward_pre_hook`.
- TransformerLens 4.4.0: the [README](https://github.com/TransformerLensOrg/TransformerLens), [Migrating to TransformerLens 4.0](https://transformerlensorg.github.io/TransformerLens/content/migrating_to_v4.html), [Migrating to TransformerLens 3](https://transformerlensorg.github.io/TransformerLens/content/migrating_to_v3.html), [TransformerBridge Model Structure](https://transformerlensorg.github.io/TransformerLens/content/model_structure.html) for the hook names, and [Compatibility Mode](https://transformerlensorg.github.io/TransformerLens/content/compatibility_mode.html).
- nnsight 0.8.3: the [quick start](https://nnsight.net/getting-started/quickstart/).
- SAELens 6.56.1: the [documentation](https://decoderesearch.github.io/SAELens/v6.56.1/), [Training SAEs](https://decoderesearch.github.io/SAELens/v6.56.1/training_saes/) and [Extending SAELens](https://decoderesearch.github.io/SAELens/v6.56.1/extending_saelens/).
- circuit-tracer, main branch: the [README](https://github.com/decoderesearch/circuit-tracer).

Papers:

- Fiotto-Kaufman et al. (2024), [NNsight and NDIF: Democratizing Access to Open-Weight Foundation Model Internals](https://arxiv.org/abs/2407.14561).
- Ameisen et al. (2025), [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html).
- Bussmann, Leask and Nanda (2024), [BatchTopK Sparse Autoencoders](https://arxiv.org/abs/2412.06410). Batch top-k dictionaries, trained with the same auxiliary loss as top-k ones against latents that stop firing.
