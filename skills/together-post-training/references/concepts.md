# Core concepts

The ideas behind the Together Post Training service: what a training session holds, what a training sample carries, and why the GRPO loop is shaped the way it is. The [SDK reference](./sdk-reference.md) gives the signatures; this page gives the reasoning behind them.

**Contents:** [Training sessions](#training-sessions) · [The training loop](#the-training-loop) · [SFT vs. GRPO](#sft-vs-grpo) · [Samples and loss masks](#samples-and-loss-masks) · [Rewards and advantages](#rewards-and-advantages) · [Clipping](#clipping) · [Overlong handling](#overlong-handling) · [LoRA](#lora) · [Checkpoints](#checkpoints) · [Learning rate schedule](#learning-rate-schedule) · [Codecs](#codecs) · [Multi-turn environments](#multi-turn-environments) · [Custom losses](#custom-losses)

---

## Training sessions

All training happens inside a **session**. A session is a server-side process that holds:

- A copy of the base model weights.
- A LoRA adapter (the trainable parameters).
- Optimizer state (Adam moments).
- A step counter.

You interact with a session through the `SessionClient` object. Creating one starts a session; stopping it releases the resources.

```
SessionClient.create() → session (TRAINING_SESSION_STATUS_RUNNING) → sample / forward_backward / optim_step / weights_sync → stop()
```

Sessions are stateful. Each `optim_step` updates the trainer's weights in-place. A later `weights_sync` makes those weights available to the generator so subsequent `sample` calls use the updated model.

Session status strings are full enum constants (`TRAINING_SESSION_STATUS_*`). Compare against them directly. See [SDK reference](./sdk-reference.md#session-status-values) for the full list.

**Generator replicas.** A model resource provisions `num_generator_replicas` samplers alongside the trainer: `>=1` (default `1`) can both `sample` and train, `0` is trainer-only (no sampler). Loops that never call `sample` — SFT and DPO — pass `num_generator_replicas=0`, which provisions less hardware.

**Model resources vs. sessions (multi-LoRA).** The service separates the *model resource* — the GPU pods plus base-model weights — from the *sessions* that run on it. A session is your trainable LoRA adapter; the resource is the (expensive) base model it trains against. When a resource is created with `lora_enabled=True`, every attached session is an independent adapter over the same frozen base weights, so N adapters cost roughly one base model's worth of GPUs instead of N. You can *own* a resource (provision it, then delete it when done) or *borrow* one a teammate provisioned (attach, leave it running). In the cookbook you express this by setting `model_resources_id` on the config and training as usual ([Multi-LoRA / shared model resources](./cookbook-guide.md#multi-lora--shared-model-resources)); at the SDK level it's `ModelResourcesClient.create(...)` + `create_session(...)` ([SDK reference: ModelResourcesClient](./sdk-reference.md#modelresourcesclient)).

---

## The training loop

Both SFT and GRPO share the same update loop:

1. **`forward_backward(samples, loss)`**. Feed training samples to the model, compute the loss, and accumulate gradients.
2. **`session.trainer.optim_step(adam_params)`**. Apply gradients to update the trainer's model weights. The learning rate lives inside `adam_params` (`adam_params.learning_rate`), not as a top-level argument.
3. **`session.trainer.weights_sync(weight_sync_type)`** (sampling loops only). Publish the updated weights for sampling. GRPO and other on-policy loops pass `"WEIGHT_SYNC_TYPE_SYNCHRONOUS"` so the next sample uses the updated policy. Trainer-only SFT/DPO (`num_generator_replicas=0`) skip this — there is no generator, and the next train step uses the trainer weights directly. See [SDK reference: weights_sync](./sdk-reference.md#sessiontrainerweights_sync) for the other modes (`BACKGROUND_PUBLISH`, `PIPELINE`) and why `PIPELINE` is rejected on LoRA sessions.
4. Repeat.

The difference between SFT and GRPO is what goes into `samples` and which `loss` you use.

In the cookbook, the generic GRPO loop lives in `post_training_cookbook.grpo.loop.training_loop`. It takes a `collect_samples(step, session) -> (samples, step_log)` callback and runs the update loop above each iteration. The two GRPO entry points live in `post_training_cookbook.grpo.train`: `train_single_turn` builds the callback via `make_single_turn_collector`, and `train_multi_turn` plugs in a collector that drives an `Env` to gather whole episodes.

A third variant splits step 1 in two: when the loss must be computed in client-side Python, `forward` scores the batch without accumulating gradients, you evaluate the loss with autograd, and `custom_forward_backward` submits the gradients. The first pass has to stay forward-only — gradients from a built-in loss would be added to yours before `optim_step` sees them. DPO uses this path. See [Custom losses](#custom-losses) below.

---

## SFT vs. GRPO

### Supervised fine-tuning (SFT)

- **Input:** pre-existing conversations (user question plus expert answer).
- **Loss:** cross-entropy on the assistant tokens.
- **Signal:** the model learns to imitate the training data.
- **Strength:** fast, stable, predictable.
- **Limitation:** bounded by the quality of the training data.

### Group relative policy optimization (GRPO)

- **Input:** prompts only. The model generates its own responses.
- **Loss:** GRPO policy gradient with clipped importance sampling.
- **Signal:** a reward function scores each response; the model learns to produce higher-reward outputs.
- **Strength:** can exceed the quality of any fixed dataset by discovering novel strategies.
- **Limitation:** slower, less stable, requires a good reward function.

### Direct preference optimization (DPO)

- **Input:** preference pairs — a `(chosen, rejected)` response for the same prompt.
- **Loss:** the DPO objective, computed in client-side Python over reference-model and policy log-probabilities (a [custom loss](#custom-losses)).
- **Signal:** the model learns to raise the relative likelihood of chosen over rejected responses.
- **Strength:** uses preference data directly without training a separate reward model or running online sampling.
- **Limitation:** bounded by the preference data; no exploration.

### When to use which

| Scenario | Approach |
|---|---|
| You have high-quality labeled data | SFT |
| You have pairwise preference data (A is better than B) | DPO |
| You can write a reward function but don't have labeled data | GRPO |
| You want the model to learn a specific format or style | SFT first, then GRPO |
| You want to maximize accuracy on a measurable task | GRPO (possibly after SFT warmup) |

---

## Samples and loss masks

A **training sample** tells the model what to learn from. You describe one in **token space** — one entry per token of the flat prompt-plus-response sequence:

- **Tokens.** The full sequence, prompt and response concatenated.
- **Loss mask.** `1` means "train on this token", `0` means "ignore". The prompt boundary is only the run of zeros at the head of the mask, so there is no separate prompt length to pass and none to get wrong.

Without the loss mask the model would be trained to predict the prompt tokens as well, which wastes computation and can hurt quality.

What goes on the wire is **target-aligned**, and it is one entry shorter: slot `i` scores the prediction made at input position `i` against `target_tokens[i]`. A run of N tokens yields N-1 slots, because the first token is an input that is never a target and the last is a target that is never an input. Below, six tokens become five slots:

```
Slot:       0         1         2           3           4
Input:      prompt_1  prompt_2  prompt_3    response_1  response_2
Target:     prompt_2  prompt_3  response_1  response_2  response_3
Loss mask:  0         0         1           1           1
```

Two things to read off it. The mask's zero run is one shorter than the prompt: slot 2 is where the model first predicts a response token, so that is the first slot that trains. And the **final slot's mask is `1`** — the last response token is a target like any other, so there is no trailing slot to zero out.

`build_cross_entropy_sample` (`post_training_cookbook.samples`) and `build_policy_sample` (`post_training_cookbook.grpo.samples`) take the token-space arrays and own the trim to slots, so you never write it yourself.

For GRPO, samples additionally carry:

- **Advantages.** How much better this completion was than the group average. You pass one scalar for the sequence and the builder broadcasts it over the trained tokens, off the same mask.
- **Generator log-probabilities.** What the model assigned to each token when it generated the response — the sampling-time baseline of the importance ratio.

---

## Rewards and advantages

In GRPO, training signal comes from **rewards**. For each prompt:

1. The model generates `group_size` completions.
2. Each completion is scored by the reward function, yielding `reward_i`.
3. Advantages are computed: `advantage_i = reward_i - mean(rewards)`.

Completions with positive advantage (above average) get reinforced. Completions with negative advantage get suppressed.

`compute_group_advantages` in `post_training_cookbook.grpo.samples` returns a `GroupAdvantages` named tuple. It distinguishes:

- **Zero-variance, all-zero**. Every completion got 0 reward; the group is skipped.
- **Zero-variance, identical-nonzero**. Every completion got the same nonzero reward; the group is also skipped, but flagged separately for diagnostics.
- **Nonzero variance**. Advantages are mean-centered rewards.

Skipped groups produce no training samples. The W&B step log surfaces both `zero_variance_skip_ratio` and `identical_nonzero_reward_skip_ratio`. See the [metrics reference](./metrics.md#group-skip-ratios) for how to read those values.

**Dynamic sampling (DAPO).** When too many groups collapse to identical rewards, a step can be left with almost nothing to train on. Setting `max_sampling_attempts > 1` turns on dynamic sampling: the step keeps drawing fresh batches and discarding zero-variance groups until it has accumulated `batch_size` usable groups, or it hits the attempt cap. With the default `max_sampling_attempts=1`, each step draws exactly one batch.

---

## Clipping

GRPO uses **clipped importance sampling** to prevent the model from changing too much in a single step.

The importance ratio is `exp(current_logprob - generator_logprob)`, which measures how much more (or less) likely the current model makes this token compared to when it generated the response.

`clip_low_threshold` and `clip_high_threshold` are the absolute bounds the ratio is clamped to — not offsets from `1.0`:

- For **PPO / GRPO**, if the ratio exceeds `clip_high_threshold` on a positive-advantage token (or falls below `clip_low_threshold` on a negative-advantage token), the surrogate zeros that token's gradient.
- For **CISPO**, the same bounds clamp a **detached** importance-sampling coefficient on `log π` — they never zero a token's gradient, only bound the weight. A positive lower bound floors that coefficient even for tokens the policy has already moved away from, which is usually undesirable off-policy.

`GRPOConfig` leaves both knobs as `None` and omits them from the payload so server defaults apply: `0.8` / `1.2` for `ppo`/`grpo`, `0.0` / `4.0` (one-sided) for `cispo`. Set either field to override. Bounds must be in `[0, 1]` / `>= 1`. A DAPO epsilon pair `(ε_low, ε_high)` maps to `(1 - ε_low, 1 + ε_high)` — e.g. DAPO Clip-Higher `(0.2, 0.28)` → `(0.8, 1.28)`. Moving either bound toward `1.0` clips harder.

`forward_backward` returns the clip fractions in `metrics` under keys like `loss/clip/high_fraction` and `loss/clip/low_fraction`. The cookbook logs them to W&B under the `fb/` namespace. See the [metrics reference](./metrics.md#service-emitted-metrics-grpo) for the full set and how to interpret each one.

**Token-level vs. sequence-level ratio.** By default the importance ratio (and clipping) is computed per token. Set `GRPOConfig.ratio_type="GRPO_LOSS_RATIO_TYPE_SEQUENCE"` (with `loss="grpo"`) for the GSPO sequence-level ratio, which scores a whole completion with a single ratio. The cookbook recipes use the token-level default.

---

## Overlong handling

Responses that exceed the token budget can be handled in two ways. Configure them through `GRPOConfig`.

### Overlong penalty (default)

A non-positive penalty is added to the reward whenever the response approaches the budget:

```
penalty = -min(exceed_tokens / buffer_len * factor, 0.0)
final_reward = raw_reward + penalty
```

The penalty starts ramping up `overlong_buffer_len` tokens before `max_sample_tokens` and is scaled by `overlong_penalty_factor`. Set `overlong_penalty_factor=0.0` to disable. The read-only `GRPOConfig.overlong_penalty` property returns `True` when the factor is positive.

### Overlong masking (alternative)

Instead of penalizing, overlong completions are dropped before sample construction — no training sample is built, so the model neither learns from nor is penalized for them (their reward still contributes to the group baseline).

Set `overlong_masking=True` (and typically `overlong_penalty_factor=0.0`) to use this approach.

---

## LoRA

Training defaults to **LoRA** (low-rank adaptation). Instead of updating all model parameters, LoRA adds small trainable matrices to each layer, which makes it:

- **Faster.** Fewer parameters to update.
- **Memory-efficient.** The base model weights are frozen.
- **Composable.** You can download only the adapter and apply it to the base model later.

Key parameters:

- **`rank`.** Dimensionality of the low-rank matrices. Higher means more expressive but more parameters.
- **`alpha`.** Scaling factor. Often set equal to rank.
- **`dropout`.** Regularization on LoRA layers.

**Full-weight fine-tuning.** `use_lora=False` (or `ModelResourcesClient.create(lora_enabled=False)`). Allowed modes: [supported-models discovery](./sdk-reference.md#supported-models-and-gpu-hardware). Use a lower LR than LoRA. See `recipes/sft/no_robots_full.py`.

**Multiple adapters on one base model.** Because the base weights are frozen and only the adapter is trainable, several LoRA adapters can share one provisioned base model. `ModelResourcesClient` exposes this directly (see [Training sessions](#training-sessions) above).

---

## Checkpoints

### Inference checkpoints

Snapshot the current model weights and upload them to the Together model registry. The checkpoint's `registration` holds the registry `id` and `revision_id`, used for:

- Downloading weights (merged or adapter-only).
- Deploying on Dedicated Model Inference.

### Training checkpoints

Snapshot the full training state (adapter plus optimizer moments plus step counter). Used to:

- Pause training and resume later on a new session (omit `load_optimizer` or pass `True` to restore optimizer and step).
- Seed a *different* session — different loss, fresh optimizer, step 0 — via `load_optimizer=False` (for example SFT→GRPO on the same adapter).
- Fork training from a known-good state with different hyperparameters.

---

## Learning rate schedule

The cookbook uses **cosine annealing with linear warmup**:

1. **Warmup phase** (first `warmup_ratio` fraction of steps): LR increases linearly from 0 to `learning_rate`.
2. **Decay phase** (remaining steps): LR follows a cosine curve from `learning_rate` down to `min_lr_ratio * learning_rate`.

```
LR
│  ╱‾‾‾‾‾‾╲
│ ╱        ╲
│╱          ╲___
└─────────────────── step
  warmup    decay
```

Configure this via `warmup_ratio` and `min_lr_ratio` in `GRPOConfig`. SFT uses simpler linear decay from `learning_rate` to 0 over the run.

---

## Codecs

Tokenization, chat-template rendering, and structured response parsing all live behind the `MessageCodec` protocol in `post_training_cookbook.codecs`. Each codec is model-specific because chat templates vary across model families.

```python
from post_training_cookbook.codecs import get_codec

codec = get_codec("Qwen/Qwen3.8-27B")
tokens = codec.encode_messages(
    [{"role": "user", "content": "What is 2+2?"}],
    add_generation_prompt=True,
)
```

A codec exposes five methods:

- `encode_messages(messages, *, tools=None, add_generation_prompt=False)` applies the chat template and returns a flat token list.
- `parse_response(tokens, *, tools=None)` decodes sampled tokens back into a structured `Message` and returns a `(message, parse_ok)` pair. `parse_ok` is `False` if tool-call JSON was malformed or `<tool_call>` tags were unclosed; GLM-5 parses a turn cut off before `</think>` as reasoning only, with `parse_ok` `True`. Pass the same `tools` you gave `encode_messages`, so each argument is parsed to its declared type. This matters for wire formats that don't carry types: the Qwen3.5 XML form renders argument values unquoted, so a string-typed zip code like `"32286"` would otherwise be read as an int. Kimi-K3 ignores `tools`: its wire format tags each argument with a type, and serving executes that type, so the codec does too. Kimi-K3 and MiniMax-M3 drop a complete but malformed call, as serving does, and fail the parse only when a call is cut off. The training loop treats parse failures as truncated episodes.
- `turn_terminator(next_role)` returns the tokens a trained assistant turn needs in order to stop, for templates that only render the stop once the next message arrives. Empty for Qwen's universal `<|im_end|>`, Kimi-K3's `<|end_of_msg|>`, and MiniMax-M3's `[e~[`, which the template writes inside the turn; the next role's marker for GLM-5 (`<|user|>`, `<|observation|>`, or `<|endoftext|>` at the end of the conversation). Raises `ValueError` when the model has no stop token for the given continuation (the caller must merge, split, or drop the turn).
- `encode(text)` tokenizes raw text without any chat-template wrapping.
- `decode(tokens)` returns a string with special tokens stripped, except the structural markers a family's response parser needs. Kimi-K3 decodes verbatim: its XTML markers (`<|open|>`, `<|close|>`, `<|sep|>`, `<|end_of_msg|>`) stay in the text.

`get_codec` resolves codecs from the model name by longest matching prefix (`qwen3.5` / `qwen3.6` / `qwen3.8` map to `Qwen35Codec`, `glm-5.3` to `Glm5Codec`, `kimi-k3` to `KimiK3Codec`, `minimax-m3` to `MiniMaxM3Codec`). An unrecognized model raises `ValueError`. To support a new model family, implement the `MessageCodec` protocol and register it in `post_training_cookbook/codecs/__init__.py` (the registry is the source of truth for which prefixes are supported).

GLM-5 and Kimi-K3 prepend a thinking-effort system block whenever thinking is on (GLM-5 always thinks), and both default to the maximum. `thinking_effort` (on `get_codec` and the GRPO/SFT configs) lowers it: `low`, `high`, or `max` for GLM-5 and Kimi-K3. `None` keeps the template default, and the Qwen and MiniMax-M3 codecs reject any value because their templates have no effort setting. At the default, runs need a generous `max_sample_tokens` (or, on Kimi-K3, `enable_thinking=False`); otherwise rollouts truncate mid-reasoning.

### Reasoning in history: `enable_thinking` and `preserve_thinking`

`get_codec` (and the GRPO/SFT configs) take two thinking options, plus `thinking_effort` (above):

- **`enable_thinking`** is the on/off switch for the model emitting `<think>…</think>` at all. `None` uses the model default. On MiniMax-M3 that default is adaptive: the model decides per turn whether to open a `<mm:think>` block, while `True`/`False` force it on or off. GLM-5 has no thinking-off mode, so its codec rejects `False`.
- **`preserve_thinking`** controls how earlier reasoning is rendered back into history. The Qwen3.5/3.6 chat templates strip `<think>` blocks from previous assistant turns (GLM-5's and Qwen3.8's keep them by default), which makes re-rendering history *not* prefix-stable — a problem when trained tokens get spliced into rendered context.

  - **Default (`preserve_thinking=None`):** keep the stock template, which strips history except on GLM-5 and Qwen3.8, where omitting the flag keeps it. `split_turns` then breaks the conversation into one trainable sequence per prefix-stable run, so each turn trains against exactly the history the model sees at inference. Faithful, at the cost of more (sometimes partly redundant) sequences.
  - **`preserve_thinking=False`:** strip history reasoning. Same render as the default where the stock template already strips. On GLM-5, Qwen3.6 and Qwen3.8 it is forwarded to the template's own flag (`clear_thinking` on GLM-5), which on GLM-5 and Qwen3.8 is how those templates drop history.
  - **`preserve_thinking=True`:** render history with a template that keeps thinking in every turn — natively for GLM-5, Qwen3.6 and Qwen3.8, via a forked template for Qwen3.5 — yielding a single prefix-stable sequence (cheaper, simpler). Only use it when inference also preserves thinking — otherwise you train on history the default inference path won't reproduce. It requires `enable_thinking != False`.

  Qwen3.6 and Qwen3.8 turn thinking on by default and prime the generation prompt with `<think>\n`, so a reasoning-less assistant turn does not extend that prompt; SFT on data without reasoning needs `enable_thinking=False` on those models, whatever `preserve_thinking` is.

  Kimi-K3's encoder and MiniMax-M3's template render the reasoning of every turn they are given, so for them the default and `True` render identically and an episode stays one sequence, unless a sampled turn has a shape the encoder normalizes on re-render (for example a `think` that never closes, which stays reasoning whatever it contains, or a tool-call argument literal that disagrees with its type). `False` drops reasoning from assistant turns before the last user message before rendering, as an inference client that does not send reasoning back would.

    GLM-5's template renders assistant content as `content.strip()`, so a GLM episode only
    collapses to one sequence when the sampled content has no whitespace after `</think>`;
    otherwise `split_turns` still emits one sequence per turn, and changing that would take a
    forked template.

    Qwen3.8 already keeps history reasoning when the flag is omitted, so the default
    matches stock inference. `True` and `False` are forwarded to that template.
    The Qwen3.5 fork is not used: it drops the reasoning-effort system turn.

For example, a Qwen model formats:

```
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
```

The codec hides those details. When building prompts for GRPO, few-shot examples are prepended as additional user/assistant turns and rendered through the same `encode_messages` call.

---

## Multi-turn environments

Single-turn GRPO scores one model response per prompt and stops. Multi-turn GRPO runs whole episodes: the model takes an action (an assistant message, possibly with tool calls), the environment responds, and the loop continues until the env signals termination or the turn limit is hit.

The `post_training_cookbook.envs` package provides the abstractions:

- `Task` is the per-episode spec (prompt, target, metadata).
- `Env` (abstract) owns conversation history, tool dispatch, and task-completion semantics. It exposes `async reset(task)`, `async step(message)`, `tool_specs`, and `messages`, plus an `async close()` hook (default no-op) for releasing resources — it runs even when an episode ends by external truncation, where the env never sees a terminal `step()`.
- `EnvFactory` (`Callable[[Task], Env]`) builds a fresh env per task; you pass one to `train_multi_turn`.
- `StepResult` reports the outcome of one step: `terminated` (task logic ended the episode), `truncated` (external policy ended it, like parse failure or turn limit), `parse_failed` (the truncation was a parse failure specifically), updated `messages`, `reward`, and `metrics`. The `done` property combines termination and truncation.
- `Tool` is the protocol for an executable tool the env exposes to the model.
- `step_env(codec, env, tokens, *, failed_parse_reward)` glues sampling to the env: it parses sampled tokens with the codec (passing `env.tool_specs` for correct argument typing), then calls `env.step(message)`. On parse failure it returns a truncated `StepResult` with `failed_parse_reward` and `parse_failed=True`.

Episode token streams are flattened into GRPO `Sample`s by the multi-turn collector in `post_training_cookbook.grpo.multi_turn`, using the episode data types (`CompletedEpisode`, `TruncationCause`) in `post_training_cookbook.envs.base`. The flattener alternates **context** segments (loss mask 0 for the prompt and tool/template tokens) with **action** segments (loss mask 1 for the assistant tokens the model sampled). When the chat template strips earlier content (for example, dropping reasoning blocks from history), it starts a new sample so tokens stay aligned with the recorded log-probabilities — the same prefix-stability concern handled by `split_turns` for SFT.

Multi-turn mode is selected by setting `GRPOConfig.max_turns` to a non-`None` value and calling `train_multi_turn(config, tasks, env_factory, val_tasks=None)`. With `max_turns=None` (the default), the recipes run single-turn.

---

## Custom losses

The built-in `LOSS_TYPE_CROSS_ENTROPY` and `LOSS_TYPE_GRPO` cover SFT and GRPO. When the objective has to be computed in client-side Python — DPO, or any loss the service doesn't implement — the SDK exposes a two-pass path:

1. **`forward`** runs a forward-only pass and returns the model's per-slot log-probabilities of the target tokens in `loss_fn_outputs`, without touching optimizer state. Slots the loss masks out (weight `0.0`) come back as `0.0` rather than as log-probabilities.
2. You treat those log-probabilities as autograd leaves, evaluate your loss, and call `.backward()` to get the gradient of the loss with respect to each token's log-probability.
3. **`custom_forward_backward`** takes those per-slot gradients and finishes the backward pass server-side, accumulating parameter gradients.
4. `optim_step` applies them, exactly as in the standard loop.

The cookbook wraps steps 1–3 in `post_training_cookbook.custom_loss.compute_logprobs` and `compute_custom_forward_backward`. `forward` is also useful on its own for snapshotting a reference policy (DPO precomputes the reference model's log-probabilities once, before training). `forward_backward` can return the same per-sample outputs with `return_loss_fn_outputs=True`, but it still accumulates gradients from its own loss — only `forward` leaves the optimizer alone. See the [custom loss how-to](./custom-losses.md) for a worked example.
