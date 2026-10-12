# Cookbook reference

The Post Training Cookbook (`post_training_cookbook`) provides high-level training recipes on top of the Together Post Training SDK. It handles dataset loading, tokenization, reward computation, sample construction, training loops, and logging, so you can focus on choosing the right model, dataset, and hyperparameters.

**Contents:** [Project structure](#project-structure) · [GRPO configuration](#grpo-configuration) · [SFT configuration](#sft-configuration) · [DPO configuration](#dpo-configuration) · [Built-in datasets](#built-in-datasets) · [Adding a custom dataset](#adding-a-custom-dataset) · [Recipes](#recipes) · [Tokenization with codecs](#tokenization-with-codecs) · [Multi-turn environments](#multi-turn-environments) · [Multi-LoRA / shared model resources](#multi-lora--shared-model-resources) · [Utility functions](#utility-functions) · [Run artifacts](#run-artifacts) · [Tests](#tests)

---

## Project structure

```
post_training_cookbook/
  grpo/              GRPO config and training loops
    config.py        GRPOConfig dataclass
    train.py         train_single_turn() and train_multi_turn() entry points
    loop.py          Shared training_loop + get_lr_cosine LR schedule
    single_turn.py   Single-turn scoring, sample collection, response budgets
    multi_turn.py    Multi-turn episode rollout and collection
    episode.py       Multi-turn episode collection (run env to completion, classify the ending)
    batching.py      Internal sample batching for dynamic sampling
    samples.py       GRPO primitives: build_policy_sample, compute_group_advantages
  sft/               SFT config and training loop
    config.py        SFTConfig dataclass
    train.py         train() entry point, render_conversation, TRAIN_ON_ALL/LAST
  dpo/               DPO config and training loop (custom-loss reference impl)
    config.py        DPOConfig dataclass
    train.py         train() entry point
    loss.py          make_dpo_loss() closure
    samples.py       build_preference_pair
  codecs/            Model-specific tokenization
    base.py          MessageCodec protocol, preserve_thinking handling
    glm5.py          Glm5Codec implementation
    kimi_k3.py       KimiK3Codec implementation
    minimax_m3.py    MiniMaxM3Codec implementation
    qwen3_5.py       Qwen35Codec implementation (Qwen3.5, 3.6, and 3.8)
    chat_templates/  Forked Jinja templates for preserve_thinking
    __init__.py      get_codec(model_name) factory
  envs/              Multi-turn environment abstractions
    base.py          Env ABC, Task, StepResult, Tool, ToolResult, EnvFactory, step_env, tool_to_spec,
                     and the episode record shared by every driver (CompletedEpisode, Action,
                     TruncationCause, classify_truncation, close_quietly)
    tool_use_game/   Reference environment for tool-use training
  serverless_eval/   Score a hosted (OpenAI-compatible) chat model on any Env — no training
    harness.py       evaluate() rollout + scoring, and the run_and_report() entry point
    completions.py   Chat-completions adapter (Message ledger <-> OpenAI request)
    single_turn.py   Exposes registered single-turn QA datasets as degenerate Envs
  datasets/          Dataset registry and reward functions
    registry.py      DATASETS registry, DatasetConfig, build_prompt_example
    rewards.py       gsm8k_reward, game24_reward, game24_accuracy
    math_grader.py   boxed_math_reward (deepmath / deepmath-lv5)
  samples.py         Loss-agnostic payload builders: build_sample owns the target trim,
                     plus build_cross_entropy_sample, make_prompt, is_trainable,
                     extract_loss_mask, trained_slots
  custom_loss.py     compute_logprobs, compute_custom_forward_backward (Python-side losses)
  overlong.py        compute_overlong_penalty, is_overlong (DAPO reward shaping)
  sequence_split.py  split_turns: prefix-stable trainable-sequence splitting
  run_artifacts.py   Run artifacts: run dir, config.yaml, metrics.jsonl, rollout dumps
  types.py           Shared TypedDicts (Message, ToolCall, ToolSpec)

recipes/
  grpo/              Ready-to-run GRPO recipes (single- and multi-turn)
  sft/               Ready-to-run SFT recipes
  dpo/               Ready-to-run DPO recipes
  sft_then_grpo.py   SFT then GRPO from the same adapter (weights-only resume)

notebooks/
  grpo_example.ipynb Beginner-friendly GRPO walkthrough
  sft_example.ipynb  Beginner-friendly SFT walkthrough
```

---

## GRPO configuration

All GRPO hyperparameters are set through `GRPOConfig`. Create one in Python or load from YAML.

Set the base model with `model` (or `model_resources_id` for multi-LoRA — exactly one is required); everything else has a default. Length parameters derive sensible defaults from `max_model_length` when left as `None`:

- `max_sample_tokens` defaults to `max_model_length` (fill the context).
- `max_prompt_length` defaults to `max_model_length - max_sample_tokens`.
- `overlong_buffer_len` defaults to `max_sample_tokens // 8`.

Overlong handling is controlled by `overlong_penalty_factor` alone. Set it to `0` to disable. The `overlong_penalty` property exposes the on/off state derived from the factor.

### Python

```python
from post_training_cookbook.grpo.config import GRPOConfig

config = GRPOConfig(
    model="Qwen/Qwen3.8-27B",
    dataset="gsm8k",
    num_examples=4000,
    batch_size=16,
    group_size=8,
    learning_rate=1e-5,
)
```

### YAML

```yaml
# config.yaml
model: "Qwen/Qwen3.8-27B"
dataset: "gsm8k"
num_examples: 4000
batch_size: 16
group_size: 8
learning_rate: 1e-5
```

```python
config = GRPOConfig.from_yaml("config.yaml")
```

Unknown keys in the YAML file raise `ValueError`.

### Full parameter reference

#### Model

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str \| None` | `None` | Base model to fine-tune. Required to provision a fresh resource; leave unset when attaching via `model_resources_id`. |
| `model_resources_id` | `str \| None` | `None` | Attach to an existing shared model resource instead of provisioning one (multi-LoRA). Set this **or** `model`, never both — a shared resource already defines its base model, and naming both raises `ValueError`. See [Multi-LoRA / shared model resources](#multi-lora--shared-model-resources). |

#### Training

| Parameter | Type | Default | Description |
|---|---|---|---|
| `batch_size` | `int` | `16` | Prompts per training step. |
| `num_steps` | `int \| None` | `None` | If set, overrides epoch-based step count. |
| `learning_rate` | `float` | `1e-5` | Peak learning rate. |
| `warmup_ratio` | `float` | `0.05` | Fraction of steps for LR warmup. |
| `min_lr_ratio` | `float` | `0.1` | Minimum LR as fraction of peak (cosine decay). |
| `temperature` | `float` | `0.7` | Sampling temperature for completions. |
| `max_sampling_attempts` | `int` | `1` | DAPO dynamic-sampling cap. `1` draws one batch per step. Higher values keep re-sampling and discarding zero-variance groups until `batch_size` groups with usable advantages are collected, or this many batches have been drawn. Guards against steps where nearly every group collapses to identical rewards. Must be `>= 1`. |
| `num_generator_replicas` | `int` | `1` | Generator replicas to provision for sampling. Raises throughput when rollouts, not the optimizer step, are the bottleneck. Provisioning only — attaching via `model_resources_id` keeps the resource's existing topology. |
| `weight_sync_type` | `WeightSyncType` | `"WEIGHT_SYNC_TYPE_SYNCHRONOUS"` | How new weights reach the generator after each optimizer step. See [Weight synchronization](#weight-synchronization). |
| `resume_from_checkpoint_id` | `str \| None` | `None` | Training-checkpoint ID to start from. See [Checkpoints and resume](#checkpoints-and-resume). |
| `load_optimizer` | `bool \| None` | `None` | Omit or `True` restores server optimizer state and counter; `False` loads weights only (fresh optimizer, step 0). Requires `resume_from_checkpoint_id`; otherwise raises `ValueError`. |
| `resume_run_dir` | `str \| None` | `None` | Source run directory whose `checkpoints.jsonl` holds the matching training record. Required when continuing (checkpoint id without `load_optimizer=False`); rejected otherwise. |

#### Reasoning (thinking)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable_thinking` | `bool \| None` | `None` | On/off switch for the model's `<think>` output. `None` uses the model's default. |
| `preserve_thinking` | `bool \| None` | `None` | `None` keeps the stock template. `True` keeps `<think>` blocks in history (requires `enable_thinking != False`). `False` strips them; on GLM-5 (as `clear_thinking`), Qwen3.6 and Qwen3.8 it is forwarded to the template, and GLM-5's and Qwen3.8's keep history when the flag is omitted. See [Core concepts: codecs](./concepts.md#codecs). |
| `thinking_effort` | `str \| None` | `None` | Thinking-effort level for GLM-5 and Kimi-K3 (`low`/`high`/`max`). `None` keeps the template default (max); other families reject any value. |

#### Policy objective

| Parameter | Type | Default | Description |
|---|---|---|---|
| `loss` | `Literal[...]` | `"grpo"` | Server objective over the same group-relative rollouts: `"ppo"`, `"grpo"`, `"cispo"`, `"dppo"`, `"dro"`, or `"importance_sampling"`. Translated by `policy_loss(config)`. PPO reduces by token-sum, so it is not a drop-in for the default. |
| `clip_low_threshold` | `float \| None` | `None` | Absolute ratio floor (not an offset from `1.0`); `[0, 1]`. `None` omits the field so the server default applies (`0.8` for `ppo`/`grpo`, `0.0` for `cispo`). See [clipping](./concepts.md#clipping). |
| `clip_high_threshold` | `float \| None` | `None` | Absolute ratio ceiling (`>= 1`). `None` omits the field (`1.2` for `ppo`/`grpo`, `4.0` for `cispo`). |
| `loss_agg_type` | `str` | `"GRPO_LOSS_AGGREGATION_TYPE_FIXED_HORIZON"` | GRPO-only loss aggregation. Use the full enum names: `GRPO_LOSS_AGGREGATION_TYPE_FIXED_HORIZON`, `GRPO_LOSS_AGGREGATION_TYPE_TOKEN_MEAN`, or `GRPO_LOSS_AGGREGATION_TYPE_SEQUENCE_MEAN`. |
| `ratio_type` | `str` | `"GRPO_LOSS_RATIO_TYPE_TOKEN"` | GRPO-only importance-ratio granularity. `GRPO_LOSS_RATIO_TYPE_SEQUENCE` is the GSPO path. |
| `routing_replay` | `bool` | `False` | Capture and replay generator MoE routing for training samples. |
| `dro_beta` | `float \| None` | `None` | Required when `loss="dro"`; rejected otherwise. |

Set `routing_replay=True` to capture MoE expert routing during sampling and forward
the returned keys unchanged to the trainer. Both single-turn and multi-turn GRPO
support replay; validation skips capture. Missing keys fail explicitly. Captures
require a supported MoE model. Single-turn completions clipped to the local context
budget cannot be replayed: reserve enough room for `max_sample_tokens`, or use
`overlong_masking=True` to drop those completions from training.

#### Token budgets

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_model_length` | `int` | `2048` | Max total sequence length (prompt plus response). |
| `max_sample_tokens` | `int \| None` | `None` (= `max_model_length`) | Max response tokens to sample. |
| `max_prompt_length` | `int \| None` | `None` (= `max_model_length - max_sample_tokens`) | Hard limit on prompt tokens (single-turn only). |
| `min_response_tokens` | `int` | `1` | Minimum response tokens required. |

#### Overlong handling

| Parameter | Type | Default | Description |
|---|---|---|---|
| `overlong_penalty_factor` | `float` | `1.0` | Scale factor for the overlong-length penalty. Set to `0.0` to disable the penalty. |
| `overlong_masking` | `bool` | `False` | Drop overlong responses before sample construction, so no training sample is built for them (alternative to the penalty). Their reward still counts toward the group baseline. |
| `overlong_buffer_len` | `int \| None` | `None` (= `max_sample_tokens // 8`) | Tokens before `max_sample_tokens` where the penalty starts ramping up. |

The read-only `overlong_penalty` property returns `True` when `overlong_penalty_factor > 0`.

#### LoRA and full-weight fine-tuning

| Parameter | Type | Default | Description |
|---|---|---|---|
| `use_lora` | `bool` | `True` | LoRA vs full-weight (`lora_enabled=False`). Allowed modes: [supported-models discovery](./sdk-reference.md#supported-models-and-gpu-hardware). |
| `lora_rank` | `int \| None` | `None` | LoRA rank. `None` defers to the platform default. |
| `lora_alpha` | `int \| None` | `None` | LoRA scaling factor. `None` defers to the platform default. |
| `lora_dropout` | `float \| None` | `None` | Dropout on LoRA layers. `None` uses the server default. |

Setting any `lora_*` field while `use_lora=False` raises `ValueError` — those fields would otherwise be silently ignored.

#### Optimizer

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimizer` | `"adam" \| "muon"` | `"adam"` | Which optimizer trains the weight matrices. `"muon"` requires full-weight training (`use_lora=False`) and raises `ValueError` otherwise. See [SDK reference: Muon optimizer](./sdk-reference.md#muon-optimizer). |
| `adam_beta1` | `float` | `0.9` | AdamW first moment decay. |
| `adam_beta2` | `float` | `0.95` | AdamW second moment decay. |
| `adam_eps` | `float` | `1e-8` | AdamW epsilon. |
| `weight_decay` | `float` | `0.01` | Weight decay coefficient. |
| `muon_scaling_strategy` | `MuonScalingStrategy \| None` | `None` | How Muon scales its updates — `"MUON_SCALING_STRATEGY_MATCH_ADAM"` or `"MUON_SCALING_STRATEGY_ORIGINAL"`. `None` defers to the server. Fixed at resource creation. |
| `muon_momentum` | `float \| None` | `None` | Muon momentum coefficient. `None` defers to the server. |
| `muon_newton_schulz_steps` | `int \| None` | `None` | Newton–Schulz orthogonalization iterations per Muon update. `None` defers to the server. |

The `muon_*` fields are ignored unless `optimizer="muon"`. Under Muon the AdamW fields above still apply: Muon tunes the weight matrices, and the parameters it does not tune (embeddings, norm layers, LM head) keep an AdamW group that inherits those settings and the step's learning rate. `recipes/grpo/gsm8k_muon.py` is a worked example.

#### Validation and logging

| Parameter | Type | Default | Description |
|---|---|---|---|
| `validation_interval` | `int` | `50` | Run validation every N steps. |
| `validation_iterations` | `int` | `1` | Validation passes per run (multi-turn averaging). |
| `val_temperature` | `float \| None` | `None` | Validation sampling temperature. `None` means greedy (`0.0`). |
| `wandb_project` | `str` | `"post-training-cookbook"` | Weights & Biases project name. |
| `wandb_entity` | `str \| None` | `None` | W&B entity (team or username). |
| `wandb_run_name` | `str \| None` | `None` | W&B run name (auto-generated if not set). |
| `wandb_group` | `str \| None` | `None` | W&B group, for clustering related runs. |
| `export_samples` | `bool` | `True` | Save per-step completions/episodes and rewards to JSON under the run directory. |
| `timeout` | `float` | `7200.0` | Timeout for SDK API calls in seconds. |
| `training_checkpoint_interval` | `int` | `0` | Save a training checkpoint (adapter + optimizer + step) every N steps. `0` disables. |
| `inference_checkpoint_interval` | `int` | `0` | Save an inference checkpoint every N steps. `0` disables intermediates; a final inference checkpoint is still written at the last step. |

The four `wandb_*` fields are also forwarded to the training session itself, as `display_name=wandb_run_name` and `metadata=SessionMetadata(wandb=...)`, so a session is identifiable by its W&B run from the API side. See [SDK reference: `SessionMetadata`](./sdk-reference.md#sessionmetadata).

#### Single-turn only

These fields are ignored when `max_turns` is set (multi-turn mode).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dataset` | `str` | `"gsm8k"` | Dataset key (must exist in `DATASETS` registry). |
| `num_examples` | `int` | `7000` | Number of training examples. |
| `num_val_examples` | `int` | `200` | Number of validation examples. |
| `group_size` | `int` | `8` | Completions sampled per prompt. |
| `num_epochs` | `int` | `1` | Passes over the dataset. |

#### Multi-turn only

These fields apply when `max_turns` is set. With `max_turns=None` (the default), training runs in single-turn mode.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_turns` | `int \| None` | `None` | Max assistant turns per episode. `None` selects single-turn mode. |
| `episodes_per_task` | `int` | `1` | Episodes sampled per task (the multi-turn analogue of `group_size`). |
| `failed_parse_reward` | `float` | `-0.1` | Reward used when codec parsing fails for a sampled response. |
| `val_max_turns` | `int \| None` | `None` | Turn limit during validation. `None` reuses `max_turns`. |
| `sequential_validation` | `bool` | `False` | Run validation episodes sequentially instead of in parallel. |
| `max_steps_off_policy` | `int \| None` | `None` | `None` keeps lockstep collection. Any integer enables asynchronous collection, and bounds how many optimizer steps old a group may be and still be trained on (`0` = accept only groups scheduled at the current policy). Requires `episodes_per_task >= 2`. See [Asynchronous multi-turn collection](#asynchronous-multi-turn-collection). |
| `groups_in_flight` | `int \| None` | `None` | Concurrent rollout workers under asynchronous collection. Defaults to `batch_size`. Peak retained groups is about `batch_size + 2 × groups_in_flight` (the admitted batch, the handoff queue, and the workers), rising by another `groups_in_flight` on the step after a validation pause. |

---

## SFT configuration

```python
from post_training_cookbook.sft.config import SFTConfig

config = SFTConfig(
    model="Qwen/Qwen3.8-27B",
    enable_thinking=False,  # SFT data has no reasoning traces
    dataset_name="HuggingFaceH4/no_robots",
    batch_size=8,
    learning_rate=1e-4,
    max_length=4096,
    max_steps=50,
    train_on_what="all_assistant_messages",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str \| None` | `None` | Base model. Required unless `model_resources_id` is set. |
| `model_resources_id` | `str \| None` | `None` | Attach to a shared model resource instead of provisioning one (multi-LoRA). See [Multi-LoRA / shared model resources](#multi-lora--shared-model-resources). |
| `dataset_name` | `str` | `"HuggingFaceH4/no_robots"` | HuggingFace dataset repo ID, or a local directory of `.jsonl` files. |
| `dataset_config` | `str \| None` | `None` | HF dataset config name. |
| `dataset_split` | `str` | `"train"` | Dataset split. |
| `batch_size` | `int` | `8` | Samples per training step. |
| `learning_rate` | `float` | `1e-4` | Learning rate. |
| `max_length` | `int` | `4096` | Max tokens per conversation. |
| `num_epochs` | `int` | `1` | Passes over the dataset, reshuffled each epoch. |
| `max_steps` | `int \| None` | `None` | Optional cap on total training steps; `None` trains every full batch of every epoch. |
| `train_on_what` | `str` | `"all_assistant_messages"` | `"all_assistant_messages"` or `"last_assistant_message"`. Use the `TRAIN_ON_ALL` and `TRAIN_ON_LAST` constants exported from `post_training_cookbook.sft.train`; other values are rejected. |
| `enable_thinking` | `bool \| None` | `None` | On/off switch for the model's `<think>` output. |
| `preserve_thinking` | `bool \| None` | `None` | `None` keeps the stock template; `True` keeps history reasoning; `False` strips it (see [codecs](./concepts.md#codecs)). |
| `thinking_effort` | `str \| None` | `None` | Thinking-effort level for GLM-5 and Kimi-K3 (`low`/`high`/`max`). `None` keeps the template default (max); other families reject any value. |
| `use_lora` | `bool` | `True` | LoRA vs full-weight. Allowed modes: [supported-models discovery](./sdk-reference.md#supported-models-and-gpu-hardware). See `recipes/sft/no_robots_full.py`. |
| `lora_rank` | `int \| None` | `None` | LoRA rank when `use_lora=True`. `None` defers to the platform default. |
| `lora_alpha` | `int \| None` | `None` | LoRA scaling factor when `use_lora=True`. `None` defers to the platform default. |
| `lora_dropout` | `float \| None` | `None` | LoRA dropout; `None` uses the server default. |
| `training_checkpoint_interval` | `int` | `0` | Save a training checkpoint every N steps. `0` disables. |
| `inference_checkpoint_interval` | `int` | `0` | Save an inference checkpoint every N steps. `0` disables intermediates; a final inference checkpoint is still written at the last step. |
| `resume_from_checkpoint_id` | `str \| None` | `None` | Training-checkpoint ID to start from. See [Checkpoints and resume](#checkpoints-and-resume). |
| `load_optimizer` | `bool \| None` | `None` | Restores server optimizer state and counter unless `False` (weights only); the SFT/DPO client loop starts at zero. `False` requires a checkpoint id. |
| `wandb_project` | `str \| None` | `None` | W&B project (disabled if `None`). |
| `wandb_entity` | `str \| None` | `None` | W&B entity (team or username). |
| `wandb_run_name` | `str \| None` | `None` | W&B run name. |
| `wandb_group` | `str \| None` | `None` | W&B group, for clustering related runs. |
| `timeout` | `float` | `7200.0` | SDK timeout in seconds. |
| `adam_beta1` | `float` | `0.9` | AdamW first moment decay. |
| `adam_beta2` | `float` | `0.95` | AdamW second moment decay. |
| `adam_eps` | `float` | `1e-8` | AdamW epsilon. |
| `weight_decay` | `float` | `0.1` | Weight decay coefficient. |

Each row must have a `messages` column (a list of `{role, content}` dicts). For any other schema, pass `train(config, to_messages=fn)`, where `fn` maps a row to its messages; put any task instruction (an answer format, for example) in that mapping. `recipes/sft_then_grpo.py` maps GSM8K this way.

SFT creates a trainer-only session (`num_generator_replicas=0`) since it never samples. Per-step metrics are written to `<run_dir>/metrics.jsonl` (see [Run artifacts](#run-artifacts)). `train()` returns the latest training-checkpoint id from `checkpoints.jsonl` (or `None`).

---

## DPO configuration

Direct preference optimization (DPO) trains on `(chosen, rejected)` pairs. The cookbook's DPO loop is also the reference implementation of the [custom-loss pattern](./custom-losses.md): it snapshots reference-model log-probabilities with `session.trainer.forward` (forward-only, so nothing accumulates into the gradients it submits later), computes the DPO loss in Python, and submits the gradients with `session.trainer.custom_forward_backward`.

```python
from post_training_cookbook.dpo.config import DPOConfig
from post_training_cookbook.dpo.train import train

config = DPOConfig(
    model="Qwen/Qwen3.8-27B",
    enable_thinking=False,  # preference data has no reasoning traces
    dataset_name="nvidia/HelpSteer3",
    batch_size=8,
    beta=0.1,
    max_steps=1000,
)

train(config)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"Qwen/Qwen3.8-27B"` | Base model. |
| `dataset_name` | `str` | `"nvidia/HelpSteer3"` | HuggingFace dataset repo ID. |
| `dataset_config` | `str \| None` | `None` | HF dataset config name. |
| `dataset_split` | `str` | `"train"` | Training split. |
| `dataset_subsample` | `int \| None` | `2000` | Cap on training pairs (`None` = use all). |
| `batch_size` | `int` | `8` | Preference pairs per step. |
| `learning_rate` | `float` | `5e-6` | Peak learning rate (cosine schedule). |
| `max_length` | `int` | `2048` | Max tokens per chosen/rejected sequence. |
| `enable_thinking` | `bool \| None` | `None` | On/off switch for the model's `<think>` output. Set `False` for Qwen models on preference data without reasoning, such as HelpSteer3. |
| `max_steps` | `int` | `1000` | Total training steps. |
| `beta` | `float` | `0.1` | DPO temperature on the implicit reward. |
| `label_smoothing` | `float` | `0.0` | Conservative-DPO label smoothing. |
| `warmup_ratio` | `float` | `0.03` | LR warmup fraction. |
| `min_lr_ratio` | `float` | `0.1` | Minimum LR as fraction of peak. |
| `precompute_ref_batch_size` | `int` | `128` | Batch size for the reference-logprob forward passes. |
| `val_split` | `str` | `"validation"` | Validation split. |
| `val_subsample` | `int \| None` | `100` | Cap on validation pairs (`0`/`None` disables). |
| `val_interval` | `int` | `100` | Validate every N steps (`<= 0` disables). |
| `wandb_project` / `wandb_entity` / `wandb_run_name` / `wandb_group` | — | `None` | W&B settings (disabled when project is `None`). |
| `timeout` | `float` | `7200.0` | SDK timeout in seconds. |
| `adam_beta1` / `adam_beta2` / `adam_eps` / `weight_decay` | — | `0.9` / `0.95` / `1e-8` / `0.1` | AdamW parameters. |
| `use_lora` | `bool` | `True` | LoRA vs full-weight. Allowed modes: [supported-models discovery](./sdk-reference.md#supported-models-and-gpu-hardware). |
| `lora_rank` | `int \| None` | `None` | LoRA rank when `use_lora=True`. `None` defers to the platform default. |
| `lora_alpha` | `int \| None` | `None` | LoRA scaling factor when `use_lora=True`. `None` defers to the platform default. |
| `lora_dropout` | `float \| None` | `None` | LoRA dropout; `None` uses the server default. |
| `training_checkpoint_interval` | `int` | `0` | Save a training checkpoint every N steps. `0` disables. |
| `inference_checkpoint_interval` | `int` | `0` | Save an inference checkpoint every N steps. `0` disables intermediates; a final inference checkpoint is still written at the last step. |
| `resume_from_checkpoint_id` | `str \| None` | `None` | Training-checkpoint ID to start from. See [Checkpoints and resume](#checkpoints-and-resume). |
| `load_optimizer` | `bool \| None` | `None` | Restores server optimizer state and counter unless `False` (weights only); the SFT/DPO client loop starts at zero. `False` requires a checkpoint id. |

DPO defaults to LoRA (same as SFT/GRPO) and uses a trainer-only session. `train(config, to_pair=fn)` maps a row to `(chosen_messages, rejected_messages)`, or `None` to skip it. The default, `helpsteer3_pair` in `post_training_cookbook.dpo.train`, turns HelpSteer3's `overall_preference` score into a pair and skips ties; pass your own for other preference datasets. Validation reports an implicit-reward **preference accuracy** rather than a token metric — see the [metrics reference](./metrics.md#dpo-metrics).

### Checkpoints and resume

`training_checkpoint_interval` and `inference_checkpoint_interval` default to `0` (off). A training checkpoint is written only when its interval hits — set it equal to `max_steps` / `num_steps` if you need one at the end. Ids go in `runs/<run>/checkpoints.jsonl`. A final inference checkpoint is always written at the last step.

There is no dedicated "resume" recipe. Pause and continue are later processes: copy a training-checkpoint id into `resume_from_checkpoint_id` on the same recipe config and re-run.

**Continue a GRPO run.** Omit `load_optimizer` or pass `True` to restore the server's adapter and optimizer state. Set `resume_run_dir` to the original run directory. The matching training checkpoint record restores both the **client iteration** and **data position**. Empty batches advance the client iteration without an optimizer update, so the server counter cannot substitute for it. A missing record or incompatible data state fails before provisioning.

```python
config = GRPOConfig(
    model="<same-base-model>",
    resume_from_checkpoint_id="<training-checkpoint-id>",
    resume_run_dir="runs/<original-run>",  # reads the matching checkpoints.jsonl record
    num_steps=100,  # total target, including completed iterations
    # Keep the original learning-rate schedule and checkpoint intervals.
)
```

Both single-turn and multi-turn GRPO (including asynchronous collection) continue the learning-rate schedule, validation, logs, and checkpoint cadence from the saved client iteration. A checkpoint recorded at step 40 with `num_steps=100` leaves 60 iterations; `num_steps` is not an additional-step count. A checkpoint at or beyond the target performs no training and provisions no session. Async policy versions restart at zero for the new session and remain separate from this client counter.

Keep the original schedule settings and total target to retain the same LR trajectory. Each training record includes a versioned `data_state`, recorded with its checkpoint ID and client step in `checkpoints.jsonl`:

- Synchronous collectors restore the next drawn row/task index, including extra dynamic-sampling attempts and batches that yielded no training samples. For example, after 40 single-draw steps of batch size 8, the next batch is rows 320–327 (wrapping by dataset size), not rows 0–7.
- Async collectors restore outstanding task IDs before issuing new work. Queued and in-flight rollouts are regenerated from the resumed model; already-consumed tasks are not replayed, even if workers completed out of order. Async completion order, sampled tokens, and environment/RNG state are not reproduced exactly.
- Data content/order, dataset size, batch size and collector mode must match; a mismatch fails before provisioning. Runs that save training checkpoints need JSON-serializable training inputs (including multi-turn targets/metadata) so their ordered identity can be checked.

Preserve or copy the source run directory. Older checkpoints lacking `data_state` cannot reconstruct variable batch consumption and are rejected for continuation; use `load_optimizer=False` to explicitly start a fresh data stream from their weights. Keep custom environment behavior and sampling settings consistent across runs. SFT and DPO restore server state only; their client schedules restart at zero and do not accept `resume_run_dir`.

**New objective from the same adapter (typically SFT→GRPO).** Pass `load_optimizer=False`. Keep the same LoRA rank/alpha. SFT is trainer-only and GRPO needs a generator, so the phases use separate sessions, usually on a **new** resource of the same base model. Worked example: `recipes/sft_then_grpo.py`. Cookbook SFT returns the latest training-checkpoint id from `sft.train()`.

**Serving.** Inference checkpoints are for deployment and download. They do not seed new training runs, so use a training checkpoint with `load_optimizer=False` for that.

---

## Built-in datasets

The cookbook includes four pre-configured datasets for GRPO training. Each defines how to format prompts, extract targets, and score model responses.

### `gsm8k`

Grade-school math word problems from [OpenAI GSM8K](https://huggingface.co/datasets/openai/gsm8k).

- **HF dataset:** `openai/gsm8k`, config `main`, split `train`.
- **Prompt format:** question plus instruction to show work and end the final line with `#### <answer>`.
- **Reward:** `gsm8k_reward`. `1.0` if the extracted numeric answer matches the target, `0.0` otherwise.
- **Few-shot:** 1 example demonstrating the expected format.

```python
config = GRPOConfig(model="Qwen/Qwen3.8-27B", dataset="gsm8k", num_examples=4000)
```

### `deepmath`

Competition-level math from [DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K).

- **HF dataset:** `zwhe99/DeepMath-103K`, config `default`, split `train`.
- **Prompt format:** question plus instruction to provide the final answer inside `\boxed{}`.
- **Reward:** `boxed_math_reward` (`datasets/math_grader.py`). `1.0` if the boxed answer matches, `0.0` otherwise.
- **Few-shot:** 1 example.

```python
config = GRPOConfig(model="Qwen/Qwen3.8-27B", dataset="deepmath", num_examples=7000)
```

### `deepmath-lv5`

Same as `deepmath` but filtered to difficulty level 5 and below.

```python
config = GRPOConfig(model="Qwen/Qwen3.8-27B", dataset="deepmath-lv5", num_examples=5000)
```

### `game24`

Arithmetic puzzle: use four numbers with `+`, `-`, `*`, `/` to make 24. From [nlile/24-game](https://huggingface.co/datasets/nlile/24-game).

- **HF dataset:** `nlile/24-game`, config `default`, split `train`. Filtered to solvable rows only.
- **Prompt format:** numbers plus instruction to produce a single `\boxed{<expression>}` line.
- **Reward:** `game24_reward`. Tiered scoring covering format, closeness to 24, and exact match.
- **Metric:** `game24_accuracy`. Strict exact-match accuracy used for validation.

```python
config = GRPOConfig(model="Qwen/Qwen3.8-27B", dataset="game24", num_examples=3000)
```

---

## Adding a custom dataset

To add your own dataset for GRPO training:

### 1. Write a reward function

Add your scoring logic to `post_training_cookbook/datasets/rewards.py`:

```python
def my_reward(prediction: str, target: str) -> float:
    """Return a float in [0, 1]. Higher = better."""
    answer = extract_answer(prediction)
    if answer is None:
        return 0.0
    return 1.0 if answer.strip() == target.strip() else 0.0
```

### 2. Register the dataset

Add a `DatasetConfig` entry in `post_training_cookbook/datasets/registry.py`:

```python
from post_training_cookbook.datasets.rewards import my_reward

DATASETS["my_dataset"] = DatasetConfig(
    hf_name="my-org/my-dataset",            # HuggingFace repo ID
    hf_config=None,                         # HF config name (or None)
    hf_split="train",
    format_question=lambda row: row["question"] + " Answer concisely.",
    extract_target=lambda row: row["answer"],
    compute_reward=my_reward,
    compute_metric=None,                    # optional separate accuracy function
    filter_fn=None,                         # optional row filter
    few_shot_messages=[                     # optional few-shot examples
        {"role": "user", "content": "What is 1+1? Answer concisely."},
        {"role": "assistant", "content": "2"},
    ],
)
```

**Fields:**

| Field | Type | Description |
|---|---|---|
| `hf_name` | `str` | HuggingFace dataset repo ID. |
| `hf_config` | `str \| None` | HF dataset config (pass `None` if the dataset has no configs). |
| `hf_split` | `str` | Dataset split to use. |
| `format_question` | `Callable[[dict], str]` | Takes a dataset row, returns the prompt text. |
| `extract_target` | `Callable[[dict], Any]` | Takes a dataset row, returns the ground truth. |
| `compute_reward` | `Callable[[str, Any], float]` | Takes (model_response, target), returns reward in `[0, 1]`. |
| `compute_metric` | `Callable[[str, Any], float] \| None` | Optional separate accuracy metric (for validation). Falls back to `compute_reward` when `None`. |
| `filter_fn` | `Callable[[dict], bool] \| None` | Optional row filter (return `True` to keep). |
| `few_shot_messages` | `list[dict[str, str]]` | Few-shot examples prepended to every prompt. |

### 3. Create a recipe

Create a new script in `recipes/grpo/`:

```python
"""GRPO on my custom dataset."""

from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.grpo.train import train_single_turn

config = GRPOConfig(
    model="Qwen/Qwen3.8-27B",
    dataset="my_dataset",
    num_examples=5000,
    batch_size=16,
    group_size=8,
    learning_rate=1e-5,
)

if __name__ == "__main__":
    train_single_turn(config)
```

Run it:

```bash
uv run python recipes/grpo/my_dataset.py
```

---

## Recipes

Recipes are self-contained Python scripts. Each one creates a config object and calls the entry point. Edit the script directly to change hyperparameters.

### Available recipes

| Recipe | Path | Entry point | Description |
|---|---|---|---|
| GSM8K GRPO | `recipes/grpo/gsm8k.py` | `train_single_turn` | Math reasoning on grade-school problems. |
| GSM8K GRPO (Muon) | `recipes/grpo/gsm8k_muon.py` | `train_single_turn` | The same task trained with the [Muon optimizer](./sdk-reference.md#muon-optimizer) — `optimizer="muon"`, full-weight. |
| DeepMath GRPO | `recipes/grpo/deepmath.py` | `train_single_turn` | Math reasoning on competition problems. |
| DeepMath Lv5 GRPO | `recipes/grpo/deepmath-lv5.py` | `train_single_turn` | DeepMath filtered to difficulty ≤ 5 (the easier subset). |
| Game of 24 GRPO | `recipes/grpo/game24.py` | `train_single_turn` | Arithmetic puzzle solving. |
| Tool-use game GRPO | `recipes/grpo/tool_use_game.py` | `train_multi_turn` | Multi-turn tool use against a toy environment. |
| DeepMath GRPO (shared resource) | `recipes/grpo/deepmath_shared.py` | `train_single_turn` | Multi-LoRA tenant: attach to a resource someone else provisioned. |
| DeepMath GRPO sweep | `recipes/grpo/deepmath_sweep.py` | `train_single_turn` | Multi-LoRA owner: LR sweep, all arms on one shared resource. |
| No Robots SFT | `recipes/sft/no_robots.py` | `post_training_cookbook.sft.train.train` | Instruction-following via supervised learning. |
| No Robots SFT (full-weight) | `recipes/sft/no_robots_full.py` | `post_training_cookbook.sft.train.train` | Same as above with `use_lora=False`. |
| SFT then GRPO | `recipes/sft_then_grpo.py` | `sft.train` then `train_single_turn` | Weights-only resume: SFT, then GRPO from those weights (`load_optimizer=False`). SFT trains GSM8K's `####` answer format, which GRPO then scores. |
| HelpSteer3 DPO | `recipes/dpo/helpsteer3.py` | `post_training_cookbook.dpo.train.train` | Preference tuning; custom-loss reference implementation. |
| GSM8K serverless eval | `recipes/serverless_eval/gsm8k.py` | `run_dataset_eval` | Score a hosted model on a dataset's held-out split — no training. |
| DeepMath serverless eval | `recipes/serverless_eval/deepmath.py` | `run_dataset_eval` | As above for competition math. |
| Game of 24 serverless eval | `recipes/serverless_eval/game24.py` | `run_dataset_eval` | As above; reports shaped reward and 0/1 accuracy. |
| Tool-use game serverless eval | `recipes/serverless_eval/tool_use_game.py` | `run_and_report` | Score a hosted model on any `Env`, multi-turn. |

The `train_single_turn` and `train_multi_turn` entry points both live in `post_training_cookbook.grpo.train`;
`run_dataset_eval` and `run_and_report` live in `post_training_cookbook.serverless_eval` and need no session.

### Running a recipe

```bash
uv run python recipes/grpo/gsm8k.py
```

### What happens when a GRPO recipe runs

1. **Codec resolution.** `get_codec(config.model, hf_token=...)` returns a `MessageCodec` matched on the model name prefix.
2. **Dataset loading.** Downloads from HuggingFace, applies filters, splits into train and val.
3. **Prompt filtering.** `filter_by_prompt_length` removes examples that exceed `max_prompt_length` or leave less than `min_response_tokens` of room for a response.
4. **Session creation.** `SessionClient.create()` starts a training session on Together's infrastructure and waits for the `TRAINING_SESSION_STATUS_RUNNING` state.
5. **Per-step collection.** `make_single_turn_collector` builds a `collect_samples(step, session)` callback that:
   - Builds tokenized prompts with few-shot examples.
   - Samples `group_size` completions per prompt.
   - Scores completions via the dataset's `compute_reward`, applies the overlong penalty.
   - Computes mean-centered advantages within each group, skipping zero-variance groups.
   - Constructs GRPO `Sample`s with loss masks, target tokens, advantages, and generator log-probabilities.
6. **Training loop.** `training_loop` iterates `total_steps`. Each step calls the collector, then `session.trainer.forward_backward` with `policy_loss(config)` (`LOSS_TYPE_GRPO` by default), applies the update with `session.trainer.optim_step` and a cosine-annealed learning rate, and calls `session.trainer.weights_sync(weight_sync_type="WEIGHT_SYNC_TYPE_SYNCHRONOUS")` so the next step samples on-policy.
7. **Validation.** Every `validation_interval` steps, samples greedy completions from the validation set and reports accuracy.
8. **Checkpointing.** Training and intermediate inference checkpoints fire on their intervals (`0` = off). A final inference checkpoint is always written.
9. **Logging.** Logs metrics to Weights & Biases and writes per-step completion JSON when `export_samples=True`. The full set of metrics, including the `fb/`-prefixed ones returned by the service, is documented in the [metrics reference](./metrics.md).

---

## Tokenization with codecs

Tokenization lives behind the `MessageCodec` protocol in `post_training_cookbook.codecs`. Each model family has a codec that knows how to apply its chat template and how to parse sampled tokens back into a structured `Message`.

```python
from post_training_cookbook.codecs import get_codec

codec = get_codec(
    "Qwen/Qwen3.8-27B",
    hf_token="...",
    enable_thinking=None,        # None = model default; True/False to force
    preserve_thinking=None,      # None = stock template; True keeps <think> in history
    thinking_effort=None,        # None = template default; GLM-5 / Kimi-K3 only
)

# Encode raw text without chat-template wrapping
tokens = codec.encode("Hello, world!")

# Encode a chat conversation using the model's chat template
messages = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
]
tokens = codec.encode_messages(messages, add_generation_prompt=True)

# Decode raw tokens back to text (skips special tokens)
text = codec.decode(tokens)

# Parse a sampled response into a Message + parse-success flag
message, parse_ok = codec.parse_response(response_tokens)
```

`get_codec` resolves the codec from the model name by the longest matching prefix (after the org prefix):

- `qwen3.5` / `qwen3.6` / `qwen3.8` → `Qwen35Codec` (for example, `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.8-27B`). Qwen3.6's and Qwen3.8's own templates take `preserve_thinking`, so `True` and `False` are forwarded to them and the Qwen3.5 fork is not used; Qwen3.8's keeps history reasoning when the flag is omitted.
- `glm-5.3` → `Glm5Codec` (`zai-org/GLM-5.3`).
- `kimi-k3` → `KimiK3Codec`.
- `minimax-m3` → `MiniMaxM3Codec` (for example, `MiniMaxAI/MiniMax-M3`). Its template omits a `null` tool-call argument, so a parsed call has no such key.

The registered prefixes in `post_training_cookbook/codecs/__init__.py` are the source of truth. An unrecognized model raises `ValueError`. To add a new model family, implement the `MessageCodec` protocol and register it in `post_training_cookbook/codecs/__init__.py`.

The `MessageCodec` protocol:

| Method | Signature | Purpose |
|---|---|---|
| `encode_messages` | `(messages, *, tools=None, add_generation_prompt=False) -> list[int]` | Apply the chat template and tokenize. |
| `parse_response` | `(response_tokens, *, tools=None) -> tuple[Message, bool]` | Decode sampled tokens into a `Message`, extracting tool calls and reasoning content. Pass the same `tools` given to `encode_messages` so each argument is parsed to its declared type (matters for formats that serialize values untyped, like the Qwen3.5 XML body; formats that carry types on the wire ignore it). The bool is `False` if a tool call was cut off (unclosed tags) or its body is missing a function name, while Kimi-K3 and MiniMax-M3 drop a complete but malformed call, as their serving parsers do. GLM-5 parses a turn cut off before `</think>` as reasoning only (`True`). |
| `turn_terminator` | `(next_role) -> list[int]` | Tokens a trained assistant turn needs in order to stop, given what follows it. Empty for Qwen's universal `<|im_end|>`, Kimi-K3's `<|end_of_msg|>`, and MiniMax-M3's `[e~[`, which the template writes inside the turn; the next role's marker for GLM-5 (`<|user|>`, `<|observation|>`, or `<|endoftext|>` at the end of the conversation). Raises `ValueError` when the family declares no stop token for that continuation. |
| `encode` | `(text) -> list[int]` | Tokenize raw text without chat-template wrapping. |
| `decode` | `(tokens) -> str` | Decode tokens, skipping special tokens other than the markers the family's parser needs (Kimi-K3 decodes verbatim). |

### Conversation rendering with loss masks

`render_conversation` (in `post_training_cookbook.sft.train`) tokenizes a multi-turn conversation into one or more **trainable sequences**, each carrying its own loss mask:

```python
from post_training_cookbook.codecs import get_codec
from post_training_cookbook.sft.train import render_conversation, TRAIN_ON_ALL, TRAIN_ON_LAST

codec = get_codec("Qwen/Qwen3.8-27B", enable_thinking=False)
messages = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "And 3+3?"},
    {"role": "assistant", "content": "6"},
]

sequences = render_conversation(
    messages,
    codec=codec,
    train_on_what=TRAIN_ON_ALL,     # train on all assistant messages
    max_length=4096,
)
# Each item is a TrainableSequence(tokens, loss_mask, logprobs):
#   loss_mask is 0 for user/header tokens, 1 for assistant content tokens.
```

| `train_on_what` value | Behavior |
|---|---|
| `"all_assistant_messages"` (`TRAIN_ON_ALL`) | Loss mask = 1 for every assistant turn. |
| `"last_assistant_message"` (`TRAIN_ON_LAST`) | Loss mask = 1 only for the final assistant turn. |

`render_conversation` returns a **list** because a conversation may not encode prefix-stably in a single pass — chat templates that strip `<think>` blocks from history mean a later turn's render no longer extends an earlier one's. `split_turns` (in `post_training_cookbook.sequence_split`) breaks the conversation into one sequence per prefix-stable run, so each turn trains against the history the model actually sees at inference. With `preserve_thinking=True` the history-preserving template (native for GLM-5, a fork for Qwen) keeps thinking in history, usually collapsing to a single sequence. `render_conversation` raises `ValueError` if the codec rewrites the assistant prompt within a single turn (a contract violation, distinct from history stripping), or if the codec has no stop token for what follows a trained turn (GLM-5 rejects back-to-back assistant turns — merge, split, or drop them).

---

## Multi-turn environments

The `post_training_cookbook.envs` package provides an async multi-turn environment abstraction for tool-use and agentic GRPO. The training loop owns tokenization and sampling; the `Env` owns conversation history, tool dispatch, and task-completion semantics.

### Public API

```python
from post_training_cookbook.envs import (
    Env,
    EnvFactory,
    Task,
    StepResult,
    Tool,
    ToolResult,
    tool_to_spec,
    step_env,
    ToolUseGameEnv,
    TOOL_USE_GAME_TASKS,
    TOOL_USE_GAME_SYSTEM_PROMPT,
)
```

### Core types

| Type | Purpose |
|---|---|
| `Task` (frozen dataclass) | One episode spec: `prompt: str`, `target: Any`, `metadata: dict`. |
| `StepResult` (dataclass) | Outcome of one env step: `terminated: bool`, `truncated: bool`, `messages: list[Message]`, `reward: float`, `metrics: dict`, `parse_failed: bool`. `terminated` = task logic ended the episode; `truncated` = external policy ended it (parse failure, turn limit). `parse_failed` flags parse-induced truncations. The `done` property returns `terminated or truncated`. |
| `Tool` (Protocol) | Executable tool: `name`, `description`, `parameters_schema`, `async execute(arguments: str) -> ToolResult`. |
| `ToolResult` (dataclass) | `content: str`, `should_stop: bool = False`. |
| `Env` (ABC) | Async multi-turn env. Subclass to implement `async reset(task) -> list[Message]`, `async step(message) -> StepResult`, `tool_specs`, `messages`. Override `async close()` to release resources — it runs even when the episode ends by external truncation (context overflow, max-turns), where the env never sees a terminal `step()`. |
| `EnvFactory` | `Callable[[Task], Env]` — builds a fresh env per task. Passed to `train_multi_turn`. |

### Helpers

- `tool_to_spec(tool)` converts an executable `Tool` into a `ToolSpec` for codec rendering.
- `step_env(codec, env, tokens, *, failed_parse_reward=-0.1)` parses sampled tokens with the codec (passing `env.tool_specs` so arguments are typed correctly) and steps the env. On parse failure it returns a truncated `StepResult` with the configured penalty and `parse_failed=True`. This is the helper the multi-turn collector uses to drive an env from sampled tokens.

### Reference environment

`ToolUseGameEnv` is a reference tool-use environment. `TOOL_USE_GAME_TASKS` and `TOOL_USE_GAME_SYSTEM_PROMPT` are the bundled task list and system prompt. Use it as a template for your own environments.

### Running multi-turn training

Multi-turn mode is selected by setting `GRPOConfig.max_turns` to a non-`None` value, then calling `train_multi_turn` with your tasks and an env factory:

```python
from post_training_cookbook.envs.tool_use_game import TOOL_USE_GAME_TASKS, ToolUseGameEnv
from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.grpo.train import train_multi_turn

tasks = list(TOOL_USE_GAME_TASKS)
config = GRPOConfig(model="Qwen/Qwen3.8-27B", max_turns=6, batch_size=4, episodes_per_task=8)

train_multi_turn(
    config,
    tasks=tasks,
    env_factory=lambda task: ToolUseGameEnv(),
    val_tasks=tasks,
)
```

`train_multi_turn(config, tasks, env_factory, val_tasks=None)` raises if `max_turns` is `None`. See `post_training_cookbook/envs/base.py` for the episode data types (`CompletedEpisode`, `TruncationCause`) and `post_training_cookbook/grpo/multi_turn.py` for the rollout/flattening logic that aligns sampled tokens and log-probabilities across turns.

### Asynchronous multi-turn collection

By default collection is lockstep: every rollout in a step finishes before the optimizer runs, so the generators sit idle through forward/backward and the whole step waits on its slowest episode. Setting `max_steps_off_policy` switches to a bounded worker pool that keeps collecting *through* the optimizer step:

```python
config = GRPOConfig(
    model="Qwen/Qwen3.8-27B",
    max_turns=6,
    batch_size=4,
    episodes_per_task=8,
    max_steps_off_policy=1,   # accept groups up to one optimizer step old
    groups_in_flight=8,       # concurrent rollout workers (defaults to batch_size)
    num_generator_replicas=2, # more sampling capacity to actually fill them
)
```

`groups_in_flight` workers each collect one task's group at a time and hand it to the trainer through a bounded queue. Every group is tagged with the policy version current when its rollout *started*; each step then admits groups whose age is within `max_steps_off_policy` and discards the rest. `max_steps_off_policy=0` still overlaps collection with training but trains only on groups scheduled at the current policy.

Two things bound the work a step will do, and they are deliberately asymmetric:

- **Zero-variance groups** are capped at `batch_size * max_sampling_attempts` discards per step, exactly as in lockstep dynamic sampling. A reward function that never separates episodes therefore yields a short (possibly empty) step that logs its metrics and moves on, rather than refilling forever while the generators keep burning capacity.
- **Stale groups** do not consume that budget, because they cannot cause a spin: the policy version is frozen for the whole of a step's admission, so every group a worker schedules *while the step is filling* has age 0. Only groups already queued or mid-rollout when the step began can be stale, capping stale drops at roughly `2 × groups_in_flight` per step.

That cap is per step, not per run. At `max_steps_off_policy=0` a worker re-reads the policy version before the step it just fed has trained, so the cohort it collects next is already one version behind by the time it arrives and is discarded — roughly one cohort per step, for the life of the run. `0` is therefore the strictest setting rather than the natural starting one; prefer `1` unless the extra on-policy purity is worth that throughput.

Accounting keeps the two drop reasons distinct. Zero-variance groups count toward `zero_variance_skip_ratio` exactly as in lockstep; stale groups do not inflate it. Both still contribute to `reward_mean` and reach the rollout export, so reward statistics stay comparable with a lockstep run. Watch `async/train_wait_s` and `async/dropped_stale_groups` to tell whether the overlap is paying off — see [Asynchronous collection metrics](metrics.md#asynchronous-collection).

Validation needs a quiescent generator, so it cancels the workers and awaits their environments. Groups the workers had already finished are carried over to the next step; rollouts that were mid-flight are discarded, which is the standing cost of pausing every `validation_interval` steps. `recipes/grpo/tool_use_game_async.py` is a runnable example.

### Weight synchronization

`weight_sync_type` chooses how an optimizer step's weights reach the generator:

| Mode | Behavior |
|---|---|
| `WEIGHT_SYNC_TYPE_SYNCHRONOUS` (default) | Publishes inline. Rollouts drain, the broadcast lands on an idle engine, and the next sample always uses the policy just trained. |
| `WEIGHT_SYNC_TYPE_BACKGROUND_PUBLISH` | Hands the publish to a server-side worker and returns immediately. Rollouts still drain before the broadcast lands. |
| `WEIGHT_SYNC_TYPE_PIPELINE` | Also deferred, but brackets the broadcast with a KV-cache-preserving scheduler pause instead of draining rollouts. A generation already in flight survives the publish and finishes under the new weights. Requires `use_lora=False`. |

The deferred modes require asynchronous collection and `GRPOConfig` rejects the combination without it. Under lockstep collection the next step's rollouts would start before the publish had necessarily landed, with no staleness budget bounding the gap and no `async/*` metric revealing it; `max_steps_off_policy` is what makes the resulting drift both bounded and measured.

`PIPELINE` is the one mode where a *single* sampled response can span two policies. Those spans are reported per action as `Action.policy_segments`. `async/mid_rollout_drift_ratio` measures both kinds of drift: a publish landing mid-generation (PIPELINE only) and one landing between an episode's turns, which leaves a later action wholly on a newer policy. The second kind happens under *every* mode once collection is async — including the synchronous default, which is what the shipped async recipe uses — because a worker between turns is not generating and the publish simply lands. This is the pairing worth reaching for with long rollouts: the generators never stop decoding for a weight swap, and the tokens produced across the swap are accounted for rather than discarded. Note that the server treats policy segments as metadata only and does not gate sampling on staleness — the client's `max_steps_off_policy` budget is the only enforcement.

---

## Multi-LoRA / shared model resources

Together Post Training separates the **model resource** — the GPU pods plus base-model weights — from the **training sessions** that run on it. A session is your trainable LoRA adapter. When a resource is created with `lora_enabled=True`, every attached session is an independent adapter against the same frozen base weights, so N adapters cost roughly one base model's worth of GPUs instead of N. The resource lifecycle (provision / attach / stop) lives in the SDK's [`ModelResourcesClient`](./sdk-reference.md#modelresourcesclient); the cookbook turns a recipe config into a session attached to it.

Before provisioning, [pick a supported model and GPU](./sdk-reference.md#supported-models-and-gpu-hardware) and pass the selected `gpu_type` in the resource's `ComputeConfig`.

On-demand billing starts when a newly provisioned model resource becomes READY; initial provisioning before READY is not billable. Once READY, the resource remains billable even while idle or after an individual session detaches. See [when billing starts and stops](./sdk-reference.md#when-billing-starts-and-stops) when estimating the shared resource's cost.

### The config field

Both `GRPOConfig` and `SFTConfig` carry `model_resources_id: str | None = None`. A config names **exactly one** model source:

- `model` set, `model_resources_id` unset (default) — provision a fresh session for `config.model`.
- `model_resources_id` set, `model` unset — attach a session to that existing resource instead.

`training_session` raises `ValueError` if both are set (a shared resource already defines its base model) or if neither is.

Nothing else about the training call changes. `train_single_turn` / `train_multi_turn` / `sft.train.train` route through `post_training_cookbook.multi_lora.training_session(...)`, which — when `model_resources_id` is set — checks the resource is `MODEL_RESOURCES_STATUS_READY`, is `lora_enabled` (when `use_lora=True`), and has a generator (for GRPO sampling) before attaching, so a mismatch fails fast instead of mid-run. An attached session leaves the resource running on exit (it only detaches); it never stops a resource it didn't create.

### Pattern 1: tenant (attach to a resource you don't own)

```python
import os
from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.grpo.train import train_single_turn

config = GRPOConfig(
    model_resources_id=os.environ["MODEL_RESOURCES_ID"],  # the resource defines the base model — don't set `model`
    dataset="gsm8k",
)
train_single_turn(config)                                 # resource left running for others
```

Discover available resources without leaving Python:

```python
from together import Together

for r in Together().post_training.model_resources.list(status="MODEL_RESOURCES_STATUS_READY").data:
    print(r.id, r.base_model)
```

### Pattern 2: owner / sweep (provision once, train N arms, tear down)

```python
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from together.post_training import ModelResourcesClient
from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.grpo.train import train_single_turn

base = GRPOConfig(model="Qwen/Qwen3.8-27B", dataset="gsm8k", num_steps=100,
                  wandb_project="grpo-gsm8k", wandb_group="gsm8k-sweep")
SWEEP = [{"learning_rate": lr, "lora_rank": rank} for lr in (1e-5, 5e-6) for rank in (8, 16)]

with ModelResourcesClient.create(base_model=base.model, lora_enabled=True) as resource:   # `base.model` provisions the GPUs
    configs = [
        replace(base, model=None,                    # drop `model` when attaching — the resource defines the base
                model_resources_id=resource.model_resources_id,
                wandb_run_name=f"lr{hp['learning_rate']:.0e}-rank{hp['lora_rank']}", **hp)
        for hp in SWEEP
    ]
    with ThreadPoolExecutor(max_workers=len(configs)) as pool:   # one thread per arm
        arms = {c.wandb_run_name: pool.submit(train_single_turn, c) for c in configs}
    failed = {n: f.exception() for n, f in arms.items() if f.exception() is not None}
# context exit stops the resource exactly once, even if an arm raised
```

Notes:

- **Concurrency.** `train_single_turn` blocks, so each arm runs in its own thread; each `SessionClient` owns a private event loop, so the sessions don't interfere. One arm raising doesn't stop its siblings — collect `future.exception()` after the pool drains.
- **Model selection is the per-arm validation.** Put the arms in one `wandb_group` and compare their `val/accuracy` curves side by side; no extra eval code.
- **Teardown.** The `ModelResourcesClient` context manager stops the resource once on exit. Without it, call `resource.stop()` yourself.

See `recipes/grpo/deepmath_shared.py` (tenant), `recipes/grpo/deepmath_sweep.py` (owner/sweep), and `notebooks/multi_lora_example.ipynb`. For the low-level attach API (`resources.create_session(...)`), see [SDK reference: ModelResourcesClient](./sdk-reference.md#modelresourcesclient).

---

## Utility functions

### Learning rate schedule

Cosine annealing with linear warmup, in `post_training_cookbook.grpo.loop`:

```python
from post_training_cookbook.grpo.loop import get_lr_cosine

lr = get_lr_cosine(
    step=50,
    total_steps=1000,
    base_lr=1e-5,
    warmup_ratio=0.05,     # warmup for first 5% of steps
    min_lr_ratio=0.1,      # decay to 10% of base_lr
)
```

### Overlong penalty

Penalizes responses that exceed the token budget (DAPO soft-overlong shaping), in `post_training_cookbook.overlong`:

```python
from post_training_cookbook.overlong import compute_overlong_penalty, is_overlong

penalty = compute_overlong_penalty(
    response_length=550,
    max_response_length=512,
    overlong_buffer_len=48,
    penalty_factor=1.0,
)
# Returns a non-positive value (0.0 if not over budget).

over = is_overlong(response_length=550, max_response_length=512)
```

---

## Run artifacts

Every recipe writes a self-contained record of the run to `runs/<run_name>/`, built by `post_training_cookbook.run_artifacts`. The run name is derived from the model, project/kind, a hyperparameter slug, and a UTC timestamp (or `wandb_run_name` if you set one).

```
runs/<run_name>/
  config.yaml        round-trippable via <Config>.from_yaml(...) (GRPOConfig / SFTConfig / DPOConfig)
  manifest.json      wandb url, session id, model, start time, git sha
  metrics.jsonl      one JSON line per step (mirrors what goes to W&B)
  checkpoints.jsonl  training / inference checkpoint ids when an interval fires
  step_NNNN_*.json   per-step rollouts / episodes (when export_samples=True)
  val_NNNN_*.json    validation rollouts (when export_samples=True)
```

`metrics.jsonl` is always written; the bulky `step_*`/`val_*` rollout dumps are skipped when `export_samples=False`. Values that aren't JSON-serializable (for example, the W&B example-output table) are forwarded to W&B but dropped from `metrics.jsonl`.

If you build your own SDK-level loop, the public helpers are `start_run`, `log_step`, `export_samples`, and `export_episodes`. `validate_lora_config` and `lora_config_from` translate a config's `use_lora`/`lora_*` fields into the SDK's `LoraConfig`.

---

## Tests

Run the test suite:

```bash
uv run pytest tests/
```

Tests cover:

- Codec encoding/decoding, tool-call parsing, and the forked thinking templates (`tests/codecs/`).
- Reward functions for all built-in datasets (`tests/datasets/test_rewards.py`).
- Env contract and reference environment (`tests/envs/`).
- GRPO sample primitives, advantages, batching, the loop, and single-/multi-turn collection (`tests/grpo/`).
- Multi-turn episode handling (`tests/grpo/test_episode.py`).
- GRPO configuration loading and validation (`tests/grpo/test_config.py`).
- SFT conversation rendering and loss masking (`tests/sft/test_rendering.py`).
- DPO loss and preference-pair construction (`tests/dpo/`).
- Custom-loss helper, LoRA config translation, overlong penalty, sequence splitting, and run artifacts (`tests/test_custom_loss.py`, `tests/test_lora_config.py`, `tests/test_overlong.py`, `tests/test_sequence_split.py`, `tests/test_run_artifacts.py`).
