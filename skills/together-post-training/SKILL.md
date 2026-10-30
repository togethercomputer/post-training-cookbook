---
name: together-post-training
description: "Use this skill for Together AI post-training fine-tuning workflows: GRPO training (single- and multi-turn) with custom reward functions, supervised fine-tuning (SFT), DPO preference tuning, custom client-side losses, LoRA and full-weight fine-tuning, training session management, checkpointing, weight downloads, and dedicated endpoint deployment — all via the Together Post Training SDK (`together.post_training`) and Post Training Cookbook. Reach for it whenever the user wants to fine-tune a language model with GRPO/SFT/DPO, write reward or custom loss functions, build a multi-turn or tool-use environment, configure training hyperparameters, manage training sessions, or deploy a fine-tuned model. Also use when the user mentions post-training-cookbook, together.post_training, SessionClient.create, ModelResourcesClient, train_single_turn, train_multi_turn, GRPOConfig, SFTConfig, DPOConfig, forward_backward, custom_forward_backward, or asks about LoRA or full-weight training on Together AI."
---

# Together AI post-training fine-tuning

The post-training fine-tuning workflow on Together AI, from configuring a training run to deploying the result. Two layers are involved:

- **Post Training Cookbook** (`post_training_cookbook`) — high-level recipes with built-in datasets, rewards, and training loops for GRPO (single- and multi-turn), SFT, and DPO.
- **Together Post Training SDK** (`together.post_training`) — low-level session-based API for sampling, gradient computation, and weight updates.

Most users start with the Cookbook. Drop to the SDK for custom training loops and custom losses.

## When to read reference files

This skill bundles five reference files. The body below is enough for most tasks; open a reference only when you need the detail in the right-hand column, and jump straight to the named section rather than reading top to bottom. Each reference opens with a `**Contents:**` line of section links.

| When the task is about... | Open | Jump to section |
|---|---|---|
| A GRPO/SFT/DPO **config field** — its type, default, or what it does | `references/cookbook-guide.md` | `GRPO configuration` / `SFT configuration` / `DPO configuration` |
| Writing a **custom reward** or registering a dataset | `references/cookbook-guide.md` | `Adding a custom dataset` |
| Building a **multi-turn environment** (`Env` API, `step_env`, episodes) | `references/cookbook-guide.md` | `Multi-turn environments` |
| Which **built-in dataset/recipe** to start from | `references/cookbook-guide.md` | `Built-in datasets` / `Recipes` |
| A **custom loss** (DPO and beyond): forward-only `forward` → autograd → `custom_forward_backward` | `references/custom-losses.md` | (whole file is short) |
| An **SDK call signature**, sample/loss type, `ModelResourcesClient`, Muon/AdamW params | `references/sdk-reference.md` | named method/type (e.g. `SessionClient`, `ModelResourcesClient`, `Loss configurations`) |
| **Why** something works the way it does (sessions, advantages, clipping, LoRA, codecs, thinking) | `references/concepts.md` | named concept |
| **Reading or debugging a metric** (GRPO, multi-turn, DPO, SFT) | `references/metrics.md` | metric's section |

Don't load a reference for a task the body already answers, and don't load all five — pick the one row that matches.

## Bundled scripts

These scripts cover the common boilerplate. Run them directly, or copy them as starting points:

| Script | What it does | When to use |
|---|---|---|
| `scripts/scaffold_dataset.py` | Generates reward function, registry entry, and recipe for a new dataset | User wants to add a custom dataset for GRPO training |
| `scripts/custom_grpo_template.py` | Complete SDK-level GRPO training loop with cosine LR, checkpointing, reward placeholder | User needs a custom training loop outside the Cookbook |
| `scripts/custom_loss_template.py` | Two-pass custom-loss loop (forward-only `forward` → autograd → `custom_forward_backward`), DPO worked example | User needs DPO or any loss the service doesn't ship |
| `scripts/multi_turn_env_template.py` | `Env` subclass starter (tool, reset/step/score) wired to `train_multi_turn` | User is building a multi-turn / tool-use environment |
| `scripts/multi_lora_sweep_template.py` | Hyperparameter sweep that shares one model resource across N adapter sessions | User wants to train several adapters / a sweep on one base model |
| `scripts/deploy_checkpoint.py` | CLI to deploy a checkpoint on Dedicated Model Inference and test it | User wants to deploy a trained model |
| `scripts/prepare_for_fp4.py` | Estimates, starts, and watches an FP4 preparation job, then prints the prepared model ID | User wants to deploy an adapter whose base model serves in FP4 |

**Scaffold a new dataset:**
```bash
python scripts/scaffold_dataset.py trivia my-org/trivia-10k --question-col question --answer-col answer
```
Generates `reward_trivia.py`, `register_trivia.py`, and `recipe_trivia.py` — edit the reward function, then integrate into the cookbook.

**Custom GRPO loop:** Copy `scripts/custom_grpo_template.py` and replace the TODO sections (data loading, tokenizer, reward function). Session management, sample construction, the LR schedule, and checkpointing are already in place.

**Custom loss / DPO:** Copy `scripts/custom_loss_template.py`. It implements the two-pass pattern end-to-end with DPO as the worked loss — replace `load_preference_pairs` with your data and `make_dpo_loss` with your objective (any function of the per-slot log-probabilities).

**Multi-turn environment:** Copy `scripts/multi_turn_env_template.py`. Fill in the tool(s), the `Env.step` scoring, and the task list; it's already wired to `train_multi_turn`.

**Multi-LoRA sweep:** Copy `scripts/multi_lora_sweep_template.py`. It provisions one shared model resource, trains N adapter arms concurrently (each its own LoRA session), and deletes the resource when the sweep finishes. Edit the base config and the sweep grid.

**Deploy a checkpoint:**
```bash
python scripts/deploy_checkpoint.py <model_id> <revision_id> --config <config_id>  # IDs from ckpt.registration
```

**Prepare an adapter for FP4 inference:**
```bash
python scripts/prepare_for_fp4.py <adapter_id> [--calibration-file file-...] --yes
```

---

## Setup

The Post Training Cookbook lives at the root of the project. Install with:

```bash
uv sync --locked
```

Required environment variables:

```bash
export TOGETHER_API_KEY="..."
export HF_TOKEN="..."           # for gated models on HuggingFace
export WANDB_API_KEY="..."      # optional, enables W&B logging
```

---

## Training methods

The cookbook ships three high-level methods — GRPO, SFT, and DPO. All three accept a `use_lora` flag (default `True`); set `use_lora=False` for full-weight fine-tuning (see [Full-weight fine-tuning](#full-weight-fine-tuning)).

### GRPO (reinforcement learning)

GRPO generates multiple completions per prompt, scores them with a reward function, and reinforces the better ones. The loop:

1. **Sample** — generate `group_size` completions per prompt at `temperature`.
2. **Score** — run the reward function on each completion.
3. **Compute advantages** — `advantage_i = reward_i - mean(group_rewards)`.
4. **Forward-backward** — send samples with GRPO loss to compute gradients.
5. **Optim step** — update weights with cosine-annealed learning rate.

Run an existing recipe:

```bash
uv run python recipes/grpo/gsm8k.py
```

Or create a new one:

```python
from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.grpo.train import train_single_turn

config = GRPOConfig(
    model="Qwen/Qwen3.8-27B",
    dataset="gsm8k",
    num_examples=4000,
    batch_size=16,
    group_size=8,
    learning_rate=1e-5,
)

if __name__ == "__main__":
    train_single_turn(config)
```

The GRPO entry points live in `post_training_cookbook.grpo.train`: `train_single_turn(config)` for prompt→response→reward, and `train_multi_turn(config, tasks, env_factory, val_tasks=None)` for multi-turn/tool-use (see [Multi-turn training](#multi-turn-tool-use-training) below). For the full GRPOConfig parameter reference, read `references/cookbook-guide.md`.

### SFT (supervised fine-tuning)

SFT trains on pre-existing conversation data with cross-entropy loss. Assistant-only masking is always applied: system and user tokens are never trained on, regardless of config. `train_on_what` only chooses *which* assistant turns count: `TRAIN_ON_ALL` (every assistant turn) or `TRAIN_ON_LAST` (only the final one).

```python
from post_training_cookbook.sft.config import SFTConfig
from post_training_cookbook.sft.train import TRAIN_ON_ALL, train

config = SFTConfig(
    model="Qwen/Qwen3.8-27B",
    dataset_name="HuggingFaceH4/no_robots",
    batch_size=8,
    learning_rate=1e-4,
    max_steps=50,
    train_on_what=TRAIN_ON_ALL,
)

if __name__ == "__main__":
    train(config)
```

SFT never samples, so it runs on a trainer-only session automatically (`num_generator_replicas=0`). Use the `TRAIN_ON_ALL` / `TRAIN_ON_LAST` constants from `post_training_cookbook.sft.train` for `train_on_what`.

### DPO (preference tuning)

DPO trains on `(chosen, rejected)` preference pairs. The cookbook's DPO loop is also the reference implementation of the [custom-loss pattern](#custom-losses-dpo-and-beyond): it snapshots reference-model log-probabilities with `session.trainer.forward` (forward-only, so nothing accumulates into the gradients it submits later), computes the DPO loss in Python, and submits the gradients with `session.trainer.custom_forward_backward`.

```python
from post_training_cookbook.dpo.config import DPOConfig
from post_training_cookbook.dpo.train import train

config = DPOConfig(
    model="Qwen/Qwen3.8-27B",
    dataset_name="nvidia/HelpSteer3",
    batch_size=8,
    beta=0.1,
    max_steps=1000,
)

if __name__ == "__main__":
    train(config)
```

`train(config, to_pair=fn)` maps a row to `(chosen_messages, rejected_messages)`, or `None` to skip it. The default, `helpsteer3_pair` in `post_training_cookbook.dpo.train`, turns HelpSteer3's `overall_preference` score into a pair and skips ties; pass your own for other preference datasets. Validation reports an implicit-reward **preference accuracy** rather than a token metric. For DPOConfig fields, read `references/cookbook-guide.md`; for the loss mechanics, read `references/custom-losses.md`.

---

## Multi-turn (tool-use) training

For agentic / tool-use tasks, the model runs whole episodes against an environment instead of producing a single response. Set `GRPOConfig.max_turns` and call `train_multi_turn`:

```python
from post_training_cookbook.envs.tool_use_game import TOOL_USE_GAME_TASKS, ToolUseGameEnv
from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.grpo.train import train_multi_turn

tasks = list(TOOL_USE_GAME_TASKS)
config = GRPOConfig(
    model="Qwen/Qwen3.8-27B",
    max_turns=6,
    batch_size=4,
    episodes_per_task=8,      # the multi-turn analogue of group_size
    num_steps=200,
)

if __name__ == "__main__":
    train_multi_turn(config, tasks=tasks, env_factory=lambda task: ToolUseGameEnv(), val_tasks=tasks)
```

To author a new environment, subclass `post_training_cookbook.envs.Env` and implement `async reset(task)`, `async step(message)`, `tool_specs`, and `messages` (plus `async close()` to release resources). Pass an `EnvFactory` (`Callable[[Task], Env]`) to `train_multi_turn`. See `post_training_cookbook/envs/tool_use_game/` for a complete reference env, and `references/cookbook-guide.md` (Multi-turn environments) and `references/concepts.md` for the full API.

---

## Common patterns

### Add a custom dataset and reward function

Three steps:

1. **Write a reward function** in `post_training_cookbook/datasets/rewards.py`:

```python
def my_reward(prediction: str, target: str) -> float:
    """Return float in [0, 1]. Higher = better."""
    answer = extract_answer(prediction)
    return 1.0 if answer == target else 0.0
```

2. **Register the dataset** in `post_training_cookbook/datasets/registry.py`:

```python
DATASETS["my_dataset"] = DatasetConfig(
    hf_name="my-org/my-dataset",
    hf_config=None,
    hf_split="train",
    format_question=lambda row: row["question"] + " Answer concisely.",
    extract_target=lambda row: row["answer"],
    compute_reward=my_reward,
    few_shot_messages=[
        {"role": "user", "content": "Example question. Answer concisely."},
        {"role": "assistant", "content": "Example answer"},
    ],
)
```

3. **Create a recipe** in `recipes/grpo/`:

```python
from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.grpo.train import train_single_turn

config = GRPOConfig(model="Qwen/Qwen3.8-27B", dataset="my_dataset", num_examples=5000)

if __name__ == "__main__":
    train_single_turn(config)
```

For `DatasetConfig` field details, read `references/cookbook-guide.md`. (`model` is required on `GRPOConfig`.)

### Use the SDK directly (custom training loop)

When the Cookbook's built-in loop doesn't fit, use the SDK:

```python
import os
from together.post_training import (
    ModelResourcesClient, AdamParams, LossConfig, GrpoLossParams,
    ModelInput, ModelInputChunk, EncodedTextChunk, SamplingParams,
)
from post_training_cookbook.grpo.samples import build_policy_sample
from post_training_cookbook.samples import is_trainable

# 1. Provision a resource and open a session
resources = ModelResourcesClient.create(
    base_model="Qwen/Qwen3.8-27B",
    api_key=os.environ.get("TOGETHER_API_KEY"),
    base_url=os.environ.get("TOGETHER_BASE_URL"),
)
session = resources.create_session()

# 2. Sample completions
model_input = ModelInput(chunks=[
    ModelInputChunk(encoded_text=EncodedTextChunk(tokens=prompt_tokens))
])
result = session.generator.sample(
    model_input,
    num_samples=8,
    sampling_params=SamplingParams(temperature=0.7, max_tokens=512),
)

# 3. Build training samples. Every argument is in token space and length
#    len(tokens); the builder owns the trim and derives target_tokens.
samples = []
for seq in result.sequences:
    response_tokens = [int(t) for t in seq.tokens]
    loss_mask = [0.0] * len(prompt_tokens) + [1.0] * len(response_tokens)
    # A completion that stopped immediately trains no slot; building it would
    # raise and take the whole run down.
    if not is_trainable(loss_mask):
        continue
    samples.append(
        build_policy_sample(
            tokens=prompt_tokens + response_tokens,
            logprobs=[0.0] * len(prompt_tokens) + [float(v) for v in seq.logprobs],
            loss_mask=loss_mask,
            advantage=adv,
        )
    )

# 4. Forward-backward. Omit clip thresholds to use server defaults (0.8 / 1.2).
fb = session.trainer.forward_backward(
    samples=samples,
    loss=LossConfig(
        type="LOSS_TYPE_GRPO",
        grpo_params=GrpoLossParams(agg_type="GRPO_LOSS_AGGREGATION_TYPE_FIXED_HORIZON"),
    ),
)

# 5. Optimizer step
session.trainer.optim_step(
    adam_params=AdamParams(beta1=0.9, beta2=0.95, weight_decay=0.1, learning_rate=1e-5),
)
session.trainer.weights_sync(weight_sync_type="WEIGHT_SYNC_TYPE_SYNCHRONOUS")
```

Step 3 uses the cookbook's builder even in an otherwise SDK-only loop, because it
validates the payload. To build the `Sample` without the cookbook, read
`references/sdk-reference.md` ("If you build by hand").

`optim_step` updates the trainer's parameters but does not publish them to the generator. Call `weights_sync` afterwards when subsequent samples should use the update (GRPO and other on-policy loops). Trainer-only SFT/DPO with `num_generator_replicas=0` skip it. Synchronous sampling loops pass `"WEIGHT_SYNC_TYPE_SYNCHRONOUS"`, which publishes before `weights_sync` returns. The other values (`WEIGHT_SYNC_TYPE_BACKGROUND_PUBLISH`, `WEIGHT_SYNC_TYPE_PIPELINE`) overlap the publish with in-flight rollouts; `PIPELINE` is rejected on LoRA sessions. See `references/sdk-reference.md` (`weights_sync`).

`SampleResult.policy_segments` is sampling-response metadata for client-side staleness tracking — not part of a training `Sample`. Lockstep GRPO ignores it; async multi-turn uses it for `async/mid_rollout_drift_ratio`.

For the complete SDK API surface (all methods, parameters, return types), read `references/sdk-reference.md`.

### Custom losses (DPO and beyond)

When neither the built-in cross-entropy nor GRPO loss fits, compute the loss in client-side Python with the two-pass pattern: `session.trainer.forward` runs a forward-only pass and returns per-slot log-probabilities in `loss_fn_outputs`, you evaluate the loss with autograd, and `session.trainer.custom_forward_backward` submits the gradients. The cookbook wraps this in `post_training_cookbook.custom_loss`:

```python
from post_training_cookbook.custom_loss import compute_logprobs, compute_custom_forward_backward

# Snapshot a frozen reference policy once (read-only, no optimizer change).
ref = compute_logprobs(session, batch)

def my_loss(logprobs):                      # list of (S_i,) float32 tensors, one per sample
    ...                                     # must return a scalar torch.Tensor
    return loss

result = compute_custom_forward_backward(session, batch, my_loss)
session.trainer.optim_step(adam_params=...)  # learning_rate lives inside adam_params
# Call weights_sync only if a later sample should see the update.
```

Two traps. First, `forward_backward` can also return per-sample outputs via `return_loss_fn_outputs=True`, but that flag is independent of forward-only behavior: the backward pass still runs, and its gradients are added to the ones `custom_forward_backward` submits, so `optim_step` applies the sum with no error. Score samples with `forward`. Second, only the positions a sample's `weights` tensor scores carry a real log-probability — the rest read back as exactly `0.0`, so reduce your loss over the mask (pass all-ones weights if you want log-probabilities everywhere).

DPO is the worked example. Read `references/custom-losses.md` for the full pattern and `post_training_cookbook/dpo/` for a complete implementation.

### Full-weight fine-tuning

Every config takes `use_lora` (default `True`). Set `use_lora=False` to fine-tune all model weights instead of a LoRA adapter:

```python
config = SFTConfig(model="Qwen/Qwen3.8-27B", batch_size=4, learning_rate=2e-5, use_lora=False)   # see recipes/sft/no_robots_full.py
```

Caveats: use a more conservative LR than a comparable LoRA run (compare `recipes/sft/no_robots.py` against `recipes/sft/no_robots_full.py`). Setting any `lora_rank`/`lora_alpha`/`lora_dropout` while `use_lora=False` raises `ValueError`.

### Reasoning (thinking) options

GRPO/SFT configs (and `get_codec`) take three thinking options:

- `enable_thinking` — on/off for the model's `<think>` output (`None` = model default; adaptive on MiniMax-M3; GLM-5 and Muse Spark always think and reject `False`).
- `preserve_thinking` — when `True`, render history with a template that keeps `<think>` blocks across turns — natively for GLM-5, Qwen3.6 and Qwen3.8, via a forked template for Qwen3.5 (requires `enable_thinking != False`). When `None` (default), the stock template decides: Qwen3.5/3.6 strip thinking and the conversation is split into prefix-stable sequences, while GLM-5's and Qwen3.8's keep it. `False` strips it. Kimi-K3, MiniMax-M3 and Muse Spark render history reasoning by default, so there the default and `True` render identically and `False` strips it before rendering. Only preserve thinking if inference also preserves it. See `references/concepts.md` (codecs).
- `thinking_effort` — `low`/`high`/`max` for GLM-5 and Kimi-K3 (default `max`), `low`/`medium`/`high`/`xhigh` for Muse Spark (default `xhigh`); `None` keeps the template default. Other families reject any value.

The codec is resolved from the model name by longest matching prefix: `qwen3.5` / `qwen3.6` / `qwen3.8` → `Qwen35Codec`, `glm-5.3` → `Glm5Codec`, `kimi-k3` → `KimiK3Codec`, `minimax-m3` → `MiniMaxM3Codec`, `muse-spark-open` → `MuseSparkCodec`.

At the default, GLM-5, Kimi-K3 and Muse Spark runs need a generous `max_sample_tokens` (or, on Kimi-K3, `enable_thinking=False`); otherwise rollouts truncate mid-reasoning.

### Multi-LoRA: many adapters on one base model

Provisioning a base model is the slow, GPU-heavy part of startup. A **model resource** is the GPUs plus base weights; a **session** is your LoRA adapter on top. When a resource is created with `lora_enabled=True`, every attached session is an independent adapter against the shared base weights — so N adapters cost roughly one base model's GPUs instead of N. You can *own* a resource or just *borrow* one.

A config names exactly one model source: set `model` to provision a fresh resource, or set `model_resources_id` to attach to an existing one — never both. `training_session` raises `ValueError` if both are set (a shared resource already defines its base model) or if neither is.

With `model_resources_id` set, `train_single_turn` / `train` attach for you after checking the resource is `READY` and LoRA/generator-compatible. Leave it `None` to provision a fresh session as usual.

**Tenant** — attach to a resource someone else provisioned; never create or delete it:

```python
config = GRPOConfig(
    model_resources_id=os.environ["MODEL_RESOURCES_ID"],   # the resource defines the base model — don't set `model`
    dataset="gsm8k",
)
train_single_turn(config)                          # resource left running — see recipes/grpo/deepmath_shared.py
```

**Owner / sweep** — provision once, train N adapter arms concurrently, tear down on exit:

```python
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from together.post_training import ModelResourcesClient

base = GRPOConfig(model="Qwen/Qwen3.8-27B", dataset="gsm8k", num_steps=100)

with ModelResourcesClient.create(base_model=base.model, lora_enabled=True) as resource:   # `base.model` provisions the GPUs
    arms = [
        replace(base, model=None,                  # drop `model` when attaching — the resource defines the base
                model_resources_id=resource.model_resources_id,
                learning_rate=lr, lora_rank=rank, wandb_run_name=f"lr{lr:.0e}-rank{rank}")
        for lr in (1e-5, 5e-6) for rank in (8, 16)
    ]
    with ThreadPoolExecutor(max_workers=len(arms)) as pool:   # one thread per arm
        futures = [pool.submit(train_single_turn, cfg) for cfg in arms]
# context exit stops the resource exactly once
```

`base` keeps `model` set only so `ModelResourcesClient.create(base_model=base.model)` can provision; each arm sets `model=None` and attaches via `model_resources_id`. Each arm is its own `SessionClient` driving its own event loop; group them in W&B with `wandb_group` so the per-arm validation curves do the model selection.

Teardown: the owner's `ModelResourcesClient` context manager stops the resource exactly once on `__exit__`, and attached tenant sessions only detach — they never stop what they didn't create.

Copy `scripts/multi_lora_sweep_template.py`, or see `recipes/grpo/deepmath_sweep.py`. Find existing resources with `Together().post_training.model_resources.list(status="MODEL_RESOURCES_STATUS_READY")`; resource enumeration lives on the client, and `ModelResourcesClient` itself has no `.list`.

For the low-level path outside the cookbook, `resources.create_session(lora_config=...)` performs the same attach. See `references/cookbook-guide.md` (Multi-LoRA) and `references/sdk-reference.md` (ModelResourcesClient).

### Checkpoint and download

```python
from pathlib import Path
from together import Together
from together.post_training import download_checkpoint

# Snapshot for inference: weights go to the model registry
ckpt = session.create_inference_checkpoint()
print(ckpt.id, ckpt.registration)  # registration.adapter for LoRA, registration.model for full weights

# Download merged weights
checkpoint_id = session.retrieve().inference_checkpoints[-1].id
paths = download_checkpoint(
    Together(),
    checkpoint_id,
    variant="CHECKPOINT_VARIANT_MERGED",    # or "CHECKPOINT_VARIANT_ADAPTER"
    output_dir=Path("./my_model"),
)
```

### Save training state and resume

```python
save = session.create_training_checkpoint()
session.stop()

# Later, on a new resource:
resources = ModelResourcesClient.create(base_model="Qwen/Qwen3.8-27B")
session = SessionClient.create(
    model_resources_id=resources.model_resources_id,
    resume_from_checkpoint_id=save.id,
    # load_optimizer omitted/True: restore optimizer + step
    # load_optimizer=False: weights only, fresh optimizer, step 0
)
```

Omit `load_optimizer` or pass `True` to restore optimizer moments and the step counter. `load_optimizer=False` loads adapter weights only — fresh optimizer, step 0 — and is valid only for **training** checkpoints, not inference or HuggingFace ones. Both `SessionClient.create` and `resources.create_session(...)` take the same pair of kwargs.

Two continuation paths:

1. **Continue the same run** — omit `load_optimizer` or pass `True`. Cookbook: set `training_checkpoint_interval` (default `0` = off), then `resume_from_checkpoint_id` from `runs/<run>/checkpoints.jsonl` on the same recipe config and re-run. GRPO also requires `resume_run_dir` with the matching training record’s saved data state. The client iteration includes empty batches and is distinct from the server optimizer counter. See `references/cookbook-guide.md` for configuration and replay limitations. Not its own recipe.
2. **Same adapter, new objective** — resume a training checkpoint with `load_optimizer=False`. Cookbook: `SFTConfig`/`GRPOConfig`/`DPOConfig` have `resume_from_checkpoint_id` and `load_optimizer`; `load_optimizer=False` requires a checkpoint id; `sft.train()` returns the last training-checkpoint id. SFT is trainer-only, so GRPO provisions a new resource (same base, same LoRA rank/alpha). Worked example: `recipes/sft_then_grpo.py`.
Inference checkpoints are for serving and download. They do not seed new training runs.

### Deploy on Dedicated Model Inference

Three paths for a LoRA checkpoint (`registration.adapter`). The CLI commands need together-py 3.0+.

Choose the path from what the platform reports, not from a list of models. The adapter's base model is `client.models.retrieve(adapter_id).base_model_id`.

1. Run `tg training fp4 estimate <adapter_id>`. If it returns a price, the base model serves in FP4: [prepare the adapter](#prepare-an-adapter-for-fp4-inference) and deploy the prepared model. If it reports that the base model isn't supported, deploy the adapter with `--merge`.
2. To compare candidates first, run `tg models configs <base_model_id>`. A config whose `adapter_mode` is `dynamic` means the base model supports multi-LoRA deployment.

- **Production, one adapter:** `tg endpoints deploy <adapter_id> --endpoint <name> --merge --config <config_id> --project <project_id>`, with a config of the base model whose `adapter_mode` is `disabled`. Replicas merge the adapter at startup. Guide: https://docs.together.ai/docs/post-training/deployment
- **Compare candidates:** `tg endpoints deploy <adapter_id> --endpoint <name> --attach-adapter [--config <config_id>]` per adapter, with a config whose `adapter_mode` is `dynamic`; call `model="<project_slug>/<endpoint>:<adapter_id>"` on `https://api-inference.together.ai/v1`. `deploy` fails with "No fixed or dynamic LoRA config found" when the base model has none. Guide: https://docs.together.ai/docs/post-training/multi-lora-deployment
- **FP4 preparation:** `tg training fp4 estimate|create <adapter_id> [--calibration-file f.jsonl] --watch` (or `scripts/prepare_for_fp4.py`), then deploy the prepared model without `--merge`. See [Prepare an adapter for FP4 inference](#prepare-an-adapter-for-fp4-inference). Guide: https://docs.together.ai/docs/post-training/prepare-for-fp4-inference

Full-weight checkpoints (`registration.model`) and FP4-prepared models are merged models: deploy them directly, without `--merge`, on a config whose `adapter_mode` is `disabled` (`tg models configs <model_id>`). Always pass `--project`: without it the CLI resolves a private model to its base model and deploys the base.

The SDK flow below matches `--merge`; `scripts/deploy_checkpoint.py` wraps it. Both need `TOGETHER_PROJECT_ID` (or `Together(project_id=...)`) for `client.models` / `client.endpoints`.

```python
import time
from together import Together

client = Together()
ckpt = session.retrieve().inference_checkpoints[-1]
artifact = ckpt.registration.adapter or ckpt.registration.model

model = client.models.retrieve(artifact.id)
# Use a config whose adapter_mode is disabled (`tg models configs <base_model_id>`), not a multi-LoRA one.
configs = client.models.configs.list(reference_model_id=model.base_model_id or model.id).data
config = next(c for c in configs if c.id == "<config_id>")

endpoint = client.endpoints.create(name="my-rl-model")
deployment = client.endpoints.deployments.create(
    endpoint.id,
    name="my-rl-model-v1",
    model=f"projects/{model.project_id}/models/{model.id}/revisions/{artifact.revision_id}",
    config=f"projects/{config.project_id}/configs/{config.id}",
    autoscaling={"min_replicas": 1, "max_replicas": 1},
    inactive_timeout=30,
)
client.endpoints.update(
    endpoint.id,
    traffic_split=[{"deployment_id": deployment.id, "weight": 1.0}],
    update_mask="trafficSplit",
)

while deployment.status.state != "DEPLOYMENT_STATE_READY":
    time.sleep(15)
    deployment = client.endpoints.deployments.retrieve(deployment.id, endpoint_id=endpoint.id)

# Routing can lag READY by a minute or two: retry on 404 or 503.
response = client.chat.completions.create(
    model=endpoint.name,
    messages=[{"role": "user", "content": "What is 2+2?"}],
    max_tokens=256,
)
```

### Prepare an adapter for FP4 inference

Adapters on a base model that serves in FP4 don't take `--merge`. An FP4 preparation job merges the adapter into the BF16 weights, calibrates, quantizes to FP4, and registers the result as a private model in the adapter's project. Guide: https://docs.together.ai/docs/post-training/prepare-for-fp4-inference. CLI reference: https://docs.together.ai/reference/cli/training.

- **Input:** the adapter's `registration.adapter.id` (`ml_...`), optionally `revision_id` (`rv_...`; defaults to the latest). Full-weight models aren't accepted. Use a key from the adapter's project.
- **Time and price:** several hours, depending on the base model, plus any wait for capacity. Run `estimate` first: it prints the price and hours, checks that the adapter is eligible, and creates nothing.
- **Calibration (optional):** `--calibration-file` takes a local JSONL (the CLI uploads it with purpose `calibration`) or a `file-...` ID. Rows use the SFT chat format (`messages`, optional `tools`), at most 64 MiB, same project as the adapter. It's mixed 1:1 with a general text corpus, and must supply at least 598,016 tokens after the chat template, or the job ends in `user_error`.
- **Statuses:** `pending` → `queued` → `running` → `completed`; failures end in `user_error` (fix the input, see `list-events`) or `error`. `cancel` moves a job to `cancel_requested`, then `cancelled`.
- **Repeat jobs:** each `create` starts a new paid job, even for an adapter already prepared, and uploads to the same model. Reuse the earlier result.

```bash
tg training fp4 estimate ml_abc123
tg training fp4 create ml_abc123 --calibration-file ./calibration.jsonl --watch   # -y skips the confirmation
tg training fp4 list-events shp-quant-...    # why a job failed
```

`create` doesn't ask for confirmation outside an interactive terminal (CI, redirected I/O, coding agents), so always run `estimate` first.

```python
import time
from together import Together

client = Together()
fp4 = client.post_training.prepare_for_fp4_inference

estimate = fp4.estimate_cost(inputs={"adapter_object_id": "ml_abc123"})
print(estimate.price_usd, estimate.time_hours, estimate.allowed_to_proceed)

job = fp4.create(inputs={"adapter_object_id": "ml_abc123"})  # add "calibration_file_id": "file-..."
while job.status not in {"completed", "cancelled", "error", "user_error"}:
    time.sleep(300)
    job = fp4.retrieve(job.id)
print(job.status, job.results.api_model_object_id if job.status == "completed" else None)
```

The prepared model is a merged model, already validated for deployment. Deploy it without `--merge`, on a config whose `adapter_mode` is `disabled`, and pass `--project`:

```bash
tg models configs <prepared_model_id>
tg endpoints deploy <prepared_model_id> --endpoint my-fp4-model --config <config_id> --project <project_id>
```

The prepared model registers against the base model that serves it in FP4, which can differ from the adapter's base, so list the configs of the prepared model, not of the adapter's base.

---

## Built-in datasets

| Key | Source | Reward | Description |
|---|---|---|---|
| `gsm8k` | `openai/gsm8k` | Exact numeric match after `####` | Grade-school math word problems |
| `deepmath` | `zwhe99/DeepMath-103K` | `boxed_math_reward` (`math_grader.py`) | Competition math |
| `deepmath-lv5` | Same, filtered to difficulty <= 5 | Same | Easier subset |
| `game24` | `nlile/24-game` | Tiered: format + closeness + exact | Arithmetic puzzle |

---

## Key files

```
post_training_cookbook/
  grpo/config.py          GRPOConfig dataclass (all hyperparameters)
  grpo/train.py           train_single_turn() and train_multi_turn() entry points
  grpo/loop.py            Shared training_loop + get_lr_cosine LR schedule
  grpo/single_turn.py     Single-turn scoring, advantages, sample building
  grpo/multi_turn.py      Multi-turn episode rollout and collection
  envs/base.py            Env contract + episode data types (CompletedEpisode, TruncationCause)
  grpo/samples.py         build_policy_sample, group-relative advantages
  sft/config.py           SFTConfig dataclass
  sft/train.py            SFT loop, render_conversation, TRAIN_ON_ALL/LAST
  dpo/                    DPOConfig, DPO loop, loss, preference-pair builders
  datasets/registry.py    DatasetConfig, DATASETS dict, loading helpers
  datasets/rewards.py     gsm8k / game24 rewards
  datasets/math_grader.py boxed_math_reward (deepmath)
  codecs/                 get_codec, Qwen3.5/GLM-5/Kimi-K3/MiniMax-M3/Muse Spark codecs, chat templates
  envs/                   Env ABC, Task, step_env, tool_use_game reference env
  custom_loss.py          compute_logprobs, compute_custom_forward_backward
  multi_lora.py           training_session: attach to a shared model resource or provision
  overlong.py             compute_overlong_penalty, is_overlong
  sequence_split.py       split_turns (prefix-stable trainable sequences)
  run_artifacts.py        run dir, config.yaml, metrics.jsonl, rollout dumps

recipes/grpo/             Ready-to-run GRPO recipes (single- and multi-turn)
recipes/sft/              Ready-to-run SFT recipes (incl. no_robots_full)
recipes/dpo/              Ready-to-run DPO recipes
```

Runs write a self-contained record to `runs/<run_name>/` (config, `manifest.json`, `metrics.jsonl`, per-step rollouts). See `references/cookbook-guide.md` (Run artifacts).

---

## Troubleshooting

**High skip ratio (> 50%)**: All completions in most groups get the same reward — no learning signal. Either the task is too easy/hard, or `group_size` is too small. Try increasing `group_size` to 8-16, adjusting difficulty, or raising `temperature`.

**Accuracy stuck at 0**: The reward function can't parse the model's output. Test `compute_reward` manually on sample completions. Check that few-shot examples match the expected format.

**Timeout errors**: Increase `timeout` in the config. Reduce `batch_size` or `max_model_length` if payloads are too large.

**Session enters ERROR state**: Check `session.retrieve().status` for details. Common causes: invalid model name, LoRA config mismatch when resuming, API key issues.

**Multi-turn episodes never finish**: A low `terminated_ratio` with high `context_overflow_ratio` or `parse_failure_ratio` means episodes hit the turn/context limit or emit malformed tool calls before completing. Raise `max_turns`/`max_model_length`, check the tool schemas and system prompt, and make sure parse failures aren't dominating. See `references/metrics.md` (multi-turn metrics).

**`use_lora=False` rejected with ValueError**: You set a `lora_rank`/`lora_alpha`/`lora_dropout` alongside `use_lora=False`. Remove those fields (they don't apply to full-weight training) or set `use_lora=True`.

For how to read and interpret every metric (GRPO, multi-turn, DPO, SFT), see `references/metrics.md`.
