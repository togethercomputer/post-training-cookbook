# Post Training Cookbook

Training recipes for Together Post Training.

## Documentation

See the [Together Post Training documentation](https://docs.together.ai/docs/post-training) for quickstarts, concepts, the SDK reference, and deployment guides.

## Install

```bash
uv sync --locked
```

Set environment variables:

```bash
export TOGETHER_API_KEY=...
export HF_TOKEN=...           # for gated models
export WANDB_API_KEY=...      # optional
```

## Quick start

**New to the service?** Open a notebook:

- `notebooks/grpo_example.ipynb` — GRPO training on GSM8K
- `notebooks/sft_example.ipynb` — Supervised fine-tuning
- `notebooks/multi_lora_example.ipynb` — Multi-LoRA: two adapter sessions sharing one model resource

Add a notebook kernel with `uv sync --locked --group notebooks`, then open the file in an IDE.

**Running experiments?** Use a recipe:

```bash
# GRPO on GSM8K
uv run python recipes/grpo/gsm8k.py

# List each model's supported GPU types, modes, and context lengths
uv run python scripts/supported_models.py

# Provision a shared model resource and leave it running for others to attach to
uv run python scripts/multi_lora/launch.py

# Multi-LoRA hyperparameter sweep (provisions one shared resource, deletes it when done)
uv run python recipes/grpo/deepmath_sweep.py

# GRPO against an existing resource (attach as tenant)
MODEL_RESOURCES_ID=... python recipes/grpo/deepmath_shared.py

# List your model resources (to attach to one) or delete them to free GPUs
uv run python scripts/multi_lora/fetch.py
uv run python scripts/multi_lora/clean_up.py <id>

# GRPO on DeepMath
uv run python recipes/grpo/deepmath.py
uv run python recipes/grpo/deepmath-lv5.py

# GRPO on Game of 24
uv run python recipes/grpo/game24.py

# Multi-turn GRPO on the tool-use toy environment
uv run python recipes/grpo/tool_use_game.py

# SFT on No Robots
uv run python recipes/sft/no_robots.py

# DPO on HelpSteer3 (demo of compute_custom_forward_backward for custom losses)
uv run python recipes/dpo/helpsteer3.py

# Hosted-model eval (no training)
uv run python recipes/serverless_eval/gsm8k.py
uv run python recipes/serverless_eval/tool_use_game.py
```

Each recipe is a self-contained Python script with a `GRPOConfig` / `SFTConfig` / `DPOConfig` — edit the file directly to change hyperparameters.

**Model resources vs. sessions:** a *model resource* is the GPUs plus base weights — you may own it or just borrow it; a *session* is your LoRA adapter trained on top. `deepmath_sweep.py` provisions and owns a resource (deleting it when the sweep ends), `scripts/multi_lora/launch.py` provisions one and leaves it running, and `deepmath_shared.py` attaches as a tenant to one that already exists. Use `scripts/supported_models.py` to discover the GPU types available for each model. List the resources you own with `scripts/multi_lora/fetch.py` and delete them with `scripts/multi_lora/clean_up.py`. A recipe attaches by setting `model_resources_id` (and leaving `model` unset — the resource defines the base model); set `model` instead to provision a fresh resource.

## Adding a new dataset

Single-turn (prompt → response → reward):

1. Add a reward function in `post_training_cookbook/datasets/rewards.py`
2. Add a `DatasetConfig` entry in `post_training_cookbook/datasets/registry.py`
3. Create a recipe in `recipes/grpo/`

Multi-turn (tool use, interactive environments):

1. Implement an `Env` subclass under `post_training_cookbook/envs/` exposing tools and a per-step reward
2. Create a recipe in `recipes/grpo/` that calls `train_multi_turn` with a task list and `env_factory` (see `tool_use_game.py`)

## Deploy a trained model

Each inference checkpoint registers its weights in the model registry. Serve them on Dedicated Model Inference:

```bash
# Production: one adapter, merged into the base model when each replica starts
tg endpoints deploy <adapter_id> --endpoint my-rl-model --merge --config <config_id> --project <project_id>

# Comparison: attach several adapters to one multi-LoRA deployment, pick one per request
tg endpoints deploy <adapter_id> --endpoint my-sweep --attach-adapter

# FP4-served base models: if `estimate` accepts the adapter, prepare it, then deploy the prepared model
tg training fp4 estimate <adapter_id>          # price and expected hours; creates nothing
tg training fp4 create <adapter_id> --watch    # prints the prepared model ID when it completes
tg endpoints deploy <prepared_model_id> --endpoint my-fp4-model --config <config_id> --project <project_id>
```

`<adapter_id>` is `ckpt.registration.adapter.id`. For configs, the SDK flow, and FP4 preparation, see:

- [Deploy a checkpoint](https://docs.together.ai/docs/post-training/deployment)
- [Multi-LoRA deployment](https://docs.together.ai/docs/post-training/multi-lora-deployment)
- [Prepare for FP4 inference](https://docs.together.ai/docs/post-training/prepare-for-fp4-inference)

## Claude Code skills

This repo bundles a [Claude Code skill](skills/) that teaches the assistant the full post-training fine-tuning workflow (GRPO/SFT/DPO, custom losses, deployment). Install it with:

```bash
./skills/install.sh
```

See [`skills/README.md`](skills/README.md) for details.

## Tests

```bash
uv run pytest tests/
```
