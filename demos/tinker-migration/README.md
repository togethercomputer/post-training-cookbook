# Tinker compatibility demos

Standalone training examples using Together Post Training's Tinker-compatible API
and `tinker_cookbook` renderers and dataset utilities:

- `grpo_gsm8k.py` — GRPO on GSM8K, based on `tinker_cookbook/recipes/rl_loop.py`.
- `grpo_gsm8k_async.py` — GRPO on GSM8K with sampling and training overlapping by one step.
- `sft_no_robots.py` — SFT on No Robots, based on `tinker_cookbook/recipes/sl_loop.py`.

The scripts use `import together.post_training.tinker as tinker`. This demonstrates
how Tinker-style training code runs on Together with the Together compatibility shim.

From the repository root, install the demo dependencies and set your API key:

```bash
uv sync --locked --group tinker
export TOGETHER_API_KEY=...
export HF_TOKEN=...  # for gated models
```

Run a demo:

```bash
uv run --group tinker python demos/tinker-migration/grpo_gsm8k.py
uv run --group tinker python demos/tinker-migration/grpo_gsm8k_async.py
uv run --group tinker python demos/tinker-migration/sft_no_robots.py
```

Use `--help` to see each script's options. Set `--wandb-project` to log metrics
to Weights & Biases.
