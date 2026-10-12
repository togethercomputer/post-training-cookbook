"""Evaluate a hosted/serverless chat model on DeepMath.

Same held-out split knobs as ``recipes/grpo/deepmath.py``. Hosted chat-completions
sanity check — not Generator step-0 parity.

Thinking is off: with it on, Qwen3.5 spends the whole budget reasoning and gets cut
before it emits ``\\boxed{}``, so the run reports truncation rather than accuracy.
To evaluate thinking-on, raise ``max_tokens`` to the training recipe's
``max_sample_tokens`` (15360) instead.

Requires TOGETHER_API_KEY. Leave TOGETHER_BASE_URL unset for public serverless.

Run: uv run python recipes/serverless_eval/deepmath.py
"""

from post_training_cookbook.serverless_eval import SamplingConfig, run_dataset_eval

if __name__ == "__main__":
    run_dataset_eval(
        model="Qwen/Qwen3.5-9B",
        dataset="deepmath",
        num_train=7000,
        num_val=200,
        sampling=SamplingConfig(temperature=0.0, max_tokens=2048, enable_thinking=False),
    )
