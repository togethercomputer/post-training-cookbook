"""Evaluate a hosted/serverless chat model on GSM8K.

Same held-out split knobs as ``recipes/grpo/gsm8k.py``. Hosted chat-completions
sanity check — not Generator step-0 parity.

Thinking is off: 768 tokens is a direct-answer budget, and Qwen3.5 with thinking on
spends it reasoning and gets cut before the answer. Raise ``max_tokens`` first if
you turn it back on.

Requires TOGETHER_API_KEY. Leave TOGETHER_BASE_URL unset for public serverless.

Run: uv run python recipes/serverless_eval/gsm8k.py
"""

from post_training_cookbook.serverless_eval import SamplingConfig, run_dataset_eval

if __name__ == "__main__":
    run_dataset_eval(
        model="Qwen/Qwen3.5-9B",
        dataset="gsm8k",
        num_train=4000,
        num_val=200,
        sampling=SamplingConfig(temperature=0.0, max_tokens=768, enable_thinking=False),
    )
