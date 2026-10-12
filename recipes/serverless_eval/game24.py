"""Evaluate a hosted/serverless chat model on Game24.

Same held-out split knobs as ``recipes/grpo/game24.py``. Hosted chat-completions
sanity check — not Generator step-0 parity. Shaped ``reward_mean``; true 0/1
score is ``accuracy`` on the printed ``metrics:`` line.

Thinking is off: with it on, Qwen3.5 exhausts the token budget on the search and is
cut before it emits ``\\boxed{}``, which drags the score to near zero regardless
of ability.

Requires TOGETHER_API_KEY. Leave TOGETHER_BASE_URL unset for public serverless.

Run: uv run python recipes/serverless_eval/game24.py
"""

from post_training_cookbook.serverless_eval import SamplingConfig, run_dataset_eval

if __name__ == "__main__":
    run_dataset_eval(
        model="Qwen/Qwen3.5-9B",
        dataset="game24",
        num_train=1200,
        num_val=100,
        sampling=SamplingConfig(temperature=0.0, max_tokens=1536, enable_thinking=False),
    )
