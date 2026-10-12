"""Evaluate a hosted/serverless chat model on the tool-use game.

Thinking is on here — the game needs it — so ``max_tokens`` is a per-turn budget
generous enough to think and still call a tool. Watch ``eval/length_capped`` if you
lower it.

Requires TOGETHER_API_KEY. Leave TOGETHER_BASE_URL unset for public serverless.

Run: uv run python recipes/serverless_eval/tool_use_game.py
"""

from post_training_cookbook.envs.tool_use_game import TOOL_USE_GAME_TASKS, ToolUseGameEnv
from post_training_cookbook.serverless_eval import SamplingConfig, run_and_report

if __name__ == "__main__":
    run_and_report(
        model="Qwen/Qwen3.5-9B",
        tasks=list(TOOL_USE_GAME_TASKS),
        env_factory=lambda _task: ToolUseGameEnv(),
        max_turns=6,
        sampling=SamplingConfig(temperature=0.0, max_tokens=4096, enable_thinking=True),
        label="tool-use-game",
    )
