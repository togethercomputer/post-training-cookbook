"""Async multi-turn GRPO on the tool-use game — edit config values below and run."""

from post_training_cookbook.envs.tool_use_game import TOOL_USE_GAME_TASKS, ToolUseGameEnv
from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.grpo.train import train_multi_turn

tasks = list(TOOL_USE_GAME_TASKS)

config = GRPOConfig(
    model="Qwen/Qwen3.8-27B",
    num_generator_replicas=2,  # Increase to provision more generation capacity.
    batch_size=4,
    episodes_per_task=8,
    num_steps=50,
    max_steps_off_policy=1,
    validation_interval=10,
    max_turns=6,
    max_model_length=2048,
    learning_rate=2e-5,
    wandb_project="grpo-tool-use-game-async",
    sequential_validation=True,
    max_sampling_attempts=2,
)

if __name__ == "__main__":
    train_multi_turn(
        config,
        tasks=tasks,
        env_factory=lambda task: ToolUseGameEnv(),
        val_tasks=tasks,
    )
