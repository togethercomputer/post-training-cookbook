"""Tests for the single-turn dataset → Env adapter."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import patch

import pytest

from post_training_cookbook.datasets.registry import DatasetConfig
from post_training_cookbook.envs.base import Task
from post_training_cookbook.serverless_eval.single_turn import SingleTurnEnv, _dataset_tasks, run_dataset_eval
from post_training_cookbook.types import Message


def _config(
    *,
    compute_reward: Callable[[str, object], float] | None = None,
    compute_metric: Callable[[str, object], float] | None = None,
    few_shot: list[Message] | None = None,
) -> DatasetConfig:
    return DatasetConfig(
        hf_name="dummy",
        hf_config=None,
        hf_split="train",
        format_question=lambda row: f"Q: {row['q']}",
        extract_target=lambda row: row["a"],
        compute_reward=compute_reward or (lambda pred, target: 1.0 if pred == target else 0.0),
        compute_metric=compute_metric,
        few_shot_messages=few_shot or [],
    )


def test_dataset_tasks_uses_config_formatting():
    tasks = _dataset_tasks(_config(), [{"q": "2+2", "a": "4"}, {"q": "3+3", "a": "6"}])
    assert [(t.prompt, t.target) for t in tasks] == [("Q: 2+2", "4"), ("Q: 3+3", "6")]


@pytest.mark.asyncio
async def test_reset_seeds_few_shot_then_user_prompt():
    few_shot = [
        Message(role="user", content="example?"),
        Message(role="assistant", content="example."),
    ]
    env = SingleTurnEnv(_config(few_shot=few_shot))

    messages = await env.reset(Task(prompt="solve this", target="x"))

    assert messages == [*few_shot, {"role": "user", "content": "solve this"}]
    assert env.tool_specs == []


@pytest.mark.asyncio
async def test_step_scores_and_terminates():
    env = SingleTurnEnv(_config())
    await env.reset(Task(prompt="Q", target="4"))

    result = await env.step(Message(role="assistant", content="4"))

    assert result.terminated and not result.truncated
    assert result.reward == 1.0
    assert result.metrics == {"accuracy": 1.0}


@pytest.mark.asyncio
async def test_accuracy_falls_back_to_reward_without_compute_metric():
    env = SingleTurnEnv(_config())
    await env.reset(Task(prompt="Q", target="4"))
    result = await env.step(Message(role="assistant", content="wrong"))
    assert result.metrics["accuracy"] == 0.0


@pytest.mark.asyncio
async def test_compute_metric_overrides_shaped_reward():
    cfg = _config(
        compute_reward=lambda _pred, _target: 0.4,
        compute_metric=lambda pred, target: 1.0 if pred == target else 0.0,
    )
    env = SingleTurnEnv(cfg)
    await env.reset(Task(prompt="Q", target="4"))
    result = await env.step(Message(role="assistant", content="not 4"))
    assert result.reward == 0.4
    assert result.metrics["accuracy"] == 0.0


def test_run_dataset_eval_no_val_raises():
    with (
        patch.dict("post_training_cookbook.serverless_eval.single_turn.DATASETS", {"dummy": _config()}),
        patch("post_training_cookbook.serverless_eval.single_turn.load_grpo_dataset", return_value=(None, None)),
    ):
        with pytest.raises(ValueError, match="No validation rows"):
            run_dataset_eval(model="m", dataset="dummy", num_train=10, num_val=0)


def test_run_dataset_eval_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        run_dataset_eval(model="m", dataset="nope", num_train=10, num_val=1)


@pytest.mark.asyncio
async def test_step_before_reset_raises():
    with pytest.raises(RuntimeError, match="reset\\(\\) must be called"):
        await SingleTurnEnv(_config()).step(Message(role="assistant", content="hi"))
