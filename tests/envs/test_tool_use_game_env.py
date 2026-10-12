from __future__ import annotations

import json

import pytest

from post_training_cookbook.envs.base import Task
from post_training_cookbook.envs.tool_use_game import ToolUseGameEnv, answer_matches_target
from post_training_cookbook.types import Message


@pytest.fixture
def tool_use_game_env() -> ToolUseGameEnv:
    return ToolUseGameEnv()


@pytest.mark.anyio
async def test_executes_multiple_tool_calls_in_order(tool_use_game_env: ToolUseGameEnv):
    env = tool_use_game_env
    task = Task(
        prompt="Synthetic lookup task.",
        target="0",
        metadata={
            "facts": {
                "synthetic_fact_alpha": "68",
                "synthetic_fact_beta": "357",
            }
        },
    )
    await env.reset(task)

    message = Message(
        role="assistant",
        content="I should look up both values first.",
        tool_calls=[
            {
                "type": "function",
                "id": "lookup_1",
                "function": {"name": "lookup", "arguments": json.dumps({"key": "synthetic_fact_alpha"})},
            },
            {
                "type": "function",
                "id": "lookup_2",
                "function": {"name": "lookup", "arguments": json.dumps({"key": "synthetic_fact_beta"})},
            },
        ],
    )

    result = await env.step(message)

    assert [msg["content"] for msg in result.messages[-2:]] == ["68", "357"]
    assert [msg["tool_call_id"] for msg in result.messages[-2:]] == ["lookup_1", "lookup_2"]


@pytest.mark.anyio
async def test_list_fact_keys_returns_sorted_keys(tool_use_game_env: ToolUseGameEnv):
    env = tool_use_game_env
    task = Task(
        prompt="Synthetic discovery task.",
        target="x",
        metadata={
            "facts": {
                "alpha_one": "1",
                "alpha_two": "2",
                "beta": "3",
            }
        },
    )
    await env.reset(task)

    result = await env.step(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "type": "function",
                    "id": "list_1",
                    "function": {"name": "list_fact_keys", "arguments": "{}"},
                },
            ],
        )
    )
    assert result.messages[-1]["content"] == "alpha_one\nalpha_two\nbeta"


@pytest.mark.parametrize(
    ("response", "target", "expected"),
    [
        (r"\boxed{2}", "2", True),
        (r"The answer is \boxed{2}.", "2", True),
        ("63", "63", False),
        ("The result of 100001 minus 99999 is 2.", "2", False),
        ("The result of $100001 - 99999$ is $2$.", "2", False),
        ("<think>\nThe result is 2.", "2", False),
        ("The answer is 3.", "2", False),
        (r"\boxed{3}", "2", False),
        (r"\boxed{X9K2M}", "X9K2M", True),
        (r"The parcel code is \boxed{X9K2M}.", "X9K2M", True),
        ("The parcel code is X9K2M.", "X9K2M", False),
        ("The parcel code is WRONG.", "X9K2M", False),
    ],
)
def test_answer_matches_target_requires_boxed(response: str, target: str, expected: bool):
    assert answer_matches_target(response, target) is expected
