from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from post_training_cookbook.envs.base import Env, Task
from post_training_cookbook.envs.tool_use_game import ToolUseGameEnv
from post_training_cookbook.envs.tool_use_game.env import TOOL_USE_GAME_SYSTEM_PROMPT
from post_training_cookbook.types import Message, ToolCall, ToolCallFunction


@dataclass(frozen=True)
class EnvContractCase:
    """Concrete env instance plus the interactions used by contract tests."""

    factory: Callable[[], Env]
    task: Task
    tool_call_message: Message
    final_answer_message: Message
    expected_tool_content: str
    system_prompt: str
    expected_tool_call_id: str
    expected_tool_name: str
    expected_terminal_reward: float
    expected_terminal_metrics: dict[str, Any]


ENV_CASES = [
    pytest.param(
        EnvContractCase(
            factory=ToolUseGameEnv,
            task=Task(
                prompt="What is 7 * (8 + 1)?",
                target="63",
                metadata={"facts": {}},
            ),
            tool_call_message=Message(
                role="assistant",
                content="I should calculate this.",
                tool_calls=[
                    ToolCall(
                        type="function",
                        id="calc_1",
                        function=ToolCallFunction(
                            name="calculator",
                            arguments=json.dumps({"expression": "7 * (8 + 1)"}),
                        ),
                    )
                ],
            ),
            final_answer_message=Message(role="assistant", content=r"\boxed{63}"),
            expected_tool_content="63",
            system_prompt=TOOL_USE_GAME_SYSTEM_PROMPT,
            expected_tool_call_id="calc_1",
            expected_tool_name="calculator",
            expected_terminal_reward=1.0,
            expected_terminal_metrics={"exact_match": 1.0},
        ),
        id="tool_use_game",
    ),
]


@pytest.fixture(params=ENV_CASES)
def env_case(request) -> EnvContractCase:
    return request.param


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
