from __future__ import annotations

import pytest

from post_training_cookbook.envs.base import StepResult, step_env
from post_training_cookbook.types import Message, ToolSpec


@pytest.mark.anyio
async def test_tool_specs_available_after_reset(env_case):
    env = env_case.factory()
    await env.reset(env_case.task)

    specs = env.tool_specs
    assert len(specs) > 0
    for spec in specs:
        assert spec["name"]
        assert spec["description"]
        assert "type" in spec["parameters"]


@pytest.mark.anyio
async def test_reset_returns_initial_history(env_case):
    env = env_case.factory()

    messages = await env.reset(env_case.task)

    assert messages == env.messages
    assert messages == [
        Message(role="system", content=env_case.system_prompt),
        Message(role="user", content=env_case.task.prompt),
    ]


@pytest.mark.anyio
async def test_step_appends_assistant_and_tool_messages(env_case):
    env = env_case.factory()
    await env.reset(env_case.task)

    result = await env.step(env_case.tool_call_message)

    assert not result.done
    assert result.reward == 0.0
    assert result.messages == env.messages
    assert result.messages[-2] == env_case.tool_call_message
    assert result.messages[-1]["role"] == "tool"
    assert result.messages[-1]["tool_call_id"] == env_case.expected_tool_call_id
    assert result.messages[-1]["name"] == env_case.expected_tool_name
    assert result.messages[-1]["content"] == env_case.expected_tool_content


@pytest.mark.anyio
async def test_final_answer_terminates_episode(env_case):
    env = env_case.factory()
    await env.reset(env_case.task)
    await env.step(env_case.tool_call_message)

    result = await env.step(env_case.final_answer_message)

    assert result == StepResult(
        terminated=True,
        truncated=False,
        messages=env.messages,
        reward=env_case.expected_terminal_reward,
        metrics=env_case.expected_terminal_metrics,
    )
    assert result.done


@pytest.mark.anyio
async def test_step_result_done_matches_termination_flags(env_case):
    env = env_case.factory()
    await env.reset(env_case.task)

    in_progress = await env.step(env_case.tool_call_message)

    assert in_progress.done is False
    assert (
        StepResult(
            terminated=False,
            truncated=True,
            messages=in_progress.messages,
            reward=-0.1,
        ).done
        is True
    )


class _FailingParseCodec:
    def parse_response(self, tokens: list[int], *, tools: list[ToolSpec] | None = None) -> tuple[Message, bool]:
        return Message(role="assistant", content="broken <tool_call>"), False


@pytest.mark.anyio
async def test_step_env_truncates_on_parse_failure(env_case):
    env = env_case.factory()
    await env.reset(env_case.task)

    result = await step_env(_FailingParseCodec(), env, [1, 2, 3], failed_parse_reward=-0.25)

    assert result.parse_failed is True
    assert result.terminated is False
    assert result.truncated is True
    assert result.done is True
    assert result.reward == -0.25
    assert result.messages == [
        Message(role="system", content=env_case.system_prompt),
        Message(role="user", content=env_case.task.prompt),
        Message(role="assistant", content="broken <tool_call>"),
    ]
