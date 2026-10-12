"""Testing that forked jinja templates generally match upstream."""

from __future__ import annotations

import pytest

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
]


EQUIVALENCE_SCENARIOS: list[dict] = [
    {
        "id": "single_user_no_genprompt",
        "messages": [{"role": "user", "content": "hello"}],
        "kwargs": {"add_generation_prompt": False},
    },
    {
        "id": "single_user_with_genprompt",
        "messages": [{"role": "user", "content": "hello"}],
        "kwargs": {"add_generation_prompt": True},
    },
    {
        "id": "sys_user_assistant_user_no_reasoning",
        "messages": [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ],
        "kwargs": {"add_generation_prompt": True},
    },
    {
        "id": "trailing_assistant_with_reasoning_no_genprompt",
        "messages": [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "reasoning_content": "deliberation", "content": "a1"},
        ],
        "kwargs": {"add_generation_prompt": False},
    },
    {
        "id": "trailing_assistant_with_reasoning_with_genprompt",
        "messages": [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "reasoning_content": "deliberation", "content": "a1"},
        ],
        "kwargs": {"add_generation_prompt": True},
    },
    {
        "id": "tool_call_then_response_then_user",
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
                    }
                ],
            },
            {"role": "tool", "content": '{"temp": 20}'},
            {"role": "user", "content": "thanks"},
        ],
        "kwargs": {"add_generation_prompt": True},
    },
    {
        "id": "multiple_tool_calls_one_assistant",
        "messages": [
            {"role": "user", "content": "two cities?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
                    },
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": {"city": "London"}},
                    },
                ],
            },
            {"role": "tool", "content": '{"temp": 20}'},
            {"role": "user", "content": "and?"},
        ],
        "kwargs": {"add_generation_prompt": True},
    },
    {
        "id": "consecutive_tool_responses",
        "messages": [
            {"role": "user", "content": "two cities?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
                    },
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": {"city": "London"}},
                    },
                ],
            },
            {"role": "tool", "content": '{"temp": 20}'},
            {"role": "tool", "content": '{"temp": 15}'},
            {"role": "user", "content": "thanks"},
        ],
        "kwargs": {"add_generation_prompt": True},
    },
    {
        "id": "enable_thinking_false_with_genprompt",
        "messages": [
            {"role": "user", "content": "hello"},
        ],
        "kwargs": {"add_generation_prompt": True, "enable_thinking": False},
    },
    {
        "id": "with_tools_provided",
        "messages": [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
        ],
        "kwargs": {"add_generation_prompt": True, "tools": _TOOLS},
    },
]


@pytest.mark.parametrize("scenario", EQUIVALENCE_SCENARIOS, ids=lambda s: s["id"])
def test_fork_matches_upstream_when_diff_path_inactive(codec_preserve_thinking, scenario):
    fork_str = getattr(codec_preserve_thinking, "_chat_template", None)
    if fork_str is None:
        pytest.skip("codec drives preserve_thinking natively; no fork to diff against upstream")

    tokenizer = codec_preserve_thinking._tokenizer

    upstream = tokenizer.apply_chat_template(scenario["messages"], tokenize=False, **scenario["kwargs"])
    forked = tokenizer.apply_chat_template(
        scenario["messages"], chat_template=fork_str, tokenize=False, **scenario["kwargs"]
    )

    assert forked == upstream, (
        f"fork drifted from upstream on scenario "
        f"{scenario['id']!r}.\n--- upstream ---\n{upstream!r}\n--- forked ---\n{forked!r}"
    )
