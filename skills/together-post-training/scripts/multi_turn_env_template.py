#!/usr/bin/env python3
"""Multi-turn (tool-use) environment scaffold for GRPO.

Copy this file and fill in the TODOs to train an agent that takes several turns
against an environment instead of producing one response. The training loop owns
tokenization and sampling; your `Env` owns conversation history, tool dispatch,
and when/how the episode ends and is scored.

The contract (see references/cookbook-guide.md → Multi-turn environments):
  - `async reset(task)`  -> initial message list (system + user)
  - `async step(message)` -> StepResult after consuming one assistant message
  - `tool_specs`          -> tool schemas for the chat template
  - `messages`            -> current full history
  - `async close()`       -> release resources (override if reset() spawns any)

`step()` receives the assistant message the model produced. If it carries
`tool_calls`, execute them and append `role="tool"` results, returning a
non-terminal StepResult. If it's a plain answer, score it and terminate.

Usage:
    export TOGETHER_API_KEY="..."
    export HF_TOKEN="..."
    python multi_turn_env_template.py
"""

from __future__ import annotations

import json
from typing import Any

from post_training_cookbook.envs.base import Env, StepResult, Task, ToolResult, tool_to_spec
from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.grpo.train import train_multi_turn
from post_training_cookbook.types import Message, ToolSpec

# ---------------------------------------------------------------------------
# TODO 1: Define your tool(s)
# ---------------------------------------------------------------------------
# A Tool exposes name / description / parameters_schema (JSON schema) and an
# async execute(arguments: str) -> ToolResult. `arguments` is the raw JSON
# string the model emitted; parse it yourself.


class LookupTool:
    """Example tool: read a value from the task's private fact table."""

    def __init__(self, facts: dict[str, str]) -> None:
        self._facts = facts

    @property
    def name(self) -> str:
        return "lookup"

    @property
    def description(self) -> str:
        return "Fetch a value from the task's fact table by exact key."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"key": {"type": "string", "description": "Fact key to retrieve."}},
            "required": ["key"],
        }

    async def execute(self, arguments: str) -> ToolResult:
        key = json.loads(arguments).get("key", "")
        if key not in self._facts:
            return ToolResult(content=f"lookup_error: unknown key {key!r}")
        # Set should_stop=True if a tool result should end the episode itself.
        return ToolResult(content=self._facts[key], should_stop=False)


# ---------------------------------------------------------------------------
# TODO 2: Implement your environment
# ---------------------------------------------------------------------------


class MyEnv(Env):
    """Minimal tool-use env: the model looks up facts, then gives a final answer."""

    SYSTEM_PROMPT = (
        "You can call tools to gather information. When you know the answer, reply "
        "with it inside \\boxed{...} and no tool call."
    )

    def __init__(self) -> None:
        self._messages: list[Message] = []
        self._task: Task | None = None
        self._tools: dict[str, Any] = {}

    async def reset(self, task: Task) -> list[Message]:
        self._task = task
        facts = task.metadata.get("facts", {})
        self._tools = {"lookup": LookupTool(facts)}  # TODO: register your tools
        self._messages = [
            Message(role="system", content=self.SYSTEM_PROMPT),
            Message(role="user", content=task.prompt),
        ]
        return list(self._messages)

    async def step(self, message: Message) -> StepResult:
        assert self._task is not None, "reset() must run before step()"
        self._messages.append(message)

        tool_calls = message.get("tool_calls", [])
        if not tool_calls:
            return self._score(message)  # no tool call -> final answer

        # Dispatch each tool call and append a tool-result message.
        stop = False
        for tc in tool_calls:
            name = tc["function"]["name"]
            tool = self._tools.get(name)
            result = (
                await tool.execute(tc["function"]["arguments"])
                if tool is not None
                else ToolResult(content=f"tool_error: unknown tool {name!r}")
            )
            self._messages.append(Message(role="tool", content=result.content, tool_call_id=tc["id"], name=name))
            stop = stop or result.should_stop

        return StepResult(
            terminated=stop,
            truncated=False,
            messages=list(self._messages),
            reward=0.0,
            metrics={"tool_calls": float(len(tool_calls))},
        )

    def _score(self, message: Message) -> StepResult:
        # TODO 3: score the final answer however your task demands.
        assert self._task is not None
        answer = message.get("content", "")
        correct = str(self._task.target).strip().lower() in answer.lower()
        reward = 1.0 if correct else 0.0
        return StepResult(
            terminated=True,
            truncated=False,
            messages=list(self._messages),
            reward=reward,
            metrics={"exact_match": reward},
        )

    @property
    def tool_specs(self) -> list[ToolSpec]:
        return [tool_to_spec(t) for t in self._tools.values()]

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    async def close(self) -> None:
        # Override if reset() opens connections / subprocesses / threads. Runs even
        # when an episode is truncated externally (context overflow, max-turns).
        return None


# ---------------------------------------------------------------------------
# TODO 4: Provide tasks (one per episode spec)
# ---------------------------------------------------------------------------

TASKS = [
    Task(
        prompt="What is the capital of the country in the fact table?",
        target="Paris",
        metadata={"facts": {"country": "France", "capital": "Paris"}},
    ),
    Task(
        prompt="Look up the mascot and report it.",
        target="otter",
        metadata={"facts": {"mascot": "otter"}},
    ),
]


def main() -> None:
    config = GRPOConfig(
        model="Qwen/Qwen3.8-27B",
        max_turns=6,  # >0 selects multi-turn mode
        batch_size=4,
        episodes_per_task=8,  # multi-turn analogue of group_size
        num_steps=200,
        max_model_length=2048,
        learning_rate=2e-5,
        wandb_project="grpo-my-env",
        sequential_validation=True,
    )
    train_multi_turn(
        config,
        tasks=TASKS,
        env_factory=lambda task: MyEnv(),  # fresh env per episode
        val_tasks=TASKS,
    )


if __name__ == "__main__":
    main()
