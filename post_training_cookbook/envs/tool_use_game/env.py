from __future__ import annotations

import ast
import json
import operator
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from post_training_cookbook.datasets.rewards import extract_boxed, normalize_numeric
from post_training_cookbook.envs.base import Env, StepResult, Task, Tool, ToolResult, tool_to_spec
from post_training_cookbook.types import Message, ToolSpec

_TASKS_FILE = Path(__file__).parent / "tasks.json"

# Paired blocks, then stray opening tags (e.g. unclosed `<think>...`).
_THINK_BLOCK = re.compile(
    r"<think\b[^>]*>.*?</think>",
    re.DOTALL | re.IGNORECASE,
)
_THINK_OPEN = re.compile(r"<think\b[^>]*>\s*", re.IGNORECASE)


def _load_tool_use_game(path: Path) -> tuple[str, tuple[Task, ...]]:
    raw = json.loads(path.read_text())
    system = str(raw["system"])
    tasks = tuple(
        Task(
            prompt=t["prompt"],
            target=t["target"],
            metadata={"facts": t.get("facts", {})},
        )
        for t in raw["tasks"]
    )
    return system, tasks


TOOL_USE_GAME_SYSTEM_PROMPT, TOOL_USE_GAME_TASKS = _load_tool_use_game(_TASKS_FILE)


@dataclass(frozen=True)
class ListFactKeysTool:
    """List key names for the current task's fact table (values stay private)."""

    facts: dict[str, str]

    @property
    def name(self) -> str:
        return "list_fact_keys"

    @property
    def description(self) -> str:
        return "List all key names in this task's private fact table. Does not return values—use lookup for that."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, arguments: str) -> ToolResult:
        raw = arguments.strip() or "{}"
        json.loads(raw)
        keys = sorted(self.facts.keys())
        if not keys:
            return ToolResult(content="(no keys)")
        return ToolResult(content="\n".join(keys))


@dataclass(frozen=True)
class LookupTool:
    """Read a value from a small in-memory fact table."""

    facts: dict[str, str]

    @property
    def name(self) -> str:
        return "lookup"

    @property
    def description(self) -> str:
        return (
            "Fetch a value from this task's private fact table by exact key. "
            "Use list_fact_keys if you need to see which keys exist."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Fact key to retrieve."},
            },
            "required": ["key"],
        }

    async def execute(self, arguments: str) -> ToolResult:
        args = json.loads(arguments)
        key = args["key"]
        if key not in self.facts:
            return ToolResult(content=f"lookup_error: unknown key {key!r}")
        return ToolResult(content=self.facts[key])


class CalculatorTool:
    """Evaluate a simple arithmetic expression."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Evaluate a basic arithmetic expression."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression using numbers and +, -, *, /, and parentheses.",
                },
            },
            "required": ["expression"],
        }

    async def execute(self, arguments: str) -> ToolResult:
        args = json.loads(arguments)
        try:
            value = _safe_eval(args["expression"])
        except ValueError as exc:
            return ToolResult(content=f"calculator_error: {exc}")
        if value.is_integer():
            return ToolResult(content=str(int(value)))
        return ToolResult(content=str(value))


class ToolUseGameEnv(Env):
    """Minimal in-memory tool-use environment.

    The episode begins with a system message and the user prompt. Assistant messages that
    contain tool calls dispatch those calls and append tool responses to the conversation.
    Assistant messages with no tool calls are treated as final answers and end the episode
    with a binary reward.
    """

    def __init__(self) -> None:
        self._messages: list[Message] = []
        self._task: Task | None = None
        self._tools: dict[str, Tool] = {}

    async def reset(self, task: Task) -> list[Message]:
        """Start a new episode from the provided task prompt."""
        self._task = task
        facts = task.metadata["facts"]
        self._tools = {
            "list_fact_keys": ListFactKeysTool(facts),
            "lookup": LookupTool(facts),
            "calculator": CalculatorTool(),
        }
        self._messages = [
            Message(role="system", content=TOOL_USE_GAME_SYSTEM_PROMPT),
            Message(role="user", content=task.prompt),
        ]
        return list(self._messages)

    async def step(self, message: Message) -> StepResult:
        """Append the assistant message, then either dispatch tools or score it."""
        if self._task is None:
            raise RuntimeError("reset() must be called before step().")

        self._messages.append(message)
        tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            return self._score_final_answer(message)

        return await self._dispatch_tool_calls(tool_calls)

    def _score_final_answer(self, message: Message) -> StepResult:
        assert self._task is not None
        matched = answer_matches_target(message.get("content", ""), self._task.target)
        reward = 1.0 if matched else 0.0
        return StepResult(
            terminated=True,
            truncated=False,
            messages=list(self._messages),
            reward=reward,
            metrics={"exact_match": reward},
        )

    async def _dispatch_tool_calls(self, tool_calls: list[dict]) -> StepResult:
        stop_requested = False
        tool_names: list[str] = []

        for tc in tool_calls:
            name = tc["function"]["name"]
            tool_names.append(name)
            tool = self._tools.get(name)
            result = (
                await tool.execute(tc["function"]["arguments"])
                if tool is not None
                else ToolResult(content=f"tool_error: unknown tool {name!r}")
            )
            self._messages.append(Message(role="tool", content=result.content, tool_call_id=tc["id"], name=name))
            stop_requested = stop_requested or result.should_stop

        return StepResult(
            terminated=stop_requested,
            truncated=False,
            messages=list(self._messages),
            reward=0.0,
            metrics={
                "tool_calls": float(len(tool_calls)),
                "unique_tools": float(len(set(tool_names))),
            },
        )

    @property
    def tool_specs(self) -> list[ToolSpec]:
        return [tool_to_spec(t) for t in self._tools.values()]

    @property
    def messages(self) -> list[Message]:
        """Current conversation history."""
        return list(self._messages)


def _normalize_answer(value: Any) -> str:
    return str(value).strip().lower()


def _is_numeric_target(target: str) -> bool:
    t = target.strip().replace(",", "")
    if not t:
        return False
    try:
        float(t)
    except ValueError:
        return False
    return True


def _strip_for_scoring(content: str) -> str:
    text = _THINK_BLOCK.sub("", content.strip())
    text = _THINK_OPEN.sub("", text).strip()
    return text


def answer_matches_target(content: str, target: str) -> bool:
    r"""Return True if the last ``\boxed{...}`` in the visible reply matches the target.

    Only the boxed region is considered; surrounding prose is ignored.
    """
    boxed = extract_boxed(_strip_for_scoring(content))
    if boxed is None:
        return False
    if _is_numeric_target(target):
        return normalize_numeric(boxed) == normalize_numeric(target)
    return _normalize_answer(boxed) == _normalize_answer(target)


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(expression: str) -> float:
    try:
        node = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid syntax") from exc
    return float(_eval_ast(node.body))


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        return float(_BINARY_OPERATORS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return float(_UNARY_OPERATORS[type(node.op)](_eval_ast(node.operand)))
    raise ValueError("unsupported expression")
