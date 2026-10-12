from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple, Protocol, runtime_checkable

from post_training_cookbook.codecs.base import MessageCodec
from post_training_cookbook.types import Message, ToolSpec

logger = logging.getLogger(__name__)


class TruncationCause(str, Enum):
    """Why an episode ended early.

    ``CONTEXT_OVERFLOW`` and ``MAX_TURNS`` are length-budget exhaustions, and
    the only causes that trigger the DAPO soft-overlong reward penalty.
    ``ENV`` covers env-side decisions (turn-limit, tool-induced abort).
    ``PARSE_FAILURE`` covers codec parse errors on sampled tokens.
    """

    CONTEXT_OVERFLOW = "context_overflow"
    MAX_TURNS = "max_turns"
    ENV = "env"
    PARSE_FAILURE = "parse_failure"


@dataclass(frozen=True)
class Task:
    """Single multi-turn episode specification.

    `prompt` seeds the initial user turn, `target` carries the expected final
    answer or evaluation payload, and `metadata` stores environment-specific
    context such as lookup tables or difficulty labels.
    """

    prompt: str
    target: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Outcome of one environment step.

    `terminated` means the episode ended because of task logic, such as the
    model giving a final answer or a tool requesting a stop. `truncated` means
    the episode was cut short by external policy, such as a parse failure or a
    turn limit. `parse_failed` is set by the codec/env wrapper when the
    sampled tokens could not be parsed into a valid message; envs leave it
    as the default. `messages` always contains the full updated conversation
    history.
    """

    terminated: bool
    truncated: bool
    messages: list[Message]
    reward: float
    metrics: dict[str, Any] = field(default_factory=dict)
    parse_failed: bool = False

    @property
    def done(self) -> bool:
        """Whether the episode ended for any reason."""
        return self.terminated or self.truncated


@dataclass
class ToolResult:
    """Result returned by an executed tool call."""

    content: str
    should_stop: bool = False


@runtime_checkable
class Tool(Protocol):
    """Executable tool exposed to the model through a chat template."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def parameters_schema(self) -> dict[str, Any]: ...

    async def execute(self, arguments: str) -> ToolResult: ...


def tool_to_spec(tool: Tool) -> ToolSpec:
    """Convert an executable tool into the schema shape used by codecs."""
    return ToolSpec(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters_schema,
    )


class Env(ABC):
    """Async multi-turn environment over structured messages.

    The env owns conversation history, tool dispatch, and task-completion
    semantics. The training loop is responsible for tokenization and sampling
    through a `MessageCodec`.
    """

    @abstractmethod
    async def reset(self, task: Task) -> list[Message]:
        """Start a new episode and return the initial message history."""

    @abstractmethod
    async def step(self, message: Message) -> StepResult:
        """Consume one assistant message and return the updated episode state."""

    @property
    @abstractmethod
    def tool_specs(self) -> list[ToolSpec]:
        """Tool schemas for codec prompt rendering, derived from the env's own tools."""

    @property
    @abstractmethod
    def messages(self) -> list[Message]:
        """Current full conversation history."""

    async def close(self) -> None:
        """Release any resources held by the env (threads, connections, subprocesses).

        Default is a no-op. Envs that spawn background work during ``reset``
        must override this to guarantee cleanup when the episode ends — including
        external truncation (context overflow, max-turns, exceptions), where the
        env never sees a terminal ``step()``.
        """
        return None


class PolicySpan(NamedTuple):
    """A run of generated tokens attributed to one policy version.

    Declared here, not reused from the training SDK, so the Env contract stays
    importable without it.
    """

    start_token: int
    version: int


class Action(NamedTuple):
    """A single assistant turn's generated tokens and their log-probabilities.

    ``policy_segments`` is empty for rollouts that did not come from a training
    sampler; see https://docs.together.ai/docs/post-training/monitoring for when it holds more than one span.

    ``routed_experts_key`` identifies the generator routing capture for this
    turn's prompt and completion, when replay is enabled.
    """

    tokens: list[int]
    logprobs: list[float]
    policy_segments: tuple[PolicySpan, ...] = ()
    routed_experts_key: str | None = None


@dataclass(kw_only=True)
class CompletedEpisode:
    """Record of a completed episode, whatever drove it.

    Carries no ``Env`` reference so the env can be GC'd and the record can be
    safely serialized. ``actions`` and ``tool_specs`` stay empty for rollouts
    with nothing to train on — an endpoint returns structured messages, not
    sampled tokens.
    """

    task: Task
    messages: list[Message]
    reward: float
    terminated: bool
    truncation_cause: TruncationCause | None
    num_turns: int
    actions: list[Action] = field(default_factory=list)
    tool_specs: list[ToolSpec] = field(default_factory=list)
    num_context_assistants: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        """True iff the episode ended for any non-terminal reason."""
        return self.truncation_cause is not None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON export. Actions are omitted (tokens/logprobs are bulky)."""
        return {
            "task": {"prompt": self.task.prompt, "target": str(self.task.target)},
            "messages": [dict(m) for m in self.messages],
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "truncation_cause": self.truncation_cause.value if self.truncation_cause else None,
            "num_turns": self.num_turns,
            "num_actions": len(self.actions),
            "metrics": self.metrics,
        }


def classify_truncation(result: StepResult) -> TruncationCause | None:
    """Map an env's terminal ``StepResult`` to a truncation cause (or None for clean termination)."""
    if result.terminated:
        return None
    if result.parse_failed:
        return TruncationCause.PARSE_FAILURE
    return TruncationCause.ENV


async def close_quietly(env: Env) -> None:
    """Run ``env.close()`` and log — never raise — on failure."""
    try:
        await env.close()
    except Exception as exc:
        logger.warning("env.close() raised %s: %s", type(exc).__name__, exc)


EnvFactory = Callable[[Task], Env]
"""Creates a fresh Env instance for a given task."""


async def step_env(
    codec: MessageCodec,
    env: Env,
    tokens: list[int],
    *,
    failed_parse_reward: float = -0.1,
) -> StepResult:
    """Parse sampled tokens and step the env, truncating on parse failure.

    Parse failures are treated as external truncations rather than task
    terminations. The partially parsed assistant message is still appended to
    the returned history so callers can inspect what failed. The result's
    ``parse_failed`` flag distinguishes parse-induced truncations from
    env-driven ones.
    """
    message, ok = codec.parse_response(tokens, tools=env.tool_specs)
    if not ok:
        return StepResult(
            terminated=False,
            truncated=True,
            messages=[*env.messages, message],
            reward=failed_parse_reward,
            parse_failed=True,
        )
    return await env.step(message)
