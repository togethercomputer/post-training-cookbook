"""Shared fakes for GRPO tests (codecs, envs, session clients, config helper)."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Coroutine, Iterable, Iterator
from types import SimpleNamespace
from typing import Any

from together.post_training import ModelInput, SampledSequence, SampleResult, SamplingParams

from post_training_cookbook.envs.base import Env, StepResult, Task
from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.types import Message, ToolSpec


def _msg(role: str, content: str = "") -> Message:
    m: Message = {"role": role}
    if content:
        m["content"] = content
    return m


def _base_config(**kwargs: Any) -> GRPOConfig:
    """GRPOConfig with multi-turn-friendly defaults; ``max_sample_tokens`` is
    left unset so it derives from ``max_model_length`` per the config's rules."""
    defaults: dict = {
        "model": "test-model",
        "batch_size": 1,
        "max_turns": 4,
        "max_model_length": 4096,
        "episodes_per_task": 1,
    }
    defaults.update(kwargs)
    return GRPOConfig(**defaults)


class FakeCodec:
    """Prefix-stable codec used across GRPO tests.

    Each message encodes as ``[role_token, *content_tokens, END_TOKEN]``, and
    ``add_generation_prompt`` appends the assistant role token. This mirrors
    real chat templates (e.g. Llama3) so ``flatten_episode`` can diff against
    the accumulated prefix.
    """

    ROLE_TOKENS = {"system": 100, "user": 200, "assistant": 300, "tool": 400}
    END_TOKEN = 500

    def __init__(self, parse_ok: Callable[[list[int]], bool] | None = None) -> None:
        self._parse_ok = parse_ok or (lambda _tokens: True)

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens if t < 128)

    def encode_messages(self, messages, *, tools=None, add_generation_prompt=False) -> list[int]:
        out: list[int] = []
        for msg in messages:
            out.append(self.ROLE_TOKENS[msg["role"]])
            out.extend(self.encode(msg.get("content", "")))
            out.append(self.END_TOKEN)
        if add_generation_prompt:
            out.append(self.ROLE_TOKENS["assistant"])
        return out

    def parse_response(self, tokens: list[int], *, tools: list[ToolSpec] | None = None) -> tuple[Message, bool]:
        return Message(role="assistant", content=self.decode(tokens)), self._parse_ok(tokens)


class BreakingCodec(FakeCodec):
    """Codec where re-encoding mutates assistant content (simulates thinking-strip)."""

    def __init__(self) -> None:
        super().__init__()
        self._call_count = 0

    def encode_messages(self, messages, *, tools=None, add_generation_prompt=False) -> list[int]:
        self._call_count += 1
        out: list[int] = []
        for msg in messages:
            out.append(self.ROLE_TOKENS[msg["role"]])
            content = msg.get("content", "")
            if msg["role"] == "assistant" and self._call_count > 1:
                out.extend(t + 1000 for t in self.encode(content))
            else:
                out.extend(self.encode(content))
            out.append(self.END_TOKEN)
        if add_generation_prompt:
            out.append(self.ROLE_TOKENS["assistant"])
        return out


class OneStepEnv(Env):
    """Terminates after one successful assistant step with fixed reward."""

    def __init__(self, reward: float = 0.5, gate: asyncio.Event | None = None) -> None:
        self._reward = reward
        self._messages: list[Message] = []
        self._gate = gate

    async def reset(self, task: Task) -> list[Message]:
        if self._gate is not None:
            await self._gate.wait()
        self._messages = [_msg("user", task.prompt)]
        return self._messages

    async def step(self, message: Message) -> StepResult:
        self._messages = [*self._messages, dict(message)]
        return StepResult(
            terminated=True,
            truncated=False,
            messages=self._messages,
            reward=self._reward,
            metrics={},
        )

    @property
    def tool_specs(self) -> list[ToolSpec]:
        return []

    @property
    def messages(self) -> list[Message]:
        return self._messages


class NeverDoneEnv(Env):
    """Env that never terminates; surfaces max-turns length truncation."""

    def __init__(self) -> None:
        self._messages: list[Message] = []

    async def reset(self, task: Task) -> list[Message]:
        self._messages = [_msg("user", task.prompt)]
        return list(self._messages)

    async def step(self, message: Message) -> StepResult:
        self._messages = [*self._messages, dict(message), _msg("tool", "noop")]
        return StepResult(terminated=False, truncated=False, messages=list(self._messages), reward=0.0)

    @property
    def tool_specs(self) -> list[ToolSpec]:
        return []

    @property
    def messages(self) -> list[Message]:
        return self._messages


class ParseFailEnv(Env):
    """Env whose step() is never called — codec parse_response fails first."""

    def __init__(self) -> None:
        self._messages: list[Message] = []

    async def reset(self, task: Task) -> list[Message]:
        self._messages = [_msg("user", task.prompt)]
        return self._messages

    async def step(self, message: Message) -> StepResult:
        raise AssertionError("step should not be called on parse failure")

    @property
    def tool_specs(self) -> list[ToolSpec]:
        return []

    @property
    def messages(self) -> list[Message]:
        return self._messages


class ScriptedEnv(Env):
    """Test env that delegates to caller-provided ``step_fn``."""

    def __init__(
        self,
        initial_messages: list[Message],
        step_fn: Callable[[int, Message, list[Message]], StepResult],
    ) -> None:
        self._messages: list[Message] = list(initial_messages)
        self._step_fn = step_fn
        self._turn = 0

    async def reset(self, task: Task) -> list[Message]:
        return list(self._messages)

    async def step(self, message: Message) -> StepResult:
        self._turn += 1
        result = self._step_fn(self._turn, message, self._messages)
        self._messages = list(result.messages)
        return result

    @property
    def tool_specs(self) -> list[ToolSpec]:
        return []

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)


def _sample_result(tokens: list[int], *, routed: bool = False) -> SampleResult:
    """One prompt's completions: a single sequence over ``tokens``."""
    return SampleResult(
        policy_segments=[],
        sequences=[
            SampledSequence(
                tokens=list(tokens),
                logprobs=[-0.1] * len(tokens),
                stop_reason="STOP_REASON_STOP",
                prompt_cache_hit_tokens=0,
                routed_experts_key=f"routing/test/{uuid.uuid4().hex}" if routed else None,
            )
        ],
    )


class _FakeGenerator:
    """The ``session.generator`` surface, with the real ``Generator`` signatures.

    Deliberately a separate object rather than the session itself: production code
    must reach generation through ``session.generator``, and a self-aliasing fake
    would let a call that skipped the sub-client still pass.
    """

    def __init__(self, session: MockSession) -> None:
        self._session = session

    def sample(
        self,
        prompt: ModelInput,
        num_samples: int | None = None,
        sampling_params: SamplingParams | None = None,
        *,
        prompt_logprobs: bool | None = None,
        return_routed_experts: bool | None = None,
        timeout: float | None = None,
        interval: float = 0.5,
    ) -> SampleResult:
        return self._session.run(
            self.sample_async(
                prompt,
                num_samples,
                sampling_params,
                prompt_logprobs=prompt_logprobs,
                return_routed_experts=return_routed_experts,
                timeout=timeout,
                interval=interval,
            )
        )

    def sample_batch(
        self,
        prompts: Iterable[ModelInput],
        num_samples: int | None = None,
        sampling_params: SamplingParams | None = None,
        *,
        prompt_logprobs: bool | None = None,
        return_routed_experts: bool | None = None,
        timeout: float | None = None,
        interval: float = 0.5,
    ) -> list[SampleResult]:
        return self._session.run(
            self.sample_batch_async(
                prompts,
                num_samples,
                sampling_params,
                prompt_logprobs=prompt_logprobs,
                return_routed_experts=return_routed_experts,
                timeout=timeout,
                interval=interval,
            )
        )

    async def sample_async(
        self,
        prompt: ModelInput,
        num_samples: int | None = None,
        sampling_params: SamplingParams | None = None,
        *,
        prompt_logprobs: bool | None = None,
        return_routed_experts: bool | None = None,
        timeout: float | None = None,
        interval: float = 0.5,
    ) -> SampleResult:
        results = await self.sample_batch_async(
            [prompt],
            num_samples,
            sampling_params,
            prompt_logprobs=prompt_logprobs,
            return_routed_experts=return_routed_experts,
            timeout=timeout,
            interval=interval,
        )
        return results[0]

    async def sample_batch_async(
        self,
        prompts: Iterable[ModelInput],
        num_samples: int | None = None,
        sampling_params: SamplingParams | None = None,
        *,
        prompt_logprobs: bool | None = None,
        return_routed_experts: bool | None = None,
        timeout: float | None = None,
        interval: float = 0.5,
    ) -> list[SampleResult]:
        self._session.record_sample_call(num_samples, sampling_params)
        self._session.last_return_routed_experts = return_routed_experts
        routed = bool(return_routed_experts) and self._session.returns_routing
        return [_sample_result(self._session.next_tokens(), routed=routed) for _ in prompts]


class _FakeTrainer:
    """The ``session.trainer`` surface, with the real ``Trainer`` signatures.

    Separate from the session for the same reason as :class:`_FakeGenerator`, and
    records every call so tests can assert on what the loop submitted.
    """

    def __init__(self) -> None:
        self.forward_backward_calls: list[dict[str, Any]] = []
        self.optim_step_calls: list[dict[str, Any]] = []
        self.weights_sync_calls: list[dict[str, Any]] = []

    def forward_backward(
        self,
        *,
        samples: Iterable[Any],
        loss: Any,
        timeout: float | None = None,
        interval: float = 0.5,
    ) -> SimpleNamespace:
        self.forward_backward_calls.append({"samples": list(samples), "loss": loss, "timeout": timeout})
        return SimpleNamespace(loss=0.5, metrics={"grad_norm": 1.0})

    def optim_step(
        self,
        *,
        adam_params: Any | None = None,
        muon_params: Any | None = None,
        timeout: float | None = 300.0,
        interval: float = 0.5,
    ) -> SimpleNamespace:
        self.optim_step_calls.append(
            {
                "adam_params": adam_params,
                "muon_params": muon_params,
                "timeout": timeout,
            }
        )
        return SimpleNamespace()

    def weights_sync(
        self,
        *,
        weight_sync_type: str,
        timeout: float | None = 300.0,
        interval: float = 0.5,
    ) -> SimpleNamespace:
        self.weights_sync_calls.append(
            {
                "weight_sync_type": weight_sync_type,
                "timeout": timeout,
            }
        )
        return SimpleNamespace()

    async def forward_backward_async(self, **kwargs: Any) -> SimpleNamespace:
        return self.forward_backward(**kwargs)

    async def optim_step_async(self, **kwargs: Any) -> SimpleNamespace:
        return self.optim_step(**kwargs)

    async def weights_sync_async(self, **kwargs: Any) -> SimpleNamespace:
        return self.weights_sync(**kwargs)


class MockSession:
    """SessionClient stub: sub-client surfaces reachable only via ``.generator``/``.trainer``."""

    session_id = "test0000-0000-0000-0000-000000000000"

    def __init__(self, response_tokens: list[int] | None = None) -> None:
        self._tokens = response_tokens if response_tokens is not None else [ord("x")]
        self.returns_routing = True
        self._generator = _FakeGenerator(self)
        self._trainer = _FakeTrainer()
        self.checkpoints: list[str] = []
        # Mirrors PolicyState's version strings. Tests that exercise deferred
        # publishes overwrite ``policy_states``.
        self.policy_states: Iterator[tuple[bool, int, int]] | None = None
        self.retrieve_calls: list[tuple[bool, int]] = []

    @property
    def generator(self) -> _FakeGenerator:
        return self._generator

    @property
    def trainer(self) -> _FakeTrainer:
        return self._trainer

    def create_training_checkpoint(self) -> SimpleNamespace:
        self.checkpoints.append("training")
        return SimpleNamespace(id="tckpt-1")

    def create_inference_checkpoint(self) -> SimpleNamespace:
        self.checkpoints.append("inference")
        return SimpleNamespace(id="model-1")

    async def create_training_checkpoint_async(self) -> SimpleNamespace:
        self.checkpoints.append("training")
        return SimpleNamespace(id="tckpt-1")

    async def create_inference_checkpoint_async(self) -> SimpleNamespace:
        self.checkpoints.append("inference")
        return SimpleNamespace(id="model-1")

    def next_tokens(self) -> list[int]:
        """The completion the next sampled prompt receives."""
        return self._tokens

    async def retrieve_async(self) -> SimpleNamespace:
        """Session snapshot, shaped like the SDK's ``policy_state``."""
        pending, applied, target = next(self.policy_states) if self.policy_states else (False, 0, 0)
        self.retrieve_calls.append((pending, applied))
        return SimpleNamespace(
            policy_state=SimpleNamespace(
                pending_publish=pending,
                applied_weights_version=str(applied),
                target_weights_version=str(target),
            )
        )

    def record_sample_call(self, num_samples: int | None, sampling_params: SamplingParams | None) -> None:
        """Hook for subclasses that assert on what the caller requested."""

    def run(self, coro: Coroutine[Any, Any, Any]) -> Any:
        return asyncio.run(coro)


class RecordingSession(MockSession):
    """``MockSession`` that captures what the last sample call requested."""

    last_num_samples: int | None = None
    last_sampling_params: SamplingParams | None = None

    def record_sample_call(self, num_samples: int | None, sampling_params: SamplingParams | None) -> None:
        self.last_num_samples = num_samples
        self.last_sampling_params = sampling_params


class ScriptedSession(MockSession):
    """Returns canned token sequences from a queue, one per sampled prompt."""

    def __init__(self, replies: list[str]) -> None:
        super().__init__()
        codec = FakeCodec()
        self._queue = [codec.encode(reply) for reply in replies]
        self.calls = 0

    def next_tokens(self) -> list[int]:
        tokens = self._queue[self.calls] if self.calls < len(self._queue) else [ord("z")]
        self.calls += 1
        return tokens
