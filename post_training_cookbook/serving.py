"""A small OpenAI-compatible HTTP facade over an injected async RL sampler.

Install the ``serving`` extra. Sampling and optional capture finish before either
JSON or buffered SSE is returned; this module does not own the sampling session.
"""

from __future__ import annotations

import asyncio
import copy
import hmac
import json
import logging
import math
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal, Self

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from together import APIStatusError, APITimeoutError
from together.post_training import SampleResult, SamplingParams

from post_training_cookbook.samples import make_prompt
from post_training_cookbook.types import Message, ToolCall, ToolCallFunction, ToolSpec

if TYPE_CHECKING:
    from post_training_cookbook.codecs import MessageCodec

DEFAULT_MAX_TOKENS = 1024
DEFAULT_MAX_CONCURRENT = 256
DEFAULT_SAMPLE_TIMEOUT = 7200.0

logger = logging.getLogger(__name__)
SampleFn = Callable[..., Awaitable[SampleResult]]
GenerationHook = Callable[..., Awaitable[None]]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _Text(_StrictModel):
    type: Literal["text"]
    text: str


class _CallFunction(_StrictModel):
    name: str = Field(min_length=1)
    arguments: str


class _Call(_StrictModel):
    id: str = Field(min_length=1)
    type: Literal["function"] = "function"
    function: _CallFunction
    index: int | None = Field(default=None, ge=0)


class _Message(_StrictModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[_Text] | None = None
    tool_calls: list[_Call] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    reasoning_content: str | None = None

    @model_validator(mode="after")
    def _validate_role(self) -> Self:
        if self.reasoning_content is not None and self.role != "assistant":
            raise ValueError("Only assistant messages can contain reasoning_content")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("Only assistant messages can contain tool_calls")
        if self.tool_call_id is not None and self.role != "tool":
            raise ValueError("Only tool messages can contain tool_call_id")
        if self.content is None and not (
            self.role == "assistant" and (self.tool_calls or self.reasoning_content is not None)
        ):
            raise ValueError("Messages require text content or assistant tool calls")
        return self


class _Function(_StrictModel):
    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    strict: bool | None = None


class _Tool(_StrictModel):
    type: Literal["function"]
    function: _Function


class _StreamOptions(_StrictModel):
    include_usage: bool = False


class _ChatRequest(_StrictModel):
    model: str
    messages: list[_Message] = Field(min_length=1)
    tools: list[_Tool] = Field(default_factory=list)
    tool_choice: Literal["auto", "none"] | None = None
    parallel_tool_calls: bool | None = None
    n: int = Field(default=1, ge=1, le=1)
    stream: bool = False
    stream_options: _StreamOptions | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2, allow_inf_nan=False)
    top_p: float | None = Field(default=None, gt=0, le=1, allow_inf_nan=False)
    seed: int | str | None = None
    stop: str | list[str] | None = None
    # Neutral values are commonly sent by OpenAI-compatible clients.
    presence_penalty: float = Field(default=0, ge=0, le=0)
    frequency_penalty: float = Field(default=0, ge=0, le=0)
    logprobs: Literal[False] | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=0)
    response_format: dict[str, Any] | None = None
    user: str | None = None
    store: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_semantics(self) -> Self:
        if self.parallel_tool_calls is False:
            raise ValueError("parallel_tool_calls=false is unsupported; omit it")
        if self.response_format not in (None, {"type": "text"}):
            raise ValueError("Structured response formats are unsupported")
        if self.store:
            raise ValueError("Completion storage is unsupported; use the capture hook")
        if any(tool.function.strict for tool in self.tools):
            raise ValueError("Strict tool-schema enforcement is unsupported")
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            if self.max_tokens != self.max_completion_tokens:
                raise ValueError("Conflicting max_tokens and max_completion_tokens")
        call_ids: set[str] = set()
        for message in self.messages:
            if message.tool_calls:
                call_ids.update(call.id for call in message.tool_calls)
            if message.role == "tool" and message.tool_call_id not in call_ids:
                raise ValueError("Tool results must reference a preceding assistant tool call")
        return self


class _HTTPError(Exception):
    def __init__(self, status: int, message: str, code: str = "invalid_request_error") -> None:
        self.status, self.message, self.code = status, message, code


def validate_bind(host: str, api_key: str | None) -> str:
    """Require a nonempty proxy credential before binding any listener."""
    token = (api_key or "").strip() or None
    if token is None:
        raise ValueError(f"A proxy API key is required before binding {host!r}")
    return token


def _messages_and_tools(body: _ChatRequest) -> tuple[list[Message], list[ToolSpec]]:
    messages: list[Message] = []
    call_names: dict[str, str] = {}
    for raw in body.messages:
        content = raw.content
        if isinstance(content, list):
            content = "".join(block.text for block in content)
        message = Message(role=raw.role, content=content or "")
        if raw.reasoning_content is not None:
            message["reasoning_content"] = raw.reasoning_content
        if raw.tool_calls:
            message["tool_calls"] = [
                ToolCall(id=call.id, function=ToolCallFunction(**call.function.model_dump()))
                for call in raw.tool_calls
            ]
            call_names.update((call.id, call.function.name) for call in raw.tool_calls)
        if raw.role == "tool":
            message["tool_call_id"] = raw.tool_call_id
            message["name"] = raw.name or call_names[raw.tool_call_id]
        elif raw.name is not None:
            message["name"] = raw.name
        messages.append(message)
    tools = [
        ToolSpec(name=t.function.name, description=t.function.description, parameters=t.function.parameters)
        for t in body.tools
    ]
    return messages, tools


async def _build_response(
    *,
    request_id: str,
    model: str,
    body: _ChatRequest,
    tools: list[ToolSpec],
    prompt_ids: list[int],
    result: SampleResult,
    codec: MessageCodec,
    call_codec: Callable[..., Awaitable[Any]],
    include_reasoning: bool,
) -> dict[str, Any]:
    try:
        if len(result.sequences) != 1:
            raise ValueError("Expected one sampled sequence")
        sequence = result.sequences[0]
        tokens = [int(t) for t in sequence.tokens]
        parsed, ok = await call_codec(codec.parse_response, tokens, tools=tools)
        if not ok:
            if sequence.stop_reason != "STOP_REASON_LENGTH":
                raise ValueError("Malformed sampled response")
            # A length cap can interrupt a tool block. Return the codec's
            # partial text, but do not execute calls from an incomplete turn.
            parsed = dict(parsed)
            parsed.pop("tool_calls", None)
        calls = [] if body.tool_choice == "none" else parsed.get("tool_calls", [])
        declared = {tool["name"] for tool in tools}
        if any(call["function"]["name"] not in declared for call in calls):
            raise ValueError("Undeclared sampled tool")
    except (ValueError, TypeError, IndexError) as exc:
        raise _HTTPError(502, "The sampled response could not be parsed", "backend_error") from exc
    message: dict[str, Any] = {
        "role": "assistant",
        "content": parsed.get("content") or (None if calls else ""),
    }
    if include_reasoning and parsed.get("reasoning_content") is not None:
        message["reasoning_content"] = parsed["reasoning_content"]
    if calls:
        message["tool_calls"] = [
            {"id": f"call_{uuid.uuid4().hex}", "type": "function", "function": call["function"]} for call in calls
        ]
    reason = "length" if sequence.stop_reason == "STOP_REASON_LENGTH" else ("tool_calls" if calls else "stop")
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": reason}],
        "usage": {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(tokens),
            "total_tokens": len(prompt_ids) + len(tokens),
        },
    }


def _stream(
    response: dict[str, Any],
    include_usage: bool,
    on_close: Callable[[], None] | None = None,
):
    # Buffered SSE: emit OpenAI-compatible chunks after sampling and capture complete.
    try:
        base = {key: response[key] for key in ("id", "created", "model")}
        base["object"] = "chat.completion.chunk"
        choice = response["choices"][0]
        message = choice["message"]
        deltas: list[dict[str, Any]] = [{"role": "assistant"}]
        if message.get("reasoning_content") is not None:
            deltas.append({"reasoning_content": message["reasoning_content"]})
        if message["content"] is not None:
            deltas.append({"content": message["content"]})
        if message.get("tool_calls"):
            deltas.append({"tool_calls": [{"index": i, **call} for i, call in enumerate(message["tool_calls"])]})
        for delta in deltas:
            chunk = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
            if include_usage:
                chunk["usage"] = None
            yield f"data: {json.dumps(chunk)}\n\n"
        chunk = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}]}
        if include_usage:
            chunk["usage"] = None
        yield f"data: {json.dumps(chunk)}\n\n"
        if include_usage:
            yield f"data: {json.dumps({**base, 'choices': [], 'usage': response['usage']})}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        if on_close is not None:
            on_close()


def create_app(
    sample_fn: SampleFn,
    codec: MessageCodec,
    *,
    model: str,
    max_model_length: int,
    default_max_tokens: int = DEFAULT_MAX_TOKENS,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    sample_timeout: float = DEFAULT_SAMPLE_TIMEOUT,
    api_key: str | None = None,
    session_id: str | None = None,
    on_generation: GenerationHook | None = None,
    include_reasoning: bool = True,
) -> FastAPI:
    """Wrap a sampler without owning its session or adding generation retries.

    ``sample_fn`` accepts the same arguments as ``session.generator.sample_async``.
    The optional async hook is called with keyword arguments ``metadata``,
    ``prompt_token_ids``, and ``result`` (the unchanged SDK ``SampleResult``).
    Capture must finish before a successful HTTP response, including SSE headers.
    A nonempty ``api_key`` is required even when embedding on loopback.
    By default HTTP messages round-trip separate ``reasoning_content`` with the
    codec's history policy. Set ``include_reasoning=False`` to carry answer text
    and tool calls only. Train from the captured tokens in either mode.
    """
    if (
        not model
        or min(max_model_length, default_max_tokens, max_concurrent) < 1
        or not math.isfinite(sample_timeout)
        or sample_timeout <= 0
    ):
        raise ValueError("Model and positive context, token, concurrency, and timeout limits are required")
    token = (api_key or "").strip()
    if not token:
        raise ValueError("A nonempty proxy API key is required")
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    capacity = asyncio.Semaphore(max_concurrent)
    codec_gate = asyncio.Semaphore(1)

    async def call_codec(method, *args, **kwargs):
        await codec_gate.acquire()
        try:
            task = asyncio.create_task(asyncio.to_thread(method, *args, **kwargs))
        except BaseException:
            codec_gate.release()
            raise

        def _release_codec(done: asyncio.Task[Any]) -> None:
            if not done.cancelled():
                done.exception()
            codec_gate.release()

        task.add_done_callback(_release_codec)
        return await asyncio.shield(task)

    app.state.active_requests = 0
    app.state.waiting_requests = 0
    app.state.peak_requests = 0
    app.state.active_samples = 0
    app.state.peak_samples = 0

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        presented = request.headers.get("authorization", "")
        if not hmac.compare_digest(presented.encode(), f"Bearer {token}".encode()):
            return JSONResponse({"error": {"type": "authentication_error", "message": "Invalid proxy API key"}}, 401)
        return await call_next(request)

    @app.exception_handler(_HTTPError)
    async def http_error(_request: Request, exc: _HTTPError):
        headers = {"Retry-After": "1"} if exc.status == 429 else None
        return JSONResponse({"error": {"type": exc.code, "message": exc.message}}, exc.status, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError):
        details = "; ".join(
            f"{'.'.join(map(str, error['loc'][1:] or ('body',)))}: {error['msg']}" for error in exc.errors()
        )
        return JSONResponse({"error": {"type": "invalid_request_error", "message": details}}, 400)

    @app.post("/v1/chat/completions")
    async def complete(body: _ChatRequest):
        if body.model != model:
            raise _HTTPError(404, "Unknown served model", "model_not_found")
        if not include_reasoning and any(message.reasoning_content is not None for message in body.messages):
            raise _HTTPError(400, "reasoning_content requires include_reasoning=True")
        messages, tools = _messages_and_tools(body)
        app.state.waiting_requests += 1
        try:
            await asyncio.wait_for(capacity.acquire(), timeout=sample_timeout)
        except TimeoutError as exc:
            raise _HTTPError(504, "Timed out waiting for generation capacity", "timeout_error") from exc
        finally:
            app.state.waiting_requests -= 1
        app.state.active_requests += 1
        app.state.peak_requests = max(app.state.peak_requests, app.state.active_requests)
        slot_held = True
        defer_release = False

        def release_slot() -> None:
            nonlocal slot_held
            if slot_held:
                slot_held = False
                app.state.active_requests -= 1
                capacity.release()

        try:
            request_id = f"chatcmpl-{uuid.uuid4().hex}"
            metadata = {
                "request_id": request_id,
                "request": copy.deepcopy(body.metadata),
                "model": model,
                "session_id": session_id,
            }
            try:
                prompt_ids = await call_codec(
                    codec.encode_messages,
                    messages,
                    tools=[] if body.tool_choice == "none" else tools,
                    add_generation_prompt=True,
                )
            except (ValueError, NotImplementedError) as exc:
                raise _HTTPError(400, "Messages or tools are unsupported by the configured codec") from exc
            remaining = max_model_length - len(prompt_ids)
            if remaining <= 0:
                raise _HTTPError(
                    400, "Prompt is too long for the configured context window", "context_length_exceeded"
                )
            params = SamplingParams(
                max_tokens=min(body.max_completion_tokens or body.max_tokens or default_max_tokens, remaining)
            )
            for key in ("temperature", "top_p", "seed"):
                value = getattr(body, key)
                if value is not None:
                    params[key] = value
            if body.stop is not None:
                params["stop"] = [body.stop] if isinstance(body.stop, str) else body.stop
            metadata["sampling_params"] = dict(params)
            app.state.active_samples += 1
            app.state.peak_samples = max(app.state.peak_samples, app.state.active_samples)
            try:
                result = await sample_fn(
                    make_prompt(prompt_ids), num_samples=1, sampling_params=params, timeout=sample_timeout
                )
            except (TimeoutError, APITimeoutError) as exc:
                raise _HTTPError(504, "Sampling timed out", "timeout_error") from exc
            except APIStatusError as exc:
                status = exc.status_code if exc.status_code in (400, 429, 503) else 502
                raise _HTTPError(status, "RL sampling backend rejected the request", "backend_error") from exc
            except Exception as exc:
                logger.error("Sampling failed for %s (%s)", request_id, type(exc).__name__)
                raise _HTTPError(502, "RL sampling failed", "backend_error") from exc
            finally:
                app.state.active_samples -= 1
            if on_generation is not None:
                try:
                    await on_generation(metadata=metadata, prompt_token_ids=list(prompt_ids), result=result)
                except Exception as exc:
                    logger.error("Capture failed for %s (%s)", request_id, type(exc).__name__)
                    raise _HTTPError(500, "Generation capture failed", "capture_error") from exc
            response = await _build_response(
                request_id=request_id,
                model=model,
                body=body,
                tools=tools,
                prompt_ids=prompt_ids,
                result=result,
                codec=codec,
                call_codec=call_codec,
                include_reasoning=include_reasoning,
            )
            if body.stream:
                include_usage = bool(body.stream_options and body.stream_options.include_usage)
                defer_release = True
                return StreamingResponse(
                    _stream(response, include_usage, release_slot),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Request-ID": request_id},
                )
            return JSONResponse(response, headers={"X-Request-ID": request_id})
        finally:
            if not defer_release:
                release_slot()

    return app
