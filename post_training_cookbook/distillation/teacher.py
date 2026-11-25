"""Teacher generation through a hosted chat-completions endpoint (Together serverless).

Every completed request is appended to a JSONL cache keyed by ``TeacherRequest.key``. A rerun
reuses successful rows for the requested model and retries missing or failed requests. A line torn
by a crash mid-write is skipped with a warning.
Requests run concurrently, ``TeacherConfig.concurrency`` at a time per API key. Listing several
keys from separate projects in ``api_key_envs`` adds their rate limits together. Rate limits and
transient errors are retried by the SDK (``max_retries``), and an API or timeout error that still
fails is cached with ``ok=False`` so the next run retries it. Any other exception is a bug and raises.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from together import APIError, AsyncTogether

from post_training_cookbook.types import Message

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TeacherConfig:
    """How the teacher is called.

    ``chat_template_kwargs`` is passed through to the endpoint, e.g. ``{"thinking": False}`` turns
    reasoning off on Kimi-K3.1-Pro. ``price_per_million`` is (input, output) USD per million tokens; set it
    together with ``budget_usd`` to stop sending new requests once the estimated spend reaches the
    budget. Requests already in flight can still complete and add to the spend.
    """

    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    chat_template_kwargs: dict[str, Any] | None = None
    reasoning_effort: str | None = None
    concurrency: int = 8
    api_key_envs: tuple[str, ...] = ("TOGETHER_API_KEY",)
    max_retries: int = 8
    timeout: float = 3600.0
    price_per_million: tuple[float, float] | None = None
    budget_usd: float | None = None

    def __post_init__(self) -> None:
        """Validate the config."""
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if not self.api_key_envs:
            raise ValueError("api_key_envs must name at least one environment variable")
        if self.budget_usd is not None and self.price_per_million is None:
            raise ValueError("budget_usd needs price_per_million to estimate spend")


@dataclass(frozen=True)
class TeacherRequest:
    """One teacher call. ``key`` must be unique and stable across runs; it is the cache key."""

    key: str
    messages: list[Message]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TeacherOutput:
    key: str
    ok: bool
    content: str = ""
    reasoning_content: str = ""
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    # The teacher that wrote this row; a row is reused only for the same model.
    model: str | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


@dataclass
class Spend:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    failures: int = 0

    def with_output(self, output: TeacherOutput) -> Spend:
        """Return the run totals after counting one completed teacher call."""
        return Spend(
            input_tokens=self.input_tokens + int(output.usage.get("prompt_tokens") or 0),
            output_tokens=self.output_tokens + int(output.usage.get("completion_tokens") or 0),
            calls=self.calls + 1,
            failures=self.failures + int(not output.ok),
        )

    def usd(self, price_per_million: tuple[float, float] | None) -> float:
        if price_per_million is None:
            return 0.0
        price_in, price_out = price_per_million
        return (self.input_tokens * price_in + self.output_tokens * price_out) / 1e6


def load_cache(path: Path) -> dict[str, TeacherOutput]:
    """Read a teacher cache; for a key written more than once, the last line wins.

    A line that does not parse (typically the last one, torn by a crash mid-write) is skipped with a
    warning, so the rows already paid for stay readable and the request is simply sent again.
    """
    if not path.exists():
        return {}
    out: dict[str, TeacherOutput] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skipping unreadable line %d of teacher cache %s", number, path)
            continue
        out[row["key"]] = TeacherOutput(**row)
    return out


def _prepare_cache_for_append(path: Path) -> None:
    """Create the cache directory and finish any unterminated line before appending."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with path.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
        if last != b"\n":
            with path.open("a", encoding="utf-8") as f:
                f.write("\n")


def _append_output(path: Path, output: TeacherOutput) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(output), ensure_ascii=False) + "\n")


def _api_keys(config: TeacherConfig) -> list[str]:
    keys: list[str] = []
    for name in config.api_key_envs:
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"{name} is not set")
        if value not in keys:
            keys.append(value)
    return keys


async def _request_completion(client: AsyncTogether, config: TeacherConfig, request: TeacherRequest) -> TeacherOutput:
    kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": request.messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.chat_template_kwargs is not None:
        kwargs["chat_template_kwargs"] = config.chat_template_kwargs
    if config.reasoning_effort is not None:
        kwargs["reasoning_effort"] = config.reasoning_effort
    try:
        resp = await client.chat.completions.create(**kwargs)
    except (APIError, TimeoutError) as exc:  # cached as a failure and retried on the next run
        return TeacherOutput(key=request.key, ok=False, error=f"{type(exc).__name__}: {exc}"[:500], model=config.model)
    choice = resp.choices[0]
    message = choice.message
    usage = resp.usage.model_dump() if getattr(resp, "usage", None) is not None else {}
    return TeacherOutput(
        key=request.key,
        ok=True,
        content=message.content or "",
        reasoning_content=getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None) or "",
        finish_reason=choice.finish_reason,
        usage=usage,
        model=config.model,
    )


async def _generate_async(
    requests: Iterable[TeacherRequest],
    config: TeacherConfig,
    cache_path: Path,
) -> tuple[dict[str, TeacherOutput], Spend]:
    requests_by_key = {request.key: request for request in requests}
    cache = load_cache(cache_path)
    outputs = {
        key: cached
        for key in requests_by_key
        if (cached := cache.get(key)) is not None and cached.model == config.model
    }
    pending = deque(request for key, request in requests_by_key.items() if key not in outputs or not outputs[key].ok)
    spend = Spend()
    if not pending:
        return outputs, spend

    _prepare_cache_for_append(cache_path)
    total = len(pending)
    budget_reached = False
    start = time.monotonic()
    clients = [
        AsyncTogether(api_key=key, max_retries=config.max_retries, timeout=config.timeout) for key in _api_keys(config)
    ]

    # Workers only yield during the API call, so claiming work and recording spend need no lock.
    async def worker(client: AsyncTogether) -> None:
        nonlocal spend, budget_reached
        while pending and not budget_reached:
            request = pending.popleft()
            output = await _request_completion(client, config, request)
            _append_output(cache_path, output)
            outputs[output.key] = output
            spend = spend.with_output(output)
            cost = spend.usd(config.price_per_million)
            if config.budget_usd is not None and cost >= config.budget_usd:
                budget_reached = True
            if spend.calls % 50 == 0 or spend.calls == total:
                rate = spend.calls / max(time.monotonic() - start, 1e-9) * 60
                print(
                    f"  [teacher {config.model.split('/')[-1]}] {spend.calls}/{total} done, {rate:.1f}/min, "
                    f"${cost:.2f}, failures={spend.failures}",
                    flush=True,
                )

    try:
        await asyncio.gather(*(worker(c) for c in clients for _ in range(config.concurrency)))
    finally:
        for client in clients:
            await client.close()
    if budget_reached:
        print(f"  teacher budget ${config.budget_usd} reached; {len(pending)} requests left unsent", flush=True)
    return {key: outputs[key] for key in requests_by_key if key in outputs}, spend


def generate(
    requests: Iterable[TeacherRequest], config: TeacherConfig, cache_path: Path
) -> tuple[dict[str, TeacherOutput], Spend]:
    """Return this model's recorded outputs in request order and the spend for new calls.

    Outputs include recorded failures. Keys without a recorded output are omitted, such as when
    the budget is reached before a request is sent.
    """
    return asyncio.run(_generate_async(requests, config, cache_path))
