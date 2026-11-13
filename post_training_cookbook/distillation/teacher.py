"""Teacher generation through a hosted chat-completions endpoint (Together serverless).

Every completed request is appended to a JSONL cache keyed by ``TeacherRequest.key``, so an
interrupted run resumes where it stopped and a rerun with the same keys makes no new calls. Keys
don't include the model, so give each teacher model its own cache file.
Requests run concurrently, ``TeacherConfig.concurrency`` at a time per API key. Listing several
keys from separate projects in ``api_key_envs`` adds their rate limits together. Rate limits and
transient errors are retried by the SDK (``max_retries``), and a request that still fails is cached
with ``ok=False`` so the next run retries it.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

from together import AsyncTogether

from post_training_cookbook.types import Message


@dataclass(frozen=True)
class TeacherConfig:
    """How the teacher is called.

    ``chat_template_kwargs`` is passed through to the endpoint, e.g. ``{"thinking": False}`` turns
    reasoning off on Kimi-K3.1-Pro. ``price_per_million`` is (input, output) USD per million tokens; set it
    together with ``budget_usd`` to stop generation once the estimated spend reaches the budget.
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


class TeacherRequest(NamedTuple):
    """One teacher call. ``key`` must be unique and stable across runs; it is the cache key."""

    key: str
    messages: list[Message]
    meta: dict[str, Any] = {}


@dataclass
class TeacherOutput:
    key: str
    ok: bool
    content: str = ""
    reasoning_content: str = ""
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


@dataclass
class Spend:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    failures: int = 0

    def usd(self, price_per_million: tuple[float, float] | None) -> float:
        if price_per_million is None:
            return 0.0
        price_in, price_out = price_per_million
        return (self.input_tokens * price_in + self.output_tokens * price_out) / 1e6


def load_cache(path: Path) -> dict[str, TeacherOutput]:
    """Read a teacher cache; for a key written more than once, the last line wins."""
    if not path.exists():
        return {}
    out: dict[str, TeacherOutput] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["key"]] = TeacherOutput(**row)
    return out


def _api_keys(config: TeacherConfig) -> list[str]:
    keys: list[str] = []
    for name in config.api_key_envs:
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"{name} is not set")
        if value not in keys:
            keys.append(value)
    return keys


async def _call(client: Any, config: TeacherConfig, request: TeacherRequest) -> TeacherOutput:
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
    except Exception as exc:  # noqa: BLE001 — cached as a failure and retried on the next run
        return TeacherOutput(key=request.key, ok=False, error=f"{type(exc).__name__}: {exc}"[:500])
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
    )


async def generate_async(
    requests: Iterable[TeacherRequest],
    config: TeacherConfig,
    cache_path: Path,
    *,
    client_factory: Callable[..., Any] = AsyncTogether,
    progress_every: int = 50,
) -> tuple[dict[str, TeacherOutput], Spend]:
    """Run every request not already cached as ``ok``; return outputs for all requests and this run's spend."""
    requests = list({r.key: r for r in requests}.values())
    cache = load_cache(cache_path)
    todo = [r for r in requests if not (r.key in cache and cache[r.key].ok)]
    spend = Spend()
    if not todo:
        return {r.key: cache[r.key] for r in requests if r.key in cache}, spend

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    queue: asyncio.Queue[TeacherRequest] = asyncio.Queue()
    for r in todo:
        queue.put_nowait(r)
    lock = asyncio.Lock()
    stop = asyncio.Event()
    done = 0
    start = time.monotonic()
    clients = [
        client_factory(api_key=key, max_retries=config.max_retries, timeout=config.timeout)
        for key in _api_keys(config)
    ]

    async def worker(client: Any) -> None:
        nonlocal done
        while not stop.is_set():
            try:
                request = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            output = await _call(client, config, request)
            async with lock:
                cache[request.key] = output
                with cache_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(output), ensure_ascii=False) + "\n")
                done += 1
                spend.calls += 1
                spend.failures += not output.ok
                spend.input_tokens += int(output.usage.get("prompt_tokens") or 0)
                spend.output_tokens += int(output.usage.get("completion_tokens") or 0)
                if config.budget_usd is not None and spend.usd(config.price_per_million) >= config.budget_usd:
                    stop.set()
                if done % progress_every == 0 or done == len(todo):
                    rate = done / max(time.monotonic() - start, 1e-9) * 60
                    print(
                        f"  [teacher {config.model.split('/')[-1]}] {done}/{len(todo)} done, {rate:.1f}/min, "
                        f"${spend.usd(config.price_per_million):.2f}, failures={spend.failures}",
                        flush=True,
                    )

    try:
        await asyncio.gather(*(worker(c) for c in clients for _ in range(config.concurrency)))
    finally:
        for client in clients:
            close = getattr(client, "close", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result
    if stop.is_set():
        print(f"  teacher budget ${config.budget_usd} reached; {queue.qsize()} requests left unsent", flush=True)
    return {r.key: cache[r.key] for r in requests if r.key in cache}, spend


def generate(
    requests: Iterable[TeacherRequest], config: TeacherConfig, cache_path: Path, **kwargs: Any
) -> tuple[dict[str, TeacherOutput], Spend]:
    """Synchronous wrapper around :func:`generate_async`."""
    return asyncio.run(generate_async(requests, config, cache_path, **kwargs))
