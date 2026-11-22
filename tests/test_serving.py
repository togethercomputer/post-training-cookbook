"""HTTP interoperability and bounded concurrency, with the real pinned SDK types."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import threading
from functools import partial
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI
from together.post_training import Generator, SampledSequence, SampleResult, SessionClient

pytest.importorskip("fastapi")

from post_training_cookbook.serving import create_app as _create_app  # noqa: E402
from post_training_cookbook.serving import validate_bind  # noqa: E402
from recipes.serving import openai_endpoint  # noqa: E402

MODEL = "Qwen/Qwen3.8-27B"
TOOL = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
REQUEST = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}
PROXY_KEY = "test-proxy-key"
THREAD_STARVATION_BURST = 64
create_app = partial(_create_app, api_key=PROXY_KEY)


class Codec:
    def __init__(self):
        self.requests = []
        self.parsed = {"role": "assistant", "content": "answer", "reasoning_content": "thinking"}
        self.ok = True

    def encode_messages(self, messages, **kwargs):
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
        return [1, 2, 3]

    def parse_response(self, tokens, *, tools):
        return copy.deepcopy(self.parsed), self.ok


def sample_result(*, stop="STOP_REASON_STOP", logprobs=None):
    return SampleResult(
        policy_segments=[{"start_token": 0, "version": 7}],
        sequences=[
            SampledSequence(
                tokens=[8, 9],
                stop_reason=stop,
                prompt_cache_hit_tokens=0,
                logprobs=logprobs,
                routed_experts_key="opaque-key",
            )
        ],
    )


@pytest.fixture
def setup():
    codec, calls, captures = Codec(), [], []
    result = sample_result()

    async def sample(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return result

    async def capture(**kwargs):
        captures.append(kwargs)

    app = create_app(sample, codec, model=MODEL, session_id="borrowed", max_model_length=10, on_generation=capture)
    return SimpleNamespace(app=app, codec=codec, calls=calls, captures=captures, sample=sample, result=result)


def client_for(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://wrapper",
        headers={"Authorization": f"Bearer {PROXY_KEY}"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("logprobs", [None, [-0.1, -0.2]])
async def test_json_and_capture(setup, logprobs):
    setup.result.sequences[0].logprobs = logprobs
    async with client_for(setup.app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                **REQUEST,
                "max_tokens": 20,
                "max_completion_tokens": 20,
                "temperature": 0.3,
                "top_p": 0.9,
                "seed": 42,
                "stop": "END",
                "metadata": {"rollout": 5},
                "response_format": {"type": "text"},
                "parallel_tool_calls": True,
                "store": False,
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"] == {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "thinking",
    }
    assert payload["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    assert setup.calls[0][0]["chunks"][0]["encoded_text"]["tokens"] == [1, 2, 3]
    assert setup.calls[0][1]["sampling_params"] == {
        "max_tokens": 7,
        "temperature": 0.3,
        "top_p": 0.9,
        "seed": 42,
        "stop": ["END"],
    }
    assert setup.captures[0]["result"] is setup.result
    assert setup.captures[0]["result"].sequences[0].logprobs == logprobs
    assert setup.captures[0]["result"].sequences[0].tokens == [8, 9]
    assert setup.captures[0]["result"].sequences[0].routed_experts_key == "opaque-key"
    assert setup.captures[0]["result"].policy_segments[0].version == 7
    assert setup.captures[0]["metadata"]["request"] == {"rollout": 5}
    assert setup.captures[0]["metadata"]["session_id"] == "borrowed"
    assert setup.captures[0]["metadata"]["request_id"] == payload["id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_incomplete_length_cap(setup, stream):
    setup.result.sequences[0].stop_reason = "STOP_REASON_LENGTH"
    setup.codec.ok = False
    setup.codec.parsed["tool_calls"] = [{"id": None, "function": {"name": "lookup", "arguments": "{}"}}]
    async with client_for(setup.app) as client:
        response = await client.post("/v1/chat/completions", json={**REQUEST, "stream": stream})
    assert response.status_code == 200
    assert setup.captures[0]["result"] is setup.result
    if stream:
        chunks = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
        assert chunks[-1]["choices"][0]["finish_reason"] == "length"
        assert all("tool_calls" not in chunk["choices"][0]["delta"] for chunk in chunks)
        assert response.text.endswith("data: [DONE]\n\n")
    else:
        choice = response.json()["choices"][0]
        assert choice["finish_reason"] == "length"
        assert choice["message"]["content"] == "answer"
        assert "tool_calls" not in choice["message"]


@pytest.mark.asyncio
async def test_text_response_replays(setup):
    async with client_for(setup.app) as client:
        first = (await client.post("/v1/chat/completions", json=REQUEST)).json()
        message = first["choices"][0]["message"]
        replay = {"model": MODEL, "messages": [*REQUEST["messages"], message, {"role": "user", "content": "continue"}]}
        assert (await client.post("/v1/chat/completions", json=replay)).status_code == 200
        assert setup.codec.requests[-1][0][1] == {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "thinking",
        }


@pytest.mark.asyncio
async def test_tool_cycle_and_none(setup):
    setup.codec.parsed = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "thinking",
        "tool_calls": [{"id": None, "function": {"name": "lookup", "arguments": '{"x":1}'}}],
    }
    async with client_for(setup.app) as client:
        first = await client.post("/v1/chat/completions", json={**REQUEST, "tools": [TOOL]})
        choice = first.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert set(choice["message"]) == {"role", "content", "reasoning_content", "tool_calls"}
        call = choice["message"]["tool_calls"][0]
        setup.codec.parsed = {"role": "assistant", "content": "done"}
        messages = [
            *REQUEST["messages"],
            choice["message"],
            {"role": "tool", "tool_call_id": call["id"], "content": "result"},
        ]
        final = await client.post("/v1/chat/completions", json={"model": MODEL, "messages": messages, "tools": [TOOL]})
        assert final.json()["choices"][0]["message"]["content"] == "done"
        assert setup.codec.requests[-1][0][-1]["name"] == "lookup"
        assert setup.codec.requests[-1][1]["tools"][0]["name"] == "lookup"
        setup.codec.parsed = {
            "role": "assistant",
            "content": "text",
            "tool_calls": [{"id": None, "function": {"name": "lookup", "arguments": "{}"}}],
        }
        suppressed = await client.post(
            "/v1/chat/completions", json={**REQUEST, "tools": [TOOL], "tool_choice": "none"}
        )
    assert "tool_calls" not in suppressed.json()["choices"][0]["message"]
    assert setup.codec.requests[-1][1]["tools"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("content", [None, "answer"])
async def test_reasoning_roundtrip(setup, stream, content):
    setup.codec.parsed["content"] = content

    async def capture(**kwargs):
        setup.captures.append(kwargs)

    app = create_app(setup.sample, setup.codec, model=MODEL, max_model_length=10, on_generation=capture)
    async with client_for(app) as client:
        first = await client.post("/v1/chat/completions", json=REQUEST)
        message = first.json()["choices"][0]["message"]
        assert message["reasoning_content"] == "thinking"
        message["content"] = content
        response = await client.post(
            "/v1/chat/completions",
            json={"model": MODEL, "messages": [*REQUEST["messages"], message], "stream": stream},
        )
    assert response.status_code == 200
    assert setup.codec.requests[-1][0][-1]["reasoning_content"] == "thinking"
    assert setup.captures[-1]["result"] is setup.result
    if stream:
        chunks = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
        assert [
            chunk["choices"][0]["delta"]["reasoning_content"]
            for chunk in chunks
            if "reasoning_content" in chunk["choices"][0]["delta"]
        ] == ["thinking"]
    else:
        assert response.json()["choices"][0]["message"]["reasoning_content"] == "thinking"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_openai_client(setup, stream):
    setup.codec.parsed["tool_calls"] = [{"id": None, "function": {"name": "lookup", "arguments": "{}"}}]
    app = create_app(setup.sample, setup.codec, model=MODEL, max_model_length=10, include_reasoning=False)
    async with AsyncOpenAI(
        api_key=PROXY_KEY, base_url="http://wrapper/v1", max_retries=0, http_client=client_for(app)
    ) as client:
        kwargs = {"model": MODEL, "messages": REQUEST["messages"], "tools": [TOOL], "stream": stream}
        if stream:
            kwargs["stream_options"] = {"include_usage": True}
        response = await client.chat.completions.create(**kwargs)
        if stream:
            chunks = [chunk async for chunk in response]
            assert all(
                "reasoning_content" not in chunk.model_dump_json() and "reasoning" not in chunk.model_dump_json()
                for chunk in chunks
            )
            assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == "answer"
            assert chunks[0].choices[0].delta.role == "assistant"
            assert chunks[-2].choices[0].finish_reason == "tool_calls"
            assert chunks[-1].usage.total_tokens == 5
            assert chunks[-1].choices == []
            calls = [
                chunk.choices[0].delta.tool_calls
                for chunk in chunks
                if chunk.choices and chunk.choices[0].delta.tool_calls
            ]
            assert calls[0][0].index == 0
            assert len({chunk.id for chunk in chunks}) == 1
        else:
            assert response.choices[0].message.tool_calls[0].function.name == "lookup"
            assert set(response.choices[0].message.model_dump(exclude_unset=True)) == {"role", "content", "tool_calls"}


@pytest.mark.asyncio
async def test_reasoning_disabled_rejects_input(setup):
    app = create_app(setup.sample, setup.codec, model=MODEL, max_model_length=10, include_reasoning=False)
    async with client_for(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "assistant", "content": "answer", "reasoning_content": "thinking"}],
            },
        )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert setup.codec.requests == []
    assert setup.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"n": 2},
        {"n": True},
        {"max_tokens": 0},
        {"max_tokens": True},
        {"temperature": -1},
        {"tool_choice": "required"},
        {"response_format": {"type": "json_object"}},
        {"response_format": {}},
        {"parallel_tool_calls": False},
        {"store": True},
        {"logprobs": True},
        {"presence_penalty": 1},
        {"stream_options": []},
        {"max_tokens": 1, "max_completion_tokens": 2},
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": "secret"}]}]},
        {"messages": [{"role": "tool", "tool_call_id": "missing", "content": "hi"}]},
        {"messages": [{"role": "user", "content": None}]},
        {"messages": [{"role": "assistant", "content": None}]},
        {"messages": [{"role": "user", "content": "hi", "tool_call_id": "call_1"}]},
        {
            "messages": [
                {
                    "role": "user",
                    "content": "hi",
                    "tool_calls": [{"id": "call_1", "function": {"name": "lookup", "arguments": "{}"}}],
                }
            ]
        },
        {
            "messages": [
                {"role": "tool", "tool_call_id": "call_1", "content": "result"},
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "call_1", "function": {"name": "lookup", "arguments": "{}"}}],
                },
            ]
        },
        {"messages": [{"role": "assistant", "content": "answer", "reasoning": "thinking"}]},
        {"messages": [{"role": "user", "content": "hi", "reasoning_content": "thinking"}]},
        {"tools": [{"type": "function", "function": {"name": "f", "strict": True}}]},
    ],
)
async def test_invalid_before_sampling(setup, change):
    async with client_for(setup.app) as client:
        response = await client.post("/v1/chat/completions", json={**REQUEST, **change})
    assert response.status_code == 400, response.text
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert not response.json()["error"]["message"].startswith(":")
    assert setup.codec.requests == []
    assert setup.calls == []


@pytest.mark.asyncio
async def test_auth_and_routes(setup):
    app = create_app(setup.sample, setup.codec, model=MODEL, max_model_length=10, api_key="key")
    async with client_for(app) as client:
        assert (await client.post("/v1/chat/completions", json=REQUEST)).status_code == 401
        client.headers["Authorization"] = "Bearer key"
        for path in ("/healthz", "/v1/models", "/docs", "/redoc", "/openapi.json"):
            assert (await client.get(path)).status_code == 404
        assert (await client.post("/v1/responses", json={})).status_code == 404
        assert (await client.post("/v1/chat/completions", json={**REQUEST, "model": "wrong"})).status_code == 404
        assert setup.calls == []
        assert (await client.post("/v1/chat/completions", json=REQUEST)).status_code == 200
    for host in ("0.0.0.0", "::", "external.example"):
        with pytest.raises(ValueError):
            validate_bind(host, " ")
    with pytest.raises(ValueError):
        validate_bind("127.0.0.1", None)
    assert validate_bind("::1", " token ") == "token"


@pytest.mark.parametrize("token", [None, "", "   "])
def test_embedded_app_requires_key(setup, token):
    with pytest.raises(ValueError, match="proxy API key"):
        _create_app(setup.sample, setup.codec, model=MODEL, max_model_length=10, api_key=token)


@pytest.mark.asyncio
async def test_unauthenticated_sampling_rejected(setup):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=setup.app), base_url="http://wrapper") as client:
        response = await client.post("/v1/chat/completions", json=REQUEST)
    assert response.status_code == 401
    assert setup.calls == []


@pytest.mark.asyncio
async def test_failures_and_recovery(setup):
    async def capture(**kwargs):
        raise OSError("private capture details")

    app = create_app(setup.sample, setup.codec, model=MODEL, max_model_length=10, on_generation=capture)
    async with client_for(app) as client:
        response = await client.post("/v1/chat/completions", json={**REQUEST, "stream": True})
        assert response.status_code == 500
        assert response.headers["content-type"] == "application/json"
        assert "private" not in response.text
        assert len(setup.calls) == 1
        assert app.state.active_requests == 0
    setup.codec.ok = False
    async with client_for(setup.app) as client:
        assert (await client.post("/v1/chat/completions", json=REQUEST)).status_code == 502
    assert len(setup.captures) == 1
    app = create_app(setup.sample, setup.codec, model=MODEL, max_model_length=3)
    async with client_for(app) as client:
        assert (await client.post("/v1/chat/completions", json=REQUEST)).status_code == 400


@pytest.fixture
def blocking_codec():
    entered, release = threading.Event(), threading.Event()
    state = SimpleNamespace(entered=entered, release=release, active=0, peak=0, calls=0)

    class BlockingCodec(Codec):
        def encode_messages(self, *args, **kwargs):
            state.active += 1
            state.peak = max(state.active, state.peak)
            state.calls += 1
            try:
                if state.calls == 1:
                    entered.set()
                    assert release.wait(5)
                return [1, 2, 3]
            finally:
                state.active -= 1

    state.codec = BlockingCodec()
    return state


@pytest.fixture
def paused_sse_stream():
    chunk_sent, release_stream = asyncio.Event(), asyncio.Event()
    sample_calls = 0

    async def sample(*args, **kwargs):
        nonlocal sample_calls
        sample_calls += 1
        return sample_result()

    app = create_app(sample, Codec(), model=MODEL, max_model_length=10, max_concurrent=1)

    async def wrapped_asgi(scope, receive, send):
        async def gated_send(message):
            await send(message)
            if message.get("type") == "http.response.body" and not chunk_sent.is_set():
                chunk_sent.set()
                await release_stream.wait()

        await app(scope, receive, gated_send)

    return SimpleNamespace(
        app=app,
        asgi=wrapped_asgi,
        chunk_sent=chunk_sent,
        release_stream=release_stream,
        sample_calls=lambda: sample_calls,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [128, 256])
async def test_concurrency_and_isolation(limit):
    entered, release = asyncio.Event(), asyncio.Event()
    active = 0
    captures = []

    async def sample(prompt, **kwargs):
        nonlocal active
        active += 1
        if active == limit:
            entered.set()
        await release.wait()
        return sample_result()

    async def capture(**kwargs):
        captures.append(kwargs["metadata"]["request"]["rollout"])

    app = create_app(sample, Codec(), model=MODEL, max_model_length=10, max_concurrent=limit, on_generation=capture)
    async with client_for(app) as client:
        jobs = [
            asyncio.create_task(client.post("/v1/chat/completions", json={**REQUEST, "metadata": {"rollout": i}}))
            for i in range(limit)
        ]
        excess = None
        try:
            await asyncio.wait_for(entered.wait(), timeout=10)
            excess = asyncio.create_task(
                client.post("/v1/chat/completions", json={**REQUEST, "metadata": {"rollout": limit}})
            )
            while app.state.waiting_requests < 1:
                await asyncio.sleep(0.01)
            assert not excess.done()
            assert app.state.active_requests == limit
            assert app.state.waiting_requests == 1
        finally:
            release.set()
            responses = await asyncio.gather(*jobs, *([excess] if excess is not None else []))
    assert all(response.status_code == 200 for response in responses)
    assert sorted(captures) == list(range(limit + 1))
    assert app.state.peak_samples == limit
    assert app.state.peak_requests == limit
    assert app.state.active_requests == 0
    assert app.state.waiting_requests == 0


@pytest.mark.asyncio
async def test_queue_timeout_returns_504():
    entered, release = asyncio.Event(), asyncio.Event()

    async def sample(*args, **kwargs):
        entered.set()
        await release.wait()
        return sample_result()

    app = create_app(sample, Codec(), model=MODEL, max_model_length=10, max_concurrent=1, sample_timeout=0.05)
    async with client_for(app) as client:
        first = asyncio.create_task(client.post("/v1/chat/completions", json=REQUEST))
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            timed_out = await client.post("/v1/chat/completions", json=REQUEST)
            assert timed_out.status_code == 504
            assert timed_out.json()["error"]["type"] == "timeout_error"
            assert app.state.waiting_requests == 0
        finally:
            release.set()
            assert (await first).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("error,status", [(TimeoutError(), 504), (RuntimeError("private"), 502)])
async def test_sampling_failure(error, status):
    async def fail(*args, **kwargs):
        raise error

    app = create_app(fail, Codec(), model=MODEL, max_model_length=10)
    async with client_for(app) as client:
        response = await client.post("/v1/chat/completions", json=REQUEST)
    assert response.status_code == status
    assert "private" not in response.text
    assert app.state.active_requests == app.state.active_samples == 0


@pytest.mark.asyncio
async def test_cancel_releases_capacity():
    entered = asyncio.Event()

    async def sample(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    app = create_app(sample, Codec(), model=MODEL, max_model_length=10, max_concurrent=1)
    async with client_for(app) as client:
        task = asyncio.create_task(client.post("/v1/chat/completions", json=REQUEST))
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert app.state.active_requests == app.state.active_samples == 0


@pytest.mark.asyncio
async def test_cancel_keeps_codec_serialized(blocking_codec):
    async def sample(*args, **kwargs):
        return sample_result()

    app = create_app(sample, blocking_codec.codec, model=MODEL, max_model_length=10, max_concurrent=1)
    async with client_for(app) as client:
        first = asyncio.create_task(client.post("/v1/chat/completions", json=REQUEST))
        assert await asyncio.to_thread(blocking_codec.entered.wait, 5)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(client.post("/v1/chat/completions", json=REQUEST))
        try:
            await asyncio.sleep(0.1)
            assert not second.done()
            assert blocking_codec.peak == 1
        finally:
            blocking_codec.release.set()
            response = await second
        assert response.status_code == 200
        assert blocking_codec.peak == 1


@pytest.mark.asyncio
async def test_codec_queue_does_not_starve_thread_pool(blocking_codec):
    async def sample(*args, **kwargs):
        return sample_result()

    app = create_app(
        sample,
        blocking_codec.codec,
        model=MODEL,
        max_model_length=10,
        max_concurrent=THREAD_STARVATION_BURST,
    )
    async with client_for(app) as client:
        jobs = [
            asyncio.create_task(client.post("/v1/chat/completions", json=REQUEST))
            for _ in range(THREAD_STARVATION_BURST)
        ]
        try:
            assert await asyncio.to_thread(blocking_codec.entered.wait, 5)
            while app.state.active_requests < THREAD_STARVATION_BURST:
                await asyncio.sleep(0.01)
            side_work = await asyncio.wait_for(
                asyncio.gather(*(asyncio.to_thread(lambda i=i: i * 2) for i in range(16))),
                timeout=1.0,
            )
            assert side_work == [i * 2 for i in range(16)]
            assert blocking_codec.peak == 1
        finally:
            blocking_codec.release.set()
            responses = await asyncio.gather(*jobs)
    assert all(response.status_code == 200 for response in responses)
    assert blocking_codec.peak == 1


def test_pinned_sdk_contract():
    assert hasattr(SessionClient, "attach_async")
    assert hasattr(SessionClient, "detach_async")
    signature = inspect.signature(Generator.sample_async)
    assert {"prompt", "num_samples", "sampling_params", "timeout"} <= signature.parameters.keys()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("preserve_thinking", [None, True, False])
async def test_recipe_detaches(monkeypatch, fail, preserve_thinking):
    events = []

    async def retrieve():
        return SimpleNamespace(base_model=MODEL)

    async def detach():
        events.append("detach")

    async def attach(**kwargs):
        events.append("attach")
        return SimpleNamespace(
            has_generator=True,
            retrieve_async=retrieve,
            detach_async=detach,
            session_id="borrowed",
            generator=SimpleNamespace(sample_async=None),
        )

    class Server:
        def __init__(self, config):
            pass

        async def serve(self):
            events.append("serve")
            if fail:
                raise RuntimeError("server failure")

    monkeypatch.setattr(openai_endpoint.SessionClient, "attach_async", attach)

    def get_codec(*args, **kwargs):
        assert kwargs["preserve_thinking"] is preserve_thinking
        return Codec()

    monkeypatch.setattr(openai_endpoint, "get_codec", get_codec)
    monkeypatch.setattr(openai_endpoint.uvicorn, "Server", Server)
    args = SimpleNamespace(
        host="127.0.0.1",
        api_key=PROXY_KEY,
        session_id="borrowed",
        enable_thinking=None,
        preserve_thinking=preserve_thinking,
        thinking_effort=None,
        max_model_length=10,
        default_max_tokens=1,
        max_concurrent=256,
        sample_timeout=30,
        port=8000,
    )
    if fail:
        with pytest.raises(RuntimeError, match="server failure"):
            await openai_endpoint.serve(args)
    else:
        await openai_endpoint.serve(args)
    assert events == ["attach", "serve", "detach"]


@pytest.mark.asyncio
async def test_capture_holds_admission():
    entered, release = asyncio.Event(), asyncio.Event()
    samples = 0

    async def sample(*args, **kwargs):
        nonlocal samples
        samples += 1
        return sample_result()

    async def capture(**kwargs):
        if samples == 1:
            entered.set()
            await release.wait()

    app = create_app(sample, Codec(), model=MODEL, max_model_length=10, max_concurrent=1, on_generation=capture)
    async with client_for(app) as client:
        pending = asyncio.create_task(client.post("/v1/chat/completions", json={**REQUEST, "stream": True}))
        second = None
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert not pending.done()
            second = asyncio.create_task(client.post("/v1/chat/completions", json=REQUEST))
            while app.state.waiting_requests < 1:
                await asyncio.sleep(0.01)
            assert not second.done()
            assert samples == 1
        finally:
            release.set()
            first_resp, second_resp = await asyncio.gather(pending, *([second] if second is not None else []))
        assert first_resp.status_code == 200
        assert second_resp.status_code == 200
        assert samples == 2


@pytest.mark.asyncio
async def test_streaming_holds_slot_until_body_consumed(paused_sse_stream):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=paused_sse_stream.asgi),
        base_url="http://wrapper",
        headers={"Authorization": f"Bearer {PROXY_KEY}"},
    ) as client:
        first = asyncio.create_task(client.post("/v1/chat/completions", json={**REQUEST, "stream": True}))
        second = None
        try:
            await asyncio.wait_for(paused_sse_stream.chunk_sent.wait(), timeout=5)
            assert paused_sse_stream.app.state.active_requests == 1
            second = asyncio.create_task(client.post("/v1/chat/completions", json=REQUEST))
            while paused_sse_stream.app.state.waiting_requests < 1:
                await asyncio.sleep(0.01)
            assert not second.done()
            assert paused_sse_stream.sample_calls() == 1
            assert paused_sse_stream.app.state.active_requests == 1
        finally:
            paused_sse_stream.release_stream.set()
            first_resp, second_resp = await asyncio.gather(first, *([second] if second is not None else []))
    assert first_resp.status_code == 200
    assert first_resp.text.endswith("data: [DONE]\n\n")
    assert second_resp.status_code == 200
    assert paused_sse_stream.sample_calls() == 2
    assert paused_sse_stream.app.state.active_requests == 0


@pytest.mark.asyncio
async def test_backend_length(setup):
    async def sample(*args, **kwargs):
        return sample_result(stop="STOP_REASON_LENGTH")

    app = create_app(sample, setup.codec, model=MODEL, max_model_length=10)
    async with client_for(app) as client:
        response = await client.post("/v1/chat/completions", json=REQUEST)
    assert response.json()["choices"][0]["finish_reason"] == "length"
