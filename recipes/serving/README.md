# OpenAI-compatible rollout endpoint

Wrap the native sampler of a live training session so HTTP clients can generate
rollouts. The wrapper renders and parses messages with the same
cookbook codecs used by training. It creates no model resource or training
session, exports no checkpoint, and detaches its handle on exit.

```bash
uv sync --locked --extra serving
export TOGETHER_API_KEY=...
export ROLLOUT_API_KEY=...  # a separate credential for callers of this proxy
uv run --extra serving python recipes/serving/openai_endpoint.py \
  --session-id <existing-session-id> --max-model-length 131072
```

Use the resource's actual context limit. Thinking options should match your
rollout policy; the recipe accepts `--enable-thinking`, `--no-enable-thinking`,
`--preserve-thinking`, `--no-preserve-thinking`, and `--thinking-effort` where
the codec supports them (`preserve_thinking` defaults to `None`, matching the
codec's stock history policy). The recipe resolves the model from the attached
session.

The default host is loopback. `ROLLOUT_API_KEY` is required for every deployment,
including embedded apps on loopback. To accept external connections, set
`--host 0.0.0.0`. Put TLS in front of the service for remote access. Give HTTP
clients the proxy key rather than the Together training key.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $ROLLOUT_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.8-27B","messages":[{"role":"user","content":"Hello"}],"max_tokens":128}'
```

## HTTP clients

Set the client's Base URL to `http://<reachable-host>:8000/v1`, use the proxy API
key, and specify the attached session's exact model ID. Use Chat Completions with
plain text responses and automatic tool selection. The only route is
`POST /v1/chat/completions`.

Streaming may be enabled: `stream=true` returns valid **buffered SSE** after
the complete native sample and capture hook finish. This gives streaming-client
compatibility, with no reduction in time to first output. Increase the client
request timeout and any proxy read/idle timeouts
to cover sampling; the SDK sampling timeout defaults to 7,200 seconds. Set
client retries to zero during validation to expose failures without duplicate
generations. Check a tool call, its tool result, and the final answer in both
JSON and streaming modes when integrating a client.

## Embedding and capture

```python
import asyncio
import json
from pathlib import Path

from post_training_cookbook.serving import create_app

capture_lock = asyncio.Lock()
capture_path = Path("runs/rollout-captures.jsonl")

def append_record(record):
    capture_path.parent.mkdir(parents=True, exist_ok=True)
    with capture_path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")

async def capture(*, metadata, prompt_token_ids, result):
    record = dict(metadata, prompt_token_ids=prompt_token_ids, result=result.model_dump(mode="json"))
    async with capture_lock:
        await asyncio.to_thread(append_record, record)

app = create_app(
    session.generator.sample_async, codec,
    model=model_name, session_id=session.session_id,
    max_model_length=131072, api_key=proxy_key,
    on_generation=capture,
)
```

Construct `codec` with `get_codec(model_name, ...)` using the same thinking and
history options as your rollout policy. By default (`include_reasoning=True`),
JSON/SSE responses include `reasoning_content` when present and accept it in
assistant history. Replay that field unchanged; do not inline it into answer
text. Set `include_reasoning=False` to carry answer text and structured tool
calls only. Train from the captured prompt and completion token IDs, including
the original reasoning tokens, rather than retokenizing the HTTP response. The
standalone CLI has no capture hook; embed the app as above to retain RL data.

The callback receives the **unchanged** `SampleResult`, including completion
tokens, optional logprobs, policy-version segments, and optional routing keys.
To request routing capture, inject a callable that forwards
`return_routed_experts=True` to the native sampler. Missing metadata remains
missing. The wrapper never substitutes zero logprobs.

Request `metadata` can carry run/task/group/rollout identifiers. It is copied
before sampling, under the hook metadata's
`request` key. Server-assigned request ID, model, session ID, and effective
sampling parameters are separate keys. A hook is awaited before success; its
failure produces an HTTP error without resampling. The example serializes
local file writes; it is not a crash-durable storage service.

## Supported subset and limits

- `/v1/chat/completions`: text messages/blocks, function tools, assistant tool
  calls and matching tool results, `tool_choice=auto/none`, `n=1`, temperature,
  top-p, seed, string stops, and `max_tokens`/`max_completion_tokens`.
- The effective generation limit is the smaller of the requested/default
  limit and remaining context. A prompt with no generation space is rejected.
  The backend retains responsibility for model EOS stopping.
- By default JSON and SSE return answer text, structured tool calls, and
  `reasoning_content` when present (`include_reasoning=False` omits
  `reasoning_content`). Reasoning also stays in the captured sampled tokens,
  along with its original token logprobs.
- SSE includes indexed tool-call deltas, finish reason, `[DONE]`, and an
  optional final usage chunk with `stream_options.include_usage=true`.
- A token limit that interrupts tool syntax returns available answer text
  with `finish_reason="length"`; incomplete turns do not emit tool calls.
- Other routes, including models, health, Responses, and API docs, return 404
  for authenticated requests. Unsupported chat semantics return 400: forced tool choice,
  strict schemas/JSON mode, multimodal input, nonzero penalties, requested
  public logprobs, `parallel_tool_calls=false`, and completion storage.
- `--max-concurrent` defaults to 256 local in-flight requests. Excess calls
  queue for capacity up to `--sample-timeout` (returning 504 on timeout). Use
  **one server worker** for a global limit. The SDK's
  `TOGETHER_RL_MAX_CONNECTIONS` defaults to 256; set it before startup if you
  need more connection-pool headroom. Generator capacity remains a property of
  the model resource and its generator replicas.
- Local cancellation does not guarantee remote cancellation. Interrupted
  requests can have unknown remote outcomes; capture-before-success applies
  to completed responses. The wrapper adds no generation retries.
