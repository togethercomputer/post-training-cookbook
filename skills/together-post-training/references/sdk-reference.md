# SDK reference

The Together Post Training SDK is imported from **`together.post_training`**. It provides a session-based training interface where you create a `SessionClient`, then call methods to sample, compute gradients, update weights, and checkpoint.

**Contents:** [SessionClient](#sessionclient) · [ModelResourcesClient](#modelresourcesclient) · [Training sample](#training-sample) · [Model input](#model-input) · [Loss configurations](#loss-configurations) · [Sampling parameters](#sampling-parameters) · [Adam parameters](#adam-parameters) · [Muon optimizer](#muon-optimizer) · [Gradient](#gradient) · [LoRA configuration](#lora-configuration) · [SessionMetadata](#sessionmetadata) · [Session state](#session-state) · [Constant strings](#constant-strings) · [Async usage](#async-usage) · [Full import list](#full-import-list)

Every operation has a sync and an async variant. Use the `_async` variants from async code, including async notebook cells: the sync ones drive their own event loop and raise `RuntimeError` if one is already running.

---

## SessionClient

```python
from together.post_training import SessionClient
```

### Defaults

These constants set the polling behavior used by every operation:

| Constant | Value | Used by |
|---|---|---|
| `DEFAULT_SESSION_CREATION_TIMEOUT` | `3600.0` seconds | `SessionClient.create` |
| `DEFAULT_SESSION_CREATION_INTERVAL` | `10.0` seconds | `SessionClient.create` |
| `DEFAULT_OPERATION_TIMEOUT` | `300.0` seconds | `sample` / `sample_batch` / `forward` / `forward_backward` / `custom_forward_backward` / `optim_step` / `weights_sync` |
| `DEFAULT_OPERATION_INTERVAL` | `0.5` seconds | every non-creation operation |
| `DEFAULT_CHECKPOINT_TIMEOUT` | `7200.0` seconds | `create_inference_checkpoint` / `create_training_checkpoint` |

`timeout` is the maximum wall-clock time to wait. `interval` is the polling interval against the operations API. `timeout=None` waits indefinitely.

---

### `SessionClient.create`

Creates a training session and polls until it reaches `TRAINING_SESSION_STATUS_RUNNING`.

```python
session = SessionClient.create(
    model_resources_id=resources.model_resources_id,  # required; attach to provisioned resources
    api_key=None,                               # defaults to TOGETHER_API_KEY
    base_url=None,                              # defaults to TOGETHER_BASE_URL
    lora_config=LoraConfig(rank=16, alpha=16, dropout=0.05),
    display_name=None,                          # human-readable session name
    metadata=None,                              # SessionMetadata, e.g. W&B run details
    resume_from_checkpoint_id=None,             # pass a training checkpoint ID to resume
    load_optimizer=None,                        # omit/True: restore optimizer+step; False: weights only
    timeout=3600.0,                             # max wait for session startup
    interval=10.0,                              # session-status polling interval
)
```

All parameters are keyword-only. `SessionClient.create` is attach-only: it starts a training session on already-provisioned [`ModelResourcesClient`](#modelresourcesclient). Provision the model first with `ModelResourcesClient.create(base_model=...)` (which is where the base model, generator replicas, and optimizer selection live), then attach — either via `resources.create_session(...)`, the preferred path, or by passing `model_resources_id` here directly.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_resources_id` | `str` | _(required)_ | ID of the provisioned [`ModelResourcesClient`](#modelresourcesclient) to attach the session to. Prefer `ModelResourcesClient.create_session` over calling this directly. |
| `api_key` | `str \| None` | `None` | API key. Defaults to `TOGETHER_API_KEY` env var. |
| `base_url` | `str \| httpx.URL \| None` | `None` | Service URL. Defaults to `TOGETHER_BASE_URL` env var. |
| `lora_config` | `LoraConfig \| None` | `None` | LoRA adapter configuration. For full-weight fine-tuning, provision the resource with `lora_enabled=False` and omit this value. |
| `display_name` | `str \| None` | `None` | Human-readable name for the session. Surfaces on `Session.display_name`. |
| `metadata` | `SessionMetadata \| None` | `None` | Auxiliary session metadata, including optional W&B details. See [`SessionMetadata`](#sessionmetadata). |
| `resume_from_checkpoint_id` | `str \| None` | `None` | Training checkpoint ID to resume from. |
| `load_optimizer` | `bool \| None` | `None` | Whether to restore optimizer state and step from a training checkpoint. Omitted or `True` restores them; `False` loads weights only with a fresh optimizer and step 0. Not valid for inference or HuggingFace checkpoints. |
| `timeout` | `float \| None` | `3600.0` | Max seconds to wait for `TRAINING_SESSION_STATUS_RUNNING`. `None` waits indefinitely. |
| `interval` | `float` | `10.0` | Polling interval in seconds. |

**Returns:** `SessionClient` once the session is `TRAINING_SESSION_STATUS_RUNNING`.

**Raises:**

- `RuntimeError` if the session enters a terminal status (`TRAINING_SESSION_STATUS_STOPPED`, `TRAINING_SESSION_STATUS_STOPPING`, `TRAINING_SESSION_STATUS_ERROR`, `TRAINING_SESSION_STATUS_EXPIRED`).
- `TimeoutError` if `timeout` is exceeded.

If anything goes wrong during creation, the SDK calls `sessions.stop` on the partially created session before re-raising, so you don't leak running sessions.

**Async variant:** `SessionClient.create_async(...)`.

---

### Properties

| Property | Type | Description |
|---|---|---|
| `session_id` | `str` | The session ID held by this handle. Cheap, no API call. |
| `trainer` | `Trainer` | Forward / backward / optimizer operations. |
| `generator` | `Generator` | Generation operations. Raises `RuntimeError` when the resource has no generator replicas. |
| `has_generator` | `bool` | Whether the resource was provisioned with at least one generator replica. |

Session state is not a property — fetch it with the [`retrieve()`](#session-state) method, which is a live API call.

`session.generator` raises `RuntimeError` when the resource was provisioned with `num_generator_replicas=0`, which is exactly what the docs recommend for loops that never sample ([SFT/DPO and custom losses](./custom-losses.md)). Guard with `session.has_generator` if the code path is shared.

---

### `session.generator.sample`

Generates text completions with per-token log-probabilities from the current model weights, for **one** model input.

```python
result = session.generator.sample(
    prompt,                         # one tokenized model input (positional)
    num_samples=8,                  # completions to generate
    sampling_params=SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
    ),
    timeout=300.0,
)
for seq in result.sequences:
    tokens = [int(t) for t in seq.tokens]
```

`prompt`, `num_samples`, and `sampling_params` are positional-or-keyword; `prompt_logprobs`, `timeout`, and `interval` are keyword-only.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | `ModelInput` | _(required, positional)_ | One tokenized [model input](#model-input). |
| `num_samples` | `int \| None` | `None` | Completions to generate. `None` defers to the server default (1). |
| `sampling_params` | `SamplingParams \| None` | `None` | Sampling configuration. |
| `prompt_logprobs` | `bool \| None` | `None` | Also compute teacher-forced log-probabilities of the model-input tokens and return them in [`SampleResult.prompt_logprobs`](#sampleresult). |
| `timeout` | `float \| None` | `300.0` | Operation timeout in seconds. |
| `interval` | `float` | `0.5` | Operation-status polling interval in seconds. |

**Returns:** a single [`SampleResult`](#sampleresult) holding `num_samples` completions.

**Async variant:** `session.generator.sample_async(...)`.

---

### `session.generator.sample_batch`

The same operation for many model inputs in one round trip. Returns one `SampleResult` per input, in the order they were passed.

```python
results = session.generator.sample_batch(
    prompts,                        # iterable of tokenized model inputs
    num_samples=8,                  # completions per model input
    sampling_params=SamplingParams(temperature=0.7, max_tokens=512),
    timeout=300.0,
)
for prompt_example, result in zip(prompt_examples, results, strict=True):
    for seq in result.sequences:
        ...
```

`prompts`, `num_samples`, and `sampling_params` are positional-or-keyword; `prompt_logprobs`, `timeout`, and `interval` are keyword-only.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompts` | `Iterable[ModelInput]` | _(required, positional)_ | Tokenized [model inputs](#model-input). |
| `num_samples` | `int \| None` | `None` | Completions per model input. `None` defers to the server default (1). |
| `sampling_params` | `SamplingParams \| None` | `None` | Sampling configuration, applied to every input. |
| `prompt_logprobs` | `bool \| None` | `None` | Also compute teacher-forced log-probabilities of each model input's tokens and return them in [`SampleResult.prompt_logprobs`](#sampleresult). |
| `timeout` | `float \| None` | `300.0` | Operation timeout in seconds. |
| `interval` | `float` | `0.5` | Operation-status polling interval in seconds. |

**Returns:** `list[SampleResult]`, aligned position-by-position with `prompts`. This is the call the cookbook's GRPO loop uses to sample a whole batch of prompts at once (`post_training_cookbook/grpo/single_turn.py`).

**Async variant:** `session.generator.sample_batch_async(...)`.

#### `SampleResult`

Completions generated for a single model input.

| Field | Type | Description |
|---|---|---|
| `sequences` | `list[SampledSequence]` | The generated completions. |
| `policy_segments` | `list[PolicyVersionSegment]` | Policy versions that produced these completions. |
| `prompt_logprobs` | `list[float] \| None` | Teacher-forced log-probability of each model-input token. Full prompt length: entry `i` is prompt token `i`, and entry `0` is always `0.0` as a placeholder, since the first prompt token has no conditioning context. Present only when `prompt_logprobs=True` was passed to [`session.generator.sample`](#sessiongeneratorsample). |

#### `SampledSequence`

| Field | Type | Description |
|---|---|---|
| `stop_reason` | `StopReason` | Why generation stopped: `"STOP_REASON_LENGTH"` (hit the token budget) or `"STOP_REASON_STOP"` (emitted a stop token or stop string). Always present. |
| `tokens` | `list[str \| int]` | Generated token IDs. Coerce with `int(t)` before decoding. |
| `logprobs` | `list[float] \| None` | Per-token log-probabilities. |

`StopReason` is a string literal type:

```python
from together.post_training import StopReason
```

`policy_segments` is a list of `PolicyVersionSegment(version, start_token)` describing which policy version produced each span of the completions. Version `0` is the initial model; normally there is a single segment `(version, start_token=0)`, but a `WEIGHT_SYNC_TYPE_PIPELINE` publish landing mid-generation can split a rollout across versions. It is `SampleResult` metadata for client-side staleness tracking — async multi-turn uses it for `async/mid_rollout_drift_ratio`.

---

### `session.trainer.forward`

Runs `forward_backward` in forward-only mode: the loss is evaluated and per-sample outputs come back, but the backward pass is skipped, so no gradients accumulate and no weights change. Use it to snapshot reference-model or on-policy log-probabilities (for example, the reference pass in DPO) without advancing the optimizer.

```python
result = session.trainer.forward(
    samples=[sample1, sample2],     # same Sample format as forward_backward
    loss=LossConfig(type="LOSS_TYPE_CROSS_ENTROPY"),
    timeout=300.0,
)
logprobs = [list(out.tensors["logprobs"].data) for out in result.loss_fn_outputs]
```

All parameters are keyword-only.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `samples` | `Iterable[Sample]` | _(required)_ | Samples to score. |
| `loss` | `LossConfig` | _(required)_ | Loss configuration. `LOSS_TYPE_CROSS_ENTROPY` is the usual choice for reading log-probabilities. |
| `return_loss_fn_outputs` | `bool` | `True` | Return the per-sample output tensors. Set it to `False` when only the scalar loss is wanted. |
| `timeout` | `float \| None` | `300.0` | Operation timeout in seconds. |
| `interval` | `float` | `0.5` | Operation-status polling interval in seconds. |

**Returns:** `ForwardBackwardResult`. Sample `i`'s per-slot log-probabilities are `result.loss_fn_outputs[i].tensors["logprobs"].data`, in batch order.

Only the slots the loss scores carry a real log-probability: a slot whose `weights` entry is `0.0` (prompt tokens, by convention) reads back as exactly `0.0`. Pass all-ones `weights` to get a log-probability at every slot.

`return_loss_fn_outputs` is independent of forward-only behavior — it is also accepted by `forward_backward`, where the backward pass still runs and gradients still accumulate. Those gradients are added to any submitted by `custom_forward_backward`, and `optim_step` applies the sum with no error to warn you. `forward` is the read-only door: use it whenever the pass is meant purely to score samples.

**Async variant:** `session.trainer.forward_async(...)`.

The cookbook wraps this in `post_training_cookbook.custom_loss.compute_logprobs`. See [Write a custom loss function](./custom-losses.md).

---

### `session.trainer.forward_backward`

Runs a forward and backward pass to compute gradients. Batched payloads over roughly 0.5 MiB (serialized) are uploaded separately and referenced by ID; the call signature is the same either way.

```python
fb = session.trainer.forward_backward(
    samples=[sample1, sample2],     # batch of training samples
    loss=LossConfig(type="LOSS_TYPE_CROSS_ENTROPY"),
    timeout=300.0,
)
```

All parameters are keyword-only.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `samples` | `Iterable[Sample]` | _(required)_ | Batch of training samples. |
| `loss` | `LossConfig` | _(required)_ | Loss configuration. |
| `return_loss_fn_outputs` | `bool` | `False` | Return the per-sample output tensors alongside the scalar loss. It does not make the pass read-only: gradients still accumulate. For a scoring pass that accumulates nothing, call [`forward`](#sessiontrainerforward) instead. |
| `timeout` | `float \| None` | `300.0` | Operation timeout in seconds. |
| `interval` | `float` | `0.5` | Operation-status polling interval in seconds. |

**Returns:** `ForwardBackwardResult`:

| Field | Type | Description |
|---|---|---|
| `loss` | `float` | Scalar loss for the batch. |
| `metrics` | `dict[str, float] \| None` | Loss-specific metrics. Empty for `LOSS_TYPE_CROSS_ENTROPY`. For `LOSS_TYPE_GRPO`, see the [metrics reference](./metrics.md#service-emitted-metrics-grpo) for the full set (`loss/ratio/mean`, `loss/clip/high_fraction`, `loss/clip/low_fraction`, `loss/kl_policy/mean`, `loss/kl_ref/mean` when `beta > 0`, `loss/aggregate/active_fraction`). |
| `loss_fn_outputs` | `list[LossFnOutput] \| None` | Per-sample output tensors when `return_loss_fn_outputs=True`, `None` otherwise. `loss_fn_outputs[i].tensors["logprobs"].data` holds sample `i`'s per-slot log-probabilities, zeroed wherever `weights` is `0.0`. |

**Async variant:** `session.trainer.forward_backward_async(...)`.

---

### `session.trainer.custom_forward_backward`

Runs a backward pass from gradients you computed client-side, instead of from a built-in `LossConfig`. Use it when the loss must be evaluated in Python — pair it with [`forward`](#sessiontrainerforward) to snapshot log-probabilities, compute the loss with autograd, then submit the per-slot gradients. Snapshot with `forward`, not `forward_backward(return_loss_fn_outputs=True)`: the latter accumulates the built-in loss's gradients on top of the ones submitted here.

```python
fb = session.trainer.custom_forward_backward(
    samples=[sample1, sample2],     # same Sample format as forward_backward
    gradients=[grad1, grad2],       # one Gradient per sample (dL/d logprob per slot)
    timeout=300.0,
)
```

All parameters are keyword-only.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `samples` | `Iterable[Sample]` | _(required)_ | Batch of training samples. |
| `gradients` | `Iterable[Gradient]` | _(required)_ | Per-sample, per-slot gradients of the loss with respect to each slot's log-probability. |
| `timeout` | `float \| None` | `300.0` | Operation timeout in seconds. |
| `interval` | `float` | `0.5` | Operation-status polling interval in seconds. |

**Returns:** the operation result (loss and any metrics).

**Async variant:** `session.trainer.custom_forward_backward_async(...)`.

The cookbook wraps the whole forward → autograd → backward flow in `post_training_cookbook.custom_loss.compute_custom_forward_backward`. See [Write a custom loss function](./custom-losses.md).

---

### `session.trainer.optim_step`

Applies accumulated gradients and updates the trainer's model parameters. It does not make the updated parameters available for sampling; call [`weights_sync`](#sessiontrainerweights_sync) afterwards when subsequent samples should use them.

```python
result = session.trainer.optim_step(
    adam_params=AdamParams(
        beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1, learning_rate=1e-5,
    ),
    timeout=300.0,
)
print(f"step {result.step}")
```

All parameters are keyword-only. Neither the learning rate nor the gradient-norm clip is a top-level argument — both live inside the optimizer params (`adam_params.learning_rate`, `adam_params.grad_clip_norm`, or their `muon_params` equivalents).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `adam_params` | `AdamParams \| None` | `None` | Adam optimizer parameters, including `learning_rate` and `grad_clip_norm`. Use on AdamW resources (the default). |
| `muon_params` | `MuonParams \| None` | `None` | Muon optimizer parameters, including `learning_rate` and `grad_clip_norm`. Valid only on a resource created with a Muon `optimizer_config`. |
| `timeout` | `float \| None` | `300.0` | Operation timeout in seconds. |
| `interval` | `float` | `0.5` | Operation-status polling interval in seconds. |

**Returns:** `OptimStepResult` with `.step` (current training step counter).

**Async variant:** `session.trainer.optim_step_async(...)`.

---

### `session.trainer.weights_sync`

Makes the trainer's current parameters available to the generator. In a synchronous on-policy loop, call it after every `optim_step` and before the next `sample`:

```python
sync = session.trainer.weights_sync(
    weight_sync_type="WEIGHT_SYNC_TYPE_SYNCHRONOUS",
    timeout=300.0,
)
print(f"weights version {sync.weights_version}")
```

All parameters are keyword-only.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `weight_sync_type` | `WeightSyncType` | _(required)_ | How the current trained parameters are made available for sampling. The cookbook uses `"WEIGHT_SYNC_TYPE_SYNCHRONOUS"`. |
| `timeout` | `float \| None` | `300.0` | Operation timeout in seconds. |
| `interval` | `float` | `0.5` | Operation-status polling interval in seconds. |

`weight_sync_type` accepts:

| Value | Behavior |
|---|---|
| `WEIGHT_SYNC_TYPE_SYNCHRONOUS` | Publishes the weights before `weights_sync` returns. The next `sample` is guaranteed to use the updated policy. GRPO and other on-policy cookbook loops use this; trainer-only SFT/DPO skip `weights_sync`. |
| `WEIGHT_SYNC_TYPE_BACKGROUND_PUBLISH` | Queues publication and returns without waiting for it to complete. |
| `WEIGHT_SYNC_TYPE_PIPELINE` | Allows publication while rollouts are in flight. Rejected with `InvalidArgument` on LoRA sessions: a rollout is bound to its adapter at admission and cannot migrate mid-decode. |

**Returns:** `WeightsSyncResult` with `.weights_version`, the policy version made available or queued for sampling.

**Async variant:** `session.trainer.weights_sync_async(...)`.

---

### `session.create_inference_checkpoint`

Snapshots the current model state for inference or download.

```python
ckpt = session.create_inference_checkpoint(
    timeout=7200.0,
)
print(ckpt.id, ckpt.registration)         # registry artifacts for deployment
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `timeout` | `float \| None` | `7200.0` | Operation timeout. |
| `interval` | `float` | `0.5` | Operation polling interval. |

**Returns:** `InferenceCheckpoint` with `.id`, `.step`, `.created_at`, and `.registration`.

The checkpoint also appears in `session.retrieve().inference_checkpoints`.

**Async variant:** `session.create_inference_checkpoint_async(...)`.

---

### `session.create_training_checkpoint`

Saves full training state (adapter plus optimizer plus step) for later resume. Pass its `id` as `resume_from_checkpoint_id`. Omit `load_optimizer` or pass `True` to restore optimizer and step; `load_optimizer=False` loads weights only (fresh optimizer, step 0).

```python
save = session.create_training_checkpoint(
    timeout=7200.0,
)
print(save.id)                      # pass to resume_from_checkpoint_id
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `timeout` | `float \| None` | `7200.0` | Operation timeout. |
| `interval` | `float` | `0.5` | Operation polling interval. |

**Returns:** `TrainingCheckpoint` with `.id`, `.step`, `.created_at`, and `.registration`.

**Async variant:** `session.create_training_checkpoint_async(...)`.

---

### `download_checkpoint`

Downloads checkpoint files to a local directory. It is a module-level function, not a `SessionClient` method, so it works without an open session.

```python
from pathlib import Path
from together import Together
from together.post_training import download_checkpoint

paths = download_checkpoint(
    Together(),
    checkpoint_id,
    variant="CHECKPOINT_VARIANT_MERGED",    # or "CHECKPOINT_VARIANT_ADAPTER"
    output_dir=Path("./weights"),
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `client` | `Together` | _(required, positional)_ | Client used to request and download the files. |
| `checkpoint_id` | `str` | _(required, positional)_ | Inference checkpoint ID. |
| `variant` | `CheckpointVariant` | `"CHECKPOINT_VARIANT_MERGED"` | Full model or adapter-only. |
| `output_dir` | `str \| Path` | `"."` | Local directory (created if needed). |

**Returns:** `list[Path]` pointing to the downloaded files. The destination directory is created if it doesn't exist.

**Async variant:** `download_checkpoint_async(AsyncTogether(), ...)`.

---

### `session.stop`

Stops the training session and releases resources (closes the underlying HTTP client and the session handle's private event loop).

```python
session.stop()
```

`SessionClient` supports both sync and async context manager protocols:

```python
# Sync
with ModelResourcesClient.create(base_model="Qwen/Qwen3.8-27B") as resources:
    with resources.create_session() as session:
        session.generator.sample(prompt)
    # session.stop() called on inner exit
# resources.stop() called on outer exit

# Async
async with await ModelResourcesClient.create_async(base_model="Qwen/Qwen3.8-27B") as resources:
    async with await resources.create_session_async() as session:
        await session.generator.sample_async(prompt)
    # session.stop_async() called on inner exit
# resources.stop_async() called on outer exit
```

**Async variant:** `session.stop_async()`.

---

## ModelResourcesClient

Provisions shared GPU resources once, then attaches multiple LoRA training sessions to them. This is the multi-LoRA pattern: train several adapters over the same base model without re-provisioning the model for each.

```python
from together.post_training import ComputeConfig, ModelResourcesClient, LoraConfig

resources = ModelResourcesClient.create(
    base_model="Qwen/Qwen3.8-27B",
    api_key=None,                   # defaults to TOGETHER_API_KEY
    base_url=None,                  # defaults to TOGETHER_BASE_URL
    lora_enabled=True,
    compute_config=ComputeConfig(num_generator_replicas=1),  # attached sessions inherit this; omit for the server default (1)
    optimizer_config=None,                # Adam (default) or Muon; inherited by attached sessions
)

session_a = resources.create_session(lora_config=LoraConfig(rank=8, alpha=16))
session_b = resources.create_session(lora_config=LoraConfig(rank=16, alpha=32))

# ... train, checkpoint, and stop each session independently ...

session_a.stop()
session_b.stop()
resources.stop()                    # releases the shared GPUs
```

`ModelResourcesClient` is a context manager that calls `stop()` on exit; each attached `SessionClient` is also a context manager.

```python
with ModelResourcesClient.create(base_model="Qwen/Qwen3.8-27B") as resources:
    with resources.create_session() as session:
        ...
```

### Supported models and GPU hardware

Live discovery is the source of truth (modes, GPUs, context length) — do not
maintain a local catalog:

```python
from together import Together

for model in Together().post_training.supported_models.get().data:
    print(f"{model.base_model} (default: {model.default_gpu_type})")
    print(" ", [config.gpu_type for config in model.compute_configs or []])
```

```bash
uv run python scripts/supported_models.py
```

GRPO / any `sample` loop needs a row with `sampling`. Full-weight vs LoRA is
per-model in that response. Non-default GPU:

```python
from together.post_training import ComputeConfig, ModelResourcesClient

with ModelResourcesClient.create(
    base_model="Qwen/Qwen3.6-35B-A3B",
    compute_config=ComputeConfig(
        gpu_type="B200-SXM",
        num_generator_replicas=1,
    ),
) as resources:
    resource = resources.retrieve()
    print(resource.compute_config.gpu_type)
```

Omit `gpu_type` for `default_gpu_type`. Invalid model/GPU/mode combos fail at
create. Sessions inherit the resource GPU; change hardware by provisioning anew.

### Estimate cost before provisioning

Use the resource-level client to estimate the on-demand hourly price for a model resource without creating it. Pass the same model and resource configuration you plan to use when provisioning:

```python
from together import Together

with Together() as client:
    estimate = client.post_training.model_resources.estimate_cost(
        base_model="zai-org/GLM-5.3",
        compute_config={"gpu_type": "B200-SXM", "num_generator_replicas": 1},
        lora_enabled=True,
    )
    print(f"{estimate.price_per_hour:.2f} {estimate.currency}/hour")
```

`base_model` is required. `compute_config`, `lora_enabled`, and `optimizer_config` are optional and match the corresponding model-resource creation settings. The response contains:

| Field | Description |
|---|---|
| `price_per_hour` | Estimated on-demand hourly price in the currency's major unit. |
| `currency` | ISO 4217 currency code. |

Run the standalone example from the repository root:

```bash
uv run python scripts/estimate_model_resource_cost.py
```

#### When billing starts and stops

For newly provisioned model resources, on-demand GPU billing starts when the resource reaches `MODEL_RESOURCES_STATUS_READY`. Initial provisioning, including downloading and loading the base weights before the resource is READY, is not billable. Billing then covers elapsed time while the resource is READY, including session initialization, runtime warmup, validation, checkpointing, waiting for external reward services, and idle time. Billing ends when the resource transitions to `MODEL_RESOURCES_STATUS_STOPPING`, `MODEL_RESOURCES_STATUS_STOPPED`, or `MODEL_RESOURCES_STATUS_ERROR`.

GPU-hours are the allocated GPU count multiplied by billable hours. To infer GPU count from usage, divide by elapsed time since the resource became READY, using the same cutoff as the usage total. Including initial provisioning in that denominator understates the GPU count, especially for short runs.

#### Estimate a complete training run

The endpoint returns the resource's hourly price, not the total cost of a training run. Estimate the total from the number of training steps and a measured steady-state step time:

1. Determine the total number of training steps required to process the full dataset with the planned configuration.
2. Launch a short test run with the same model-resource configuration and workload settings as the planned run, including LoRA or full-weight training, generator replicas, batch and group sizes, and sequence lengths.
3. Discard the first step's timing because it can include one-time runtime warmup and cache initialization. Measure one subsequent step. For GRPO, read `timing/step_total_s` from `metrics.jsonl`. For other loops, use the difference between adjacent `elapsed_s` values from consecutive step records after the first step. Ignore step-number gaps and avoid intervals that include validation or checkpointing.
4. Multiply the measured step time by the total step count, convert seconds to hours, and multiply by the endpoint's hourly price:

```python
billable_overhead_seconds = 0
estimated_run_seconds = total_steps * steady_state_step_seconds + billable_overhead_seconds
estimated_run_cost = estimated_run_seconds / 3600 * estimate.price_per_hour
print(f"{estimated_run_cost:.2f} {estimate.currency}")
```

Leave `billable_overhead_seconds` at `0` for a steady-state training estimate. For a budget estimate, add measured overhead incurred after the resource becomes READY: session initialization, extra first-step warmup beyond the steady-state estimate, reference-logprob preprocessing, all expected validation and checkpoint runs, and idle time before the resource is stopped. Exclude initial provisioning before READY.

This formula assumes the run exclusively occupies the model resource. When several LoRA sessions share one resource, estimate the shared resource's cost once from its billable lifetime, then allocate that cost among the concurrent runs using your accounting policy. Attaching or stopping an individual session does not resize the resource's GPU allocation. Stop the model resource when all users are finished; detaching sessions leaves it running and billable while READY. Do not apply the full resource price independently to every attached session.

### `ModelResourcesClient.create`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `base_model` | `str` | _(required)_ | Base model to provision. |
| `api_key` | `str \| None` | `None` | Defaults to `TOGETHER_API_KEY`. |
| `base_url` | `str \| httpx.URL \| None` | `None` | Defaults to `TOGETHER_BASE_URL`. |
| `lora_enabled` | `bool` | `True` | Enable LoRA on the shared resources. |
| `compute_config` | `ComputeConfig \| None` | `None` | Compute layout. `gpu_type` selects an advertised hardware configuration (omitting it uses the model default); `num_generator_replicas` (server default `1`) sets generator replicas, with `0` for trainer-only. Inherited by attached sessions. |
| `optimizer_config` | `OptimizerConfig \| None` | `None` | Optimizer selection (Adam or Muon) for the resource. `None` uses Adam. Inherited by every session attached to the resource. See [Optimizer config](#optimizerconfig). |
| `timeout` | `float \| None` | _(SDK default)_ | Max seconds to wait for `MODEL_RESOURCES_STATUS_READY`, including time spent waiting in the queue for GPUs. `None` waits as long as needed. When it runs out, the client stops the resource. See [Waiting for GPUs](#waiting-for-gpus). |
| `interval` | `float` | _(SDK default)_ | Status polling interval. |

### Methods

| Method | Returns | Description |
|---|---|---|
| `create_session(*, display_name=None, metadata=None, resume_from_checkpoint_id=None, load_optimizer=None, lora_config=None, timeout=..., interval=...)` | `SessionClient` | Start a training session on the shared resources. `load_optimizer` has the same semantics as on [`SessionClient.create`](#sessionclientcreate). |
| `retrieve()` | resource model | Fetch current resource state from the API. |
| `stop()` | — | Stop the resources and release GPUs. |
| `detach()` | — | Release the local handle without stopping the remote resource. |
| `model_resources_id` (property) | `str` | ID of the provisioned resources. |

Each method has an `_async` variant. Resource status values: `MODEL_RESOURCES_STATUS_PENDING`, `MODEL_RESOURCES_STATUS_CREATING`, `MODEL_RESOURCES_STATUS_READY`, `MODEL_RESOURCES_STATUS_ERROR`, `MODEL_RESOURCES_STATUS_STOPPING`, `MODEL_RESOURCES_STATUS_STOPPED`.

### Waiting for GPUs

`ModelResourcesClient.create` returns once the resource is `MODEL_RESOURCES_STATUS_READY`. When no cluster has free GPUs of the resource's type, the resource waits in a queue with status `MODEL_RESOURCES_STATUS_PENDING`. You can queue as many resources as you like, and a queued resource waits until it gets GPUs or you stop it. A create is rejected (HTTP 429) only when it would take your organization over its GPU quota; contact support to raise the quota.

While a resource waits, `status_details.queue_position` shows its estimated place in the queue: the number of older resources of the same GPU type that are still waiting, plus one. The field is not set once the resource leaves the queue.

```python
from together import Together

resource = Together().post_training.model_resources.retrieve(model_resources_id)
position = resource.status_details.queue_position
if position is not None:
    print(f"Waiting for GPUs, queue position {position}")
```

The `timeout` of `ModelResourcesClient.create` counts time in the queue. When it runs out, the client stops the resource, so a busy queue can end a run before it starts. Pass `timeout=None` to wait as long as needed.

### Attaching with `SessionClient.create`

`resources.create_session(...)` is sugar for `SessionClient.create(model_resources_id=resources.model_resources_id, ...)`. Either path starts an adapter session on the shared resource; pass `model_resources_id` directly when you only have the ID (for example, a resource a teammate provisioned). The cookbook's [`train_*` entry points](./cookbook-guide.md#multi-lora--shared-model-resources) take this path via `config.model_resources_id`.

### Discovering and inspecting resources

The resource-level client lists and fetches resources without binding a `ModelResourcesClient` object — useful for finding an existing resource to attach to, or validating one before use:

```python
from together import Together

client = Together()
for r in client.post_training.model_resources.list(status="MODEL_RESOURCES_STATUS_READY").data:
    print(r.id, r.base_model, r.type)

resource = client.post_training.model_resources.retrieve(model_resources_id)
# compute_config includes gpu_type and num_generator_replicas (0 = trainer only)
```

| Method | Description |
|---|---|
| `client.post_training.model_resources.estimate_cost(...)` | Estimate the on-demand hourly price without creating a resource. |
| `client.post_training.model_resources.list(status=...)` | List resources, optionally filtered by status. Returns `.data`. |
| `client.post_training.model_resources.retrieve(id)` | Fetch one resource's state (`status`, `status_details`, `base_model`, `lora_enabled`, `type`). |
| `client.post_training.model_resources.stop(id)` | Stop a resource by ID. |

---

## Training sample

A `Sample` combines model input tokens with loss inputs that control what the model learns.

**The one rule.** Slot `i` scores the prediction made at input position `i` against `target_tokens[i]`. There is no server-side shift, so a run of N tokens becomes a payload of N-1 slots: `model_input` drops the sequence's last token, which is a target only, and every loss array drops its first entry, because slot 0 already predicts the second token.

```python
from together.post_training import (
    Sample,
    ModelInput, ModelInputChunk, EncodedTextChunk,
)

tokens = [101, 102, 103, 104, 105]    # token space: 2 prompt tokens, then 3 response tokens

# As sent: N-1 = 4 slots.
model_input   = tokens[:-1]           # [101, 102, 103, 104]
target_tokens = tokens[1:]            # [102, 103, 104, 105]
weights       = [0.0, 1.0, 1.0, 1.0]  # slot 1 is the first response token

sample = Sample(
    model_input=ModelInput(
        chunks=[ModelInputChunk(
            encoded_text=EncodedTextChunk(tokens=model_input)
        )]
    ),
    loss_fn_inputs={
        "weights": {"data": weights, "dtype": "float32"},
        "target_tokens": {"data": target_tokens, "dtype": "int64"},
    },
)
```

Two consequences of that rule catch people out: the zero run in `weights` is one shorter than the prompt — two prompt tokens, one leading zero — and the **last weight is `1.0`**, because the final response token is a target like any other.

The cookbook builds samples through `post_training_cookbook.grpo.samples.build_policy_sample` and `post_training_cookbook.sft.train.build_conversation_samples`. Both take their arrays in token space, length N, and own the trim, so you never write it yourself. Build samples by hand only when you need full control.

### Fields

Under the rule above, all the arrays below hold N-1 entries, one per slot.

| Field | Type | Required | Description |
|---|---|---|---|
| `model_input.chunks[].encoded_text.tokens` | `list[str \| int]` | Always | The input tokens, `tokens[:-1]`: the sequence without its final token, which is a target only. |
| `loss_fn_inputs.target_tokens.data` | `list[int]` | Always | What each slot is scored against, `tokens[1:]`, with `dtype="int64"`. |
| `loss_fn_inputs.weights.data` | `list[float]` | For cross-entropy | Non-negative per-slot weights with `dtype="float32"`. Cross-entropy supports fractional weights; policy losses require 0/1 values. Weights also select which slots return a real log-probability in `loss_fn_outputs`: a `0.0` weight reads back as `0.0`, so a caller that wants log-probabilities everywhere must pass all-ones weights. |
| `loss_fn_inputs.advantages` | tensor dictionary | Policy losses | Per-slot advantages with `dtype="float32"`. |
| `loss_fn_inputs.logprobs` | tensor dictionary | Policy losses | Per-slot rollout logprobs with `dtype="float32"`. |
| `loss_fn_inputs.reference_logprobs` | tensor dictionary | GRPO with KL | Optional per-slot reference-policy logprobs with `dtype="float32"`. Leave the key out entirely when unused, rather than zero-filling it. |

`loss_fn_inputs` is the required tensor representation. Tensor dtypes use lowercase `"int64"` and `"float32"`.

The same `Sample` shape is used by `forward`, `forward_backward`, and `custom_forward_backward`.

### If you build by hand

Which end an array loses depends on what indexes it: **loss** arrays trim from the head (`arr[1:]`), while `model_input`, the only array indexed by *input* position, narrows from the **tail**. Trim one at the wrong end and it still has the right length with the wrong contents, so no length check can catch it. What that costs, and whether you would see it:

| Array | Indexed by | Cost of trimming the wrong end | Does a metric move? |
|---|---|---|---|
| `model_input` | input position | Every slot scored against the wrong hidden state. | Yes — loss is immediately nonsense. |
| `target_tokens` | slot | Every slot scored against the previous slot's target. | Yes — loss explodes. |
| `weights` | slot | Trains the prompt's last slot, skips the first response token. | Barely — the trained set moves by one slot, which perturbs `loss/ratio/mean` without a clear signature. |
| `advantages` | slot | Each gradient's sign and size land on the neighbouring slot. | **No.** `advantages` never enters the importance ratio, so nothing you plot moves. |
| `logprobs` | slot | Every ratio pairs a trainer logprob with the wrong generator logprob. | **Yes, directly.** This is the sampling-time baseline of the ratio, so `loss/ratio/mean` leaves 1.0 and `loss/kl_policy/mean` blows up — 5.17 and 1351 at `beta=0.04` on a fully misaligned payload. |
| `reference_logprobs` | slot | KL penalty measured against the neighbouring slot. | Only at `beta > 0`, through the KL metrics. |

### GRPO inputs

For GRPO loss, each sample additionally requires per-slot advantages and the generator's log-probabilities:

```python
loss_fn_inputs = {
    "target_tokens": {"data": target_tokens, "dtype": "int64"},  # [102, 103, 104, 105]
    "weights": {"data": weights, "dtype": "float32"},            # [0.0, 1.0, 1.0, 1.0]
    "advantages": {
        # One scalar for the sequence, spread over the slots the mask trains.
        "data": [0.0, 1.5, 1.5, 1.5],
        "dtype": "float32",
    },
    "logprobs": {
        # Slot 1 holds the generator's logprob for target_tokens[1], the first
        # response token.
        "data": [0.0, -2.1, -0.5, -1.3],
        "dtype": "float32",
    },
    # Required when beta > 0; leave the key out entirely otherwise.
    # "reference_logprobs": {
    #     "data": [0.0, -2.0, -0.6, -1.4],
    #     "dtype": "float32",
    # },
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `advantages` | `TensorData` | Yes | Float32 per-slot advantage values. The leading slots covering the prompt are `0.0`; a P-token prompt leaves P-1 of them. |
| `logprobs` | `TensorData` | Yes | Float32 log-probs from the sampling model, one per slot. The leading prompt slots are `0.0`. |
| `reference_logprobs` | `TensorData` | If `beta > 0` | Float32 log-probs from the reference model for the KL penalty, one per slot. |

---

## Model input

Tokenized input for sampling and for the `model_input` field of a training [sample](#training-sample) — one type for both:

```python
from together.post_training import ModelInput, ModelInputChunk, EncodedTextChunk

prompt = ModelInput(
    chunks=[ModelInputChunk(
        encoded_text=EncodedTextChunk(tokens=[101, 102, 103])
    )]
)
```

The cookbook's `post_training_cookbook.samples.make_prompt` wraps this construction for a flat token list.

---

## Loss configurations

### Cross-entropy (SFT)

Standard next-token prediction loss. Requires `weights` and `target_tokens`.

```python
from together.post_training import LossConfig

loss = LossConfig(type="LOSS_TYPE_CROSS_ENTROPY")
```

### GRPO

Group relative policy optimization. Requires flat `target_tokens`, `advantages`, and `logprobs` entries; `weights` is optional.

```python
from together.post_training import LossConfig, GrpoLossParams

# Omit clip_* to use server defaults (0.8 / 1.2 for GRPO/PPO; 0.0 / 4.0 for CISPO).
loss = LossConfig(
    type="LOSS_TYPE_GRPO",
    grpo_params=GrpoLossParams(
        agg_type="GRPO_LOSS_AGGREGATION_TYPE_FIXED_HORIZON",
    ),
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `clip_low_threshold` | `float` | `0.8` (server; omit) | Absolute lower bound the importance-sampling ratio is clamped to. Validated `[0, 1]`; closer to `1.0` clips harder. CISPO server omit-default is `0.0`. |
| `clip_high_threshold` | `float` | `1.2` (server; omit) | Absolute upper bound. Validated `>= 1`; closer to `1.0` clips harder. CISPO server omit-default is `4.0`. |
| `beta` | `float` | `0.0` | KL penalty coefficient. When `> 0`, `reference_logprobs` must be provided. |
| `agg_type` | `Literal[...]` | `"GRPO_LOSS_AGGREGATION_TYPE_FIXED_HORIZON"` | One of `GRPO_LOSS_AGGREGATION_TYPE_FIXED_HORIZON`, `GRPO_LOSS_AGGREGATION_TYPE_TOKEN_MEAN`, `GRPO_LOSS_AGGREGATION_TYPE_SEQUENCE_MEAN`. |
| `ratio_type` | `Literal[...]` | `"GRPO_LOSS_RATIO_TYPE_TOKEN"` | Importance-ratio granularity. `GRPO_LOSS_RATIO_TYPE_TOKEN` is standard GRPO (per-token ratio); `GRPO_LOSS_RATIO_TYPE_SEQUENCE` is the GSPO sequence-level ratio. |

The two clip thresholds are the bounds themselves, not DAPO-style offsets from `1.0`: a DAPO epsilon pair `(ε_low, ε_high)` maps to `clip_low_threshold = 1 - ε_low` and `clip_high_threshold = 1 + ε_high`, so DAPO's `(0.2, 0.28)` is `(0.8, 1.28)`. That asymmetric band is DAPO's "Clip-Higher": the ceiling sits further from `1.0` than the floor. The server omit-default `0.8 / 1.2` is the symmetric PPO band; set `1.28` explicitly when you want Clip-Higher.

Valid `LossConfig.type` values for the cookbook's losses: `"LOSS_TYPE_CROSS_ENTROPY"`, `"LOSS_TYPE_GRPO"`. The SDK additionally accepts `"LOSS_TYPE_PPO"`, `"LOSS_TYPE_CISPO"`, `"LOSS_TYPE_DPPO"`, `"LOSS_TYPE_DRO"`, and `"LOSS_TYPE_IMPORTANCE_SAMPLING"`, each with its own `*_params` and `*_inputs` types where applicable.

---

## Sampling parameters

```python
from together.post_training import SamplingParams

params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
)
```

| Field | Type | Description |
|---|---|---|
| `max_tokens` | `int` | Maximum tokens to generate per completion. Unset defaults to `512` on the server. |
| `temperature` | `float` | Sampling temperature. |
| `top_p` | `float` | Nucleus sampling threshold. |
| `top_k` | `int` | Top-k limit (use `-1` to disable). |
| `stop` | `list[str]` | Stop sequences. |
| `seed` | `str \| int` | Random seed for reproducibility. |

All fields are optional; the server applies defaults.

Teacher-forced prompt log-probabilities are not a sampling parameter — request them with the `prompt_logprobs` keyword argument of [`session.generator.sample`](#sessiongeneratorsample) / [`session.generator.sample_batch`](#sessiongeneratorsample_batch).

---

## Adam parameters

```python
from together.post_training import AdamParams

params = AdamParams(
    beta1=0.9,
    beta2=0.95,
    eps=1e-8,
    weight_decay=0.1,
    learning_rate=1e-5,
)
```

| Field | Type | Description |
|---|---|---|
| `beta1` | `float` | First moment decay. |
| `beta2` | `float` | Second moment decay. |
| `eps` | `float` | Numerical stability epsilon. |
| `learning_rate` | `float` | Learning rate for the step. |
| `weight_decay` | `float` | Weight decay coefficient. |
| `grad_clip_norm` | `float` | Maximum gradient norm for this step; gradients across all model parameters are clipped to it. `0` disables clipping. Unset falls back to the session default (`1.0`). |

All fields are optional; the server applies defaults.

---

## Muon optimizer

The Muon optimizer is an alternative to AdamW for the weight matrices. Selecting it has two parts: pick Muon at resource provisioning with an `OptimizerConfig` on `ModelResourcesClient.create`, then pass `MuonParams` to each `optim_step`. The optimizer choice belongs to the resource and is inherited by every session attached to it.

```python
from together.post_training import (
    AdamParams,
    ModelResourcesClient,
    MuonConfig,
    MuonParams,
    OptimizerConfig,
)

with ModelResourcesClient.create(
    base_model="Qwen/Qwen3.8-27B",
    optimizer_config=OptimizerConfig(
        muon=MuonConfig(scaling_strategy="MUON_SCALING_STRATEGY_MATCH_ADAM")
    ),
) as resources:
    session = resources.create_session()

    session.trainer.optim_step(
        muon_params=MuonParams(
            learning_rate=2e-2,
            momentum=0.95,
            newton_schulz_steps=5,
            weight_decay=0.0,
            adam=AdamParams(learning_rate=1e-3),  # params Muon does not tune
        ),
    )
    session.trainer.weights_sync(
        weight_sync_type="WEIGHT_SYNC_TYPE_SYNCHRONOUS",
    )
```

### `OptimizerConfig`

Resource-level optimizer selection, passed to `ModelResourcesClient.create(optimizer_config=...)`. Set exactly one of `adam` or `muon`.

| Field | Type | Description |
|---|---|---|
| `adam` | `AdamConfig` | Select Adam. An empty object — pass `{}`. This is the default when `optimizer_config` is `None`; the per-step hyperparameters live in `AdamParams` on `optim_step`. |
| `muon` | `MuonConfig` | Select Muon. Its only field is `scaling_strategy: MuonScalingStrategy` — one of `"MUON_SCALING_STRATEGY_UNSPECIFIED"`, `"MUON_SCALING_STRATEGY_MATCH_ADAM"`, `"MUON_SCALING_STRATEGY_ORIGINAL"`. |

### `MuonParams`

Per-`optim_step` Muon parameters (analogous to `AdamParams`).

| Field | Type | Description |
|---|---|---|
| `learning_rate` | `float` | Learning rate / Newton–Schulz step size. |
| `momentum` | `float` | Momentum for the weight-matrix updates. |
| `newton_schulz_steps` | `int` | Iterations of the Newton–Schulz orthogonalization. |
| `weight_decay` | `float` | Weight decay for the weight-matrix updates. |
| `grad_clip_norm` | `float` | Maximum gradient norm for this step; gradients across all model parameters are clipped to it. `0` disables clipping. Unset falls back to the session default (`1.0`). |
| `adam` | `AdamParams` | Per-step overrides for the parameters Muon does not tune (embeddings, norm layers, LM head). |

All fields are optional; the server applies defaults.

---

## Gradient

Per-sample, per-slot gradients submitted to `session.trainer.custom_forward_backward`. One `Gradient` per sample, aligned to the log-probabilities `forward` returned for it: N-1 entries for a run of N tokens.

```python
from together.post_training import Gradient

grad = Gradient(
    data=[0.0, -0.12, 0.4, -0.05],     # per slot; a 2-token prompt leaves 1 leading zero
    dtype="D_TYPE_FLOAT32",
)
```

| Field | Type | Description |
|---|---|---|
| `data` | `Iterable[float]` | Gradient of the loss with respect to each slot's log-probability. |
| `dtype` | `DType` | Data type of the array (typically `"D_TYPE_FLOAT32"`). |

---

## LoRA configuration

```python
from together.post_training import LoraConfig

lora = LoraConfig(
    rank=16,
    alpha=16,
    dropout=0.05,
)
```

| Field | Type | Description |
|---|---|---|
| `rank` | `int` | Rank of LoRA adapter matrices. |
| `alpha` | `int` | LoRA scaling factor. |
| `dropout` | `float` | Dropout on LoRA layers. |

All fields are optional; the server applies defaults. `LoraConfig` has no `enable` field. For full-weight fine-tuning, provision the resource with `ModelResourcesClient.create(lora_enabled=False)` and omit `lora_config` entirely.

---

## `SessionMetadata`

Auxiliary metadata attached at session creation and echoed back on `Session.metadata`. Passed to `SessionClient.create(metadata=...)` or `create_session(metadata=...)`.

```python
from together.post_training import SessionMetadata, WandbMetadata

metadata = SessionMetadata(
    wandb=WandbMetadata(project="grpo-gsm8k", entity="my-team", run_name="lr1e-5-rank8"),
)
```

| Field | Type | Description |
|---|---|---|
| `wandb` | `WandbMetadata` | W&B run details for this session. |

`WandbMetadata` fields, all optional: `project`, `entity`, `group`, `run_name`, `run_id`, `url` — every one a `str`.

The cookbook builds this from its own W&B config in `post_training_cookbook.config.session_metadata_from`, which forwards `wandb_project` / `wandb_entity` / `wandb_group` / `wandb_run_name` and returns `None` when none are set.

---

## Session state

The `session.retrieve()` method returns a `Session`:

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Session ID. |
| `display_name` | `str \| None` | Whatever was passed as `display_name` at creation. |
| `metadata` | `SessionMetadata` | Whatever was passed as `metadata` at creation, read as attributes (`.wandb.project`). See [`SessionMetadata`](#sessionmetadata) for the request-side fields. |
| `status` | `Literal[...]` | One of the session-status constants below. |
| `step` | `str \| int` | Current training step. |
| `lora_config` | `LoraConfig \| None` | LoRA configuration, read as attributes (`.rank`). |
| `resources_id` | `str` | ID of the [`ModelResourcesClient`](#modelresourcesclient) backing this session. Always present, since every session attaches to a resource. The wire field is `model_resources_id`; the Python attribute is `resources_id`. |
| `training_checkpoints` | `list[TrainingCheckpoint]` | Saved training states. |
| `inference_checkpoints` | `list[InferenceCheckpoint]` | Saved model snapshots. |
| `resume_from_checkpoint_id` | `str \| None` | Set when this session resumed from a training checkpoint. |
| `error` | `SessionError \| None` | Failure detail once the session reaches `TRAINING_SESSION_STATUS_ERROR`. |
| `created_by` | `str` | Account that created the session. |
| `created_at` | `datetime` | Session creation time. |
| `updated_at` | `datetime` | Last update time. |

The base model, compute layout, and optimizer live on the [`ModelResourcesClient`](#modelresourcesclient) the session is attached to, not on the session itself.

### Session status values

Always compare against the full constant. The status field never holds the bare suffix.

- `"TRAINING_SESSION_STATUS_UNSPECIFIED"`
- `"TRAINING_SESSION_STATUS_CREATING"`
- `"TRAINING_SESSION_STATUS_RUNNING"`
- `"TRAINING_SESSION_STATUS_STOPPED"`
- `"TRAINING_SESSION_STATUS_STOPPING"`
- `"TRAINING_SESSION_STATUS_ERROR"`
- `"TRAINING_SESSION_STATUS_EXPIRED"`

```python
if session.retrieve().status == "TRAINING_SESSION_STATUS_RUNNING":
    ...
```

### `InferenceCheckpoint`

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Checkpoint ID (for download). |
| `step` | `str \| int` | Training step at snapshot time. |
| `created_at` | `datetime` | When the checkpoint was created. |
| `registration` | `Registration \| None` | Where the weights live in the model registry. Absent when the checkpoint was not uploaded. |
| `registration.adapter` | `ModelRegistryArtifact \| None` | LoRA adapter (`id`, `revision_id`), deployed on top of the base model. |
| `registration.model` | `ModelRegistryArtifact \| None` | Full model weights (`id`, `revision_id`), for full-weight training and LoRA on custom base weights. |

### `TrainingCheckpoint`

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Checkpoint ID (for `resume_from_checkpoint_id`). |
| `step` | `str \| int` | Training step at snapshot time. |
| `created_at` | `datetime` | When the checkpoint was created. |
| `registration` | `ModelRegistryArtifact \| None` | Model registry artifact (`id`, `revision_id`) holding the training state. |

---

## Constant strings

Reference for the common literal-string parameters and fields.

| Group | Allowed values |
|---|---|
| `LossConfig.type` | `"LOSS_TYPE_UNSPECIFIED"`, `"LOSS_TYPE_CROSS_ENTROPY"`, `"LOSS_TYPE_GRPO"`, `"LOSS_TYPE_IMPORTANCE_SAMPLING"`, `"LOSS_TYPE_PPO"`, `"LOSS_TYPE_CISPO"`, `"LOSS_TYPE_DPPO"`, `"LOSS_TYPE_DRO"` |
| `GrpoLossParams.agg_type` | `"GRPO_LOSS_AGGREGATION_TYPE_UNSPECIFIED"`, `"GRPO_LOSS_AGGREGATION_TYPE_FIXED_HORIZON"`, `"GRPO_LOSS_AGGREGATION_TYPE_TOKEN_MEAN"`, `"GRPO_LOSS_AGGREGATION_TYPE_SEQUENCE_MEAN"` |
| `GrpoLossParams.ratio_type` | `"GRPO_LOSS_RATIO_TYPE_TOKEN"`, `"GRPO_LOSS_RATIO_TYPE_SEQUENCE"` |
| `StopReason` | `"STOP_REASON_LENGTH"`, `"STOP_REASON_STOP"` |
| `WeightSyncType` | `"WEIGHT_SYNC_TYPE_SYNCHRONOUS"`, `"WEIGHT_SYNC_TYPE_BACKGROUND_PUBLISH"`, `"WEIGHT_SYNC_TYPE_PIPELINE"` |
| `MuonScalingStrategy` | `"MUON_SCALING_STRATEGY_UNSPECIFIED"`, `"MUON_SCALING_STRATEGY_MATCH_ADAM"`, `"MUON_SCALING_STRATEGY_ORIGINAL"` |
| `dtype` (`loss_fn_inputs`) | `"int64"`, `"float32"` |
| `dtype` (custom gradients) | `"D_TYPE_UNSPECIFIED"`, `"D_TYPE_INT64"`, `"D_TYPE_FLOAT32"`, `"D_TYPE_BFLOAT16"` |
| `CheckpointVariant` | `"CHECKPOINT_VARIANT_UNSPECIFIED"`, `"CHECKPOINT_VARIANT_MERGED"`, `"CHECKPOINT_VARIANT_ADAPTER"` |
| `ComputeConfig` | `gpu_type` — see [Supported models and GPU hardware](#supported-models-and-gpu-hardware) (omit for the model default); `num_generator_replicas: int` (`0` = trainer only, `>=1` = generator replicas) |
| `ModelResourcesClient` status | `"MODEL_RESOURCES_STATUS_PENDING"`, `"MODEL_RESOURCES_STATUS_CREATING"`, `"MODEL_RESOURCES_STATUS_READY"`, `"MODEL_RESOURCES_STATUS_ERROR"`, `"MODEL_RESOURCES_STATUS_STOPPING"`, `"MODEL_RESOURCES_STATUS_STOPPED"` |
| `Session.status` | See [Session status values](#session-status-values). |

---

## Async usage

Every operation has an `_async` variant with the same signature. Combine them with the async context manager when you want fully non-blocking control.

```python
import asyncio
from together.post_training import ModelResourcesClient

async def main():
    async with await ModelResourcesClient.create_async(base_model="Qwen/Qwen3.8-27B") as resources:
        async with await resources.create_session_async() as session:
            result = await session.generator.sample_async(prompt)
            # ... forward_backward_async, optim_step_async, weights_sync_async, ...

asyncio.run(main())
```

---

## Full import list

```python
from together.post_training import (
    # Main interfaces
    SessionClient, ModelResourcesClient, ModelResourcesStatus,

    # Request types
    Sample, LossConfig, SamplingParams,
    AdamParams, MuonParams,
    OptimizerConfig, AdamConfig, MuonConfig, MuonScalingStrategy,
    ComputeConfig,
    LoraConfig, CheckpointVariant, Gradient, WeightSyncType,
    SessionMetadata, WandbMetadata,

    # Model input (sampling prompts and training-sample inputs)
    ModelInput, ModelInputChunk, EncodedTextChunk,

    # Generic sample tensors
    TensorData,

    # Loss configuration
    GrpoLossParams, CrossEntropyLossParams,
    PpoLossParams, CispoLossParams, DppoLossParams, DroLossParams,

    # Response types
    Session, SessionError, SampleResult, SampledSequence, StopReason,
    PolicyVersionSegment, ForwardBackwardResult, LossFnOutput,
    OptimStepResult, TrainingCheckpointResult, InferenceCheckpointResult,
)
```
