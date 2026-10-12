# Write a custom loss function

Train with a loss the service doesn't implement natively by computing it in client-side Python. DPO is the worked example, but the same pattern fits any objective you can express over the model's log-probabilities.

!!! tip "Using a coding agent?"

    The `together-post-training` skill can scaffold a custom-loss training loop and reward/loss closures for you.

---

## When you need a custom loss

The SDK ships `LOSS_TYPE_CROSS_ENTROPY` (SFT) and a family of policy objectives — GRPO, PPO, CISPO, DPPO, DRO, and importance sampling — selectable on the GRPO loop via `GRPOConfig.loss` / `policy_loss(config)`. Reach for a custom loss when none of them fits — for example:

- **DPO** and other preference objectives over `(chosen, rejected)` pairs.
- Auxiliary or regularization terms the built-in losses don't expose.
- Research losses you want to prototype without a service change.

If your objective is a reward you can score per completion, you usually want [GRPO with a custom reward function](./cookbook-guide.md#adding-a-custom-dataset) instead — it's simpler and runs on-policy. Custom losses are for objectives that aren't a scalar reward per sample.

---

## The two-pass pattern

The service can't run your Python loss, so the gradient is split across two calls:

1. **`session.trainer.forward(samples=..., loss=...)`** runs a forward-only pass and returns the model's per-slot log-probabilities of the target tokens in `loss_fn_outputs` — no gradients, no weight change. `forward_backward` returns the same outputs with `return_loss_fn_outputs=True`, but the two flags are independent: it still runs the backward pass, and its gradients are added to the ones you submit in step 3 before `optim_step` applies them. Keep the first pass on `forward`.
2. You treat those log-probabilities as autograd leaves, evaluate your loss in PyTorch, and call `.backward()` to get `dL/d(logprob)` for every token.
3. **`session.trainer.custom_forward_backward(samples=..., gradients=...)`** sends those per-slot gradients back; the service finishes the backward pass and accumulates parameter gradients.
4. **`session.trainer.optim_step(...)`** applies the update, exactly as in any other loop.
5. **`session.trainer.weights_sync(...)`** only when a later `sample` needs the updated policy. Trainer-only custom-loss loops (DPO, CE-style) with `num_generator_replicas=0` skip it.

Because the loss is a pure function of the log-probabilities the model assigns, that's all the service needs from you.

---

## The cookbook helpers

`post_training_cookbook.custom_loss` wraps the pattern so you only write the loss:

```python
from post_training_cookbook.custom_loss import compute_custom_forward_backward, compute_logprobs

# Forward-only pass: one list of per-slot logprobs per sample, in batch order.
# Only positions the sample's `weights` mark are real logprobs; the rest read back as 0.0.
logprobs = compute_logprobs(session, batch, timeout=config.timeout)

# Full custom-loss step: forward -> your loss -> backward -> gradients submitted.
result = compute_custom_forward_backward(session, batch, custom_loss_fn, timeout=config.timeout)
session.trainer.optim_step(adam_params=AdamParams(..., learning_rate=current_lr))
# Call weights_sync only if a later sample should see the update.
```

`custom_loss_fn` receives a list with one `(S_i,)` float32 tensor of per-slot log-probabilities per sample (same order as `batch`) and must return a **scalar** `torch.Tensor`:

```python
CustomLossFn = Callable[[list[torch.Tensor]], torch.Tensor]
```

`compute_custom_forward_backward` returns a `CustomLossResult`:

| Field | Description |
|---|---|
| `loss` | The scalar value your loss function returned (a Python `float`). |
| `metrics` | Any metrics the `custom_forward_backward` operation returned. |
| `raw` | The underlying SDK response. |

`compute_logprobs` is also useful on its own for snapshotting a frozen reference policy before training (DPO does this once up front).

---

## Minimal example

A toy "stay close to the reference" loss — penalize the policy for moving its log-probabilities away from a snapshot:

```python
import torch
from post_training_cookbook.custom_loss import compute_custom_forward_backward, compute_logprobs
from post_training_cookbook.samples import extract_loss_mask

# 1. Snapshot the reference logprobs once (no gradient).
ref = compute_logprobs(session, batch)
ref_tensors = [torch.tensor(r) for r in ref]

# Masked-out positions read back as 0.0 in both snapshots and live logprobs,
# so the loss has to reduce over the scored positions only.
masks = [torch.tensor(extract_loss_mask(sample), dtype=torch.float32) for sample in batch]

# 2. Define the loss over the live (autograd) logprobs.
def stay_close_loss(logprobs: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([
        ((lp - r) ** 2 * m).sum() / m.sum()
        for lp, r, m in zip(logprobs, ref_tensors, masks)
    ]).mean()

# 3. One training step.
result = compute_custom_forward_backward(session, batch, stay_close_loss)
session.trainer.optim_step(
    adam_params=AdamParams(beta1=0.9, beta2=0.95, weight_decay=0.1, learning_rate=1e-6),
)
print(result.loss)
```

Each `Sample` in `batch` is built the same way as for SFT: you hand a builder the token sequence and a token-space loss mask marking the tokens you care about, and it trims both to target-aligned slots. That matters here — `extract_loss_mask` reads the mask back off the built sample, so it is target-aligned, one entry per slot, and lines up with what `forward` returns. See [SDK reference: training sample](./sdk-reference.md#training-sample).

---

## Worked example: DPO

The cookbook's DPO recipe (`recipes/dpo/helpsteer3.py`, with the loop in `post_training_cookbook.dpo.train`) is a complete custom-loss implementation. The pieces:

### 1. Build preference-pair samples

Each preference pair becomes two SFT-style samples — the chosen response and the rejected response — sharing the same prompt. `post_training_cookbook.dpo.samples` provides the pair builder:

```python
from post_training_cookbook.dpo.samples import build_preference_pair
from post_training_cookbook.samples import extract_loss_mask

pair = build_preference_pair(
    chosen_messages=chosen_msgs,        # [...context, {"role": "assistant", "content": chosen}]
    rejected_messages=rejected_msgs,    # [...context, {"role": "assistant", "content": rejected}]
    codec=codec,
    max_length=config.max_length,
)
if pair is None:
    return  # max_length left one side with no trained token; DPO needs both
chosen_sample, rejected_sample = pair

# Read the mask off the built sample: it is target-aligned, one entry per slot,
# so it lines up with what `forward` returns.
chosen_mask = extract_loss_mask(chosen_sample)   # 1 on response slots, 0 elsewhere
```

The training loop interleaves them into a flat batch `[chosen_0, rejected_0, chosen_1, rejected_1, …]`, so index `2k` is a chosen response and `2k+1` is its rejected counterpart.

### 2. Precompute reference log-probabilities

DPO compares the trained policy against a frozen reference. Since the reference is just the model before any updates, the loop snapshots its masked log-probability sums once, before training, with `compute_logprobs` (here summed over the response tokens via the loss mask):

```python
ref_logprobs = compute_logprobs(session, batch_samples, timeout=config.timeout)
ref_sums = [
    sum(lp * m for lp, m in zip(data, mask))
    for data, mask in zip(ref_logprobs, batch_masks)
]
```

### 3. Define the DPO loss closure

`post_training_cookbook.dpo.loss.make_dpo_loss` returns a `CustomLossFn` bound to this batch's reference sums and masks:

```python
from post_training_cookbook.dpo.loss import make_dpo_loss

loss_fn = make_dpo_loss(
    batch_ref_sums,        # reference masked-logprob sums, flat-interleaved
    batch_masks,           # per-sample response masks
    beta=config.beta,
    label_smoothing=config.label_smoothing,
)
```

Internally it reduces each sample's per-slot log-probabilities to a masked sum, forms the chosen-minus-rejected margin relative to the reference, and returns the DPO objective `-log σ(β · margin)` (with optional label smoothing).

### 4. Run the step

```python
result = compute_custom_forward_backward(session, batch_samples, loss_fn, timeout=config.timeout)
session.trainer.optim_step(
    adam_params=AdamParams(
        beta1=config.adam_beta1, beta2=config.adam_beta2,
        eps=config.adam_eps, weight_decay=config.weight_decay,
        learning_rate=current_lr,
    ),
)
```

### 5. Validate with preference accuracy

DPO has no token-level accuracy, so the loop reports an **implicit-reward preference accuracy** on held-out pairs — the fraction where the policy's reward shift prefers chosen over rejected. See the [metrics reference](./metrics.md#dpo-metrics) for `val/preference_accuracy`, `val/mean_margin`, and the per-side deltas.

Run it:

```bash
uv run python recipes/dpo/helpsteer3.py
```

---

## Tips

- **Return a scalar tensor.** `custom_loss_fn` must return a 0-dim `torch.Tensor` so `.backward()` works; `compute_custom_forward_backward` raises otherwise.
- **Gradients are per slot.** The service expects one gradient value per slot, aligned to the log-probabilities `forward` returned. The helper produces them from `.backward()` automatically; if you build gradients by hand, put `0.0` on the prompt's slots — a masked-out slot's log-probability comes back as a fabricated `0.0`, so there is nothing there to differentiate.
- **Use a trainer-only session.** Custom-loss loops that never call `sample` (DPO included) should create the session with `compute_config=ComputeConfig(num_generator_replicas=0)` to avoid provisioning a sampler.
- **Snapshot references once.** A frozen reference policy only needs `compute_logprobs` a single time, before training — recomputing it every step is wasted work.
- **Mind memory on `forward`.** `forward` runs the full batch through the model. For large preference sets, chunk the reference precompute (the DPO loop uses `precompute_ref_batch_size`).

---

## Next steps

- Need the low-level call signatures, including `return_loss_fn_outputs` and how `weights` gates which log-probabilities are real? See [SDK reference: `forward`](./sdk-reference.md#sessiontrainerforward) and [`custom_forward_backward`](./sdk-reference.md#sessiontrainercustom_forward_backward).
- Want a scalar reward instead of a full loss? See [GRPO with a custom reward function](./cookbook-guide.md#adding-a-custom-dataset).
