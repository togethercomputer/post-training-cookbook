# Training metrics reference

Every metric the cookbook logs, grouped by where it comes from: the service returns some in each `forward_backward` response, and the cookbook's training loops compute the rest. Each entry gives the formula or source, followed by how to read the value.

**Contents:** [Service-emitted metrics (GRPO)](#service-emitted-metrics-grpo) · [Custom training-loop metrics](#custom-training-loop-metrics) · [Quick reference: what to watch first](#quick-reference-what-to-watch-first)

## Service-emitted metrics (GRPO)

Every `forward_backward` call with `LossConfig(type="LOSS_TYPE_GRPO", ...)` returns a `metrics` dict. The cookbook logs each entry as `fb/<key>` to W&B.

For `LOSS_TYPE_CROSS_ENTROPY` (SFT), the service returns only `loss`, as a total summed over the
batch's trained positions. The cookbook divides it by the batch's trained-token count and logs that
mean as `loss`; the total stays available as `loss_sum`. The metrics below are GRPO-only.

### Importance ratio

| Metric | Formula | What it measures |
|---|---|---|
| `loss/ratio/mean` | masked mean of `r_t = exp(logprob_t − generator_logprob_t)` | The trainer policy's likelihood divided by the generator's likelihood, averaged over loss-mask=1 tokens. |
| `loss/kl_policy/mean` | masked mean of `−log_ratio` | Approximate KL between trainer and generator policies (k1 estimator). |

This is the most load-bearing health metric here. During on-policy GRPO, the trainer and the sampler are the same model, so the importance ratio should sit very close to `1.0` (and `loss/kl_policy/mean` close to `0`).

How to interpret:

- **`loss/ratio/mean ≈ 1.00 ± 0.01`.** Healthy. Trainer and generator agree.
- **Persistent drift away from 1.0** (for example, settling at 0.85 or 1.2) usually points to a system-side issue: trainer/generator weight skew, mismatched tokenization, or a stale rollout cache. Check this before suspecting your reward function.
- **Sudden spikes** for a single step are normal at the start of training and after warmup.
- **Continuous growth** over many steps suggests the policy is moving fast enough that captured generator logprobs are stale. Lower `learning_rate`, move `clip_high_threshold` toward `1.0`, or refresh rollouts more often.

When the loss uses the GSPO sequence-level ratio (`GrpoLossParams.ratio_type="GRPO_LOSS_RATIO_TYPE_SEQUENCE"`), these metrics report a per-sequence importance ratio instead of a per-token one; the same "sit near 1.0" intuition applies. The cookbook recipes use the default token-level ratio.

### Clipping

| Metric | What it measures |
|---|---|
| `loss/clip/clipped_ratio/mean` | Mean of the clipped ratio `clip(r, clip_low_threshold, clip_high_threshold)` over all loss-mask=1 tokens. |
| `loss/clip/high_fraction` | Fraction of tokens where `r > clip_high_threshold` **and** `advantage > 0`. These tokens have their gradient suppressed to prevent runaway up-weighting of already-favored completions. |
| `loss/clip/low_fraction` | Fraction of tokens where `r < clip_low_threshold` **and** `advantage < 0`. These tokens have their gradient suppressed to prevent runaway down-weighting of already-disfavored completions. |

The thresholds are absolute ratio bounds straddling `1.0` (`clip_low_threshold <= 1 <= clip_high_threshold`), not offsets — see [Core concepts: clipping](./concepts.md#clipping).

How to interpret:

- **Both fractions near 0 (< 1% to 2%).** Healthy. The policy is moving slowly enough that PPO-style clipping rarely triggers.
- **`high_fraction` rising while training accuracy plateaus.** The model has many positive-advantage tokens it can't push further because their probability is already high. Consider lower `learning_rate`, raising `clip_high_threshold`, or accepting the plateau as a coverage limit of the dataset.
- **`low_fraction` rising.** Symmetric to the above on negative-advantage tokens; lowering `clip_low_threshold` widens the band.
- **Both fractions large (> 10%).** Updates are being aggressively suppressed. Reduce `learning_rate` and/or widen the clip band (`clip_low_threshold` down, `clip_high_threshold` up).

### KL to reference

| Metric | What it measures |
|---|---|
| `loss/kl_ref/mean` | Masked mean KL between the trainer policy and the reference policy, using the k1/k2/k3 estimator selected by the loss. |

Only emitted when `GrpoLossParams.beta > 0` (KL penalty enabled). When `beta=0` (the default in the cookbook recipes), this metric isn't returned at all.

How to interpret:

- **Rising slowly.** The trainer is drifting from the reference; the KL penalty term in the loss pushes back. Tune `beta` to control how much.
- **Spiking.** The trainer made a large step relative to the reference. Combined with a rising `loss/ratio/mean`, this signals over-aggressive updates.

### Aggregation

| Metric | What it measures |
|---|---|
| `loss/aggregate/active_fraction` | Fraction of all tokens in the batch (across the loss mask) that contributed to the loss. |

How to interpret:

- The denominator is the full batch tensor `B × S`, including padding and prompt tokens. A typical value is whatever fraction of your sequences is response.
- If this drops sharply, your batch contains many short responses or many overlong-masked completions.

### What about `loss` itself?

`ForwardBackwardResult.loss` is the scalar GRPO loss aggregated according to `loss_agg_type`. It is not a good training-progress signal: GRPO loss can be flat, slightly negative, or non-monotonic while reward improves. Watch reward and accuracy for progress; watch the metrics above for stability.

---

## Custom training-loop metrics

The service does not emit the metrics in this section. The cookbook implements them after sampling, scoring, validation, or SFT batching finishes. Treat them as a reference implementation: if you build your own SDK-level training loop, compute and log the analogous metrics yourself.

In this repo, the reference implementation lives in `post_training_cookbook/grpo/single_turn.py` (`_build_single_turn_step_log`, `run_validation`), `post_training_cookbook/grpo/multi_turn.py` (multi-turn step log and `run_multi_turn_validation`), `post_training_cookbook/grpo/loop.py` (`training_loop`), `post_training_cookbook/dpo/train.py`, and `post_training_cookbook/sft/train.py`.

### Reward, raw reward, and accuracy

| Metric | Source | What it measures |
|---|---|---|
| `accuracy` | mean over the step of `ds_cfg.compute_metric(decoded, target)` (falls back to `compute_reward` when no metric is set). | The metric you actually care about. For binary tasks this is the success rate. |
| `reward_mean` | mean of `raw_reward + overlong_penalty` across all completions in the step. | What the GRPO loss actually consumes after length adjustments. |
| `reward_std` | std of `reward_mean` across completions. | Spread of rewards. Going to zero across all groups is the warning sign for `zero_variance_skip_ratio`. |

The W&B `example_output` table also includes `raw_reward` and `overlong_penalty` per row so you can see both before-adjustment and after-adjustment values for one example completion per step. `raw_reward` is the value the reward function produced; `reward = raw_reward + overlong_penalty`.

How to interpret:

- **`accuracy`.** Track `val/accuracy` over training; it's the only metric that survives reward-function changes.
- **`reward_mean` lower than `accuracy`** when the overlong penalty is biting. Compare against `mean_overlong_penalty`.
- **`reward_std → 0`** means most groups have all-equal rewards; expect `zero_variance_skip_ratio` to be high in the same step.

### Group skip ratios

For each prompt, GRPO samples `group_size` completions and computes mean-centered advantages. Groups where every completion got the same reward have zero variance and produce no gradient signal, so the cookbook drops them.

| Metric | What it measures |
|---|---|
| `zero_variance_skip_ratio` | Fraction of groups in the step that were dropped because reward standard deviation fell below `1e-8`. |
| `identical_nonzero_reward_skip_ratio` | Sub-fraction of those zero-variance groups where the common reward was non-zero (every completion succeeded). Useful for distinguishing "all wrong" from "all right". |

How to interpret:

- **`zero_variance_skip_ratio` low (< 30%).** Healthy.
- **`zero_variance_skip_ratio` consistently > 50%.** Most groups give no signal. Either the task is too easy (model gets it right every time, so `identical_nonzero_reward_skip_ratio` will be large), or too hard (model fails every time, so `identical_nonzero_reward_skip_ratio ≈ 0`). Adjust difficulty, raise `group_size`, or relax the reward function.
- **`identical_nonzero_reward_skip_ratio` large.** The model has saturated this slice of the data. Move to a harder dataset or stop training.

When fewer than `group_size` samples survive per step, the cookbook also prints a `[step N] skipped: only X samples` line to stdout and skips `forward_backward` entirely for that step.

### Response length and truncation

| Metric | What it measures |
|---|---|
| `mean_response_budget` | Average per-completion budget (`min(max_sample_tokens, max_model_length − len(prompt))`). |
| `mean_sampled_response_length` | Average number of tokens the model actually produced. |
| `mean_trainable_response_length` | Average number of tokens kept after clipping to the budget. |
| `truncated_ratio` | Fraction of completions whose sampler `stop_reason` was `"STOP_REASON_LENGTH"`, or whose `sampled_length` hit `max_response_tokens`, or whose tokens exceeded the budget and were clipped. |
| `post_budget_clip_ratio` | Fraction of completions where `sampled_length > response_budget`, so trailing tokens were dropped. |

How to interpret:

- **`truncated_ratio` high (> 30%).** Many completions are running out of room. Either the task needs longer responses, or the model is producing degenerate long outputs. Compare against `mean_sampled_response_length`: if it's tracking `max_sample_tokens`, the model is bumping into the limit.
- **`post_budget_clip_ratio` high.** Same as above but specifically the budget-clip cause. If it's high while `truncated_ratio` is low, the prompts are leaving very little room for responses; raise `max_model_length` or trim prompts.
- **`mean_sampled_response_length` ≈ `mean_response_budget`.** The model has learned to fill the entire budget. Pair with `truncated_ratio` to see whether outputs are also being cut off.

A truncated completion is still scored and trained on (with whatever reward the reward function returns for the partial output) unless `overlong_masking=True`, in which case no training sample is built for it at all (its reward still counts toward the group baseline).

### Overlong penalty

| Metric | What it measures |
|---|---|
| `mean_overlong_penalty` | Average penalty value across all completions in the step. Always ≤ 0. |
| `overlong_penalized_ratio` | Fraction of completions that received any overlong penalty (`penalty < 0`). |

The penalty starts ramping up `overlong_buffer_len` tokens before `max_sample_tokens` and is scaled by `overlong_penalty_factor`. Set `overlong_penalty_factor=0.0` to disable.

How to interpret:

- **`overlong_penalized_ratio = 0`.** Nothing to worry about. The model isn't approaching the budget.
- **`overlong_penalized_ratio` rising over training.** The model is producing longer responses, possibly converging on a length-runaway pattern. Keep an eye on `accuracy`.
- **`mean_overlong_penalty` strongly negative.** The penalty is dominating reward. Either lower `overlong_penalty_factor` or expand `max_sample_tokens`.

### Advantages

| Metric | What it measures |
|---|---|
| `completion_advantage_std` | Std of all per-completion advantages (`reward − group_mean`) that were retained for training (zero-variance groups excluded). |
| `mean_logprob` | Average per-token log-probability the generator assigned to its own samples. |

How to interpret:

- **`completion_advantage_std → 0`** (excluding zero-variance skips) means the surviving groups have very tight reward distributions. Updates will be small. Often paired with high `zero_variance_skip_ratio`.
- **`mean_logprob` becoming more negative over training.** The model is producing less-likely-to-itself completions. Often a temperature/exploration story, occasionally a sign of mode collapse onto rare tokens.

### Per-step training loop metrics

`training_loop` adds a few more keys around the collector's step log:

| Metric | What it measures |
|---|---|
| `loss` | Scalar from `ForwardBackwardResult.loss`. See the warning above: GRPO loss is not a great progress signal. |
| `learning_rate` | Current LR after cosine schedule. |
| `elapsed` | Wall-clock seconds since training started. |
| `timing/step_collect_s` | Seconds spent collecting samples (sampling + scoring) this step. |
| `timing/step_forward_backward_s` | Seconds spent in `forward_backward`. |
| `timing/step_optim_s` | Seconds spent in `optim_step` and the following synchronous `weights_sync`. |
| `timing/step_total_s` | Sum of the three phases above. |
| `fb/<key>` | Every key from `ForwardBackwardResult.metrics`, prefixed with `fb/`. See [Service-emitted metrics](#service-emitted-metrics-grpo) above. |

### Dynamic sampling

Logged only when `max_sampling_attempts > 1` (DAPO dynamic sampling): a step keeps drawing batches and discarding zero-variance groups until it has enough usable groups or hits the attempt cap.

| Metric | What it measures |
|---|---|
| `dynamic_sampling/attempts` | Number of sampling rounds drawn this step. |
| `dynamic_sampling/total_groups_sampled` | Total groups sampled across all attempts. |
| `dynamic_sampling/kept_groups` | Groups retained (non-zero-variance) for training. |

### Asynchronous collection

Logged every step when `max_steps_off_policy` is set (see [Asynchronous multi-turn collection](cookbook-guide.md#asynchronous-multi-turn-collection)). Together these tell "async is keeping the generators busy" apart from "async is throwing most of its rollouts away".

| Metric | What it measures |
|---|---|
| `async/accepted_groups` | Groups trained on this step. Below `batch_size` means the step ran short — it exhausted `batch_size * max_sampling_attempts` zero-variance discards. |
| `async/staleness_mean` | Mean age, in optimizer steps, of the groups trained on. Omitted entirely on a step that accepted nothing, so a gap is never mistaken for perfectly on-policy. |
| `async/staleness_max` | Oldest group trained on. Never exceeds `max_steps_off_policy`. |
| `async/dropped_stale_groups` | Groups discarded for exceeding the staleness budget. Sustained non-zero values mean rollouts are slower than the optimizer: raise `max_steps_off_policy`, raise `num_generator_replicas`, or lower `groups_in_flight`. Stale groups sit outside the `zero_variance_skip_ratio` accounting, so that ratio keeps describing the training cohort as it does in lockstep. |
| `async/dropped_zero_variance_groups` | Groups discarded for having no reward variance. Hitting `batch_size * max_sampling_attempts` every step points at the reward function, not at async. |
| `async/queue_depth` | Completed groups already waiting when the step began collecting. Persistently at `groups_in_flight` means collection outruns training and rollouts are being held (and risk going stale); persistently `0` alongside a non-zero `async/train_wait_s` means training waits on collection. |
| `async/train_wait_s` | Seconds this step spent blocked waiting for groups. The direct measure of lost overlap — near `0` is the goal. |
| `async/mid_rollout_drift_ratio` | Fraction of generated tokens produced by a policy other than the one the group was scheduled against. Non-zero whenever a publish lands during a rollout — including under the default `WEIGHT_SYNC_TYPE_SYNCHRONOUS`, since a worker between turns is not generating and the publish simply lands. `WEIGHT_SYNC_TYPE_PIPELINE` additionally splits a *single* action across policies. Actions the server reported no segments for are excluded from both sides of the ratio, so this never reads `0.0` merely for lack of data. |

### Validation metrics

`run_validation` samples the validation set once with greedy decoding (`temperature=0.0`, `num_samples=1`) and logs:

| Metric | What it measures |
|---|---|
| `val/accuracy` | Mean of `compute_metric` over the validation set. The single most informative metric here. |
| `val/reward_mean` | Mean of training-shaped reward (with penalties) over validation. |
| `val/mean_response_length` | Mean trainable response length on validation. |
| `val/elapsed_seconds` | Validation wall time. |

How to interpret:

- **`val/accuracy`** is the headline. It should rise during training and then plateau. Plateaus are normal; falling val accuracy while training reward keeps rising is reward-hacking.
- **`val/mean_response_length`** drifting upward signals length runaway even when the validation reward looks fine.

### Multi-turn metrics

Multi-turn GRPO (`train_multi_turn`) replaces the single-turn per-completion stats with per-**episode** stats. Per step:

| Metric | What it measures |
|---|---|
| `reward_mean` | Mean episode reward (training-shaped). |
| `reward_std` | Spread of episode rewards. |
| `raw_reward_mean` | Mean episode reward before shaping. |
| `num_turns_mean`, `num_turns_max` | Assistant turns per episode. |
| `terminated_ratio` | Fraction of episodes that ended on task logic (a real terminal state). |
| `truncated_ratio` | Fraction cut short by external policy (turn limit, context overflow, parse failure). |
| `parse_failure_ratio` | Fraction whose sampled tokens failed to parse into a valid message. |
| `context_overflow_ratio` | Fraction truncated because the conversation exceeded the context budget. |
| `num_episodes` | Episodes collected this step. |
| `num_training_samples` | Trainable sequences produced after flattening episodes. |
| `zero_variance_skip_ratio` | Fraction of groups dropped for zero reward variance (same meaning as single-turn). |
| `timing/episode_reset_mean_s`, `timing/sample_fraction` | Rollout timing diagnostics. |

How to interpret:

- **`terminated_ratio` low with high `truncated_ratio`/`context_overflow_ratio`.** Episodes are running out of turns or context before finishing. Raise `max_turns` or `max_model_length`, or simplify the task.
- **`parse_failure_ratio` rising.** The model is emitting malformed tool calls. Check the system prompt and tool schemas, and that `parse_response` receives the env's `tool_specs`.

Multi-turn validation (`run_multi_turn_validation`) logs `val/reward_mean`, `val/reward_iter_std`, `val/raw_reward_mean`, `val/num_turns_mean`, `val/terminated_ratio`, and `val/iter_fail_ratio` (the fraction of validation iterations that failed outright).

### DPO metrics

DPO (`post_training_cookbook/dpo/train.py`) logs per step:

| Metric | What it measures |
|---|---|
| `loss` | DPO loss for the batch (`-log σ(β·(margin))`, optionally label-smoothed). Should trend down. |
| `learning_rate` | Current LR (cosine schedule). |
| `pairs` | Preference pairs in the batch. |
| `elapsed_s` | Wall-clock seconds since training started. |
| `cfb/<key>` | Any metrics returned by the `custom_forward_backward` operation, prefixed with `cfb/`. |

Validation (every `val_interval` steps) reports an **implicit-reward preference accuracy** rather than a token metric:

| Metric | What it measures |
|---|---|
| `val/preference_accuracy` | Fraction of held-out pairs where the policy's implicit reward prefers the chosen response over the rejected one — i.e. `(logπ_θ(y_w) − logπ_ref(y_w)) > (logπ_θ(y_l) − logπ_ref(y_l))`. The headline DPO signal. |
| `val/mean_chosen_delta` | Mean shift in chosen-response logprob vs. the reference. |
| `val/mean_rejected_delta` | Mean shift in rejected-response logprob vs. the reference. |
| `val/mean_margin` | Mean `chosen_delta − rejected_delta`. Should grow positive as the policy separates chosen from rejected. |

### SFT metrics

SFT logs a much smaller set per step from `post_training_cookbook/sft/train.py`:

| Metric | What it measures |
|---|---|
| `loss` | Mean cross-entropy for each trained token (token mean). Unlike GRPO loss, SFT loss is a meaningful progress signal: it should fall steadily. Because it is normalized, steps stay comparable when the batch token count changes. |
| `loss_sum` | The unnormalized total the service returns, summed over trained positions. It scales with the batch token count, so it is not comparable across steps. |
| `trained_tokens` | Weight mass of the batch's loss mask — the denominator of `loss`. |
| `learning_rate` | Linearly decaying LR. |
| `samples` | Samples trained this step. Can differ from `batch_size`: a row may split into several sequences, and unrenderable rows are dropped. |
| `elapsed_s` | Wall-clock seconds since training started. |

(The step index itself isn't a logged value — it's the W&B step axis and the leading `step` field of each `metrics.jsonl` record.)

As with GRPO, every step is also appended to `<run_dir>/metrics.jsonl` for offline analysis (see [Run artifacts](./cookbook-guide.md#run-artifacts)).

---

## Quick reference: what to watch first

| Question | Watch |
|---|---|
| Is the system on-policy? | `fb/loss/ratio/mean` (target ≈ 1.0). |
| Is training making progress? | `val/accuracy`, then `accuracy` (per-step). |
| Are updates being clipped too aggressively? | `fb/loss/clip/high_fraction`, `fb/loss/clip/low_fraction`. |
| Is the reward signal differentiating completions? | `zero_variance_skip_ratio`, `reward_std`, `completion_advantage_std`. |
| Are responses hitting the budget? | `truncated_ratio`, `post_budget_clip_ratio`, `mean_sampled_response_length`. |
| Is the overlong penalty dominating? | `overlong_penalized_ratio`, `mean_overlong_penalty`. |
| Is the model reward-hacking? | `accuracy` rising while `val/accuracy` falls or `val/mean_response_length` rises. |
| Are multi-turn episodes finishing? | `terminated_ratio` (up), `context_overflow_ratio` / `parse_failure_ratio` (down). |
| Is DPO separating chosen from rejected? | `val/preference_accuracy`, `val/mean_margin` (both should rise). |
| Is SFT learning? | `loss` (should fall), `learning_rate` (should decay). |
