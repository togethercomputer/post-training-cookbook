"""Single-turn GRPO rollout pipeline: scoring, sample construction, collector, validation."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, NamedTuple

import numpy as np
import wandb
from datasets import Dataset
from together.post_training import (
    Sample,
    SamplingParams,
    SessionClient,
    StopReason,
)

from post_training_cookbook.codecs.base import MessageCodec
from post_training_cookbook.datasets.registry import DatasetConfig, PromptExample, build_prompt_example
from post_training_cookbook.grpo.batching import oversample_until_kept
from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.grpo.data_progress import DataProgress
from post_training_cookbook.grpo.loop import CollectSamplesFn
from post_training_cookbook.grpo.samples import (
    GroupAdvantages,
    build_policy_sample,
    compute_group_advantages,
)
from post_training_cookbook.overlong import compute_overlong_penalty, is_overlong
from post_training_cookbook.run_artifacts import Run, export_samples, log_step
from post_training_cookbook.samples import is_trainable, make_prompt

logger = logging.getLogger(__name__)


class ScoredCompletion(NamedTuple):
    tokens: list[int]
    logprobs: list[float]
    reward: float
    raw_reward: float
    metric: float
    penalty: float
    is_truncated: bool
    response_budget: int
    sampled_length: int
    trainable_length: int
    was_clipped: bool
    stop_reason: StopReason
    routed_experts_key: str | None = None


def get_response_budget(
    prompt_tokens: list[int],
    max_model_length: int,
    max_sample_tokens: int,
    min_response_tokens: int,
) -> int:
    available = max_model_length - len(prompt_tokens)
    if available < min_response_tokens:
        raise ValueError(
            f"Prompt leaves insufficient response room: "
            f"prompt_length={len(prompt_tokens)}, max_model_length={max_model_length}"
        )
    return min(max_sample_tokens, available)


def score_group(
    sequences: list,
    *,
    codec: MessageCodec,
    target: Any,
    ds_cfg: DatasetConfig,
    config: GRPOConfig,
    max_response_tokens: int,
) -> list[ScoredCompletion]:
    """Score completions using ds_cfg.compute_reward."""
    metric_fn = ds_cfg.compute_metric or ds_cfg.compute_reward
    results: list[ScoredCompletion] = []
    for seq in sequences:
        key = seq.routed_experts_key if config.routing_replay else None
        if config.routing_replay and not key:
            raise RuntimeError("routing_replay is on but the generator returned no routed_experts_key")
        sampled_tokens = [int(token) for token in (seq.tokens or [])]
        sampled_logprobs = [float(value) for value in (seq.logprobs or [])]
        tokens = sampled_tokens[:max_response_tokens]
        logprobs = sampled_logprobs[:max_response_tokens]
        decoded = codec.decode(tokens)
        raw_reward = ds_cfg.compute_reward(decoded, target)
        metric = metric_fn(decoded, target)
        stop_reason = seq.stop_reason
        was_clipped = len(sampled_tokens) > len(tokens)
        truncated = (
            was_clipped or stop_reason == "STOP_REASON_LENGTH" or is_overlong(len(sampled_tokens), max_response_tokens)
        )

        if config.overlong_penalty:
            penalty = compute_overlong_penalty(
                response_length=len(sampled_tokens),
                max_response_length=max_response_tokens,
                overlong_buffer_len=config.overlong_buffer_len,
                penalty_factor=config.overlong_penalty_factor,
            )
        else:
            penalty = 0.0

        results.append(
            ScoredCompletion(
                tokens=tokens,
                logprobs=logprobs,
                reward=raw_reward + penalty,
                raw_reward=raw_reward,
                metric=metric,
                penalty=penalty,
                is_truncated=truncated,
                response_budget=max_response_tokens,
                sampled_length=len(sampled_tokens),
                trainable_length=len(tokens),
                was_clipped=was_clipped,
                stop_reason=stop_reason,
                routed_experts_key=key,
            )
        )
    return results


def build_scored_group_samples(
    group: list[ScoredCompletion],
    advantages: list[float],
    prompt_tokens: list[int],
    config: GRPOConfig,
) -> list[Sample]:
    """Build GRPO training samples for a scored group with pre-computed advantages.

    A completion that trains no slot is dropped rather than built: the payload
    would cost a forward pass and carry no gradient.
    """
    samples: list[Sample] = []
    untrainable = 0
    for completion, advantage in zip(group, advantages, strict=True):
        if config.overlong_masking and completion.is_truncated:
            continue
        prompt_len, response_len = len(prompt_tokens), len(completion.tokens)
        loss_mask = [0.0] * prompt_len + [1.0] * response_len
        if not is_trainable(loss_mask):
            untrainable += 1
            continue
        if config.routing_replay:
            if not completion.routed_experts_key:
                raise ValueError("routing_replay is on but a training completion carries no routed_experts_key")
            if completion.was_clipped:
                raise ValueError("routing_replay cannot replay a clipped completion; leave enough response budget")
        samples.append(
            build_policy_sample(
                tokens=prompt_tokens + completion.tokens,
                logprobs=[0.0] * prompt_len + [float(logprob) for logprob in completion.logprobs],
                loss_mask=loss_mask,
                advantage=advantage,
                routed_experts_key=completion.routed_experts_key if config.routing_replay else None,
            )
        )
    if untrainable:
        logger.warning("Dropped %d of %d completions that train no slot", untrainable, len(group))
    return samples


def skip_counts(results: Sequence[GroupAdvantages]) -> tuple[int, int]:
    """Return (zero_variance_skips, identical_nonzero_reward_skips among those)."""
    zv = sum(1 for r in results if r.zero_variance)
    inz = sum(1 for r in results if r.zero_variance and r.identical_nonzero)
    return zv, inz


def _build_single_turn_step_log(
    scored_groups: list[list[ScoredCompletion]],
    group_advantages: list[list[float]],
    *,
    zero_variance_skipped_groups: int,
    identical_nonzero_skipped_groups: int,
    total_groups: int,
    codec: MessageCodec,
    question: str,
    target: str,
) -> dict[str, Any]:
    """Build the W&B step_log dict for a single-turn collection step."""
    all_completions = [sc for g in scored_groups for sc in g]
    if not all_completions:
        return {}

    n = len(all_completions)
    penalized = sum(1 for sc in all_completions if sc.penalty < 0)
    truncated = sum(1 for sc in all_completions if sc.is_truncated)
    clipped = sum(1 for sc in all_completions if sc.was_clipped)

    first = scored_groups[0][0]
    log: dict[str, Any] = {
        "accuracy": float(np.mean([sc.metric for g in scored_groups for sc in g])),
        "reward_mean": float(np.mean([sc.reward for sc in all_completions])),
        "reward_std": float(np.std([sc.reward for sc in all_completions])),
        "mean_response_budget": float(np.mean([sc.response_budget for sc in all_completions])),
        "mean_sampled_response_length": float(np.mean([sc.sampled_length for sc in all_completions])),
        "mean_trainable_response_length": float(np.mean([sc.trainable_length for sc in all_completions])),
        "overlong_penalized_ratio": penalized / max(n, 1),
        "mean_overlong_penalty": float(np.mean([sc.penalty for sc in all_completions])),
        "zero_variance_skip_ratio": zero_variance_skipped_groups / max(total_groups, 1),
        "identical_nonzero_reward_skip_ratio": identical_nonzero_skipped_groups / max(total_groups, 1),
        "truncated_ratio": truncated / max(n, 1),
        "post_budget_clip_ratio": clipped / max(n, 1),
        "completion_advantage_std": float(np.std([a for advs in group_advantages for a in advs])),
        "mean_logprob": float(np.mean([lp for sc in all_completions for lp in sc.logprobs])),
        "example_output": wandb.Table(
            columns=[
                "question",
                "target",
                "response",
                "raw_reward",
                "overlong_penalty",
                "response_budget",
                "sampled_response_length",
                "trainable_response_length",
                "stop_reason",
            ],
            data=[
                [
                    question,
                    target,
                    codec.decode(first.tokens),
                    first.raw_reward,
                    first.penalty,
                    first.response_budget,
                    first.sampled_length,
                    first.trainable_length,
                    first.stop_reason,
                ]
            ],
        ),
    }
    return log


class _PromptOutcome(NamedTuple):
    """One prompt's sampled completions, scored and tagged with group advantages.

    ``advantages.values is None`` when the group has zero reward variance —
    these outcomes contribute to skip-rate metrics but produce no training samples.
    """

    prompt_example: PromptExample
    scored_group: list[ScoredCompletion]
    advantages: GroupAdvantages


def _sample_and_score_prompts(
    prompt_examples: list[PromptExample],
    session: SessionClient,
    *,
    codec: MessageCodec,
    ds_cfg: DatasetConfig,
    config: GRPOConfig,
) -> list[_PromptOutcome]:
    """Sample completions for a batch of prompts, score them, compute advantages."""
    prompts = [make_prompt(pe.prompt_tokens) for pe in prompt_examples]
    results = session.generator.sample_batch(
        prompts,
        num_samples=config.group_size,
        return_routed_experts=config.routing_replay,
        sampling_params=SamplingParams(max_tokens=config.max_sample_tokens, temperature=config.temperature),
        timeout=config.timeout,
    )
    outcomes: list[_PromptOutcome] = []
    for prompt_example, result in zip(prompt_examples, results, strict=True):
        sequences = result.sequences
        if not sequences:
            raise RuntimeError("Sample operation returned no sequences")
        group = score_group(
            sequences,
            codec=codec,
            target=prompt_example.target,
            ds_cfg=ds_cfg,
            config=config,
            max_response_tokens=get_response_budget(
                prompt_example.prompt_tokens,
                config.max_model_length,
                config.max_sample_tokens,
                config.min_response_tokens,
            ),
        )
        outcomes.append(_PromptOutcome(prompt_example, group, compute_group_advantages([sc.reward for sc in group])))
    return outcomes


def make_single_turn_collector(
    config: GRPOConfig,
    ds_cfg: DatasetConfig,
    train_dataset: Dataset,
    codec: MessageCodec,
    run: Run,
    *,
    data_progress: DataProgress,
) -> CollectSamplesFn:
    """Build the per-step ``collect_samples`` closure for single-turn GRPO.

    When ``config.max_sampling_attempts > 1``, each step keeps drawing batches
    until ``config.batch_size`` groups have non-trivial advantages or the cap
    is reached (DAPO §3.2 dynamic sampling).
    """

    def collect_samples(step: int, session: SessionClient) -> tuple[list[Sample], dict[str, Any]]:
        def draw_batch() -> list[_PromptOutcome]:
            batch_rows = train_dataset.select(data_progress.draw_batch())
            prompt_examples = [
                build_prompt_example(
                    row,
                    ds_cfg,
                    codec,
                    max_prompt_length=config.max_prompt_length,
                    max_model_length=config.max_model_length,
                    min_response_tokens=config.min_response_tokens,
                )
                for row in batch_rows
            ]
            return _sample_and_score_prompts(prompt_examples, session, codec=codec, ds_cfg=ds_cfg, config=config)

        outcomes, attempts = oversample_until_kept(
            draw_batch,
            is_kept=lambda o: o.advantages.values is not None,
            target=config.batch_size,
            max_attempts=config.max_sampling_attempts,
        )

        all_advantages = [o.advantages for o in outcomes]
        zero_variance_skipped_groups, identical_nonzero_skipped_groups = skip_counts(all_advantages)
        total_groups = len(all_advantages)
        keepers = [o for o in outcomes if o.advantages.values is not None]

        if not keepers:
            print(
                f"[{session.session_id[:8]}] [step {step}] skipped: 0/{total_groups} groups "
                f"had non-zero reward variance (attempts={attempts})",
                flush=True,
            )
            return [], {}

        samples = [
            sample
            for o in keepers
            for sample in build_scored_group_samples(
                o.scored_group, o.advantages.values, o.prompt_example.prompt_tokens, config
            )
        ]

        if len(samples) < config.group_size:
            print(
                f"[{session.session_id[:8]}] [step {step}] skipped: only {len(samples)} samples "
                f"after overlong masking ({zero_variance_skipped_groups}/{total_groups} "
                f"zero-variance, attempts={attempts})",
                flush=True,
            )
            return [], {}

        export_samples(
            run,
            step,
            [o.prompt_example.question for o in keepers],
            [[codec.decode(sc.tokens) for sc in o.scored_group] for o in keepers],
            [[sc.reward for sc in o.scored_group] for o in keepers],
        )

        step_log = _build_single_turn_step_log(
            [o.scored_group for o in keepers],
            [o.advantages.values for o in keepers],
            zero_variance_skipped_groups=zero_variance_skipped_groups,
            identical_nonzero_skipped_groups=identical_nonzero_skipped_groups,
            total_groups=total_groups,
            codec=codec,
            question=keepers[0].prompt_example.question,
            target=str(keepers[0].prompt_example.target),
        )

        if config.max_sampling_attempts > 1:
            step_log["dynamic_sampling/attempts"] = attempts
            step_log["dynamic_sampling/total_groups_sampled"] = total_groups
            step_log["dynamic_sampling/kept_groups"] = len(keepers)

        print(
            f"[{session.session_id[:8]}] [step {step}] accuracy={step_log.get('accuracy', 0.0):.3f}"
            f" reward={step_log.get('reward_mean', 0.0):.3f}"
            f" penalized={step_log.get('overlong_penalized_ratio', 0.0):.2%}"
            f" zero_var_skip={step_log.get('zero_variance_skip_ratio', 0.0):.2%}"
            f" clipped={step_log.get('post_budget_clip_ratio', 0.0):.2%}"
            f" attempts={attempts}",
            flush=True,
        )

        return samples, step_log

    return collect_samples


def run_validation(
    step: int,
    *,
    val_dataset,
    session: SessionClient,
    codec: MessageCodec,
    ds_cfg: DatasetConfig,
    config: GRPOConfig,
    run: Run,
) -> float:
    """Greedy single-sample evaluation on the held-out set; logs val/* metrics."""
    metric_fn = ds_cfg.compute_metric or ds_cfg.compute_reward
    val_accuracy_scores: list[float] = []
    val_rewards: list[float] = []
    val_response_lengths: list[int] = []
    val_questions: list[str] = []
    val_completions: list[list[str]] = []
    val_completion_rewards: list[list[float]] = []
    val_start = time.monotonic()

    prompt_examples = [
        build_prompt_example(
            row,
            ds_cfg,
            codec,
            max_prompt_length=config.max_prompt_length,
            max_model_length=config.max_model_length,
            min_response_tokens=config.min_response_tokens,
        )
        for row in val_dataset
    ]
    prompts = [make_prompt(pe.prompt_tokens) for pe in prompt_examples]

    val_temperature = config.val_temperature if config.val_temperature is not None else 0.0
    results = session.generator.sample_batch(
        prompts,
        num_samples=1,
        return_routed_experts=False,
        sampling_params=SamplingParams(max_tokens=config.max_sample_tokens, temperature=val_temperature),
        timeout=config.timeout,
    )

    val_config = replace(config, routing_replay=False)
    for prompt_example, result in zip(prompt_examples, results):
        sequences = result.sequences
        if not sequences:
            val_questions.append(prompt_example.question)
            val_completions.append([])
            val_completion_rewards.append([])
            continue

        group = score_group(
            sequences[:1],
            codec=codec,
            target=prompt_example.target,
            ds_cfg=ds_cfg,
            config=val_config,
            max_response_tokens=get_response_budget(
                prompt_example.prompt_tokens,
                config.max_model_length,
                config.max_sample_tokens,
                config.min_response_tokens,
            ),
        )
        scored = group[0]
        decoded = codec.decode(scored.tokens)
        reward = scored.raw_reward
        accuracy = metric_fn(decoded, prompt_example.target)
        val_accuracy_scores.append(accuracy)
        val_rewards.append(reward)
        val_response_lengths.append(scored.trainable_length)
        val_questions.append(prompt_example.question)
        val_completions.append([decoded])
        val_completion_rewards.append([reward])

    export_samples(run, step, val_questions, val_completions, val_completion_rewards, prefix="val")

    val_accuracy = float(np.mean(val_accuracy_scores)) if val_accuracy_scores else 0.0
    val_mean_len = float(np.mean(val_response_lengths)) if val_response_lengths else 0.0
    val_elapsed = time.monotonic() - val_start

    log_step(
        run,
        step,
        {
            "val/accuracy": val_accuracy,
            "val/reward_mean": float(np.mean(val_rewards)) if val_rewards else 0.0,
            "val/mean_response_length": val_mean_len,
            "val/elapsed_seconds": val_elapsed,
        },
    )

    print(
        f"[{session.session_id[:8]}] [step {step}] VALIDATION accuracy={val_accuracy:.3f} "
        f"mean_len={val_mean_len:.0f} elapsed={val_elapsed:.1f}s",
        flush=True,
    )
    return val_accuracy
