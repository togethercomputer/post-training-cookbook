"""MOPD fusion: distill one student from one teacher per domain, all on one model resource.

Each step:

1. **Sample.** The student answers ``prompts_per_domain`` training prompts from every domain,
   ``samples_per_prompt`` times each, at ``temperature`` (1.0 by default), keeping its sampling
   log-probabilities.
2. **Score.** Each completion goes to its domain's teacher, which scores the student's exact
   tokens teacher-forced (:mod:`post_training_cookbook.mopd.teachers`). Teachers score in parallel.
3. **Train.** Per-token advantage ``kl_coef * (teacher - student)`` log-probability, shared
   between domains per ``balance``; one ``forward_backward`` with the built-in importance-sampling
   loss, then ``optim_step`` (cosine schedule) and ``weights_sync`` so the next step samples the
   updated student.

Per-domain metrics show what the student learns from each teacher: ``<domain>/reverse_kl`` (the
per-token reverse-KL estimate, which falls as the student matches its teacher),
``<domain>/token_frac`` (the domain's share of the step's completion tokens) and
``<domain>/loss_weight_frac`` (its share of the update after balancing).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from together.post_training import (
    AdamParams,
    LoraConfig,
    LossConfig,
    ModelResourcesClient,
    SamplingParams,
    SessionClient,
)

from post_training_cookbook.codecs import get_codec
from post_training_cookbook.codecs.base import MessageCodec
from post_training_cookbook.datasets.registry import PromptExample, build_prompt_example
from post_training_cookbook.grpo.loop import get_lr_cosine
from post_training_cookbook.mopd.config import MOPDConfig
from post_training_cookbook.mopd.distill import Completion, balance_weights, build_distillation_sample
from post_training_cookbook.mopd.domains import Domain, load_splits
from post_training_cookbook.mopd.evaluate import EvalItem, build_eval_items, evaluate_session
from post_training_cookbook.mopd.teachers import open_teacher, score_with_generator, score_with_trainer
from post_training_cookbook.multi_lora import model_resource, resolve_base_model
from post_training_cookbook.run_artifacts import Run, checkpoint_if_due, latest_checkpoint_ref, log_step, track_run
from post_training_cookbook.samples import make_prompt

load_dotenv(override=True)

__all__ = ["MOPDResult", "teacher_sessions", "train", "train_student"]

ScoreFn = Callable[..., list[list[float]]]
_IMPORTANCE_SAMPLING = LossConfig(type="LOSS_TYPE_IMPORTANCE_SAMPLING")
_EXPORT_PROMPTS_PER_DOMAIN = 2


@dataclass(frozen=True)
class MOPDResult:
    """Where the distilled student ended up."""

    run_dir: Path
    student_session_id: str
    training_checkpoint_id: str | None
    inference_checkpoint_id: str | None


class _PromptCursor:
    """Cycles through a domain's prompts, reshuffling (seeded) at every epoch boundary."""

    def __init__(self, examples: Sequence[PromptExample], seed: int) -> None:
        if not examples:
            raise ValueError("a domain needs at least one training prompt")
        self._examples = list(examples)
        self._seed = seed
        self._epoch = 0
        self._order = self._shuffled(0)
        self._pos = 0

    def _shuffled(self, epoch: int) -> list[int]:
        return [int(i) for i in np.random.default_rng(self._seed + epoch).permutation(len(self._examples))]

    def take(self, k: int) -> list[PromptExample]:
        out: list[PromptExample] = []
        while len(out) < k:
            if self._pos == len(self._order):
                self._epoch += 1
                self._order = self._shuffled(self._epoch)
                self._pos = 0
            out.append(self._examples[self._order[self._pos]])
            self._pos += 1
        return out


def _render(domain: Domain, rows, codec: MessageCodec, config: MOPDConfig) -> tuple[list[PromptExample], int]:
    examples: list[PromptExample] = []
    skipped = 0
    for row in rows:
        try:
            examples.append(
                build_prompt_example(
                    row,
                    domain.dataset,
                    codec,
                    max_prompt_length=config.max_prompt_length,
                    max_model_length=config.max_model_length,
                )
            )
        except ValueError:
            skipped += 1
    return examples, skipped


def _lora(rank: int | None, alpha: int | None, dropout: float | None = None) -> LoraConfig | None:
    pairs = (("rank", rank), ("alpha", alpha), ("dropout", dropout))
    values = {key: value for key, value in pairs if value is not None}
    return LoraConfig(**values) if values else None


def _evaluate(
    step: int,
    student: SessionClient,
    codec: MessageCodec,
    domains: Sequence[Domain],
    eval_items: dict[str, list[EvalItem]],
    config: MOPDConfig,
    run: Run,
) -> None:
    t0 = time.monotonic()
    metrics: dict[str, Any] = {}
    for domain in domains:
        result = evaluate_session(
            student,
            codec,
            domain,
            eval_items[domain.name],
            max_sample_tokens=config.max_sample_tokens,
            timeout=config.timeout,
        )
        metrics[f"val/{domain.name}/accuracy"] = result.accuracy
        metrics[f"val/{domain.name}/mean_response_tokens"] = result.mean_response_tokens
        metrics[f"val/{domain.name}/truncated_rate"] = result.truncated_rate
        if run.rollouts_enabled:
            path = run.dir / f"val_{step:04d}_{domain.name}.jsonl"
            path.write_text("".join(json.dumps(r, default=str) + "\n" for r in result.records))
    metrics["val/mean_accuracy"] = float(np.mean([metrics[f"val/{d.name}/accuracy"] for d in domains]))
    metrics["val/elapsed_s"] = time.monotonic() - t0
    log_step(run, step, metrics)
    summary = " ".join(f"{d.name}={metrics[f'val/{d.name}/accuracy']:.3f}" for d in domains)
    elapsed = metrics["val/elapsed_s"]
    print(f"[{student.session_id[:8]}] [step {step}] VALIDATION {summary} ({elapsed:.0f}s)", flush=True)


def _score_domains(
    teachers: dict[str, SessionClient],
    score: ScoreFn,
    groups: dict[str, tuple[list[list[int]], list[list[int]]]],
    timeout: float,
) -> dict[str, list[list[float]]]:
    """Score every domain's completions with its own teacher, all teachers at once."""
    with ThreadPoolExecutor(max_workers=max(len(groups), 1)) as pool:
        futures = {
            name: pool.submit(score, teachers[name], prompts, completions, timeout=timeout)
            for name, (prompts, completions) in groups.items()
        }
        return {name: future.result() for name, future in futures.items()}


def _train_step(
    step: int,
    total_steps: int,
    *,
    student: SessionClient,
    teachers: dict[str, SessionClient],
    score: ScoreFn,
    cursors: dict[str, _PromptCursor],
    codec: MessageCodec,
    domains: Sequence[Domain],
    config: MOPDConfig,
    run: Run,
) -> None:
    step_start = time.monotonic()
    batch: list[tuple[Domain, PromptExample]] = [
        (domain, example) for domain in domains for example in cursors[domain.name].take(config.prompts_per_domain)
    ]

    # 1. Sample: the student answers every prompt, samples_per_prompt times.
    t0 = time.monotonic()
    results = student.generator.sample_batch(
        [make_prompt(example.prompt_tokens) for _, example in batch],
        num_samples=config.samples_per_prompt,
        sampling_params=SamplingParams(max_tokens=config.max_sample_tokens, temperature=config.temperature),
        timeout=config.timeout,
    )
    sample_s = time.monotonic() - t0

    raw: list[dict[str, Any]] = []
    for (domain, example), result in zip(batch, results, strict=True):
        for sequence in result.sequences:
            tokens = [int(token) for token in sequence.tokens]
            logprobs = [float(value) for value in (sequence.logprobs or [])]
            if not tokens:
                continue
            if len(logprobs) != len(tokens):
                raise RuntimeError(f"generator returned {len(logprobs)} logprobs for {len(tokens)} tokens")
            raw.append(
                {
                    "domain": domain,
                    "example": example,
                    "tokens": tokens,
                    "logprobs": logprobs,
                    "truncated": sequence.stop_reason == "STOP_REASON_LENGTH",
                }
            )
    if not raw:
        raise RuntimeError(f"step {step}: the student produced no non-empty completion")

    # 2. Score: route each completion to its domain's teacher.
    t0 = time.monotonic()
    groups: dict[str, tuple[list[list[int]], list[list[int]]]] = {}
    for item in raw:
        prompts, completions = groups.setdefault(item["domain"].name, ([], []))
        prompts.append(item["example"].prompt_tokens)
        completions.append(item["tokens"])
    teacher_scores = _score_domains(teachers, score, groups, config.timeout)
    teacher_s = time.monotonic() - t0
    position = dict.fromkeys(groups, 0)
    completions: list[Completion] = []
    for item in raw:
        name = item["domain"].name
        completions.append(
            Completion(
                domain=name,
                prompt_tokens=item["example"].prompt_tokens,
                completion_tokens=item["tokens"],
                student_logprobs=item["logprobs"],
                teacher_logprobs=teacher_scores[name][position[name]],
            )
        )
        position[name] += 1

    # 3. Train: one importance-sampling pass on the per-token distillation advantages.
    weights = balance_weights(completions, config.balance)
    samples = [
        build_distillation_sample(completion, kl_coef=config.kl_coef, weight=weight)
        for completion, weight in zip(completions, weights, strict=True)
    ]
    t0 = time.monotonic()
    fb = student.trainer.forward_backward(samples=samples, loss=_IMPORTANCE_SAMPLING, timeout=config.timeout)
    fb_s = time.monotonic() - t0
    lr = get_lr_cosine(step, total_steps, config.learning_rate, config.warmup_ratio, config.min_lr_ratio)
    t0 = time.monotonic()
    student.trainer.optim_step(
        adam_params=AdamParams(
            learning_rate=lr,
            beta1=config.adam_beta1,
            beta2=config.adam_beta2,
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        ),
        timeout=config.timeout,
    )
    student.trainer.weights_sync(timeout=config.timeout)
    optim_s = time.monotonic() - t0

    metrics = _step_metrics(raw, completions, weights, codec, domains, config)
    metrics.update(
        {
            "loss": fb.loss,
            "learning_rate": lr,
            "timing/sample_s": sample_s,
            "timing/teacher_score_s": teacher_s,
            "timing/forward_backward_s": fb_s,
            "timing/optim_sync_s": optim_s,
            "timing/step_s": time.monotonic() - step_start,
            **{f"fb/{key}": value for key, value in (fb.metrics or {}).items()},
        }
    )
    logged_step = step + 1
    log_step(run, logged_step, metrics)
    _export(run, logged_step, raw, completions, codec, domains)
    kl = " ".join(f"{d.name}={metrics[f'{d.name}/reverse_kl']:.4f}" for d in domains)
    print(
        f"[{student.session_id[:8]}] [step {logged_step}] reverse_kl {kl} | loss={fb.loss:.4f} lr={lr:.2e} "
        f"| sample={sample_s:.0f}s teacher={teacher_s:.0f}s fb={fb_s:.0f}s optim={optim_s:.0f}s",
        flush=True,
    )


def _step_metrics(
    raw: list[dict[str, Any]],
    completions: list[Completion],
    weights: list[float],
    codec: MessageCodec,
    domains: Sequence[Domain],
    config: MOPDConfig,
) -> dict[str, Any]:
    total_tokens = sum(len(c.completion_tokens) for c in completions)
    total_weight = sum(w * len(c.completion_tokens) for c, w in zip(completions, weights))
    all_kl = np.concatenate([c.reverse_kl for c in completions])
    advantages = -config.kl_coef * np.concatenate([w * c.reverse_kl for c, w in zip(completions, weights)])
    metrics: dict[str, Any] = {
        "reverse_kl": float(all_kl.mean()),
        "completion_tokens": total_tokens,
        "num_completions": len(completions),
        "advantage/mean": float(advantages.mean()),
        "advantage/p01": float(np.percentile(advantages, 1)),
        "advantage/p99": float(np.percentile(advantages, 99)),
        "advantage/min": float(advantages.min()),
    }
    for domain in domains:
        idx = [i for i, c in enumerate(completions) if c.domain == domain.name]
        if not idx:
            continue
        kl = np.concatenate([completions[i].reverse_kl for i in idx])
        tokens = sum(len(completions[i].completion_tokens) for i in idx)
        metric_fn = domain.dataset.compute_metric or domain.dataset.compute_reward
        scores = [
            float(metric_fn(codec.decode(completions[i].completion_tokens), raw[i]["example"].target)) for i in idx
        ]
        metrics.update(
            {
                f"{domain.name}/reverse_kl": float(kl.mean()),
                f"{domain.name}/reverse_kl_per_sequence": float(
                    np.mean([completions[i].reverse_kl.sum() for i in idx])
                ),
                f"{domain.name}/token_frac": tokens / total_tokens,
                f"{domain.name}/loss_weight_frac": sum(weights[i] * len(completions[i].completion_tokens) for i in idx)
                / total_weight,
                f"{domain.name}/mean_completion_tokens": tokens / len(idx),
                f"{domain.name}/truncated_rate": float(np.mean([raw[i]["truncated"] for i in idx])),
                f"{domain.name}/sample_accuracy": float(np.mean(scores)),
            }
        )
    return metrics


def _export(
    run: Run,
    step: int,
    raw: list[dict[str, Any]],
    completions: list[Completion],
    codec: MessageCodec,
    domains: Sequence[Domain],
) -> None:
    """Dump the first prompts of each domain with every completion and its per-token signal summary."""
    if not run.rollouts_enabled:
        return
    entries = []
    for domain in domains:
        seen: list[str] = []
        for item, completion in zip(raw, completions):
            if completion.domain != domain.name:
                continue
            question = item["example"].question
            if question not in seen:
                if len(seen) == _EXPORT_PROMPTS_PER_DOMAIN:
                    continue
                seen.append(question)
            entries.append(
                {
                    "domain": domain.name,
                    "prompt": question,
                    "completion": codec.decode(completion.completion_tokens),
                    "completion_tokens": len(completion.completion_tokens),
                    "reverse_kl_mean": float(completion.reverse_kl.mean()),
                    "reverse_kl_max": float(completion.reverse_kl.max()),
                    "truncated": item["truncated"],
                }
            )
    (run.dir / f"step_{step:04d}.json").write_text(json.dumps(entries, indent=2))


def _check_teachers(config: MOPDConfig, domains: Sequence[Domain]) -> list[str]:
    names = [domain.name for domain in domains]
    if sorted(names) != sorted(config.teacher_checkpoints):
        raise ValueError(f"teacher_checkpoints {sorted(config.teacher_checkpoints)} must name every domain {names}")
    return names


def _stop(session: SessionClient) -> None:
    try:
        session.stop()
    except Exception as exc:  # noqa: BLE001 — teardown is best-effort; stopping the resource covers it
        print(f"[mopd] WARN stopping session {session.session_id}: {type(exc).__name__}: {exc}", flush=True)


@contextmanager
def teacher_sessions(resources: ModelResourcesClient, config: MOPDConfig) -> Iterator[dict[str, SessionClient]]:
    """Open one frozen teacher session per domain (concurrently) and stop them all on exit.

    Several students can share the same teachers: hold this context once and call
    :func:`train_student` for each student, from one thread per student if they should run at once.
    """
    teacher_lora = _lora(config.teacher_lora_rank, config.teacher_lora_alpha)
    teachers: dict[str, SessionClient] = {}
    try:
        with ThreadPoolExecutor(max_workers=max(len(config.teacher_checkpoints), 1)) as pool:
            futures = {
                name: pool.submit(
                    open_teacher, resources, name=name, training_checkpoint_id=checkpoint, lora_config=teacher_lora
                )
                for name, checkpoint in config.teacher_checkpoints.items()
            }
            errors = []
            for name, future in futures.items():
                try:
                    teachers[name] = future.result()
                except Exception as exc:  # noqa: BLE001 — stop the teachers that did open, then re-raise
                    errors.append(exc)
            if errors:
                raise errors[0]
        print("[mopd] teachers " + ", ".join(f"{n}={s.session_id}" for n, s in teachers.items()), flush=True)
        yield teachers
    finally:
        for session in teachers.values():
            _stop(session)


def train_student(
    config: MOPDConfig,
    domains: Sequence[Domain],
    resources: ModelResourcesClient,
    teachers: dict[str, SessionClient],
) -> MOPDResult:
    """Distill one student from ``teachers`` on ``resources``; the student session is stopped on exit."""
    _check_teachers(config, domains)
    codec = get_codec(
        resolve_base_model(config), hf_token=os.environ.get("HF_TOKEN"), enable_thinking=config.enable_thinking
    )
    cursors: dict[str, _PromptCursor] = {}
    eval_items: dict[str, list[EvalItem]] = {}
    for index, domain in enumerate(domains):
        train_rows, test_rows = load_splits(domain)
        examples, skipped = _render(domain, train_rows, codec, config)
        cursors[domain.name] = _PromptCursor(examples, seed=domain.seed + index)
        items, eval_skipped = build_eval_items(
            domain,
            test_rows.select(range(min(config.eval_examples_per_domain, len(test_rows)))),
            codec,
            max_prompt_length=config.max_prompt_length,
            max_model_length=config.max_model_length,
        )
        eval_items[domain.name] = items
        print(
            f"[mopd] {domain.name}: {len(examples)} training prompts ({skipped} too long), "
            f"{len(items)} monitoring prompts ({eval_skipped} too long)",
            flush=True,
        )

    score: ScoreFn = score_with_generator if config.teacher_scoring == "generator" else score_with_trainer
    total_steps = config.num_steps
    student = resources.create_session(
        display_name=config.wandb_run_name or "mopd-student",
        resume_from_checkpoint_id=config.resume_from_checkpoint_id,
        load_optimizer=config.load_optimizer,
        lora_config=_lora(config.lora_rank, config.lora_alpha, config.lora_dropout),
    )
    try:
        print(f"[mopd] student {student.session_id} ({config.wandb_run_name})", flush=True)
        with track_run(config, kind="mopd", session_id=student.session_id, total_steps=total_steps) as run:
            (run.dir / "sessions.json").write_text(
                json.dumps(
                    {
                        "model_resources_id": resources.model_resources_id,
                        "student": student.session_id,
                        "teachers": {n: s.session_id for n, s in teachers.items()},
                        "teacher_checkpoints": config.teacher_checkpoints,
                        "domains": {d.name: {"num_train": d.num_train, "num_test": d.num_test} for d in domains},
                    },
                    indent=2,
                )
            )
            if config.eval_interval:
                _evaluate(0, student, codec, domains, eval_items, config, run)
            for step in range(total_steps):
                _train_step(
                    step,
                    total_steps,
                    student=student,
                    teachers=teachers,
                    score=score,
                    cursors=cursors,
                    codec=codec,
                    domains=domains,
                    config=config,
                    run=run,
                )
                logged = step + 1
                last = logged == total_steps
                checkpoint_if_due(
                    student,
                    run,
                    step=logged,
                    total_steps=total_steps,
                    training_interval=total_steps if last else config.training_checkpoint_interval,
                    inference_interval=0,
                )
                if config.eval_interval and (last or logged % config.eval_interval == 0):
                    _evaluate(logged, student, codec, domains, eval_items, config, run)
            return MOPDResult(
                run_dir=run.dir,
                student_session_id=student.session_id,
                training_checkpoint_id=latest_checkpoint_ref(run, kind="training"),
                inference_checkpoint_id=latest_checkpoint_ref(run, kind="inference"),
            )
    finally:
        _stop(student)


def train(config: MOPDConfig, domains: Sequence[Domain]) -> MOPDResult:
    """Run the MOPD fusion stage: open the teachers, distill one student, stop every session."""
    _check_teachers(config, domains)
    with (
        model_resource(
            model=config.model,
            model_resources_id=config.model_resources_id,
            num_generator_replicas=config.num_generator_replicas,
        ) as resources,
        teacher_sessions(resources, config) as teachers,
    ):
        return train_student(config, domains, resources, teachers)
