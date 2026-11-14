"""Score a session on each domain's held-out rows with greedy decoding.

The fusion loop uses this for held-out monitoring, and the evaluation recipe uses it to fill the
model-by-domain table: base model, each expert, the mixed baseline and the MOPD student, all
scored on the same rows with the same prompts and the same metric.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from together import RateLimitError
from together.post_training import LoraConfig, ModelResourcesClient, SamplingParams, SessionClient

from post_training_cookbook.codecs.base import MessageCodec
from post_training_cookbook.datasets.registry import build_prompt_example
from post_training_cookbook.mopd.domains import Domain
from post_training_cookbook.samples import make_prompt

__all__ = ["DomainEval", "EvalItem", "build_eval_items", "evaluate_checkpoints", "evaluate_session", "write_eval"]

_SAMPLE_CHUNK = 256


@dataclass(frozen=True)
class EvalItem:
    """One held-out row, rendered to the prompt every model sees."""

    index: int
    question: str
    target: Any
    prompt_tokens: list[int]


@dataclass
class DomainEval:
    """Greedy-decoding results of one session on one domain."""

    domain: str
    n: int
    accuracy: float
    mean_reward: float
    mean_response_tokens: float
    truncated_rate: float
    skipped_prompts: int
    records: list[dict[str, Any]]

    def summary(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if key != "records"}


def build_eval_items(
    domain: Domain,
    rows: Sequence[dict],
    codec: MessageCodec,
    *,
    max_prompt_length: int,
    max_model_length: int,
) -> tuple[list[EvalItem], int]:
    """Render ``rows`` into prompts; returns the items and how many rows were skipped as too long."""
    items: list[EvalItem] = []
    skipped = 0
    for index, row in enumerate(rows):
        try:
            example = build_prompt_example(
                row, domain.dataset, codec, max_prompt_length=max_prompt_length, max_model_length=max_model_length
            )
        except ValueError:
            skipped += 1
            continue
        items.append(EvalItem(index, example.question, example.target, example.prompt_tokens))
    return items, skipped


def evaluate_session(
    session: SessionClient,
    codec: MessageCodec,
    domain: Domain,
    items: Sequence[EvalItem],
    *,
    max_sample_tokens: int,
    skipped_prompts: int = 0,
    timeout: float | None = None,
) -> DomainEval:
    """Greedy-decode one completion per item on ``session``'s generator and score it.

    Accuracy uses the domain's ``compute_metric`` when it has one (Game of 24's strict check), else
    its ``compute_reward``; ``mean_reward`` always uses ``compute_reward``.
    """
    cfg = domain.dataset
    metric_fn = cfg.compute_metric or cfg.compute_reward
    kwargs = {"timeout": timeout} if timeout is not None else {}
    records: list[dict[str, Any]] = []
    for start in range(0, len(items), _SAMPLE_CHUNK):
        chunk = items[start : start + _SAMPLE_CHUNK]
        results = session.generator.sample_batch(
            [make_prompt(item.prompt_tokens) for item in chunk],
            num_samples=1,
            sampling_params=SamplingParams(max_tokens=max_sample_tokens, temperature=0.0),
            **kwargs,
        )
        for item, result in zip(chunk, results, strict=True):
            sequence = result.sequences[0]
            tokens = [int(token) for token in sequence.tokens]
            response = codec.decode(tokens)
            records.append(
                {
                    "index": item.index,
                    "question": item.question,
                    "target": item.target,
                    "response": response,
                    "metric": float(metric_fn(response, item.target)),
                    "reward": float(cfg.compute_reward(response, item.target)),
                    "response_tokens": len(tokens),
                    "stop_reason": sequence.stop_reason,
                }
            )
    n = len(records)
    return DomainEval(
        domain=domain.name,
        n=n,
        accuracy=float(np.mean([r["metric"] for r in records])) if n else 0.0,
        mean_reward=float(np.mean([r["reward"] for r in records])) if n else 0.0,
        mean_response_tokens=float(np.mean([r["response_tokens"] for r in records])) if n else 0.0,
        truncated_rate=float(np.mean([r["stop_reason"] == "STOP_REASON_LENGTH" for r in records])) if n else 0.0,
        skipped_prompts=skipped_prompts,
        records=records,
    )


def write_eval(result: DomainEval, path: Path) -> None:
    """Write per-item records as JSONL next to a ``.summary.json`` with the aggregate numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in result.records:
            f.write(json.dumps(record, default=str) + "\n")
    path.with_suffix(".summary.json").write_text(json.dumps(result.summary(), indent=2))


def _create_session_when_free(
    resources: ModelResourcesClient, *, wait_s: float = 1200.0, **kwargs: Any
) -> SessionClient:
    """``resources.create_session``, retried while the resource is at its session cap.

    A stopped session keeps its slot for a few minutes while it is STOPPING, so a session created
    right after another one stopped can be refused with HTTP 429 "at capacity".
    """
    deadline = time.monotonic() + wait_s
    while True:
        try:
            return resources.create_session(**kwargs)
        except RateLimitError:
            if time.monotonic() > deadline:
                raise
            time.sleep(30)


def evaluate_checkpoints(
    resources: ModelResourcesClient,
    models: dict[str, str | None],
    domains: Sequence[Domain],
    items: dict[str, list[EvalItem]],
    codec: MessageCodec,
    *,
    lora_config: LoraConfig | None,
    out_dir: Path,
    max_sample_tokens: int,
    max_concurrent: int = 4,
    skipped: dict[str, int] | None = None,
    timeout: float | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Score several models on every domain, one session per model, on one resource.

    ``models`` maps a label to a *training* checkpoint ID, or to ``None`` for the base model (a
    fresh adapter). Up to ``max_concurrent`` sessions evaluate at once; keep it within the
    resource's session cap (a session refused as "at capacity" is retried for up to 20 min).
    Writes ``<out_dir>/<label>/<domain>.jsonl`` (plus ``.summary.json``) and returns
    ``{label: {domain: summary}}``.
    """

    def run_one(label: str, checkpoint_id: str | None) -> dict[str, dict[str, Any]]:
        session = _create_session_when_free(
            resources,
            display_name=f"eval-{label}",
            resume_from_checkpoint_id=checkpoint_id,
            load_optimizer=False if checkpoint_id else None,
            lora_config=lora_config,
        )
        try:
            summaries: dict[str, dict[str, Any]] = {}
            for domain in domains:
                result = evaluate_session(
                    session,
                    codec,
                    domain,
                    items[domain.name],
                    max_sample_tokens=max_sample_tokens,
                    skipped_prompts=(skipped or {}).get(domain.name, 0),
                    timeout=timeout,
                )
                write_eval(result, out_dir / label / f"{domain.name}.jsonl")
                summaries[domain.name] = result.summary()
                print(f"[eval] {label} {domain.name}: accuracy={result.accuracy:.3f} (n={result.n})", flush=True)
            return summaries
        finally:
            session.stop()

    with ThreadPoolExecutor(max_workers=max(1, min(max_concurrent, len(models)))) as pool:
        futures = {label: pool.submit(run_one, label, checkpoint_id) for label, checkpoint_id in models.items()}
        return {label: future.result() for label, future in futures.items()}
