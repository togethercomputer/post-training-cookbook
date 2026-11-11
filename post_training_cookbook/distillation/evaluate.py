"""Score a student session or the teacher on the same eval items.

An eval item is a dict with at least ``"id"`` and ``"messages"`` (the prompt). An :class:`EvalSet`
pairs items with a metrics function over ``(items, outputs)``, where ``outputs`` are the final
response texts in item order. The student is sampled on its session's generator, so call
``session.trainer.weights_sync()`` first if the trainer has stepped since the last sync.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from together.post_training import SamplingParams

from post_training_cookbook.distillation.teacher import TeacherConfig, TeacherRequest, generate
from post_training_cookbook.samples import make_prompt

Metrics = Callable[[Sequence[dict[str, Any]], Sequence[str]], dict[str, float]]


@dataclass(frozen=True)
class EvalSet:
    """Held-out items and how to score them.

    ``every_eval=False`` scores the set only at step 0 and at the last step, for a large final set.
    """

    name: str
    items: Sequence[dict[str, Any]]
    metrics: Metrics
    every_eval: bool = True


def student_outputs(
    session: Any,
    codec: Any,
    items: Sequence[dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float,
    timeout: float | None,
    chunk: int = 128,
) -> tuple[list[str], list[bool]]:
    """Sample one completion per item on the session's generator; return final texts and truncation flags."""
    texts: list[str] = []
    truncated: list[bool] = []
    params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
    for start in range(0, len(items), chunk):
        batch = items[start : start + chunk]
        prompts = [make_prompt(codec.encode_messages(it["messages"], add_generation_prompt=True)) for it in batch]
        results = session.generator.sample_batch(prompts, num_samples=1, sampling_params=params, timeout=timeout)
        for result in results:
            seq = result.sequences[0]
            message, _ok = codec.parse_response([int(t) for t in seq.tokens])
            texts.append(message.get("content", "") or "")
            truncated.append(seq.stop_reason == "STOP_REASON_LENGTH")
    return texts, truncated


def teacher_outputs(items: Sequence[dict[str, Any]], config: TeacherConfig, cache_path: Path) -> list[str | None]:
    """The teacher's final texts for ``items``, cached under the item IDs.

    A call that failed after the SDK's retries is ``None``, so callers can score the teacher on the
    items it answered instead of counting a failure as an empty translation.
    """
    requests = [TeacherRequest(key=it["id"], messages=list(it["messages"])) for it in items]
    outputs, _ = generate(requests, config, cache_path)
    return [outputs[it["id"]].content if it["id"] in outputs and outputs[it["id"]].ok else None for it in items]


def score_answered(eval_set: EvalSet, outputs: Sequence[str | None]) -> dict[str, float]:
    """Score ``eval_set`` on the items that have an output; reports how many were scored as ``n``."""
    pairs = [(item, out) for item, out in zip(eval_set.items, outputs, strict=True) if out is not None]
    metrics = eval_set.metrics([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else {}
    return {**metrics, "n": float(len(pairs))}
