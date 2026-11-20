"""Turn teacher outputs into chat SFT rows, filter them, and mix sets.

A row is ``{"id": ..., "messages": [...prompt messages..., {"role": "assistant", ...}]}``, the same
shape ``post_training_cookbook.sft.train.build_conversation_samples`` renders, plus any request
``meta`` under ``"meta"``. Filters take ``(request, output)`` and return whether to keep the row;
task-specific ones live with the task (the translation filters are in ``recipes/distillation``).
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from post_training_cookbook.distillation.teacher import TeacherOutput, TeacherRequest

Filter = Callable[[TeacherRequest, TeacherOutput], bool]


def sft_rows(
    requests: Sequence[TeacherRequest],
    outputs: dict[str, TeacherOutput],
    *,
    filters: Iterable[Filter] = (),
    keep_reasoning: bool = True,
) -> tuple[list[dict[str, Any]], Counter]:
    """Build one SFT row per usable teacher output; return the rows and a count of why others were dropped."""
    filters = list(filters)
    rows: list[dict[str, Any]] = []
    dropped: Counter = Counter()
    for request in requests:
        output = outputs.get(request.key)
        if output is None:
            dropped["missing"] += 1
            continue
        if not output.ok:
            dropped["failed"] += 1
            continue
        if output.truncated:
            dropped["truncated"] += 1
            continue
        if not output.content.strip():
            dropped["empty"] += 1
            continue
        rejected = next((f for f in filters if not f(request, output)), None)
        if rejected is not None:
            dropped[getattr(rejected, "__name__", "filter")] += 1
            continue
        assistant: dict[str, Any] = {"role": "assistant", "content": output.content.strip()}
        if keep_reasoning and output.reasoning_content.strip():
            assistant["reasoning_content"] = output.reasoning_content.strip()
        rows.append({"id": request.key, "messages": [*request.messages, assistant], "meta": dict(request.meta)})
    return rows, dropped


def mix(*groups: Sequence[dict[str, Any]], seed: int = 0) -> list[dict[str, Any]]:
    """Concatenate row groups and shuffle them deterministically."""
    rows = [row for group in groups for row in group]
    random.Random(seed).shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
