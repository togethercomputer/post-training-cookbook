"""Batch-dispenser primitives for the per-step collector.

Both modes draw fixed-size batches from a finite source (rows or tasks) and
optionally redraw on zero-variance groups (DAPO §3.2 dynamic sampling). The
redraw machinery lives here without knowing anything about GRPO; batch indices
come from :class:`post_training_cookbook.grpo.data_progress.DataProgress`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

_T = TypeVar("_T")


def oversample_until_kept(
    draw_batch: Callable[[], Iterable[_T]],
    *,
    is_kept: Callable[[_T], bool],
    target: int,
    max_attempts: int,
) -> tuple[list[_T], int]:
    """Draw batches until ``target`` kept items accumulate, capped at ``max_attempts``.

    Non-keepers preceding the ``target``-th keeper are retained so callers can
    compute accurate skip-rate diagnostics; trailing items past it are dropped.
    """
    items: list[_T] = []
    kept = 0
    for attempts in range(1, max_attempts + 1):
        for item in draw_batch():
            items.append(item)
            if is_kept(item):
                kept += 1
                if kept >= target:
                    return items, attempts
    return items, attempts
