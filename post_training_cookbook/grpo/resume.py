"""Decide where a GRPO run starts: the progress saved with a checkpoint, or a fresh start."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.grpo.data_progress import DataProgress, fingerprint_data
from post_training_cookbook.run_artifacts import find_checkpoint_record


@dataclass(frozen=True)
class ResumePoint:
    """Completed client steps and the data position saved with a training checkpoint."""

    step: int
    data_state: dict[str, Any]


def load_resume_point(config: GRPOConfig) -> ResumePoint | None:
    """Read where the source run stood at ``resume_from_checkpoint_id``.

    ``None`` means start fresh at step 0: no checkpoint, or a weights-only load.
    The server's optimizer step cannot stand in for the client step, because empty
    batches advance the client schedule without an optimizer update.
    """
    if not config.continues_run:
        return None
    checkpoint_id, run_dir = config.resume_from_checkpoint_id, Path(config.resume_run_dir)
    record = find_checkpoint_record(run_dir, kind="training", ref=checkpoint_id)
    if record is None:
        raise ValueError(f"No training checkpoint {checkpoint_id} in {run_dir / 'checkpoints.jsonl'}")
    if "data_state" not in record:
        raise ValueError(
            f"Training checkpoint {checkpoint_id} has no data_state, so its data position is unknown; "
            "use load_optimizer=False to start a new run from the weights"
        )
    return ResumePoint(step=record["step"], data_state=record["data_state"])


def start_progress(
    config: GRPOConfig, resume: ResumePoint | None, *, kind: str, size: int, rows: Iterable[Any]
) -> tuple[int, DataProgress]:
    """Return the client step and data cursor this run starts from.

    ``rows`` are hashed only when a training checkpoint may record or restore their
    position, so only checkpointed runs need JSON-serializable inputs.
    """
    fingerprint = fingerprint_data(rows) if resume is not None or config.training_checkpoint_interval else None
    source = {"kind": kind, "size": size, "batch_size": config.batch_size, "fingerprint": fingerprint}
    if resume is None:
        return 0, DataProgress(**source)
    return resume.step, DataProgress.restore(resume.data_state, **source)
