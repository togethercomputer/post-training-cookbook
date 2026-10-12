"""Run artifacts: a self-contained record of a training run.

A ``Run`` bundles a directory, an optional wandb handle, and a flag for
whether bulky rollout dumps are written. Callers pass the same ``Run`` to
``log_step``, ``export_samples``, and ``export_episodes`` — exports are
silently skipped when ``rollouts_enabled=False``, so collectors don't need
to branch on a config toggle.

Layout written to ``runs/<run_name>/``:

    config.yaml      — round-trippable via ``Config.from_yaml``
    manifest.json    — wandb url, session id, model, start time, git sha
    metrics.jsonl    — one line per step (mirrors ``wandb_run.log``)
    step_NNNN_*.json — per-step rollouts (when ``rollouts_enabled=True``)
    val_NNNN_*.json  — validation rollouts (when ``rollouts_enabled=True``)
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import wandb
import yaml
from together.error import APIError
from together.post_training import SessionClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Run:
    """Artifact destination for one training run.

    ``snapshot_data_state`` supplies the ``data_state`` field of training-checkpoint
    records: an independent JSON snapshot taken before the checkpoint is requested
    and recorded with its returned ID. Inference records carry no data state.
    """

    dir: Path
    wandb: Any | None = None
    rollouts_enabled: bool = True
    snapshot_data_state: Callable[[], dict[str, Any]] | None = None


def run_name(config: Any, *, kind: str, ts: datetime) -> str:
    """Compose a unique run-name from config + UTC timestamp."""
    model_id = config.model or getattr(config, "model_resources_id", None) or "model"
    model_short = model_id.split("/")[-1]
    stamp = ts.strftime("%Y%m%dT%H%M%SZ")
    project = getattr(config, "wandb_project", None) or kind
    return f"{model_short}_{project}_{config.run_name_suffix()}_{stamp}"


def config_to_dict(config: Any) -> dict[str, Any]:
    if not is_dataclass(config):
        raise TypeError(f"Expected a dataclass, got {type(config).__name__}")
    return asdict(config)


def start_run(
    config: Any,
    *,
    kind: str,
    wandb_run: Any | None = None,
    session_id: str | None = None,
    root: Path | str = "runs",
) -> Run:
    """Create ``runs/<name>/``, write config.yaml and manifest.json, return Run.

    ``Run.rollouts_enabled`` is read from ``config.export_samples`` (defaulting
    to True if absent), so collectors can hand the same ``Run`` to the export
    functions without branching.
    """
    ts = datetime.now(UTC)
    name = getattr(config, "wandb_run_name", None) or run_name(config, kind=kind, ts=ts)
    run_dir = Path(root) / name
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.yaml").write_text(yaml.safe_dump(config_to_dict(config), sort_keys=False))
    manifest = _manifest(config, kind=kind, ts=ts, wandb_run=wandb_run, name=name, session_id=session_id)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Run artifacts: {run_dir}")
    return Run(dir=run_dir, wandb=wandb_run, rollouts_enabled=getattr(config, "export_samples", True))


@contextmanager
def track_run(config: Any, *, kind: str, session_id: str, total_steps: int) -> Iterator[Run]:
    """Scope a wandb run (when ``config.wandb_project`` is set) and a run-artifact ``Run``.

    Starts the wandb run, yields the ``Run`` to log against, and finishes the
    wandb run on exit. ``total_steps`` is the computed step cap recorded in the
    wandb config alongside the rest of the config. The finish is best-effort so a
    teardown failure can't mask an exception from the ``with`` body.
    """
    wandb_run = None
    if config.wandb_project:
        wandb_run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name,
            group=config.wandb_group,
            config={"total_steps": total_steps, **config_to_dict(config)},
            reinit="create_new",
        )
    try:
        yield start_run(config, kind=kind, wandb_run=wandb_run, session_id=session_id)
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception as exc:
                logger.warning("wandb.finish() failed (%s: %s)", type(exc).__name__, exc)


class DueCheckpoints(NamedTuple):
    training: bool
    inference: bool


def checkpoints_due(step: int, total_steps: int, *, training_interval: int, inference_interval: int) -> DueCheckpoints:
    """Which checkpoints are due at ``step`` (1-indexed)."""
    training = bool(training_interval) and step % training_interval == 0
    inference = step == total_steps or (bool(inference_interval) and step % inference_interval == 0)
    return DueCheckpoints(training, inference)


_CHECKPOINT_ERRORS = (APIError, RuntimeError, TimeoutError)


def checkpoint_if_due(
    session: SessionClient, run: Run, *, step: int, total_steps: int, training_interval: int, inference_interval: int
) -> None:
    """Save the checkpoints due at ``step`` (the 1-indexed just-completed optimizer step).

    Training checkpoints capture optimizer state for resuming; inference
    checkpoints register a servable, downloadable model.
    """
    session.run(
        checkpoint_if_due_async(
            session,
            run,
            step=step,
            total_steps=total_steps,
            training_interval=training_interval,
            inference_interval=inference_interval,
        )
    )


async def checkpoint_if_due_async(
    session: SessionClient, run: Run, *, step: int, total_steps: int, training_interval: int, inference_interval: int
) -> None:
    """Async counterpart to :func:`checkpoint_if_due` for session event-loop callers."""
    due = checkpoints_due(
        step, total_steps, training_interval=training_interval, inference_interval=inference_interval
    )
    if due.training:
        await _save_checkpoint_async(session, run, step, kind="training", required=False)
    if due.inference:
        await _save_checkpoint_async(session, run, step, kind="inference", required=step == total_steps)


async def _save_checkpoint_async(session: SessionClient, run: Run, step: int, *, kind: str, required: bool) -> None:
    """Create and record one checkpoint without re-entering the session event loop."""
    # Freeze before the remote await: async collectors may keep issuing work.
    data_state = run.snapshot_data_state() if kind == "training" and run.snapshot_data_state is not None else None
    try:
        if kind == "training":
            checkpoint = await session.create_training_checkpoint_async()
        elif kind == "inference":
            checkpoint = await session.create_inference_checkpoint_async()
        else:
            raise ValueError(f"unknown checkpoint kind: {kind}")
    except _CHECKPOINT_ERRORS as exc:
        if required:
            raise
        logger.warning("[step %d] %s checkpoint failed, continuing (%s: %s)", step, kind, type(exc).__name__, exc)
        return
    _record_checkpoint(run, step, kind=kind, ref=checkpoint.id, data_state=data_state)


def latest_checkpoint_ref(run: Run, *, kind: str) -> str | None:
    """The last recorded checkpoint ``ref`` of ``kind`` in this run, if any."""
    refs = [record["ref"] for record in _checkpoint_records(run.dir) if record["kind"] == kind]
    return refs[-1] if refs else None


def find_checkpoint_record(run_dir: Path, *, kind: str, ref: str) -> dict[str, Any] | None:
    """The last recorded checkpoint of ``kind`` named ``ref`` in ``run_dir``, if any."""
    matches = [record for record in _checkpoint_records(run_dir) if record["kind"] == kind and record["ref"] == ref]
    return matches[-1] if matches else None


def _checkpoint_records(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "checkpoints.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _record_checkpoint(run: Run, step: int, *, kind: str, ref: str, data_state: dict[str, Any] | None = None) -> None:
    """Append the checkpoint to ``checkpoints.jsonl`` and announce it on stdout."""
    record = {
        "step": step,
        "kind": kind,
        "ref": ref,
        "saved_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if data_state is not None:
        record["data_state"] = data_state
    with (run.dir / "checkpoints.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[step {step}] {kind} checkpoint: {ref}")


def log_step(run: Run, step: int, metrics: dict[str, Any]) -> None:
    """Forward to wandb (if present) and append one JSON line to metrics.jsonl.

    Non-JSON-serializable values (e.g. ``wandb.Table``) are forwarded to wandb
    but dropped from metrics.jsonl. wandb gets the full metrics dict.
    """
    if run.wandb is not None:
        run.wandb.log(metrics, step=step)
    record = {"step": step} | _drop_non_jsonable(metrics)
    with (run.dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def export_samples(
    run: Run,
    step: int,
    questions: list[str],
    batch_completions: list[list[str]],
    batch_rewards: list[list[float]],
    *,
    prefix: str = "step",
) -> None:
    """Write a per-step rollout dump. No-op when ``run.rollouts_enabled`` is False."""
    if not run.rollouts_enabled:
        return
    entries = [
        {"prompt": q, "completions": cs, "rewards": rs}
        for q, cs, rs in zip(questions, batch_completions, batch_rewards)
    ]
    path = _rollout_path(run, prefix, step)
    path.write_text(json.dumps(entries, indent=2))
    logger.info("Saved %d sample groups to %s", len(entries), path)


def export_episodes(
    run: Run,
    step: int,
    groups: list[list[dict]],
    *,
    prefix: str = "step",
) -> None:
    """Write a per-step episode dump. No-op when ``run.rollouts_enabled`` is False."""
    if not run.rollouts_enabled:
        return
    path = _rollout_path(run, prefix, step)
    path.write_text(json.dumps(groups, indent=2))
    logger.info("Saved %d episodes (%d groups) to %s", sum(len(g) for g in groups), len(groups), path)


def _rollout_path(run: Run, prefix: str, step: int) -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return run.dir / f"{prefix}_{step:04d}_{ts}.json"


def _manifest(
    config: Any,
    *,
    kind: str,
    ts: datetime,
    wandb_run: Any | None,
    name: str,
    session_id: str | None,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "kind": kind,
        "run_name": name,
        "started_at": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": getattr(config, "model", None),
        "git": _git_info(),
    }
    if session_id is not None:
        info["session_id"] = session_id
    if wandb_run is not None:
        info["wandb"] = {
            "url": getattr(wandb_run, "url", None),
            "project": getattr(wandb_run, "project", None),
            "entity": getattr(wandb_run, "entity", None),
            "run_id": getattr(wandb_run, "id", None),
        }
    return info


def _git_info() -> dict[str, Any]:
    def _run(args: list[str]) -> str:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()

    try:
        sha = _run(["git", "rev-parse", "--short", "HEAD"])
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        dirty = bool(_run(["git", "status", "--porcelain"]))
        return {"sha": sha, "branch": branch, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}


def _drop_non_jsonable(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return ``metrics`` minus any values ``json.dumps`` can't encode."""
    return {k: v for k, v in metrics.items() if _is_jsonable(v)}


def _is_jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
    except TypeError:
        return False
    return True
