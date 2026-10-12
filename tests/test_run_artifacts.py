"""Tests for the run-artifacts module."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
import yaml

from post_training_cookbook.dpo.config import DPOConfig
from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.run_artifacts import (
    DueCheckpoints,
    Run,
    checkpoint_if_due,
    checkpoints_due,
    config_to_dict,
    export_episodes,
    export_samples,
    latest_checkpoint_ref,
    log_step,
    run_name,
    start_run,
)
from post_training_cookbook.sft.config import SFTConfig


@dataclass
class _FakeWandb:
    logged: list[tuple[int, dict]]
    url: str = "https://wandb.ai/team/proj/runs/abc"
    project: str = "proj"
    entity: str = "team"
    id: str = "abc"

    def log(self, metrics: dict, step: int) -> None:
        self.logged.append((step, metrics))


def test_run_name_grpo_uses_group_size():
    cfg = GRPOConfig(model="org/Big-Model-7B", batch_size=4, group_size=8, learning_rate=1e-5)
    ts = datetime(2026, 5, 1, 18, 38, 33, tzinfo=UTC)
    name = run_name(cfg, kind="grpo-single-turn", ts=ts)
    assert name == "Big-Model-7B_post-training-cookbook_1.00e-05_4x8_20260501T183833Z"


def test_run_name_sft_omits_group():
    cfg = SFTConfig(model="org/Tiny-1B", batch_size=4, learning_rate=1e-4)
    ts = datetime(2026, 5, 1, 18, 38, 33, tzinfo=UTC)
    name = run_name(cfg, kind="sft", ts=ts)
    assert name == "Tiny-1B_sft_1.00e-04_4_20260501T183833Z"


def test_config_to_dict_round_trip(tmp_path):
    cfg = SFTConfig(model="org/M", batch_size=4, learning_rate=1e-4, max_steps=10)
    run = start_run(cfg, kind="sft", root=tmp_path)
    loaded = SFTConfig(**yaml.safe_load((run.dir / "config.yaml").read_text()))
    assert loaded == cfg


def test_start_run_writes_manifest(tmp_path):
    cfg = SFTConfig(model="org/M", batch_size=4, learning_rate=1e-4, max_steps=10)
    fake = _FakeWandb(logged=[])
    run = start_run(cfg, kind="sft", wandb_run=fake, root=tmp_path)

    manifest = json.loads((run.dir / "manifest.json").read_text())
    assert manifest["kind"] == "sft"
    assert manifest["model"] == "org/M"
    assert manifest["wandb"]["url"] == fake.url
    assert manifest["wandb"]["run_id"] == fake.id
    assert "started_at" in manifest


def test_log_step_writes_jsonl_and_forwards(tmp_path):
    cfg = SFTConfig(model="org/M", batch_size=4, learning_rate=1e-4, max_steps=10)
    fake = _FakeWandb(logged=[])
    run = start_run(cfg, kind="sft", wandb_run=fake, root=tmp_path)

    log_step(run, 0, {"loss": 1.5, "lr": 1e-4})
    log_step(run, 1, {"loss": 1.2, "lr": 9e-5})

    lines = (run.dir / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == [
        {"step": 0, "loss": 1.5, "lr": 1e-4},
        {"step": 1, "loss": 1.2, "lr": 9e-5},
    ]
    assert fake.logged == [(0, {"loss": 1.5, "lr": 1e-4}), (1, {"loss": 1.2, "lr": 9e-5})]


def test_log_step_drops_non_jsonable_from_jsonl(tmp_path):
    """wandb.Table-style values forwarded to wandb but skipped in metrics.jsonl."""
    cfg = SFTConfig(model="org/M", batch_size=4, learning_rate=1e-4, max_steps=10)
    fake = _FakeWandb(logged=[])
    run = start_run(cfg, kind="sft", wandb_run=fake, root=tmp_path)

    bad = object()
    log_step(run, 0, {"loss": 1.0, "table": bad})

    line = json.loads((run.dir / "metrics.jsonl").read_text().strip())
    assert line == {"step": 0, "loss": 1.0}
    assert fake.logged[0][1] == {"loss": 1.0, "table": bad}


def test_log_step_handles_no_wandb(tmp_path):
    cfg = SFTConfig(model="org/M", batch_size=4, learning_rate=1e-4, max_steps=10)
    run = start_run(cfg, kind="sft", root=tmp_path)
    log_step(run, 0, {"loss": 1.0})
    assert json.loads((run.dir / "metrics.jsonl").read_text().strip()) == {"step": 0, "loss": 1.0}


def test_config_to_dict_rejects_non_dataclass():
    with pytest.raises(TypeError):
        config_to_dict({"model": "x"})


def test_export_samples_filename_format(tmp_path):
    cfg = SFTConfig(model="org/M", batch_size=4, learning_rate=1e-4, max_steps=10)
    run = start_run(cfg, kind="sft", root=tmp_path)

    export_samples(run, 7, ["q1"], [["a1"]], [[0.5]])
    files = sorted(p.name for p in run.dir.glob("step_*.json"))
    assert len(files) == 1
    assert files[0].startswith("step_0007_")


def test_export_episodes_writes_groups(tmp_path):
    cfg = SFTConfig(model="org/M", batch_size=4, learning_rate=1e-4, max_steps=10)
    run = start_run(cfg, kind="sft", root=tmp_path)

    export_episodes(run, 3, [[{"reward": 1.0}], [{"reward": 0.0}]])
    files = list(run.dir.glob("step_*.json"))
    assert len(files) == 1
    contents = json.loads(files[0].read_text())
    assert contents == [[{"reward": 1.0}], [{"reward": 0.0}]]


def test_run_is_frozen():
    run = Run(dir="x", wandb=None)  # type: ignore[arg-type]
    with pytest.raises(Exception):
        run.dir = "y"  # type: ignore[misc]


class _CountingSession:
    """Records which steps requested training vs inference checkpoints."""

    def __init__(self) -> None:
        self.training_steps: list[int] = []
        self.inference_steps: list[int] = []

    async def create_training_checkpoint_async(self):
        self.training_steps.append(len(self.training_steps) + 1)
        return SimpleNamespace(id=f"train-{len(self.training_steps)}")

    async def create_inference_checkpoint_async(self):
        self.inference_steps.append(len(self.inference_steps) + 1)
        return SimpleNamespace(id=f"model-{len(self.inference_steps)}")

    def run(self, coro):
        return asyncio.run(coro)


@pytest.mark.parametrize(
    "step, total_steps, training_interval, inference_interval, expected",
    [
        (20, 40, 20, 20, DueCheckpoints(True, True)),
        (40, 40, 20, 20, DueCheckpoints(True, True)),  # final on boundary: still one inference
        (10, 40, 20, 20, DueCheckpoints(False, False)),
        (5, 5, 0, 0, DueCheckpoints(False, True)),  # intervals off, final still due
        (3, 5, 0, 3, DueCheckpoints(False, True)),  # interval hit, not final
        (4, 5, 0, 3, DueCheckpoints(False, False)),
        (5, 5, 0, 3, DueCheckpoints(False, True)),  # final, off boundary
    ],
)
def test_checkpoints_due(step, total_steps, training_interval, inference_interval, expected):
    assert (
        checkpoints_due(step, total_steps, training_interval=training_interval, inference_interval=inference_interval)
        == expected
    )


def _checkpoint_if_due(session, run, cfg, *, step, total_steps):
    checkpoint_if_due(
        session,
        run,
        step=step,
        total_steps=total_steps,
        training_interval=cfg.training_checkpoint_interval,
        inference_interval=cfg.inference_checkpoint_interval,
    )


def test_checkpoint_if_due_saves_and_records(tmp_path):
    """The effectful wrapper maps due-decisions to save calls and ordered jsonl records."""
    cfg = SFTConfig(training_checkpoint_interval=20, inference_checkpoint_interval=20)
    run = start_run(cfg, kind="sft", root=tmp_path)
    session = _CountingSession()

    for step in range(40):
        _checkpoint_if_due(session, run, cfg, step=step + 1, total_steps=40)

    saved = [json.loads(line) for line in (run.dir / "checkpoints.jsonl").read_text().splitlines()]
    assert [(r["kind"], r["step"], r["ref"]) for r in saved] == [
        ("training", 20, "train-1"),
        ("inference", 20, "model-1"),
        ("training", 40, "train-2"),
        ("inference", 40, "model-2"),
    ]
    assert latest_checkpoint_ref(run, kind="training") == "train-2"
    assert latest_checkpoint_ref(run, kind="inference") == "model-2"


def test_latest_checkpoint_ref_missing_file(tmp_path):
    run = start_run(SFTConfig(), kind="sft", root=tmp_path)
    assert latest_checkpoint_ref(run, kind="training") is None


@pytest.mark.parametrize(
    "make_config",
    [
        lambda **kw: SFTConfig(**kw),
        lambda **kw: DPOConfig(**kw),
        lambda **kw: GRPOConfig(model="org/M", **kw),
    ],
)
@pytest.mark.parametrize("field", ["training_checkpoint_interval", "inference_checkpoint_interval"])
def test_negative_checkpoint_interval_rejected(make_config, field):
    with pytest.raises(ValueError):
        make_config(**{field: -1})


class _FailingSession:
    """Training checkpoints succeed; inference checkpoints raise ``exc``."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.inference_attempts = 0

    async def create_training_checkpoint_async(self):
        return SimpleNamespace(id="train-ok")

    async def create_inference_checkpoint_async(self):
        self.inference_attempts += 1
        raise self.exc

    def run(self, coro):
        return asyncio.run(coro)


def test_intermediate_checkpoint_failure_is_swallowed(tmp_path):
    """A tolerated failure on an intermediate checkpoint logs and continues, writing no record."""
    cfg = SFTConfig(inference_checkpoint_interval=2)
    run = start_run(cfg, kind="sft", root=tmp_path)
    session = _FailingSession(RuntimeError("operation failed"))

    _checkpoint_if_due(session, run, cfg, step=2, total_steps=10)

    assert session.inference_attempts == 1
    assert not (run.dir / "checkpoints.jsonl").exists()


def test_final_checkpoint_failure_propagates(tmp_path):
    """The mandatory final inference checkpoint must not be silently swallowed."""
    cfg = SFTConfig()
    run = start_run(cfg, kind="sft", root=tmp_path)
    session = _FailingSession(RuntimeError("operation failed"))

    with pytest.raises(RuntimeError):
        _checkpoint_if_due(session, run, cfg, step=10, total_steps=10)


def test_unexpected_checkpoint_error_always_propagates(tmp_path):
    """Errors outside the tolerated set abort even for intermediate checkpoints."""
    cfg = SFTConfig(inference_checkpoint_interval=2)
    run = start_run(cfg, kind="sft", root=tmp_path)
    session = _FailingSession(ValueError("unexpected bug"))

    with pytest.raises(ValueError):
        _checkpoint_if_due(session, run, cfg, step=2, total_steps=10)
