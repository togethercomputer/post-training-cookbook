"""Training entry points without creating remote sessions."""

import json
from contextlib import nullcontext
from dataclasses import asdict
from unittest.mock import ANY, AsyncMock, Mock

import pytest

from post_training_cookbook.envs.base import Task
from post_training_cookbook.grpo import train
from post_training_cookbook.grpo.data_progress import DataProgress, fingerprint_data
from post_training_cookbook.run_artifacts import Run
from tests.grpo._fakes import FakeCodec, MockSession, OneStepEnv, _base_config


@pytest.mark.parametrize(
    ("weight_sync_type", "use_lora"),
    [
        ("WEIGHT_SYNC_TYPE_SYNCHRONOUS", True),
        ("WEIGHT_SYNC_TYPE_BACKGROUND_PUBLISH", True),
        ("WEIGHT_SYNC_TYPE_PIPELINE", False),
    ],
)
def test_single_turn_rejects_async_collection_before_setup(monkeypatch, weight_sync_type, use_lora):
    get_codec = Mock()
    training_session = Mock()
    monkeypatch.setattr(train, "get_codec", get_codec)
    monkeypatch.setattr(train, "training_session", training_session)
    config = _base_config(
        max_turns=None,
        max_steps_off_policy=0,
        weight_sync_type=weight_sync_type,
        use_lora=use_lora,
    )

    with pytest.raises(ValueError, match="only supported by multi-turn"):
        train.train_single_turn(config)

    get_codec.assert_not_called()
    training_session.assert_not_called()


@pytest.mark.parametrize("budget", [None, 0, 1])
def test_multi_turn_dispatch_preserves_zero_budget(monkeypatch, tmp_path, budget):
    session = MockSession()
    run = Run(dir=tmp_path, wandb=None, rollouts_enabled=False)
    codec = FakeCodec()
    monkeypatch.setattr(train, "get_codec", lambda *args, **kwargs: codec)
    session_factory = Mock(return_value=nullcontext(session))
    monkeypatch.setattr(train, "training_session", session_factory)
    monkeypatch.setattr(train, "track_run", lambda *args, **kwargs: nullcontext(run))
    sync_loop, async_loop = Mock(), AsyncMock()
    monkeypatch.setattr(train, "training_loop", sync_loop)
    monkeypatch.setattr(train, "async_training_loop", async_loop)
    config = _base_config(max_steps_off_policy=budget, episodes_per_task=2, num_steps=1, num_generator_replicas=2)
    tasks = [Task(prompt="q", target="a")]

    def factory(task):
        return OneStepEnv()

    train.train_multi_turn(config, tasks, factory)

    if budget is None:
        sync_loop.assert_called_once()
        async_loop.assert_not_called()
    else:
        sync_loop.assert_not_called()
        async_loop.assert_awaited_once_with(
            1, session, config, codec, tasks, factory, ANY, None, data_progress=ANY, start_step=0
        )
        assert (
            async_loop.call_args.args[6].snapshot_data_state()
            == async_loop.call_args.kwargs["data_progress"].snapshot()
        )
    assert session.checkpoints == []
    assert session_factory.call_args.kwargs["num_generator_replicas"] == 2


def test_single_turn_forwards_generator_replicas(monkeypatch, tmp_path):
    session = MockSession()
    run = Run(dir=tmp_path, wandb=None, rollouts_enabled=False)
    session_factory = Mock(return_value=nullcontext(session))
    monkeypatch.setattr(train, "get_codec", lambda *args, **kwargs: FakeCodec())
    monkeypatch.setattr(train, "training_session", session_factory)
    monkeypatch.setattr(train, "track_run", lambda *args, **kwargs: nullcontext(run))
    monkeypatch.setattr(train, "load_grpo_dataset", lambda *args: ([{}], None))
    monkeypatch.setattr(train, "filter_by_prompt_length", lambda dataset, *args, **kwargs: dataset)
    monkeypatch.setattr(train, "make_single_turn_collector", Mock())
    loop = Mock()
    monkeypatch.setattr(train, "training_loop", loop)

    train.train_single_turn(_base_config(max_turns=None, num_steps=1, num_generator_replicas=3))

    assert session_factory.call_args.kwargs["num_generator_replicas"] == 3
    loop.assert_called_once()
    assert session.checkpoints == []


_KINDS = {"single": "single_turn", "multi": "multi_turn", "async": "multi_turn_async"}


def _progress(mode):
    """Fresh progress over the one-row data the stubbed entry points train on."""
    rows = [{}] if mode == "single" else [asdict(Task(prompt="q", target="a"))]
    return DataProgress(kind=_KINDS[mode], size=1, batch_size=1, fingerprint=fingerprint_data(rows))


def _patch_entry_points(monkeypatch, tmp_path):
    """Stub remote setup and data loading; return the session factory and the sync/async loop mocks."""
    session = MockSession()
    run = Run(dir=tmp_path, wandb=None, rollouts_enabled=False)
    monkeypatch.setattr(train, "get_codec", lambda *args, **kwargs: FakeCodec())
    session_factory = Mock(return_value=nullcontext(session))
    monkeypatch.setattr(train, "training_session", session_factory)
    monkeypatch.setattr(train, "track_run", lambda *args, **kwargs: nullcontext(run))
    monkeypatch.setattr(train, "load_grpo_dataset", lambda *args: ([{}], None))
    monkeypatch.setattr(train, "filter_by_prompt_length", lambda dataset, *args, **kwargs: dataset)
    monkeypatch.setattr(train, "make_single_turn_collector", Mock())
    sync_loop, async_loop = Mock(), AsyncMock()
    monkeypatch.setattr(train, "training_loop", sync_loop)
    monkeypatch.setattr(train, "async_training_loop", async_loop)
    return session_factory, sync_loop, async_loop


def _resume_from_record(tmp_path, mode, record):
    (tmp_path / "checkpoints.jsonl").write_text(json.dumps({"kind": "training", "ref": "checkpoint", **record}) + "\n")
    config = _base_config(
        num_steps=10,
        max_turns=None if mode == "single" else 4,
        max_steps_off_policy=0 if mode == "async" else None,
        resume_from_checkpoint_id="checkpoint",
        resume_run_dir=str(tmp_path),
    )
    if mode == "single":
        train.train_single_turn(config)
    else:
        train.train_multi_turn(config, [Task(prompt="q", target="a")], OneStepEnv)


@pytest.mark.parametrize("mode", ["single", "multi", "async"])
def test_entry_points_forward_checkpoint_progress(monkeypatch, tmp_path, mode):
    progress = _progress(mode)
    for _ in range(7):
        if mode == "async":
            progress.consume(progress.claim())
        else:
            progress.draw_batch()
    session_factory, sync_loop, async_loop = _patch_entry_points(monkeypatch, tmp_path)
    _resume_from_record(tmp_path, mode, {"step": 7, "data_state": progress.snapshot()})
    loop = async_loop if mode == "async" else sync_loop
    assert loop.call_args.args[0] == 10
    assert loop.call_args.kwargs["start_step"] == 7
    assert session_factory.call_args.kwargs["resume_from_checkpoint_id"] == "checkpoint"


@pytest.mark.parametrize("mode", ["single", "multi", "async"])
@pytest.mark.parametrize("step", [10, 11])
def test_completed_target_allocates_no_session(monkeypatch, tmp_path, mode, step):
    record = {"step": step, "data_state": _progress(mode).snapshot()}
    session_factory, sync_loop, async_loop = _patch_entry_points(monkeypatch, tmp_path)
    _resume_from_record(tmp_path, mode, record)
    session_factory.assert_not_called()
    sync_loop.assert_not_called()
    async_loop.assert_not_called()


@pytest.mark.parametrize("mode", ["single", "multi", "async"])
@pytest.mark.parametrize(
    "record,message",
    [
        ({"step": 1}, "no data_state"),
        ({"step": 1, "data_state": DataProgress(kind="other", size=1, batch_size=1).snapshot()}, "does not match"),
    ],
)
def test_unusable_progress_rejected_before_allocation(monkeypatch, tmp_path, mode, record, message):
    session_factory, _, _ = _patch_entry_points(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match=message):
        _resume_from_record(tmp_path, mode, record)
    session_factory.assert_not_called()
