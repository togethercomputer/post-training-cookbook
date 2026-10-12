"""Resume real collectors from checkpoint records, including prefetched task gaps."""

import asyncio
import json
from dataclasses import asdict, replace

import pytest
from datasets import Dataset

from post_training_cookbook.envs.base import Task
from post_training_cookbook.grpo import single_turn
from post_training_cookbook.grpo.async_loop import async_training_loop
from post_training_cookbook.grpo.data_progress import DataProgress, fingerprint_data
from post_training_cookbook.grpo.loop import training_loop
from post_training_cookbook.grpo.multi_turn import make_multi_turn_collector
from post_training_cookbook.grpo.resume import load_resume_point, start_progress
from post_training_cookbook.run_artifacts import Run, checkpoint_if_due_async
from tests.grpo._fakes import FakeCodec, MockSession, OneStepEnv, _base_config
from tests.grpo.test_single_turn import _ds_cfg, _outcome

_SOURCE = {"kind": "multi_turn_async", "size": 10, "batch_size": 2, "fingerprint": "data"}


@pytest.mark.parametrize(
    "change",
    [
        {"version": 99},
        {"source": {**_SOURCE, "size": 11}},
        {"source": {**_SOURCE, "kind": "single_turn"}},
        {"source": {**_SOURCE, "batch_size": 3}},
        {"source": {**_SOURCE, "fingerprint": "other"}},
    ],
)
def test_restore_rejects_changed_source(change):
    state = {**DataProgress(**_SOURCE).snapshot(), **change}
    with pytest.raises(ValueError, match="does not match"):
        DataProgress.restore(state, **_SOURCE)


@pytest.mark.parametrize(
    "size,batch_size,expected",
    [
        (6, 3, [[0, 1, 2], [3, 4, 5], [0, 1, 2]]),
        (4, 3, [[0, 1, 2], [3, 0, 1], [2, 3, 0]]),
        (3, 5, [[0, 1, 2, 0, 1], [2, 0, 1, 2, 0]]),
    ],
)
def test_batches_wrap_around_dataset(size, batch_size, expected):
    progress = DataProgress(kind="single_turn", size=size, batch_size=batch_size)
    assert [progress.draw_batch() for _ in expected] == expected


def test_pending_tasks_replayed_once():
    progress = DataProgress(**_SOURCE)
    assert [progress.claim() for _ in range(4)] == [0, 1, 2, 3]
    progress.consume(1)
    snapshot = progress.snapshot()
    progress.consume(0)
    resumed = DataProgress.restore(snapshot, **_SOURCE)
    assert [resumed.claim() for _ in range(4)] == [0, 2, 3, 4]
    resumed.retry(2)
    assert resumed.claim() == 2
    assert resumed.claim() == 5


@pytest.mark.parametrize("checkpoint_step", [1, 3])
@pytest.mark.parametrize("max_attempts", [1, 2])
@pytest.mark.parametrize("mode", ["single_turn", "multi_turn"])
def test_collector_resumes_same_batches(tmp_path, monkeypatch, mode, max_attempts, checkpoint_step):
    dataset = Dataset.from_dict({"i": list(range(7))})
    tasks = [Task(prompt=f"q{i}", target="x") for i in range(7)]
    config = _base_config(
        batch_size=2,
        group_size=2,
        episodes_per_task=2,
        max_sampling_attempts=max_attempts,
        training_checkpoint_interval=1,
        max_prompt_length=64,
    )
    draws = []
    episode_counts = {}

    def score(prompts, session, **kwargs):
        draws.extend(prompt.question for prompt in prompts)
        return [
            _outcome(prompt.question, [0, 0] if prompt.question in {"q0", "q1", "q4"} else [0, 1])
            for prompt in prompts
        ]

    def factory(task):
        count = episode_counts.get(task.prompt, 0)
        episode_counts[task.prompt] = count + 1
        if count % 2 == 0:
            draws.append(task.prompt)
        reward = 0 if task.prompt in {"q0", "q1", "q4"} else count % 2
        return OneStepEnv(reward=reward)

    monkeypatch.setattr(single_turn, "_sample_and_score_prompts", score)

    def execute(cfg, directory):
        directory.mkdir()
        rows = dataset if mode == "single_turn" else (asdict(task) for task in tasks)
        start_step, progress = start_progress(cfg, load_resume_point(cfg), kind=mode, size=7, rows=rows)
        run = Run(directory, rollouts_enabled=False, snapshot_data_state=progress.snapshot)
        if mode == "single_turn":
            collector = single_turn.make_single_turn_collector(
                cfg, _ds_cfg(), dataset, FakeCodec(), run, data_progress=progress
            )
        else:
            collector = make_multi_turn_collector(cfg, FakeCodec(), tasks, factory, run, data_progress=progress)
        training_loop(6, collector, MockSession(), cfg, run, start_step=start_step)

    execute(config, tmp_path / "full")
    full_draws = list(draws)
    records = [json.loads(line) for line in (tmp_path / "full/checkpoints.jsonl").read_text().splitlines()]
    record = next(record for record in records if record["kind"] == "training" and record["step"] == checkpoint_step)
    # Keep just the selected record: the fake session reuses its checkpoint ID.
    source = tmp_path / "source"
    source.mkdir()
    (source / "checkpoints.jsonl").write_text(json.dumps(record) + "\n")
    draws.clear()
    resumed = replace(config, resume_from_checkpoint_id=record["ref"], resume_run_dir=str(source))
    execute(resumed, tmp_path / "resumed")
    assert draws == full_draws[record["data_state"]["next_index"] :]
    assert draws[0] != "q0"  # This checkpoint is deliberately away from the dataset start.
    if max_attempts == 2:
        assert record["data_state"]["next_index"] > record["step"] * config.batch_size


@pytest.mark.asyncio
async def test_snapshot_precedes_remote_await(tmp_path):
    progress = DataProgress(**_SOURCE)
    progress.claim()
    session = MockSession()

    async def checkpoint():
        progress.claim()  # Background worker advances while the checkpoint request is in flight.
        return session.create_training_checkpoint()

    session.create_training_checkpoint_async = checkpoint
    run = Run(tmp_path, snapshot_data_state=progress.snapshot)
    await checkpoint_if_due_async(session, run, step=1, total_steps=2, training_interval=1, inference_interval=0)
    record = json.loads((tmp_path / "checkpoints.jsonl").read_text())
    assert record["ref"] == "tckpt-1"
    assert record["data_state"]["next_index"] == 1
    assert progress.snapshot()["next_index"] == 2


@pytest.mark.asyncio
async def test_async_replays_unconsumed_tasks(tmp_path):
    tasks = [Task(prompt=str(i), target="a") for i in range(20)]
    config = _base_config(
        episodes_per_task=2, groups_in_flight=2, max_steps_off_policy=0, training_checkpoint_interval=1
    )
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    scheduled = []
    never_ready = asyncio.Event()

    def factory(task):
        scheduled.append(int(task.prompt))
        return OneStepEnv(reward=len(scheduled) % 2, gate=never_ready if task.prompt == "0" else None)

    _, progress = start_progress(
        config, None, kind="multi_turn_async", size=len(tasks), rows=(asdict(task) for task in tasks)
    )
    first_run = Run(first_dir, rollouts_enabled=False, snapshot_data_state=progress.snapshot)
    async with asyncio.timeout(5):
        await async_training_loop(
            1, MockSession(), config, FakeCodec(), tasks, factory, first_run, data_progress=progress
        )
    records = [json.loads(line) for line in (first_dir / "checkpoints.jsonl").read_text().splitlines()]
    record = next(record for record in records if record["kind"] == "training")
    state = record["data_state"]
    assert 0 in state["pending"]  # Slow earlier task must not be skipped on resume.
    assert 1 not in state["pending"]  # Later task already trained, so it must not replay.
    assert state["next_index"] >= 3  # The other worker prefetched ahead.
    scheduled.clear()

    def resumed_factory(task):
        scheduled.append(int(task.prompt))
        return OneStepEnv(reward=len(scheduled) % 2)

    resumed = replace(config, resume_from_checkpoint_id=record["ref"], resume_run_dir=str(first_dir))
    start_step, progress = start_progress(
        resumed,
        load_resume_point(resumed),
        kind="multi_turn_async",
        size=len(tasks),
        rows=(asdict(task) for task in tasks),
    )
    async with asyncio.timeout(5):
        await async_training_loop(
            2,
            MockSession(),
            resumed,
            FakeCodec(),
            tasks,
            resumed_factory,
            Run(second_dir, rollouts_enabled=False),
            data_progress=progress,
            start_step=start_step,
        )
    claimed = scheduled[::2]
    expected = state["pending"] + list(range(state["next_index"], state["next_index"] + len(claimed)))
    assert claimed == expected[: len(claimed)]
    assert claimed[0] == 0
    assert 1 not in claimed


def test_fingerprint_only_for_checkpointed_runs():
    rows = [{"id": 1}, {"id": 2}]
    assert fingerprint_data(rows) != fingerprint_data(reversed(rows))
    unhashable = [{"id": object()}]
    _, progress = start_progress(_base_config(), None, kind="single_turn", size=1, rows=unhashable)
    assert progress.snapshot()["source"]["fingerprint"] is None
    with pytest.raises(ValueError, match="JSON-serializable"):
        start_progress(_base_config(training_checkpoint_interval=1), None, kind="single_turn", size=1, rows=unhashable)
