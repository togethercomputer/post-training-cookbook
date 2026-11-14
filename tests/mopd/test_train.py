import json
from contextlib import contextmanager

import pytest

from post_training_cookbook.mopd import train
from post_training_cookbook.mopd.config import MOPDConfig

from ._fakes import CharCodec, FakeResource, logprob_of, toy_domain

ANSWER = [ord(c) for c in "ok a1"]


def _rows(prefix, n):
    return [{"q": f"{prefix}{i}?", "a": f"{prefix}{i}"} for i in range(n)]


@pytest.fixture
def harness(tmp_path, monkeypatch):
    resource = FakeResource(answer=ANSWER)

    @contextmanager
    def fake_model_resource(**kwargs):
        yield resource

    monkeypatch.setattr(train, "model_resource", fake_model_resource)
    monkeypatch.setattr(train, "get_codec", lambda *args, **kwargs: CharCodec())
    monkeypatch.setattr(train, "resolve_base_model", lambda config: "toy-model")
    monkeypatch.chdir(tmp_path)
    return resource


def _config(**overrides):
    values = {
        "model_resources_id": "mr-fake",
        "teacher_checkpoints": {"alpha": "ckpt-a", "beta": "ckpt-b"},
        "num_steps": 2,
        "prompts_per_domain": 2,
        "samples_per_prompt": 3,
        "max_model_length": 64,
        "max_prompt_length": 32,
        "max_sample_tokens": 16,
        "eval_interval": 1,
        "eval_examples_per_domain": 2,
        "wandb_run_name": "mopd-test",
    }
    values.update(overrides)
    return MOPDConfig(**values)


def _domains():
    return [toy_domain("alpha", _rows("a", 10), 6, 3), toy_domain("beta", _rows("b", 10), 6, 3)]


def test_fusion_routes_each_completion_to_its_teacher_and_trains_on_teacher_minus_student(harness):
    result = train.train(_config(), _domains())

    teachers = {
        s.create_kwargs["display_name"]: s
        for s in harness.sessions
        if s.create_kwargs["display_name"].startswith("teacher-")
    }
    assert {name: s.create_kwargs["resume_from_checkpoint_id"] for name, s in teachers.items()} == {
        "teacher-alpha": "ckpt-a",
        "teacher-beta": "ckpt-b",
    }
    student = next(s for s in harness.sessions if s.create_kwargs["display_name"] == "mopd-test")
    assert student.create_kwargs["resume_from_checkpoint_id"] is None

    # Each teacher scored only its own domain's 2 prompts x 3 samples per step.
    for teacher in teachers.values():
        scored = [call for call in teacher.generator.sample_calls if call["prompt_logprobs"]]
        assert len(scored) == 2 and all(len(call["prompts"]) == 6 for call in scored)

    # Two steps, each one importance-sampling pass over 2 domains x 2 prompts x 3 samples.
    calls = student.trainer.forward_backward_calls
    assert len(calls) == 2
    assert calls[0]["loss"]["type"] == "LOSS_TYPE_IMPORTANCE_SAMPLING"
    sample = calls[0]["samples"][0]
    adv = [
        a
        for a, w in zip(sample["loss_fn_inputs"]["advantages"]["data"], sample["loss_fn_inputs"]["weights"]["data"])
        if w
    ]
    expected = [logprob_of(t, offset=0.2) - logprob_of(t, offset=0.0) for t in ANSWER]
    assert adv == pytest.approx(expected)  # teacher (offset 0.2) minus student, per token
    assert len(student.trainer.optim_steps) == 2 and student.trainer.syncs == 2

    # The final step saves a training checkpoint (for resume / eval sessions) and an inference one.
    assert student.checkpoints == ["training", "inference"]
    assert result.training_checkpoint_id.startswith("ckpt-train-")
    assert result.inference_checkpoint_id.startswith("ckpt-inf-")
    assert all(s.stopped for s in harness.sessions)

    metrics = [json.loads(line) for line in (result.run_dir / "metrics.jsonl").read_text().splitlines()]
    train_steps = [m for m in metrics if "reverse_kl" in m]
    assert [m["step"] for m in train_steps] == [1, 2]
    assert train_steps[0]["alpha/reverse_kl"] == pytest.approx(0.2)
    assert train_steps[0]["alpha/token_frac"] == pytest.approx(0.5)
    assert 0.0 <= train_steps[0]["alpha/sample_accuracy"] <= 1.0
    val_steps = [m["step"] for m in metrics if "val/mean_accuracy" in m]
    assert val_steps == [0, 1, 2]
    sessions = json.loads((result.run_dir / "sessions.json").read_text())
    assert sessions["teacher_checkpoints"] == {"alpha": "ckpt-a", "beta": "ckpt-b"}


def test_fusion_rejects_teachers_that_do_not_match_the_domains(harness):
    with pytest.raises(ValueError, match="must name every domain"):
        train.train(_config(teacher_checkpoints={"alpha": "ckpt-a"}), _domains())


def test_student_can_start_from_a_training_checkpoint(harness):
    train.train(
        _config(resume_from_checkpoint_id="ckpt-sft", load_optimizer=False, num_steps=1, eval_interval=0), _domains()
    )
    student = next(s for s in harness.sessions if s.create_kwargs["display_name"] == "mopd-test")
    assert student.create_kwargs["resume_from_checkpoint_id"] == "ckpt-sft"
    assert student.create_kwargs["load_optimizer"] is False


def test_trainer_scoring_gives_the_same_signal(harness):
    result = train.train(_config(teacher_scoring="trainer", num_steps=1, eval_interval=0), _domains())
    metrics = [json.loads(line) for line in (result.run_dir / "metrics.jsonl").read_text().splitlines()]
    assert metrics[0]["beta/reverse_kl"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": "m"},  # both model sources
        {"model_resources_id": None},  # neither
        {"teacher_checkpoints": {}},
        {"balance": "prompt"},
        {"teacher_scoring": "serverless"},
        {"max_prompt_length": 60},
        {"temperature": 0.0},
        {"num_steps": 0},
        {"load_optimizer": False},  # weights-only resume needs a checkpoint to load
    ],
)
def test_config_validation(overrides):
    with pytest.raises(ValueError):
        _config(**overrides)


def test_two_students_share_one_set_of_teachers(harness):
    from concurrent.futures import ThreadPoolExecutor

    token = _config(num_steps=2, eval_interval=0, wandb_run_name="student-token")
    domain = _config(num_steps=2, eval_interval=0, wandb_run_name="student-domain", balance="domain")
    with train.teacher_sessions(harness, token) as teachers:
        assert set(teachers) == {"alpha", "beta"}
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(train.train_student, c, _domains(), harness, teachers) for c in (token, domain)]
            results = [f.result() for f in futures]
        assert not any(t.stopped for t in teachers.values())  # still open for the next student
    assert all(t.stopped for t in teachers.values())
    names = [s.create_kwargs["display_name"] for s in harness.sessions]
    assert sorted(names) == ["student-domain", "student-token", "teacher-alpha", "teacher-beta"]
    assert {r.run_dir.name for r in results} == {"student-token", "student-domain"}
    # Each teacher scored both students' completions: 2 students x 2 steps.
    for teacher in teachers.values():
        assert len([c for c in teacher.generator.sample_calls if c["prompt_logprobs"]]) == 4
