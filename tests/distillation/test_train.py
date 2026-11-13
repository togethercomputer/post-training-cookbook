"""Distillation ``train()``: step budget, eval cadence, weights sync, and failing before provisioning."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from post_training_cookbook.distillation import evaluate, train
from post_training_cookbook.distillation.config import DistillConfig
from post_training_cookbook.distillation.evaluate import EvalSet, score_answered
from tests._codec_fakes import DummyCodec


def _row(i: int) -> dict:
    return {"id": str(i), "messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": "a"}]}


ROWS = [_row(i) for i in range(10)]  # 5 full batches of 2


def _exact(items, outputs):
    return {"accuracy": sum(o == "a" for o in outputs) / len(items)}


def _eval_set(name: str, every_eval: bool = True) -> EvalSet:
    items = [{"id": f"{name}:{i}", "messages": [{"role": "user", "content": f"q{i}"}]} for i in range(3)]
    return EvalSet(name=name, items=items, metrics=_exact, every_eval=every_eval)


def _stub_remote(monkeypatch, tmp_path):
    """Stub the codec, session and run tracking; return (session, factory, events)."""
    events: list[tuple] = []
    session = Mock()
    session.trainer.forward_backward.return_value = Mock(loss=2.0)
    session.trainer.weights_sync.side_effect = lambda **kwargs: events.append(("sync",))
    factory = Mock(return_value=nullcontext(session))

    def fake_student_outputs(session, codec, items, **kwargs):
        events.append(("eval", items[0]["id"].split(":")[0]))
        return ["a"] * len(items), [False] * len(items)

    run = SimpleNamespace(dir=tmp_path)
    monkeypatch.setattr(train, "load_dotenv", lambda **kwargs: None)
    monkeypatch.setattr(train, "get_codec", lambda *args, **kwargs: DummyCodec())
    monkeypatch.setattr(train, "training_session", factory)
    monkeypatch.setattr(train, "track_run", lambda *args, **kwargs: nullcontext(run))
    monkeypatch.setattr(train, "log_step", Mock())
    monkeypatch.setattr(train, "checkpoint_if_due", Mock())
    monkeypatch.setattr(train, "latest_checkpoint_ref", Mock(return_value="ckpt-1"))
    monkeypatch.setattr(train, "student_outputs", fake_student_outputs)
    return session, factory, events


@pytest.mark.parametrize(("num_epochs", "max_steps", "expected"), [(2.0, None, 10), (1.5, None, 7), (2.0, 3, 3)])
def test_step_budget(monkeypatch, tmp_path, num_epochs, max_steps, expected) -> None:
    session, _, _ = _stub_remote(monkeypatch, tmp_path)

    train.train(DistillConfig(model="org/M", batch_size=2, num_epochs=num_epochs, max_steps=max_steps), ROWS)

    assert session.trainer.optim_step.call_count == expected


def test_evals_run_at_step_zero_each_interval_and_the_end_after_a_sync(monkeypatch, tmp_path) -> None:
    session, _, events = _stub_remote(monkeypatch, tmp_path)
    config = DistillConfig(model="org/M", batch_size=2, num_epochs=1.0, eval_interval=2)  # 5 steps

    result = train.train(config, ROWS, [_eval_set("lao"), _eval_set("full", every_eval=False)])

    # Step 0 scores everything; steps 2 and 4 only the every-eval set; step 5 (last) everything.
    assert events == [
        ("eval", "lao"),
        ("eval", "full"),
        ("sync",),
        ("eval", "lao"),
        ("sync",),
        ("eval", "lao"),
        ("sync",),
        ("eval", "lao"),
        ("eval", "full"),
    ]
    assert result.eval_metrics == {
        "lao": {"accuracy": 1.0, "truncated_rate": 0.0},
        "full": {"accuracy": 1.0, "truncated_rate": 0.0},
    }
    assert result.inference_checkpoint == "ckpt-1"
    assert (tmp_path / "eval_full_step0005.json").exists()
    assert not (tmp_path / "eval_full_step0002.json").exists()


def test_no_eval_sets_means_no_weight_syncs(monkeypatch, tmp_path) -> None:
    session, _, events = _stub_remote(monkeypatch, tmp_path)

    train.train(DistillConfig(model="org/M", batch_size=2, num_epochs=1.0, eval_interval=1), ROWS)

    assert events == []
    session.trainer.weights_sync.assert_not_called()


def test_loss_is_logged_per_trained_token(monkeypatch, tmp_path) -> None:
    _stub_remote(monkeypatch, tmp_path)

    train.train(DistillConfig(model="org/M", batch_size=2, max_steps=1), ROWS)

    (_, step, metrics), _ = train.log_step.call_args
    assert step == 1
    assert metrics["loss_sum"] == 2.0
    assert metrics["loss"] == pytest.approx(2.0 / metrics["trained_tokens"])


def test_unrenderable_rows_fail_before_a_session_is_provisioned(monkeypatch, tmp_path) -> None:
    _, factory, _ = _stub_remote(monkeypatch, tmp_path)
    prompt_only = [{"id": "x", "messages": [{"role": "user", "content": "q"}]}]

    with pytest.raises(ValueError, match="none of 1 rows rendered"):
        train.train(DistillConfig(model="org/M", batch_size=1), prompt_only)
    factory.assert_not_called()


def test_too_few_rows_for_a_batch_fail_before_provisioning(monkeypatch, tmp_path) -> None:
    _, factory, _ = _stub_remote(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="cannot fill a batch"):
        train.train(DistillConfig(model="org/M", batch_size=32), ROWS)
    factory.assert_not_called()


def test_eval_sets_need_a_generator(monkeypatch, tmp_path) -> None:
    _, factory, _ = _stub_remote(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="num_generator_replicas"):
        train.train(DistillConfig(model="org/M", batch_size=2, num_generator_replicas=0), ROWS, [_eval_set("lao")])
    factory.assert_not_called()


def test_evaluate_student_scores_the_base_model(monkeypatch, tmp_path) -> None:
    session, factory, events = _stub_remote(monkeypatch, tmp_path)

    scores = train.evaluate_student(DistillConfig(model="org/M"), [_eval_set("lao")])

    assert scores == {"lao": {"accuracy": 1.0, "truncated_rate": 0.0}}
    assert events == [("eval", "lao")]
    session.trainer.optim_step.assert_not_called()
    factory.assert_called_once()


def test_student_outputs_parse_responses_and_flag_truncation() -> None:
    codec = DummyCodec()
    codec.parse_response = lambda tokens: ({"role": "assistant", "content": "".join(map(chr, tokens))}, True)

    def sample_batch(prompts, **kwargs):
        return [
            SimpleNamespace(sequences=[SimpleNamespace(tokens=[ord("h"), ord("i")], stop_reason="STOP_REASON_STOP")]),
            SimpleNamespace(sequences=[SimpleNamespace(tokens=[ord("x")], stop_reason="STOP_REASON_LENGTH")]),
        ][: len(prompts)]

    session = SimpleNamespace(generator=SimpleNamespace(sample_batch=Mock(side_effect=sample_batch)))
    items = [{"id": str(i), "messages": [{"role": "user", "content": "q"}]} for i in range(3)]

    texts, truncated = evaluate.student_outputs(
        session, codec, items, max_tokens=8, temperature=0.0, timeout=None, chunk=2
    )

    assert texts == ["hi", "x", "hi"]
    assert truncated == [False, True, False]
    assert session.generator.sample_batch.call_count == 2


def test_score_answered_skips_failed_teacher_calls() -> None:
    eval_set = _eval_set("lao")

    assert score_answered(eval_set, ["a", None, "b"]) == {"accuracy": 0.5, "n": 2.0}
    assert score_answered(eval_set, [None, None, None]) == {"n": 0.0}


def test_teacher_outputs_map_failures_to_none(monkeypatch, tmp_path) -> None:
    from post_training_cookbook.distillation.teacher import Spend, TeacherConfig, TeacherOutput

    def fake_generate(requests, config, cache_path):
        return {
            "lao:0": TeacherOutput(key="lao:0", ok=True, content="a"),
            "lao:1": TeacherOutput(key="lao:1", ok=False, error="503"),
        }, Spend()

    monkeypatch.setattr(evaluate, "generate", fake_generate)

    outputs = evaluate.teacher_outputs(_eval_set("lao").items, TeacherConfig(model="t"), tmp_path / "t.jsonl")

    assert outputs == ["a", None, None]


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="num_epochs"):
        DistillConfig(model="m", num_epochs=0)
    with pytest.raises(ValueError, match="eval_interval"):
        DistillConfig(model="m", eval_interval=0)
    with pytest.raises(ValueError, match="max_steps"):
        DistillConfig(model="m", max_steps=0)
    with pytest.raises(ValueError, match="use_lora=False"):
        DistillConfig(model="m", use_lora=False, lora_rank=8)


def test_config_from_yaml_rejects_unknown_keys(tmp_path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("model: org/M\nnum_epochs: 1.5\n")
    assert DistillConfig.from_yaml(str(path)).num_epochs == 1.5

    path.write_text("model: org/M\nepochs: 2\n")
    with pytest.raises(ValueError, match="Unknown config keys"):
        DistillConfig.from_yaml(str(path))


def test_unset_adam_fields_are_left_to_the_server() -> None:
    params = train._adam_params(DistillConfig(model="m"), 1e-4)
    assert params == {"learning_rate": 1e-4, "weight_decay": 0.0}

    params = train._adam_params(DistillConfig(model="m", adam_beta2=0.95, adam_eps=1e-8), 2e-4)
    assert params == {"learning_rate": 2e-4, "weight_decay": 0.0, "beta2": 0.95, "eps": 1e-8}
