import pytest

from post_training_cookbook.mopd.teachers import open_teacher, score_with_generator, score_with_trainer

from ._fakes import CharCodec, FakeResource, FakeSession, logprob_of, toy_domain


def test_generator_scores_are_the_prompt_logprobs_of_the_completion_positions():
    teacher = FakeSession("t", answer=[], offset=0.2)
    prompts = [[1, 50, 51], [1, 60]]
    completions = [[70, 71], [80, 81, 82]]
    scores = score_with_generator(teacher, prompts, completions)
    assert scores == [
        [logprob_of(70, offset=0.2), logprob_of(71, offset=0.2)],
        [logprob_of(80, offset=0.2), logprob_of(81, offset=0.2), logprob_of(82, offset=0.2)],
    ]
    call = teacher.generator.sample_calls[0]
    assert call["prompt_logprobs"] is True
    assert call["params"]["max_tokens"] == 1
    assert call["prompts"][0]["chunks"][0]["encoded_text"]["tokens"] == [1, 50, 51, 70, 71]


def test_trainer_scores_match_generator_scores_on_the_same_tokens():
    teacher = FakeSession("t", answer=[], offset=0.2)
    prompts = [[1, 50, 51], [1, 60]]
    completions = [[70, 71], [80, 81, 82]]
    assert score_with_trainer(teacher, prompts, completions) == score_with_generator(teacher, prompts, completions)


def test_generator_scoring_rejects_a_short_logprob_row():
    teacher = FakeSession("t", answer=[], offset=0.2)
    original = teacher.generator.sample_batch

    def truncated(*args, **kwargs):
        results = original(*args, **kwargs)
        results[0].prompt_logprobs = results[0].prompt_logprobs[:-1]
        return results

    teacher.generator.sample_batch = truncated
    with pytest.raises(RuntimeError, match="expected 4"):
        score_with_generator(teacher, [[1, 2]], [[3, 4]])


def test_scoring_rejects_empty_or_unpaired_inputs():
    teacher = FakeSession("t", answer=[], offset=0.2)
    with pytest.raises(ValueError):
        score_with_generator(teacher, [[1]], [])
    with pytest.raises(ValueError):
        score_with_generator(teacher, [[1]], [[]])


def test_open_teacher_resumes_the_training_checkpoint_without_its_optimizer():
    resource = FakeResource(answer=[])
    teacher = open_teacher(resource, name="math", training_checkpoint_id="ckpt-1", lora_config={"rank": 32})
    assert teacher.create_kwargs == {
        "display_name": "teacher-math",
        "resume_from_checkpoint_id": "ckpt-1",
        "load_optimizer": False,
        "lora_config": {"rank": 32},
    }


def test_eval_session_creation_waits_out_a_full_resource(monkeypatch):
    import httpx
    from together import RateLimitError

    from post_training_cookbook.mopd import evaluate

    monkeypatch.setattr(evaluate.time, "sleep", lambda s: None)
    resource = FakeResource(answer=[])
    calls = {"n": 0}
    original = resource.create_session

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            response = httpx.Response(429, request=httpx.Request("POST", "https://x"))
            raise RateLimitError("at capacity", response=response, body=None)
        return original(**kwargs)

    resource.create_session = flaky
    session = evaluate._create_session_when_free(resource, display_name="eval-x")
    assert calls["n"] == 3 and session.create_kwargs["display_name"] == "eval-x"


def test_evaluate_test_splits_scores_every_model_on_every_test_split(tmp_path):
    from post_training_cookbook.mopd.evaluate import evaluate_test_splits

    short = [{"q": f"a{i}?", "a": f"a{i}"} for i in range(6)]
    long = [{"q": "x" * 200, "a": f"long{i}"} for i in range(6)]
    domains = [toy_domain("alpha", short, 3, 3), toy_domain("beta", long, 2, 4)]
    resource = FakeResource(answer=[ord(c) for c in "ok a1"])
    summaries = evaluate_test_splits(
        resource,
        {"base": None, "expert": "ckpt-e"},
        domains,
        CharCodec(),
        lora_config=None,
        out_dir=tmp_path,
        max_prompt_length=64,
        max_model_length=128,
        max_sample_tokens=8,
    )
    assert set(summaries) == {"base", "expert"}
    for label in summaries:
        assert set(summaries[label]) == {"alpha", "beta"}
        assert summaries[label]["alpha"]["n"] == 3
        assert (tmp_path / label / "alpha.jsonl").exists()
    # Prompts too long to render are counted, not silently dropped.
    assert summaries["base"]["beta"]["n"] == 0
    assert summaries["base"]["beta"]["skipped_prompts"] == 4
    # The base model is a fresh adapter; a checkpoint is resumed without its optimizer.
    by_name = {s.create_kwargs["display_name"]: s for s in resource.sessions}
    assert by_name["eval-base"].create_kwargs["resume_from_checkpoint_id"] is None
    assert by_name["eval-expert"].create_kwargs["resume_from_checkpoint_id"] == "ckpt-e"
    assert by_name["eval-expert"].create_kwargs["load_optimizer"] is False
    assert all(s.stopped for s in resource.sessions)
