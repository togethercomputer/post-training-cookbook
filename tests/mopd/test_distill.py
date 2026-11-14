import numpy as np
import pytest

from post_training_cookbook.mopd.distill import (
    Completion,
    balance_weights,
    build_distillation_sample,
    token_advantages,
)


def _completion(domain="math", n=3, student=None, teacher=None, prompt=(1, 2)):
    return Completion(
        domain=domain,
        prompt_tokens=list(prompt),
        completion_tokens=list(range(10, 10 + n)),
        student_logprobs=list(student if student is not None else [-1.0] * n),
        teacher_logprobs=list(teacher if teacher is not None else [-0.5] * n),
    )


def test_advantage_is_teacher_minus_student_scaled_by_kl_coef():
    c = _completion(student=[-1.0, -2.0, -0.1], teacher=[-0.5, -3.0, -0.1])
    assert token_advantages(c) == pytest.approx([0.5, -1.0, 0.0])
    assert token_advantages(c, kl_coef=2.0) == pytest.approx([1.0, -2.0, 0.0])
    np.testing.assert_allclose(c.reverse_kl, [-0.5, 1.0, 0.0])


def test_completion_rejects_mismatched_lengths_and_empty_inputs():
    with pytest.raises(ValueError):
        Completion("d", [1], [5, 6], [-1.0], [-1.0, -1.0])
    with pytest.raises(ValueError):
        Completion("d", [1], [], [], [])
    with pytest.raises(ValueError):
        Completion("d", [], [5], [-1.0], [-1.0])


def test_token_balance_is_uniform():
    batch = [_completion("math", n=8), _completion("ifeval", n=2)]
    assert balance_weights(batch, "token") == [1.0, 1.0]


def test_domain_balance_equalizes_each_domains_total_weight_and_keeps_the_batch_total():
    batch = [_completion("math", n=6), _completion("math", n=2), _completion("ifeval", n=2)]
    weights = balance_weights(batch, "domain")
    per_domain = {"math": 0.0, "ifeval": 0.0}
    for c, w in zip(batch, weights):
        per_domain[c.domain] += w * len(c.completion_tokens)
    assert per_domain["math"] == pytest.approx(per_domain["ifeval"])
    assert sum(per_domain.values()) == pytest.approx(10.0)  # total tokens unchanged
    assert weights[0] == weights[1]  # same domain, same multiplier


def test_unknown_balance_rejected():
    with pytest.raises(ValueError):
        balance_weights([_completion()], "sequence")


def test_sample_aligns_advantages_and_logprobs_with_the_completion_slots():
    c = _completion(prompt=(1, 2, 3), student=[-1.0, -2.0], teacher=[-0.5, -2.5], n=2)
    sample = build_distillation_sample(c, kl_coef=1.0, weight=2.0)
    inputs = sample["loss_fn_inputs"]
    # 5 tokens -> 4 slots; slot i predicts token i + 1, so the completion is predicted by slots 2 and 3.
    assert sample["model_input"]["chunks"][0]["encoded_text"]["tokens"] == [1, 2, 3, 10]
    assert inputs["target_tokens"]["data"] == [2, 3, 10, 11]
    assert inputs["weights"]["data"] == [0.0, 0.0, 1.0, 1.0]
    assert inputs["logprobs"]["data"] == [0.0, 0.0, -1.0, -2.0]
    assert inputs["advantages"]["data"] == pytest.approx([0.0, 0.0, 1.0, -1.0])
