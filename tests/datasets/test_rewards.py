"""Tests for reward functions — the core scoring abstractions used in GRPO training."""

from post_training_cookbook.datasets.rewards import (
    boxed_numeric_reward,
    game24_accuracy,
    game24_reward,
    gsm8k_reward,
)


class TestBoxedNumericReward:
    def test_correct(self):
        assert boxed_numeric_reward("\\boxed{42}", "42") == 1.0

    def test_correct_with_comma(self):
        assert boxed_numeric_reward("\\boxed{1000}", "1,000") == 1.0

    def test_wrong(self):
        assert boxed_numeric_reward("\\boxed{43}", "42") == 0.0

    def test_no_boxed(self):
        assert boxed_numeric_reward("the answer is 42", "42") == 0.0

    def test_empty_target(self):
        assert boxed_numeric_reward("\\boxed{42}", "") == 0.0


class TestGsm8kReward:
    def test_correct_hash_format(self):
        assert gsm8k_reward("#### 42", "42") == 1.0

    def test_correct_boxed_format(self):
        assert gsm8k_reward("\\boxed{42}", "42") == 1.0

    def test_wrong_answer(self):
        assert gsm8k_reward("#### 43", "42") == 0.0

    def test_no_answer(self):
        assert gsm8k_reward("I don't know", "42") == 0.0


class TestGame24Reward:
    def test_correct_expression(self):
        reward = game24_reward("\\boxed{1*2*3*4}", [1, 2, 3, 4])
        assert reward == 1.0

    def test_wrong_result(self):
        reward = game24_reward("\\boxed{1+2+3+4}", [1, 2, 3, 4])
        assert reward < 1.0  # 10 != 24, but uses all numbers

    def test_missing_number(self):
        reward = game24_reward("\\boxed{8*3}", [1, 2, 3, 4])
        assert reward == 0.0  # doesn't use all numbers

    def test_no_boxed(self):
        assert game24_reward("the answer is 1*2*3*4", [1, 2, 3, 4]) == 0.0


class TestGame24Accuracy:
    def test_correct(self):
        assert game24_accuracy("\\boxed{1*2*3*4}", [1, 2, 3, 4]) == 1.0

    def test_wrong(self):
        assert game24_accuracy("\\boxed{1+2+3+4}", [1, 2, 3, 4]) == 0.0
