"""Unit tests for the shared prefix-stable sequence splitter.

The SFT and GRPO paths both delegate their turn-splitting to this module; these
tests pin the splitting decision directly, independent of any codec.
"""

from __future__ import annotations

from post_training_cookbook.sequence_split import Completion, Context, Turn, split_turns


def _turn(prompt: list[int], completion: list[int], logprobs: list[float] | None = None) -> Turn:
    return Turn(Context(prompt), Completion(completion, logprobs))


def test_empty_turns_yields_no_sequences():
    assert split_turns([]) == []


def test_single_turn_is_one_sequence():
    sequences = split_turns([_turn([1, 2], [3, 4])])
    assert len(sequences) == 1
    assert sequences[0].tokens == [1, 2, 3, 4]
    assert sequences[0].loss_mask == [0, 0, 1, 1]


def test_prefix_stable_turns_stay_one_sequence():
    # Turn 2's prompt extends turn 1's emitted tokens (prompt+completion).
    sequences = split_turns([_turn([1], [2]), _turn([1, 2, 3], [4])])
    assert len(sequences) == 1
    assert sequences[0].tokens == [1, 2, 3, 4]
    # completion tokens of both turns are masked in; the newly exposed [3] is context.
    assert sequences[0].loss_mask == [0, 1, 0, 1]


def test_prefix_break_starts_new_sequence():
    # Turn 2's prompt does NOT extend [1,2] (history was rewritten), so it splits.
    sequences = split_turns([_turn([1], [2]), _turn([9, 9], [3])])
    assert len(sequences) == 2
    assert (sequences[0].tokens, sequences[0].loss_mask) == ([1, 2], [0, 1])
    assert (sequences[1].tokens, sequences[1].loss_mask) == ([9, 9, 3], [0, 0, 1])


def test_logprobs_ride_through_a_split():
    # Logprobs must stay aligned with their completion tokens across a split.
    sequences = split_turns([_turn([1], [2], [0.1]), _turn([8], [3], [0.2])])
    assert len(sequences) == 2
    assert sequences[0].logprobs == [0.0, 0.1]
    assert sequences[1].logprobs == [0.0, 0.2]


def test_off_policy_turns_have_no_logprobs():
    sequences = split_turns([_turn([1], [2])])
    assert sequences[0].logprobs is None


def test_no_empty_context_run_when_prompt_equals_emitted():
    # Turn 2's prompt is exactly turn 1's emitted tokens — no new context to add.
    sequences = split_turns([_turn([1], [2]), _turn([1, 2], [3])])
    assert len(sequences) == 1
    assert sequences[0].tokens == [1, 2, 3]
    assert sequences[0].loss_mask == [0, 1, 1]
