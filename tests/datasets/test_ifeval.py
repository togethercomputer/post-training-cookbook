import json

import pytest

from post_training_cookbook.datasets import ifeval
from post_training_cookbook.datasets.ifeval import (
    IF_FUNCTIONS_MAP,
    IFEVAL_FUNC_NAMES,
    ifeval_reward,
    normalize_response,
    parse_ground_truth,
)
from post_training_cookbook.datasets.registry import DATASETS


def _gt(func_name, **kwargs):
    # RLVR-IFEval rows carry every argument key, unset ones as null.
    keys = (
        "N quantifier end_phrase keyword_list word forbidden_words letter i first_word postscript_marker "
        "options section_splitter original_prompt"
    ).split()
    return json.dumps({"func_name": func_name, **dict.fromkeys(keys), **kwargs})


@pytest.mark.parametrize(
    ("func_name", "kwargs", "good", "bad"),
    [
        ("validate_lowercase", {}, "all lower here", "Not lower"),
        ("validate_uppercase", {}, "ALL UPPER", "Mixed Case"),
        ("validate_no_commas", {}, "no commas at all", "one, two"),
        ("verify_keywords", {"keyword_list": ["apple", "Pear"]}, "An APPLE and a pear", "only apple"),
        ("verify_keyword_frequency", {"word": "cat", "N": 2}, "cat dog Cat", "cat dog"),
        ("validate_forbidden_words", {"forbidden_words": ["bad"]}, "all good", "a Bad idea"),
        ("verify_letter_frequency", {"letter": "z", "N": 2}, "zz top", "z top"),
        ("verify_paragraph_count", {"N": 2}, "one\n* * *\ntwo", "one\n* * *\n* * *\ntwo"),
        ("validate_word_constraint", {"N": 3, "quantifier": "at most"}, "one two three", "one two three four"),
        ("validate_word_constraint", {"N": 10, "quantifier": "around"}, " ".join(["w"] * 11), "w w"),
        ("verify_sentence_constraint", {"N": 2, "quantifier": "at least"}, "One. Two.", "Only one"),
        ("validate_paragraphs", {"N": 2, "first_word": "Second", "i": 2}, "First\n\nSecond one", "First\n\nThird"),
        ("verify_postscript", {"postscript_marker": "P.S."}, "Body\nP.S. thanks", "Body\nP.S."),
        ("validate_placeholders", {"N": 2}, "[a] and [b]", "[a] only"),
        ("verify_bullet_points", {"N": 2}, "* one\n* two", "* one"),
        ("validate_title", {}, "<<Title>>\nbody", "no title"),
        ("validate_choice", {"options": ["yes", "no"]}, "My answer is yes", "maybe"),
        ("validate_highlighted_sections", {"N": 2}, "*a* and *b*", "*a* only"),
        ("validate_sections", {"N": 2, "section_splitter": "SECTION"}, "SECTION 1 a SECTION 2 b", "SECTION 1 a"),
        ("validate_json_format", {}, '{"a": 1}', "not json"),
        ("validate_two_responses", {}, "first ****** second", "same ****** same"),
        ("validate_end", {"end_phrase": "Any questions?"}, "Text. Any questions?", "Any questions? More"),
        ("validate_quotation", {}, '"wrapped"', "not wrapped"),
        ("validate_frequency_capital_words", {"N": 2, "quantifier": "at least"}, "NASA and ESA", "only NASA"),
    ],
)
def test_checkers(func_name, kwargs, good, bad):
    assert ifeval_reward(good, _gt(func_name, **kwargs)) == 1.0
    assert ifeval_reward(bad, _gt(func_name, **kwargs)) == 0.0


def test_every_trained_constraint_has_a_checker_and_repeat_prompt_is_excluded():
    assert "validate_repeat_prompt" in IF_FUNCTIONS_MAP
    assert "validate_repeat_prompt" not in IFEVAL_FUNC_NAMES
    assert len(IFEVAL_FUNC_NAMES) == 23
    assert IFEVAL_FUNC_NAMES <= set(IF_FUNCTIONS_MAP)


def test_normalize_drops_reasoning_answer_tags_and_whitespace():
    assert normalize_response("<think>plan</think>\n  <answer>Done</answer>  ") == "Done"


def test_empty_response_scores_zero_even_when_vacuously_true():
    assert ifeval_reward("   ", _gt("validate_lowercase")) == 0.0


def test_reward_is_whitespace_insensitive_at_the_ends():
    # The response is stripped before checking, as open-instruct does.
    assert ifeval_reward("ends here. Any questions?\n", _gt("validate_end", end_phrase="Any questions?")) == 1.0


def test_parse_ground_truth_drops_nulls_and_accepts_dicts():
    parsed = parse_ground_truth(_gt("validate_title"))
    assert parsed == {"func_name": "validate_title"}
    assert parse_ground_truth({"func_name": "x", "N": None, "word": "w"}) == {"func_name": "x", "word": "w"}
    with pytest.raises(ValueError):
        parse_ground_truth({"N": 3})


def test_letter_frequency_rejects_multi_character_letters():
    with pytest.raises(ValueError):
        ifeval.verify_letter_frequency("abc", "ab", 1)


def test_registry_entry_formats_the_single_user_turn_and_filters_repeat_prompt():
    cfg = DATASETS["ifeval"]
    row = {
        "messages": [{"role": "user", "content": "Write a haiku. Use no commas."}],
        "ground_truth": _gt("validate_no_commas"),
    }
    assert cfg.format_question(row) == "Write a haiku. Use no commas."
    assert cfg.extract_target(row) == {"func_name": "validate_no_commas"}
    assert cfg.compute_reward("a haiku with no commas", cfg.extract_target(row)) == 1.0
    assert cfg.filter_fn(row)
    assert not cfg.filter_fn({**row, "ground_truth": _gt("validate_repeat_prompt", original_prompt="x")})
