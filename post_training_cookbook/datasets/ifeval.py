"""Instruction-following rewards for AllenAI's RLVR-IFEval dataset.

Each RLVR-IFEval row is one user prompt carrying one verifiable constraint (for example "answer in
all lowercase letters" or "include exactly 3 bullet points"), and a ``ground_truth`` JSON object
naming the checker (``func_name``) and its arguments. :func:`ifeval_reward` scores a response 1.0
when it satisfies the constraint and 0.0 otherwise.

The checker functions are vendored from AllenAI's open-instruct
(``open_instruct/if_functions.py``, https://github.com/allenai/open-instruct), licensed under the
Apache License 2.0, so the rewards match the ones RLVR-IFEval was released with. Changes: type
hints, docstrings trimmed, and ``validate_response_language`` dropped (no RLVR-IFEval row uses it,
and it needs ``langdetect``). The response is normalised the way open-instruct's
``IFEvalVerifierOld`` does it: anything up to a closing ``</think>`` is dropped, ``<answer>`` tags
are removed, and surrounding whitespace is stripped.

``validate_repeat_prompt`` is excluded from :data:`IFEVAL_FUNC_NAMES`: its instruction says the
sentence asking for the repeat is *not* part of the request to repeat, but every
``original_prompt`` in the dataset is the whole prompt, so a response that follows the instruction
fails the check.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

# --- Checkers vendored from open-instruct (Apache-2.0) -------------------------------------------


def verify_keywords(text: str, keyword_list: list[str]) -> bool:
    """Include keywords {keyword1}, {keyword2} in your response."""
    response_lower = text.lower()
    return all(keyword.lower() in response_lower for keyword in keyword_list)


def verify_keyword_frequency(text: str, word: str, N: int) -> bool:
    """In your response, the word {word} should appear {N} times."""
    words = re.findall(r"\b\w+\b", text.lower())
    return sum(1 for w in words if w == word.lower()) == N


def validate_forbidden_words(text: str, forbidden_words: list[str]) -> bool:
    """Do not include keywords {forbidden words} in the response."""
    text_lower = text.lower()
    return not [word for word in forbidden_words if word.lower() in text_lower]


def verify_letter_frequency(text: str, letter: str, N: int) -> bool:
    """In your response, the letter {letter} should appear {N} times."""
    if len(letter) != 1:
        raise ValueError("Letter parameter must be a single character")
    return text.count(letter) == N


def verify_paragraph_count(text: str, N: int) -> bool:
    """Your response should contain {N} paragraphs, separated by the markdown divider * * *."""
    text = "\n".join(line.strip() for line in text.splitlines()).strip()
    paragraphs = text.split("* * *")
    valid_paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if len(valid_paragraphs) != len(paragraphs):
        return False
    return len(paragraphs) == N


def validate_word_constraint(text: str, N: int, quantifier: str) -> bool:
    """Answer with at least / around / at most {N} words."""
    actual_count = len(text.strip().split())
    tolerance = max(round(N * 0.1), 1)
    if quantifier == "at least":
        return actual_count >= N
    if quantifier == "at most":
        return actual_count <= N
    if quantifier == "around":
        return abs(actual_count - N) <= tolerance
    return False


def verify_sentence_constraint(text: str, N: int, quantifier: str) -> bool:
    """Answer with at least / around / at most {N} sentences."""
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s", text)
    actual_count = len(sentences)
    if quantifier == "at least":
        return actual_count >= N
    if quantifier == "around":
        return abs(actual_count - N) <= 1
    if quantifier == "at most":
        return actual_count <= N
    return False


def validate_paragraphs(text: str, N: int, first_word: str, i: int) -> bool:
    """There should be {N} paragraphs separated by two line breaks; the {i}-th starts with {first word}."""
    paragraphs = text.split("\n\n")
    if len(paragraphs) != N:
        return False
    return bool(paragraphs[i - 1].strip().startswith(first_word))


def verify_postscript(text: str, postscript_marker: str) -> bool:
    """At the end of your response, please explicitly add a postscript starting with {postscript marker}."""
    if postscript_marker in text:
        remaining_text = text[text.find(postscript_marker) :].strip()
        return len(remaining_text) > len(postscript_marker)
    return False


def validate_placeholders(text: str, N: int) -> bool:
    """The response must contain at least {N} placeholders in square brackets, such as [address]."""
    return len(re.findall(r"\[(.*?)\]", text)) >= N


def verify_bullet_points(text: str, N: int) -> bool:
    """Your answer must contain exactly {N} markdown bullet points."""
    bullet_points = [line.strip() for line in text.split("\n") if line.strip().startswith(("*", "-"))]
    return len(bullet_points) == N


def validate_title(text: str) -> bool:
    """Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>."""
    return len(re.findall(r"<<(.*?)>>", text)) > 0


def validate_choice(text: str, options: list[str]) -> bool:
    """Answer with one of the following options: {options}."""
    return any(option in text for option in options)


def validate_highlighted_sections(text: str, N: int) -> bool:
    """Highlight at least {N} sections in your answer with markdown, i.e. *highlighted section*."""
    return len(re.findall(r"\*(.*?)\*", text)) >= N


def validate_sections(text: str, N: int, section_splitter: str) -> bool:
    """Your response must have {N} sections. Mark the beginning of each section with {section splitter} X."""
    sections = text.split(section_splitter)
    if sections[0] == "":
        sections.pop(0)
    return len(sections) == N


def validate_json_format(text: str) -> bool:
    """Entire output should be wrapped in JSON format."""
    try:
        json.loads(text)
    except (ValueError, RecursionError):
        return False
    return True


def validate_repeat_prompt(text: str, original_prompt: str) -> bool:
    """First, repeat the request without change, then give your answer."""
    return bool(text.startswith(original_prompt))


def validate_two_responses(text: str) -> bool:
    """Give two different responses separated by 6 asterisk symbols: ******."""
    if text.count("******") == 1:
        first_response, second_response = (part.strip() for part in text.split("******"))
        return first_response != second_response
    return False


def validate_uppercase(text: str) -> bool:
    """Your entire response should be in English, capital letters only."""
    return text == text.upper()


def validate_lowercase(text: str) -> bool:
    """Your entire response should be in English, and in all lowercase letters."""
    return text == text.lower()


def validate_frequency_capital_words(text: str, N: int, quantifier: str) -> bool:
    """Words with all capital letters should appear at least / around / at most {N} times."""
    words = re.findall(r"\b[A-Z]+\b", text)
    if quantifier == "at least":
        return len(words) >= N
    if quantifier == "around":
        return abs(len(words) - N) <= max(round(N * 0.1), 1)
    if quantifier == "at most":
        return len(words) <= N
    return False


def validate_end(text: str, end_phrase: str) -> bool:
    """Finish your response with this exact phrase {end phrase}."""
    return bool(text.endswith(end_phrase))


def validate_quotation(text: str) -> bool:
    """Wrap your entire response with double quotation marks."""
    return bool(text.startswith('"') and text.endswith('"'))


def validate_no_commas(text: str) -> bool:
    """In your entire response, refrain from the use of any commas."""
    return "," not in text


IF_FUNCTIONS_MAP: dict[str, Callable[..., bool]] = {
    "verify_keywords": verify_keywords,
    "verify_keyword_frequency": verify_keyword_frequency,
    "validate_forbidden_words": validate_forbidden_words,
    "verify_letter_frequency": verify_letter_frequency,
    "verify_paragraph_count": verify_paragraph_count,
    "validate_word_constraint": validate_word_constraint,
    "verify_sentence_constraint": verify_sentence_constraint,
    "validate_paragraphs": validate_paragraphs,
    "verify_postscript": verify_postscript,
    "validate_placeholders": validate_placeholders,
    "verify_bullet_points": verify_bullet_points,
    "validate_title": validate_title,
    "validate_choice": validate_choice,
    "validate_highlighted_sections": validate_highlighted_sections,
    "validate_sections": validate_sections,
    "validate_json_format": validate_json_format,
    "validate_repeat_prompt": validate_repeat_prompt,
    "validate_two_responses": validate_two_responses,
    "validate_uppercase": validate_uppercase,
    "validate_lowercase": validate_lowercase,
    "validate_frequency_capital_words": validate_frequency_capital_words,
    "validate_end": validate_end,
    "validate_quotation": validate_quotation,
    "validate_no_commas": validate_no_commas,
}

# --- Cookbook glue --------------------------------------------------------------------------------

IFEVAL_FUNC_NAMES: frozenset[str] = frozenset(IF_FUNCTIONS_MAP) - {"validate_repeat_prompt"}
"""Constraints the cookbook trains and evaluates on (every RLVR-IFEval type but the repeat-prompt one)."""


def parse_ground_truth(ground_truth: str | dict[str, Any]) -> dict[str, Any]:
    """The constraint as ``{"func_name": ..., <arg>: <value>}``, with unset (``None``) arguments dropped."""
    raw = json.loads(ground_truth) if isinstance(ground_truth, str) else dict(ground_truth)
    if "func_name" not in raw:
        raise ValueError(f"ground truth has no func_name: {raw!r}")
    return {key: value for key, value in raw.items() if value is not None}


def normalize_response(prediction: str) -> str:
    """Strip reasoning and answer tags the way open-instruct does before checking a constraint."""
    answer = prediction.replace("<|assistant|>", "").strip()
    answer = answer.split("</think>")[-1]
    return answer.replace("<answer>", "").replace("</answer>", "").strip()


def ifeval_reward(prediction: str, target: str | dict[str, Any]) -> float:
    """1.0 when ``prediction`` satisfies the RLVR-IFEval constraint in ``target``, else 0.0.

    ``target`` is the row's ``ground_truth`` (a JSON string or the parsed dict). An empty response
    scores 0.0 even for constraints an empty string would trivially pass, such as all-lowercase.
    """
    constraint = parse_ground_truth(target)
    func_name = constraint.pop("func_name")
    answer = normalize_response(prediction)
    if not answer:
        return 0.0
    return 1.0 if IF_FUNCTIONS_MAP[func_name](answer, **constraint) else 0.0
