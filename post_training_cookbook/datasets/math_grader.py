"""Math-answer grading for free-form LaTeX answers."""

from __future__ import annotations

import re

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

from post_training_cookbook.datasets.rewards import extract_boxed

# ``math_verify`` only treats a string as LaTeX when it sits inside a math
# environment, so answers are parsed wrapped in ``$...$``.
_EXTRACTION = [LatexExtractionConfig(), ExprExtractionConfig()]

# ``math_verify`` enforces its parse/compare timeout with ``signal.alarm()``,
# which raises if called off the main thread. GRPO scores rollouts from worker
# threads (e.g. one arm per thread in a sweep), so disable the signal timeout.
_NO_SIGNAL_TIMEOUT = None


def _canonical_string(s: str) -> str:
    r"""Normalize a non-math text answer for the string-equality fallback."""
    s = re.sub(r"\\(?:text|mathrm)\s*\{([^{}]*)\}", r"\1", s.strip())
    s = s.replace("$", "").replace("\\left", "").replace("\\right", "")
    return s.replace(" ", "").lower()


def math_equal(prediction: str, target: str) -> bool:
    """Whether two free-form math answers are mathematically equal."""
    if prediction is None or not target:
        return False
    gold = parse(f"${target}$", extraction_config=_EXTRACTION, parsing_timeout=_NO_SIGNAL_TIMEOUT)
    pred = parse(f"${prediction}$", extraction_config=_EXTRACTION, parsing_timeout=_NO_SIGNAL_TIMEOUT)
    if verify(gold, pred, timeout_seconds=_NO_SIGNAL_TIMEOUT):
        return True
    return _canonical_string(prediction) == _canonical_string(target)


def boxed_math_reward(prediction: str, target: str) -> float:
    r"""1.0 when the model's ``\boxed{}`` answer matches the target, else 0.0."""
    pred = extract_boxed(prediction)
    if pred is None or not target:
        return 0.0
    return 1.0 if math_equal(pred, target) else 0.0
