"""Reward functions for dataset-specific evaluation and scoring."""

from __future__ import annotations

import re


def extract_boxed(text: str) -> str | None:
    r"""Extract the last \boxed{...} value from text."""
    boxed_strs: list[str] = []
    stack: list[int] = []
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
        elif ch == "}":
            if not stack:
                continue
            open_idx = stack.pop()
            if text[:open_idx].endswith("\\boxed"):
                boxed_strs.append(text[open_idx + 1 : i])
    if boxed_strs:
        return boxed_strs[-1]
    match = re.search(r"\\boxed\s+([a-zA-Z0-9]+)", text)
    if match:
        return match.group(1)
    return None


def normalize_numeric(value: str) -> str:
    value = value.strip().replace(",", "").replace(" ", "")
    try:
        f = float(value)
        if f == int(f):
            return str(int(f))
        return str(f)
    except (ValueError, OverflowError):
        return value


def _safe_eval_expr(expr: str) -> float | None:
    expr = expr.replace("\u00d7", "*").replace("\u00f7", "/").replace("\u2212", "-")
    tokens: list[str] = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch in " \t\n":
            i += 1
            continue
        if ch in "+-*/()":
            tokens.append(ch)
            i += 1
        elif ch.isdigit() or ch == ".":
            j = i
            while j < len(expr) and (expr[j].isdigit() or expr[j] == "."):
                j += 1
            tokens.append(expr[i:j])
            i = j
        else:
            return None
    pos = 0

    def _parse_expr() -> float | None:
        nonlocal pos
        sign = 1.0
        if pos < len(tokens) and tokens[pos] == "-":
            sign = -1.0
            pos += 1
        elif pos < len(tokens) and tokens[pos] == "+":
            pos += 1
        left = _parse_term()
        if left is None:
            return None
        left *= sign
        while pos < len(tokens) and tokens[pos] in ("+", "-"):
            op = tokens[pos]
            pos += 1
            right = _parse_term()
            if right is None:
                return None
            left = left + right if op == "+" else left - right
        return left

    def _parse_term() -> float | None:
        nonlocal pos
        left = _parse_atom()
        if left is None:
            return None
        while pos < len(tokens) and tokens[pos] in ("*", "/"):
            op = tokens[pos]
            pos += 1
            right = _parse_atom()
            if right is None:
                return None
            if op == "/":
                if right == 0:
                    return None
                left = left / right
            else:
                left = left * right
        return left

    def _parse_atom() -> float | None:
        nonlocal pos
        if pos >= len(tokens):
            return None
        if tokens[pos] == "(":
            pos += 1
            val = _parse_expr()
            if val is None or pos >= len(tokens) or tokens[pos] != ")":
                return None
            pos += 1
            return val
        try:
            val = float(tokens[pos])
            pos += 1
            return val
        except ValueError:
            return None

    try:
        result = _parse_expr()
        if result is None or pos != len(tokens):
            return None
        return result
    except (RecursionError, IndexError):
        return None


def _extract_numbers(expr: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", expr)]


_GAME24_ALLOWED_PATTERN = re.compile(r"^[\d+\-*/().\s]+$")
_GAME24_USE_ALL_NUMBERS_REWARD = 0.20
_GAME24_CLOSENESS_REWARD = 0.40
_GAME24_EXACT_MATCH_REWARD = 0.40
_GAME24_MAX_REWARD = 1.0


def _has_arithmetic_operator(expr: str) -> bool:
    return any(op in expr for op in ("+", "-", "*", "/", "\\times", "\\cdot", "\\div", "\\frac"))


def _extract_braced(text: str, start: int) -> tuple[str | None, int]:
    if start >= len(text) or text[start] != "{":
        return None, start
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : idx], idx + 1
    return None, start


def _normalize_latex_math(expr: str) -> str:
    expr = expr.strip()
    normalized: list[str] = []
    i = 0

    while i < len(expr):
        if expr.startswith("\\frac", i):
            numerator, next_idx = _extract_braced(expr, i + len("\\frac"))
            if numerator is None:
                normalized.append(expr[i:])
                break
            denominator, final_idx = _extract_braced(expr, next_idx)
            if denominator is None:
                normalized.append(expr[i:])
                break
            normalized.append(f"(({_normalize_latex_math(numerator)})/({_normalize_latex_math(denominator)}))")
            i = final_idx
            continue

        if expr.startswith("\\left", i):
            i += len("\\left")
            continue
        if expr.startswith("\\right", i):
            i += len("\\right")
            continue
        if expr.startswith("\\times", i):
            normalized.append("*")
            i += len("\\times")
            continue
        if expr.startswith("\\cdot", i):
            normalized.append("*")
            i += len("\\cdot")
            continue
        if expr.startswith("\\div", i):
            normalized.append("/")
            i += len("\\div")
            continue
        if expr.startswith("\\,", i) or expr.startswith("\\!", i):
            i += 2
            continue

        ch = expr[i]
        if ch == "{":
            normalized.append("(")
        elif ch == "}":
            normalized.append(")")
        else:
            normalized.append(ch)
        i += 1

    return "".join(normalized)


def _game24_eval_details(expr: str, target: list[int]) -> tuple[bool, float | None, float]:
    expr = _normalize_latex_math(expr)
    if not expr or not _GAME24_ALLOWED_PATTERN.fullmatch(expr):
        return False, None, float("inf")

    result = _safe_eval_expr(expr)
    if result is None:
        return False, None, float("inf")

    uses_numbers_once = sorted(_extract_numbers(expr)) == sorted(target)
    return uses_numbers_once, result, abs(result - 24.0)


def boxed_numeric_reward(prediction: str, target: str) -> float:
    pred_value = extract_boxed(prediction)
    if pred_value is None or not target:
        return 0.0
    if normalize_numeric(pred_value) == normalize_numeric(target.strip()):
        return 1.0
    if pred_value.strip() == target.strip():
        return 1.0
    return 0.0


def extract_hash_answer(text: str) -> str | None:
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1)
    return None


def extract_gsm8k_answer(text: str) -> str | None:
    answer = extract_hash_answer(text)
    if answer is None:
        answer = extract_boxed(text)
    if answer is None:
        return None
    return normalize_numeric(answer)


def gsm8k_reward(prediction: str, target: str) -> float:
    pred_value = extract_gsm8k_answer(prediction)
    if pred_value is None or not target:
        return 0.0
    return 1.0 if pred_value == normalize_numeric(target) else 0.0


def game24_reward(prediction: str, target: list[int]) -> float:
    expr = extract_boxed(prediction)
    if expr is None:
        return 0.0

    expr = expr.strip()
    uses_numbers_once, result, distance = _game24_eval_details(expr, target)
    if result is None or not uses_numbers_once:
        return 0.0

    reward = _GAME24_USE_ALL_NUMBERS_REWARD
    closeness = 0.5 ** (min(distance, 24.0) / 5.0)
    reward += _GAME24_CLOSENESS_REWARD * closeness
    if distance <= 1e-6:
        reward += _GAME24_EXACT_MATCH_REWARD
    return min(reward, _GAME24_MAX_REWARD)


def game24_accuracy(prediction: str, target: list[int]) -> float:
    expr = extract_boxed(prediction)
    if expr is None:
        return 0.0

    uses_numbers_once, result, distance = _game24_eval_details(expr, target)
    if result is None or not uses_numbers_once or distance > 1e-6:
        return 0.0
    return 1.0
