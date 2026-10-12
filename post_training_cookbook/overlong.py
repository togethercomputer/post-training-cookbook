"""DAPO soft-overlong reward shaping (§3.2)."""

from __future__ import annotations


def compute_overlong_penalty(
    response_length: int,
    max_response_length: int,
    overlong_buffer_len: int,
    penalty_factor: float = 1.0,
) -> float:
    if overlong_buffer_len <= 0:
        return 0.0
    expected_len = max_response_length - overlong_buffer_len
    exceed_len = response_length - expected_len
    return min(-exceed_len / overlong_buffer_len * penalty_factor, 0.0)


def is_overlong(response_length: int, max_response_length: int) -> bool:
    return response_length >= max_response_length
