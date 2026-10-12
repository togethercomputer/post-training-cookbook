"""Tests for the DAPO soft-overlong reward shaping."""

import pytest

from post_training_cookbook.overlong import compute_overlong_penalty


def test_penalty_scales_in_buffer():
    """Penalty is zero outside the buffer, grows as length approaches max,
    and reaches ``-penalty_factor`` exactly at max."""
    no_penalty = compute_overlong_penalty(50, 100, overlong_buffer_len=20)
    soft = compute_overlong_penalty(85, 100, overlong_buffer_len=20)
    harder = compute_overlong_penalty(95, 100, overlong_buffer_len=20)
    full = compute_overlong_penalty(100, 100, overlong_buffer_len=20, penalty_factor=1.0)

    assert no_penalty == 0.0
    assert soft < 0.0
    assert harder < soft  # more negative = harsher penalty
    assert full == pytest.approx(-1.0)
