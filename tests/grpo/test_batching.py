"""Tests for the oversample-until-kept batch dispenser."""

from post_training_cookbook.grpo.batching import oversample_until_kept


class TestOversampleUntilKept:
    """``oversample_until_kept`` repeatedly draws batches until ``target`` kept
    items accumulate, then trims past the target-th keeper. Earlier non-keepers
    stay in the prefix to feed skip-rate diagnostics."""

    @staticmethod
    def _draw_from(batches: list[list[int]]):
        """Returns a draw_batch closure that yields the supplied batches in order."""
        i = [0]

        def draw() -> list[int]:
            batch = batches[i[0]] if i[0] < len(batches) else []
            i[0] += 1
            return batch

        return draw

    def test_stops_when_target_reached_in_first_batch(self):
        items, attempts = oversample_until_kept(
            self._draw_from([[1, 1, 1]]), is_kept=lambda x: x == 1, target=2, max_attempts=3
        )
        # 2nd keeper at index 1 → cutoff=2.
        assert items == [1, 1]
        assert attempts == 1

    def test_continues_when_target_not_reached(self):
        items, attempts = oversample_until_kept(
            self._draw_from([[0, 0], [0, 1], [1, 0]]), is_kept=lambda x: x == 1, target=2, max_attempts=3
        )
        # Attempt 1: [0,0] (kept=0). Attempt 2: [0,0,0,1] (kept=1). Attempt 3: [0,0,0,1,1,0] (kept=2 → stop).
        # Trim at 2nd keeper (index 4) → first 5 items.
        assert items == [0, 0, 0, 1, 1]
        assert attempts == 3

    def test_returns_all_when_under_target_within_cap(self):
        items, attempts = oversample_until_kept(
            self._draw_from([[0, 1], [0, 0]]), is_kept=lambda x: x == 1, target=5, max_attempts=2
        )
        assert items == [0, 1, 0, 0]
        assert attempts == 2

    def test_includes_leading_non_keepers(self):
        items, attempts = oversample_until_kept(
            self._draw_from([[0, 0, 1]]), is_kept=lambda x: x == 1, target=1, max_attempts=1
        )
        assert items == [0, 0, 1]
        assert attempts == 1

    def test_drops_trailing_non_keepers_past_target(self):
        items, attempts = oversample_until_kept(
            self._draw_from([[1, 1, 0, 0]]), is_kept=lambda x: x == 1, target=2, max_attempts=1
        )
        assert items == [1, 1]
        assert attempts == 1

    def test_max_attempts_one_skips_loop_exit_check(self):
        # max_attempts=1: draw once and return whatever it produced.
        items, attempts = oversample_until_kept(
            self._draw_from([[0, 0, 0]]), is_kept=lambda x: x == 1, target=1, max_attempts=1
        )
        assert items == [0, 0, 0]
        assert attempts == 1
