"""Checkpointable input positions for synchronous batches and async task leases."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Iterable
from typing import Any

_STATE_VERSION = 1


def fingerprint_data(rows: Iterable[Any]) -> str:
    """Hash the ordered rows so a resume can tell the same data from a reordered or edited copy."""
    digest = hashlib.sha256()
    try:
        for row in rows:
            digest.update(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
            digest.update(b"\n")
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpointed training inputs must be JSON-serializable to verify data order") from exc
    return digest.hexdigest()


class DataProgress:
    """Track drawn inputs and async tasks not yet consumed by a completed step.

    Indices count absolute draws, so repeated visits to a dataset row remain
    distinct. Synchronous collectors draw whole batches; async workers lease one
    task at a time and return it to the front of the line if cancelled.
    """

    def __init__(self, *, kind: str, size: int, batch_size: int, fingerprint: str | None = None) -> None:
        self._source = {"kind": kind, "size": size, "batch_size": batch_size, "fingerprint": fingerprint}
        self._next_index = 0
        self._pending: dict[int, None] = {}
        self._replay: deque[int] = deque()

    @classmethod
    def restore(
        cls, state: dict[str, Any], *, kind: str, size: int, batch_size: int, fingerprint: str | None = None
    ) -> DataProgress:
        """Resume from a :meth:`snapshot`; saved pending tasks are issued before new ones."""
        progress = cls(kind=kind, size=size, batch_size=batch_size, fingerprint=fingerprint)
        if state["version"] != _STATE_VERSION or state["source"] != progress._source:
            raise ValueError(
                f"Checkpoint data state (version {state.get('version')!r}, source {state.get('source')}) "
                f"does not match this run's (version {_STATE_VERSION}, source {progress._source}); "
                "dataset contents and order, batch size and collector mode must stay the same"
            )
        progress._next_index = state["next_index"]
        progress._pending = dict.fromkeys(state["pending"])
        progress._replay = deque(state["pending"])
        return progress

    def draw_batch(self) -> list[int]:
        """Advance by one full drawn batch, including rows subsequently filtered out."""
        start = self._next_index
        self._next_index += self._source["batch_size"]
        return [index % self._source["size"] for index in range(start, self._next_index)]

    def claim(self) -> int:
        """Lease an absolute task index, replaying restored and cancelled tasks first."""
        if self._replay:
            return self._replay.popleft()
        index = self._next_index
        self._next_index += 1
        self._pending[index] = None
        return index

    def consume(self, index: int) -> None:
        """Record a task admitted or deliberately dropped by the current client step."""
        del self._pending[index]

    def retry(self, index: int) -> None:
        """Reissue a cancelled worker's task without losing its data position."""
        self._replay.append(index)

    def snapshot(self) -> dict[str, Any]:
        """Return independent JSON state for one training-checkpoint record."""
        return {
            "version": _STATE_VERSION,
            "source": dict(self._source),
            "next_index": self._next_index,
            "pending": list(self._pending),
        }
