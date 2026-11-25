r"""Math-answer grading for free-form LaTeX answers.

``math_verify`` turns both answers into sympy expressions, and sympy evaluates integer powers
eagerly, so a sampled answer such as ``\boxed{10^{10^{7}}}`` starts a big-integer computation. That
runs as C code holding the GIL: comparing it to ``5`` froze every other thread of the process for
about 10 s, and larger powers run far longer. ``math_verify``'s own timeout is
``signal.alarm()``-based, which can't fire off the main thread (where GRPO scores rollouts) and
can't interrupt C code anyway. In a run training several sessions from one process, one such
answer can stall all of them, and a resource that receives no operation stops for inactivity.

:func:`math_equal` therefore runs ``math_verify`` in a small pool of worker processes and waits at
most :data:`VERIFY_TIMEOUT_S`. A worker that overruns is killed (the pool is replaced) and the
answer falls back to the string comparison.
"""

from __future__ import annotations

import multiprocessing
import re
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from concurrent.futures.process import BrokenProcessPool

from post_training_cookbook.datasets.rewards import extract_boxed

VERIFY_TIMEOUT_S = 10.0
"""Longest a single ``math_verify`` comparison may run before it counts as not equal-by-math."""

_WORKERS = 4
_pool: ProcessPoolExecutor | None = None
_pool_lock = threading.Lock()


def _canonical_string(s: str) -> str:
    r"""Normalize a non-math text answer for the string-equality fallback."""
    s = re.sub(r"\\(?:text|mathrm)\s*\{([^{}]*)\}", r"\1", s.strip())
    s = s.replace("$", "").replace("\\left", "").replace("\\right", "")
    return s.replace(" ", "").lower()


def _verify(prediction: str, target: str) -> bool:
    """Run in a worker process: are the two LaTeX answers mathematically equal?"""
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

    # ``math_verify`` only treats a string as LaTeX inside a math environment, so wrap in ``$...$``.
    # Its signal-based timeouts stay off: the parent process enforces the deadline by killing us.
    extraction = [LatexExtractionConfig(), ExprExtractionConfig()]
    gold = parse(f"${target}$", extraction_config=extraction, parsing_timeout=None)
    pred = parse(f"${prediction}$", extraction_config=extraction, parsing_timeout=None)
    return bool(verify(gold, pred, timeout_seconds=None))


def _get_pool() -> ProcessPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            # "spawn": forking a process that runs threads (one GRPO arm per thread) is unsafe.
            _pool = ProcessPoolExecutor(max_workers=_WORKERS, mp_context=multiprocessing.get_context("spawn"))
        return _pool


def _discard_pool(pool: ProcessPoolExecutor) -> None:
    """Kill ``pool``'s workers (one is stuck) so the next call starts a fresh pool."""
    global _pool
    with _pool_lock:
        if _pool is pool:
            _pool = None
    for process in list(getattr(pool, "_processes", {}).values()):
        process.kill()
    pool.shutdown(wait=False, cancel_futures=True)


def _verify_with_deadline(prediction: str, target: str) -> bool | None:
    """``math_verify``'s verdict, or ``None`` if it overran :data:`VERIFY_TIMEOUT_S` or its pool broke."""
    for _ in range(2):  # a pool killed by another thread's timeout breaks in-flight calls: retry once
        pool = _get_pool()
        try:
            return pool.submit(_verify, prediction, target).result(timeout=VERIFY_TIMEOUT_S)
        except FutureTimeoutError:
            _discard_pool(pool)
            return None
        except BrokenProcessPool:
            _discard_pool(pool)
    return None


def math_equal(prediction: str, target: str) -> bool:
    """Whether two free-form math answers are mathematically equal."""
    if prediction is None or not target:
        return False
    if _verify_with_deadline(prediction, target):
        return True
    return _canonical_string(prediction) == _canonical_string(target)


def boxed_math_reward(prediction: str, target: str) -> float:
    r"""1.0 when the model's ``\boxed{}`` answer matches the target, else 0.0."""
    pred = extract_boxed(prediction)
    if pred is None or not target:
        return 0.0
    return 1.0 if math_equal(pred, target) else 0.0
