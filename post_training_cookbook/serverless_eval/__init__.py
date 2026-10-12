"""Evaluate a hosted (OpenAI-compatible) chat model on any post_training_cookbook ``Env``.

The harness core (``evaluate`` / ``run_and_report``) is env-agnostic. ``single_turn``
adds a batteries-included adapter for registered single-turn QA datasets.
"""

from post_training_cookbook.serverless_eval.harness import EvalSummary, SamplingConfig, evaluate, run_and_report
from post_training_cookbook.serverless_eval.single_turn import SingleTurnEnv, run_dataset_eval

__all__ = [
    "EvalSummary",
    "SamplingConfig",
    "SingleTurnEnv",
    "evaluate",
    "run_and_report",
    "run_dataset_eval",
]
