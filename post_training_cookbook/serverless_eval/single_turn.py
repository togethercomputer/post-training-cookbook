"""Evaluate hosted models on single-turn QA datasets through the eval core."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from post_training_cookbook.datasets.registry import DATASETS, DatasetConfig, load_grpo_dataset
from post_training_cookbook.envs.base import Env, StepResult, Task
from post_training_cookbook.serverless_eval.harness import (
    DEFAULT_CONCURRENCY,
    EvalSummary,
    SamplingConfig,
    run_and_report,
)
from post_training_cookbook.types import Message, ToolSpec


def _dataset_tasks(ds_cfg: DatasetConfig, rows: Iterable[Mapping[str, Any]]) -> list[Task]:
    return [Task(prompt=ds_cfg.format_question(row), target=ds_cfg.extract_target(row)) for row in rows]


class SingleTurnEnv(Env):
    """One assistant turn, scored by a ``DatasetConfig``'s reward.

    The conversation is the config's few-shot messages followed by the task
    prompt. The first assistant reply is scored against the task target and ends
    the episode. There are no tools.
    """

    def __init__(self, ds_cfg: DatasetConfig) -> None:
        self._ds_cfg = ds_cfg
        self._task: Task | None = None
        self._messages: list[Message] = []

    async def reset(self, task: Task) -> list[Message]:
        self._task = task
        self._messages = [
            *self._ds_cfg.few_shot_messages,
            Message(role="user", content=task.prompt),
        ]
        return list(self._messages)

    async def step(self, message: Message) -> StepResult:
        if self._task is None:
            raise RuntimeError("reset() must be called before step().")

        self._messages.append(message)
        answer = message.get("content", "")
        target = self._task.target

        reward = self._ds_cfg.compute_reward(answer, target)
        accuracy = self._ds_cfg.compute_metric(answer, target) if self._ds_cfg.compute_metric is not None else reward

        return StepResult(
            terminated=True,
            truncated=False,
            messages=list(self._messages),
            reward=reward,
            metrics={"accuracy": accuracy},
        )

    @property
    def tool_specs(self) -> list[ToolSpec]:
        return []

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)


def run_dataset_eval(
    *,
    model: str,
    dataset: str,
    num_train: int,
    num_val: int,
    sampling: SamplingConfig = SamplingConfig(),
    seed: int = 0,
    max_concurrency: int = DEFAULT_CONCURRENCY,
    out_dir: str = "runs",
) -> EvalSummary:
    """Load a registered dataset's held-out split and evaluate ``model`` on it.

    Uses the same ``num_train``/``num_val``/``seed`` as ``load_grpo_dataset`` so
    the split matches the matching GRPO recipe's. Does not apply the trainer's
    prompt-length filter — this is a hosted sanity check, not Generator step-0
    parity.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; known: {sorted(DATASETS)}")

    ds_cfg = DATASETS[dataset]
    _, val = load_grpo_dataset(ds_cfg, num_train=num_train, num_val=num_val, seed=seed)
    if val is None:
        raise ValueError(f"No validation rows for {dataset!r}; increase num_val or shrink num_train.")

    return run_and_report(
        model=model,
        tasks=_dataset_tasks(ds_cfg, val),
        env_factory=lambda _task: SingleTurnEnv(ds_cfg),
        max_turns=1,
        sampling=sampling,
        max_concurrency=max_concurrency,
        label=dataset,
        out_dir=out_dir,
    )
