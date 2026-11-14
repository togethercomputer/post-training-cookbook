"""MOPD domains: the prompts, reward, and routing key behind each teacher.

A :class:`Domain` names one area the student should master (math, instruction following, ...).
The same object drives every stage of the recipe:

- **Experts.** :func:`register` puts the domain's :class:`DatasetConfig` into the cookbook's
  dataset registry under :attr:`Domain.dataset_key`, so ``train_single_turn`` trains that domain's
  GRPO expert with no MOPD-specific code.
- **Fusion.** The student answers prompts from every domain's training split, and each answer is
  scored by the teacher whose name matches :attr:`Domain.name`.
- **Evaluation.** Every model is scored on each domain's held-out test split with the domain's
  reward.

:func:`load_splits` returns the exact rows the expert trained on (``train``) and the rows held out
from every stage (``test``), because it calls the same ``load_grpo_dataset`` with the same counts
and seed that ``train_single_turn`` uses.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from datasets import Dataset

from post_training_cookbook.datasets.registry import DATASETS, DatasetConfig, load_grpo_dataset

__all__ = [
    "MIXED_DATASET_KEY",
    "Domain",
    "default_domains",
    "load_splits",
    "mixed_dataset_config",
    "register",
]

MIXED_DATASET_KEY = "mopd/mixed"
"""Registry key of the all-domains mixture used by the single-run GRPO baseline."""


@dataclass(frozen=True)
class Domain:
    """One MOPD domain.

    Attributes:
        name: Routing key. A prompt from this domain is scored by the teacher of the same name.
        dataset: Prompt formatting, target extraction and reward for the domain.
        num_train: Rows in the training split. Experts, the mixed baseline and the fusion student
            all draw prompts from these rows only.
        num_test: Rows in the held-out test split, taken right after the training rows.
        seed: Shuffle seed, shared by every stage so the splits never overlap.
    """

    name: str
    dataset: DatasetConfig
    num_train: int
    num_test: int
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate the name (used in registry keys) and the split sizes."""
        if not self.name or "/" in self.name:
            raise ValueError(f"domain name must be non-empty and contain no '/': {self.name!r}")
        if self.num_train < 1 or self.num_test < 1:
            raise ValueError(f"{self.name}: num_train and num_test must be >= 1")

    @property
    def dataset_key(self) -> str:
        """The key :func:`register` stores this domain's dataset under, e.g. ``"mopd/math"``."""
        return f"mopd/{self.name}"


def register(domains: Sequence[Domain]) -> None:
    """Add each domain's dataset to the cookbook registry, so ``GRPOConfig(dataset=domain.dataset_key)`` works."""
    names = [domain.name for domain in domains]
    if len(set(names)) != len(names):
        raise ValueError(f"domain names must be unique: {names}")
    for domain in domains:
        DATASETS[domain.dataset_key] = domain.dataset


def load_splits(domain: Domain) -> tuple[Dataset, Dataset]:
    """``(train, test)`` rows for ``domain``.

    Raises:
        ValueError: If the dataset has too few rows for both splits at their requested sizes.
    """
    train, test = load_grpo_dataset(domain.dataset, domain.num_train, domain.num_test, seed=domain.seed)
    if len(train) < domain.num_train or test is None or len(test) < domain.num_test:
        got = f"{len(train)} train + {0 if test is None else len(test)} test"
        raise ValueError(f"{domain.name}: dataset too small for {domain.num_train} + {domain.num_test} rows ({got})")
    return train, test


def mixed_dataset_config(domains: Sequence[Domain]) -> DatasetConfig:
    """A dataset that mixes every domain's training split, for the one-run GRPO baseline.

    Each row keeps its domain, so formatting, target extraction and the reward dispatch to that
    domain's own functions: the baseline sees the same prompts and rewards as the experts, just in
    one run. Register it under :data:`MIXED_DATASET_KEY` and train with ``num_examples`` equal to
    the sum of the domains' ``num_train`` and ``num_val_examples=0`` (the mixture has no held-out
    rows; evaluate on each domain's test split instead).
    """
    by_name = {domain.name: domain for domain in domains}
    if len(by_name) != len(domains):
        raise ValueError("domain names must be unique")

    def loader() -> Dataset:
        rows: list[dict[str, str]] = []
        for domain in domains:
            train, _ = load_splits(domain)
            # Rows from different datasets have different columns, so each travels as JSON.
            rows.extend({"domain": domain.name, "row": json.dumps(row, default=str)} for row in train)
        return Dataset.from_list(rows)

    def unpack(row: dict) -> tuple[DatasetConfig, dict[str, Any]]:
        return by_name[row["domain"]].dataset, json.loads(row["row"])

    def format_question(row: dict) -> str:
        cfg, inner = unpack(row)
        return cfg.format_question(inner)

    def extract_target(row: dict) -> dict[str, Any]:
        cfg, inner = unpack(row)
        return {"domain": row["domain"], "target": cfg.extract_target(inner)}

    def compute_reward(prediction: str, target: dict[str, Any]) -> float:
        return by_name[target["domain"]].dataset.compute_reward(prediction, target["target"])

    def compute_metric(prediction: str, target: dict[str, Any]) -> float:
        cfg = by_name[target["domain"]].dataset
        return (cfg.compute_metric or cfg.compute_reward)(prediction, target["target"])

    if any(domain.dataset.few_shot_messages for domain in domains):
        raise ValueError("the mixed baseline needs zero-shot domains; clear few_shot_messages on every domain")
    return DatasetConfig(
        hf_name="+".join(domain.dataset.hf_name for domain in domains),
        hf_config=None,
        hf_split="train",
        format_question=format_question,
        extract_target=extract_target,
        compute_reward=compute_reward,
        compute_metric=compute_metric,
        loader=loader,
    )


def default_domains() -> list[Domain]:
    r"""The three domains of the cookbook's MOPD experiment, zero-shot.

    - ``math``: DeepMath-103K, answer in ``\boxed{}``, graded with ``math_verify``.
    - ``ifeval``: AllenAI RLVR-IFEval, one verifiable formatting constraint per prompt.
    - ``game24``: Game of 24, a boxed arithmetic expression that must equal 24.

    Few-shot examples are cleared so every stage (experts, mixed baseline, student, evaluation)
    sends the identical prompt for a row.
    """
    return [
        Domain(
            name="math",
            dataset=replace(DATASETS["deepmath"], few_shot_messages=[]),
            num_train=4000,
            num_test=500,
        ),
        Domain(name="ifeval", dataset=DATASETS["ifeval"], num_train=4000, num_test=500),
        Domain(
            name="game24",
            dataset=replace(DATASETS["game24"], few_shot_messages=[]),
            num_train=1000,
            num_test=362,
        ),
    ]
