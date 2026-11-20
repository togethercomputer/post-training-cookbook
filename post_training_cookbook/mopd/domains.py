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

from collections.abc import Sequence
from dataclasses import dataclass, replace

from datasets import Dataset

from post_training_cookbook.datasets.registry import DATASETS, DatasetConfig, load_grpo_dataset

__all__ = ["Domain", "default_domains", "load_splits", "register"]


@dataclass(frozen=True)
class Domain:
    """One MOPD domain.

    Attributes:
        name: Routing key. A prompt from this domain is scored by the teacher of the same name.
        dataset: Prompt formatting, target extraction and reward for the domain.
        num_train: Rows in the training split. Experts and the fusion student draw prompts from
            these rows only.
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


def default_domains() -> list[Domain]:
    r"""The three domains of the cookbook's MOPD experiment, zero-shot.

    - ``math``: DeepMath-103K, answer in ``\boxed{}``, graded with ``math_verify``.
    - ``ifeval``: AllenAI RLVR-IFEval, one verifiable formatting constraint per prompt.
    - ``game24``: Game of 24, a boxed arithmetic expression that must equal 24.

    Few-shot examples are cleared so every stage (experts, student, evaluation)
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
