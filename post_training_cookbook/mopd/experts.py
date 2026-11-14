"""Train the MOPD experts (and an optional mixed-domain baseline) side by side on one resource.

Each expert is a plain single-turn GRPO run from :func:`post_training_cookbook.grpo.train.train_single_turn`
on one domain's training split, attached as its own LoRA session to a shared model resource. The
runs train concurrently, one thread each, and every run saves a training checkpoint at its last
step: that checkpoint is what the fusion stage loads as the domain's teacher.

The optional ``mixed`` arm is the baseline MOPD is compared against: one GRPO run on the mixture of
every domain, with ``len(domains)`` times an expert's batch, so per step it samples as many
rollouts as the experts combined, over the same number of steps.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from post_training_cookbook.datasets.registry import DATASETS
from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.grpo.train import train_single_turn
from post_training_cookbook.mopd.domains import MIXED_DATASET_KEY, Domain, mixed_dataset_config, register
from post_training_cookbook.run_artifacts import Run, latest_checkpoint_ref

__all__ = ["MIXED", "ExpertRun", "expert_configs", "train_experts"]

MIXED = "mixed"
"""Name of the mixed-domain baseline arm in :func:`expert_configs` and :func:`train_experts`."""


@dataclass(frozen=True)
class ExpertRun:
    """Outcome of one arm: its run directory and checkpoints, or the error that stopped it."""

    name: str
    run_dir: Path
    training_checkpoint_id: str | None
    inference_checkpoint_id: str | None
    error: str | None = None


def expert_configs(
    domains: Sequence[Domain],
    base: GRPOConfig,
    *,
    model_resources_id: str,
    run_prefix: str,
    include_mixed: bool = False,
    checkpoint_interval: int | None = None,
    runs_root: Path | str = "runs",
) -> dict[str, GRPOConfig]:
    """One GRPO config per domain (plus ``"mixed"`` when asked), all attached to ``model_resources_id``.

    ``base`` carries the shared hyperparameters and must set ``num_steps``; each arm overrides the
    dataset, its example counts, the run name (``<run_prefix>-expert-<domain>`` or
    ``<run_prefix>-mixed``) and saves a training checkpoint at its final step, plus every
    ``checkpoint_interval`` steps when set (so a failed run can continue from its last checkpoint
    rather than restart). Registers every domain dataset (and the mixture) in the cookbook's dataset
    registry.
    """
    if base.num_steps is None:
        raise ValueError("base.num_steps must be set: experts save their training checkpoint at that step")
    interval = checkpoint_interval or base.num_steps
    if base.num_steps % interval:
        raise ValueError(f"checkpoint_interval ({interval}) must divide num_steps ({base.num_steps})")
    register(domains)
    shared = {
        "model": None,
        "model_resources_id": model_resources_id,
        "training_checkpoint_interval": interval,
    }
    configs = {
        domain.name: replace(
            base,
            **shared,
            dataset=domain.dataset_key,
            num_examples=domain.num_train,
            num_val_examples=min(base.num_val_examples, domain.num_test),
            wandb_run_name=f"{run_prefix}-expert-{domain.name}",
        )
        for domain in domains
    }
    if include_mixed:
        DATASETS[MIXED_DATASET_KEY] = mixed_dataset_config(domains)
        configs[MIXED] = replace(
            base,
            **shared,
            dataset=MIXED_DATASET_KEY,
            num_examples=sum(domain.num_train for domain in domains),
            num_val_examples=0,
            batch_size=base.batch_size * len(domains),
            wandb_run_name=f"{run_prefix}-{MIXED}",
        )
    for config in configs.values():
        if (Path(runs_root) / str(config.wandb_run_name)).exists():
            raise FileExistsError(f"run directory for {config.wandb_run_name} already exists; pick a new run_prefix")
    return configs


def train_experts(configs: dict[str, GRPOConfig], *, runs_root: Path | str = "runs") -> dict[str, ExpertRun]:
    """Train every arm concurrently (one thread each) and collect their checkpoints.

    An arm that fails does not stop the others; its :class:`ExpertRun` carries the error, and its
    checkpoint IDs are whatever it recorded before failing.
    """
    with ThreadPoolExecutor(max_workers=len(configs)) as pool:
        futures = {name: pool.submit(train_single_turn, config) for name, config in configs.items()}
        errors = {
            name: (None if (exc := future.exception()) is None else f"{type(exc).__name__}: {exc}")
            for name, future in futures.items()
        }
    results: dict[str, ExpertRun] = {}
    for name, config in configs.items():
        run = Run(dir=Path(runs_root) / str(config.wandb_run_name))
        results[name] = ExpertRun(
            name=name,
            run_dir=run.dir,
            training_checkpoint_id=latest_checkpoint_ref(run, kind="training"),
            inference_checkpoint_id=latest_checkpoint_ref(run, kind="inference"),
            error=errors[name],
        )
    return results
