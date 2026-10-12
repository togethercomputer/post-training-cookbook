"""GRPO training entry points (single-turn and multi-turn).

Each entry point owns its session lifecycle, wandb run, and run-artifact
directory; the per-mode rollout logic lives in :mod:`post_training_cookbook.grpo.single_turn`
and :mod:`post_training_cookbook.grpo.multi_turn`. Both modes share the
:func:`post_training_cookbook.grpo.loop.training_loop` core.
"""

from __future__ import annotations

import functools
import os
from dataclasses import asdict, replace

from dotenv import load_dotenv

from post_training_cookbook.codecs import get_codec
from post_training_cookbook.config import lora_config_from, optimizer_config_from, session_metadata_from
from post_training_cookbook.datasets.registry import (
    DATASETS,
    filter_by_prompt_length,
    load_grpo_dataset,
)
from post_training_cookbook.envs.base import EnvFactory, Task
from post_training_cookbook.grpo.async_loop import async_training_loop
from post_training_cookbook.grpo.config import GRPOConfig
from post_training_cookbook.grpo.loop import training_loop
from post_training_cookbook.grpo.multi_turn import (
    make_multi_turn_collector,
    run_multi_turn_validation,
    run_multi_turn_validation_async,
)
from post_training_cookbook.grpo.resume import load_resume_point, start_progress
from post_training_cookbook.grpo.single_turn import make_single_turn_collector, run_validation
from post_training_cookbook.multi_lora import resolve_base_model, training_session
from post_training_cookbook.run_artifacts import track_run

load_dotenv(override=True)


def train_single_turn(config: GRPOConfig) -> None:
    """Single-turn GRPO training entry point."""
    if config.max_steps_off_policy is not None:
        raise ValueError("max_steps_off_policy is only supported by multi-turn training")

    resume = load_resume_point(config)

    codec = get_codec(
        resolve_base_model(config),
        hf_token=os.environ.get("HF_TOKEN"),
        enable_thinking=config.enable_thinking,
        preserve_thinking=config.preserve_thinking,
        thinking_effort=config.thinking_effort,
    )

    ds_cfg = DATASETS[config.dataset]
    train_dataset, val_dataset = load_grpo_dataset(ds_cfg, config.num_examples, config.num_val_examples)
    train_dataset = filter_by_prompt_length(
        train_dataset,
        ds_cfg,
        codec,
        config.max_prompt_length,
        config.max_model_length,
        min_response_tokens=config.min_response_tokens,
    )
    if val_dataset is not None:
        val_dataset = filter_by_prompt_length(
            val_dataset,
            ds_cfg,
            codec,
            config.max_prompt_length,
            config.max_model_length,
            min_response_tokens=config.min_response_tokens,
        )

    steps_per_epoch = len(train_dataset) // config.batch_size
    total_steps = config.num_steps if config.num_steps is not None else steps_per_epoch * config.num_epochs

    start_step, data_progress = start_progress(
        config, resume, kind="single_turn", size=len(train_dataset), rows=train_dataset
    )
    if start_step >= total_steps:
        print(f"Already completed {start_step} steps (target {total_steps}); no training needed")
        return

    print(f"Dataset: {config.dataset} ({ds_cfg.hf_name})")
    print(f"Dataset size: {len(train_dataset)} train, {len(val_dataset) if val_dataset else 0} val")
    print(
        f"Training for {total_steps} steps"
        f" ({config.num_epochs} epochs, {config.batch_size} prompts x {config.group_size} samples per step)"
    )

    with (
        training_session(
            model=config.model,
            model_resources_id=config.model_resources_id,
            display_name=config.wandb_run_name,
            metadata=session_metadata_from(config),
            resume_from_checkpoint_id=config.resume_from_checkpoint_id,
            load_optimizer=config.load_optimizer,
            lora_config=lora_config_from(config),
            lora_enabled=config.use_lora,
            optimizer_config=optimizer_config_from(config),
            num_generator_replicas=config.num_generator_replicas,
        ) as session,
        track_run(config, kind="grpo-single-turn", session_id=session.session_id, total_steps=total_steps) as run,
    ):
        run = replace(run, snapshot_data_state=data_progress.snapshot)
        collect_samples = make_single_turn_collector(
            config, ds_cfg, train_dataset, codec, run, data_progress=data_progress
        )

        validate = None
        if val_dataset is not None:
            validate = functools.partial(
                run_validation,
                val_dataset=val_dataset,
                session=session,
                codec=codec,
                ds_cfg=ds_cfg,
                config=config,
                run=run,
            )

        training_loop(total_steps, collect_samples, session, config, run, validate, start_step=start_step)


def train_multi_turn(
    config: GRPOConfig,
    tasks: list[Task],
    env_factory: EnvFactory,
    val_tasks: list[Task] | None = None,
) -> None:
    """Multi-turn GRPO training entry point."""
    if config.max_turns is None:
        raise ValueError("max_turns must be set for multi-turn training")

    resume = load_resume_point(config)

    codec = get_codec(
        resolve_base_model(config),
        hf_token=os.environ.get("HF_TOKEN"),
        enable_thinking=config.enable_thinking,
        preserve_thinking=config.preserve_thinking,
        thinking_effort=config.thinking_effort,
    )
    total_steps = config.num_steps if config.num_steps is not None else len(tasks) // config.batch_size

    start_step, data_progress = start_progress(
        config,
        resume,
        kind="multi_turn" if config.max_steps_off_policy is None else "multi_turn_async",
        size=len(tasks),
        rows=(asdict(task) for task in tasks),
    )
    if start_step >= total_steps:
        print(f"Already completed {start_step} steps (target {total_steps}); no training needed")
        return

    print(f"Multi-turn training: {len(tasks)} tasks, {total_steps} steps, max_turns={config.max_turns}")

    with (
        training_session(
            model=config.model,
            model_resources_id=config.model_resources_id,
            display_name=config.wandb_run_name,
            metadata=session_metadata_from(config),
            resume_from_checkpoint_id=config.resume_from_checkpoint_id,
            load_optimizer=config.load_optimizer,
            lora_config=lora_config_from(config),
            lora_enabled=config.use_lora,
            optimizer_config=optimizer_config_from(config),
            num_generator_replicas=config.num_generator_replicas,
        ) as session,
        track_run(config, kind="grpo-multi-turn", session_id=session.session_id, total_steps=total_steps) as run,
    ):
        run = replace(run, snapshot_data_state=data_progress.snapshot)
        validation_fn = (
            run_multi_turn_validation if config.max_steps_off_policy is None else run_multi_turn_validation_async
        )
        validate = None
        if val_tasks is not None:
            validate = functools.partial(
                validation_fn,
                config=config,
                codec=codec,
                val_tasks=val_tasks,
                env_factory=env_factory,
                session=session,
                run=run,
            )

        if config.max_steps_off_policy is None:
            collect_samples = make_multi_turn_collector(
                config, codec, tasks, env_factory, run, data_progress=data_progress
            )
            training_loop(total_steps, collect_samples, session, config, run, validate, start_step=start_step)
        else:
            session.run(
                async_training_loop(
                    total_steps,
                    session,
                    config,
                    codec,
                    tasks,
                    env_factory,
                    run,
                    validate,
                    data_progress=data_progress,
                    start_step=start_step,
                )
            )
