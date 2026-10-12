"""SFT training orchestration."""

from __future__ import annotations

import itertools
import logging
import os
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

import datasets
from dotenv import load_dotenv
from together.post_training import AdamParams, LossConfig, Sample

from post_training_cookbook.codecs import MessageCodec, get_codec
from post_training_cookbook.config import lora_config_from, session_metadata_from
from post_training_cookbook.multi_lora import resolve_base_model, training_session
from post_training_cookbook.run_artifacts import (
    checkpoint_if_due,
    latest_checkpoint_ref,
    log_step,
    track_run,
)
from post_training_cookbook.samples import build_cross_entropy_sample, is_trainable, trained_slots
from post_training_cookbook.sequence_split import Completion, Context, TrainableSequence, Turn, split_turns
from post_training_cookbook.sft.config import TRAIN_ON_ALL, TRAIN_ON_LAST, SFTConfig
from post_training_cookbook.types import Message

__all__ = [
    "TRAIN_ON_ALL",
    "TRAIN_ON_LAST",
    "build_conversation_samples",
    "messages_column",
    "render_conversation",
    "train",
]

logger = logging.getLogger(__name__)

RowToMessages = Callable[[Mapping[str, Any]], list[Message]]


def _batches(dataset: datasets.Dataset, batch_size: int) -> Iterator[datasets.Dataset]:
    """Yield full batches indefinitely, reshuffling each epoch; a trailing partial batch is dropped."""
    if len(dataset) < batch_size:
        raise ValueError(f"{len(dataset)} rows cannot fill a batch of {batch_size}")
    for epoch in itertools.count():
        rows = dataset.shuffle(seed=epoch)
        for start in range(0, len(rows) - batch_size + 1, batch_size):
            yield rows.select(range(start, start + batch_size))


def messages_column(row: Mapping[str, Any]) -> list[Message]:
    """Read a row already in chat form. Pass your own ``to_messages`` to ``train`` for any other schema."""
    return row["messages"]


def render_conversation(
    messages: list[Message],
    *,
    codec: MessageCodec,
    train_on_what: str,
    max_length: int,
) -> list[TrainableSequence]:
    """Render a conversation into one or more trainable sequences, truncated to ``max_length``.

    Each trained turn is closed with ``codec.turn_terminator``, so a turn whose stop token
    is only rendered once the *next* message arrives still trains the model to stop — and
    to stop the right way (yield to the user, hand off to a tool, end the conversation).
    """
    assistant_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if not assistant_indices:
        raise ValueError("Conversation has no assistant message")

    if train_on_what == TRAIN_ON_LAST:
        assistant_indices = [max(assistant_indices)]

    turns: list[Turn] = []
    for assistant_idx in assistant_indices:
        prompt_tokens = codec.encode_messages(messages[:assistant_idx], add_generation_prompt=True)
        full_tokens = codec.encode_messages(messages[: assistant_idx + 1], add_generation_prompt=False)
        if full_tokens[: len(prompt_tokens)] != prompt_tokens:
            raise ValueError(
                f"Codec rewrote the assistant prompt while encoding turn {assistant_idx}: "
                "rendering messages[:i+1] does not extend rendering messages[:i] with "
                "add_generation_prompt=True."
            )
        next_role = messages[assistant_idx + 1]["role"] if assistant_idx + 1 < len(messages) else None
        completion = full_tokens[len(prompt_tokens) :] + codec.turn_terminator(next_role)
        turns.append(Turn(prompt=Context(prompt_tokens), completion=Completion(completion)))

    return [_truncate(seq, max_length) for seq in split_turns(turns)]


def _truncate(seq: TrainableSequence, max_length: int) -> TrainableSequence:
    """Clip a sequence to ``max_length`` tokens, keeping its arrays aligned."""
    if len(seq.tokens) > max_length:
        logger.warning(
            "Clipping a %d-token sequence to max_length=%d — the final trained turn loses its tail, "
            "including its stop token",
            len(seq.tokens),
            max_length,
        )
    return TrainableSequence(
        seq.tokens[:max_length],
        seq.loss_mask[:max_length],
        seq.logprobs[:max_length] if seq.logprobs is not None else None,
        seq.routed_experts_key,
    )


def build_conversation_samples(
    messages: list[Message],
    *,
    codec: MessageCodec,
    train_on_what: str,
    max_length: int,
) -> list[Sample]:
    """Render one conversation into the cross-entropy samples it trains on.

    A sequence that trains no slot is dropped here rather than rejected by the
    builder mid-run. Truncation to ``max_length`` is the usual cause — a prompt
    longer than the budget leaves an all-zero mask — but an assistant turn that
    renders to nothing gets there too.

    Args:
        messages: One conversation, in the chat-template message form.
        codec: The model's message codec, used to apply the chat template.
        train_on_what: Which assistant turns to train on, e.g. ``TRAIN_ON_LAST``.
        max_length: The token budget each rendered sequence is truncated to.

    Returns:
        One ``Sample`` per trained sequence, in render order. Empty when every
        sequence was dropped.
    """
    sequences = render_conversation(
        messages,
        codec=codec,
        train_on_what=train_on_what,
        max_length=max_length,
    )
    samples: list[Sample] = []
    for seq in sequences:
        if not is_trainable(seq.loss_mask):
            logger.warning(
                "Dropping a %d-token sequence that trains no slot (max_length=%d)",
                len(seq.tokens),
                max_length,
            )
            continue
        samples.append(build_cross_entropy_sample(seq.tokens, seq.loss_mask))
    return samples


def _build_batch_samples(
    rows: Iterable[Mapping[str, Any]],
    *,
    to_messages: RowToMessages,
    codec: MessageCodec,
    train_on_what: str,
    max_length: int,
) -> list[Sample]:
    """Render a batch of dataset rows, dropping the ones the codec cannot render trainably.

    A row with no stop token for what follows a trained turn, or a prompt the template
    rewrites, costs that row rather than the whole run — so the result can be empty and
    the caller must handle that before submitting a training step.
    """
    samples: list[Sample] = []
    for index, row in enumerate(rows):
        messages = to_messages(row)
        try:
            samples.extend(
                build_conversation_samples(messages, codec=codec, train_on_what=train_on_what, max_length=max_length)
            )
        except ValueError as exc:
            logger.warning("Skipping row %d: roles=%s — %s", index, [msg["role"] for msg in messages], exc)
    return samples


def train(config: SFTConfig, *, to_messages: RowToMessages = messages_column) -> str | None:
    """Run SFT and return the latest training-checkpoint id, if one was saved.

    ``to_messages`` maps a dataset row to the conversation it trains on; any task
    instruction the model should learn belongs in that mapping.
    """
    load_dotenv(override=True)
    codec = get_codec(
        resolve_base_model(config),
        hf_token=os.environ.get("HF_TOKEN"),
        enable_thinking=config.enable_thinking,
        preserve_thinking=config.preserve_thinking,
        thinking_effort=config.thinking_effort,
    )

    dataset = datasets.load_dataset(config.dataset_name, config.dataset_config, split=config.dataset_split)
    steps_per_epoch = len(dataset) // config.batch_size
    total_steps = steps_per_epoch * config.num_epochs
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)
    if total_steps <= 0:
        raise ValueError(
            f"Nothing to train: {len(dataset)} rows in split {config.dataset_split!r} with batch_size "
            f"{config.batch_size}, num_epochs {config.num_epochs} and max_steps {config.max_steps} yields 0 steps"
        )
    to_messages(dataset[0])  # A schema mismatch should fail here, before a session is provisioned.

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
            num_generator_replicas=0,
        ) as session,
        track_run(config, kind="sft", session_id=session.session_id, total_steps=total_steps) as run,
    ):
        overall_start = time.monotonic()
        trained_steps = 0
        for step, batch_rows in zip(range(total_steps), _batches(dataset, config.batch_size)):
            step_num = step + 1
            samples = _build_batch_samples(
                batch_rows,
                to_messages=to_messages,
                codec=codec,
                train_on_what=config.train_on_what,
                max_length=config.max_length,
            )

            if samples:
                lr_mult = max(0.0, 1.0 - step / max(total_steps, 1))
                current_lr = config.learning_rate * lr_mult

                t0 = time.monotonic()
                fwd_bwd_result = session.trainer.forward_backward(
                    loss=LossConfig(type="LOSS_TYPE_CROSS_ENTROPY"),
                    samples=samples,
                    timeout=config.timeout,
                )
                fb_time = time.monotonic() - t0
                loss_sum = fwd_bwd_result.loss
                batch_trained_slots = trained_slots(samples)
                loss_value = loss_sum / batch_trained_slots if batch_trained_slots else float("nan")

                t0 = time.monotonic()
                session.trainer.optim_step(
                    adam_params=AdamParams(
                        beta1=config.adam_beta1,
                        beta2=config.adam_beta2,
                        eps=config.adam_eps,
                        weight_decay=config.weight_decay,
                        learning_rate=current_lr,
                    ),
                    timeout=config.timeout,
                )
                optim_time = time.monotonic() - t0
                trained_steps += 1

                log_step(
                    run,
                    step_num,
                    {
                        "loss": loss_value,
                        "loss_sum": loss_sum,
                        # Loss-mask weight mass (trained slots): the denominator of `loss`.
                        "trained_tokens": batch_trained_slots,
                        "learning_rate": current_lr,
                        "samples": len(samples),
                        "elapsed_s": time.monotonic() - overall_start,
                        "timing/step_forward_backward_s": fb_time,
                        "timing/step_optim_s": optim_time,
                    },
                )
                print(
                    f"[step {step_num}] loss={loss_value:.4f} "
                    f"(loss_sum={loss_sum:.1f} over {batch_trained_slots:g} slots) "
                    f"lr={current_lr:.6g} "
                    f"samples={len(samples)} elapsed={time.monotonic() - overall_start:.1f}s "
                    f"(fb={fb_time:.1f}s optim={optim_time:.1f}s)",
                    flush=True,
                )
            else:
                # A whole batch dropping out is a dataset/config error, not a per-row one, so
                # say so loudly rather than stepping the optimizer on an empty gradient.
                logger.warning("Skipping step %d: no row in the batch was renderable", step_num)

            # The final step's inference checkpoint is the run's only servable model, so a
            # skipped step must not swallow it — but a run that trained nothing must not
            # publish one either.
            if trained_steps > 0:
                checkpoint_if_due(
                    session,
                    run,
                    step=step_num,
                    total_steps=total_steps,
                    training_interval=config.training_checkpoint_interval,
                    inference_interval=config.inference_checkpoint_interval,
                )

        if trained_steps == 0:
            raise ValueError(
                f"Trained 0 of {total_steps} batches — nothing was renderable. Check dataset_split, "
                "train_on_what, and whether the codec has a stop token for this data's turn order."
            )
        training_ckpt_id = latest_checkpoint_ref(run, kind="training")
    print(f"Training completed in {time.monotonic() - overall_start:.1f}s")
    return training_ckpt_id
