"""Teachers: frozen expert sessions that score the student's tokens.

A teacher is an ordinary LoRA session on the student's own model resource, started from the
expert's *training* checkpoint (on a resource with a trainer, a LoRA session can only start from
a training checkpoint). It never trains. Its only job is to return the log-probability it assigns
to each token of a student completion, teacher-forced on the student's exact tokens.

:func:`score_with_generator` (the default) reads those log-probabilities from the teacher's
generator with ``prompt_logprobs=True``, the same inference engine that produced the student's
sampling log-probabilities, so the per-token difference measures the two policies rather than any
numeric gap between the trainer and the generator. :func:`score_with_trainer` reads them from the
trainer's forward-only pass instead, for a teacher session without a generator.
"""

from __future__ import annotations

from collections.abc import Sequence

from together.post_training import LoraConfig, ModelResourcesClient, SamplingParams, SessionClient

from post_training_cookbook.custom_loss import compute_logprobs
from post_training_cookbook.samples import build_cross_entropy_sample, make_prompt

__all__ = ["open_teacher", "score_with_generator", "score_with_trainer"]


def open_teacher(
    resources: ModelResourcesClient,
    *,
    name: str,
    training_checkpoint_id: str,
    lora_config: LoraConfig | None,
) -> SessionClient:
    """Start a frozen teacher session from an expert's training checkpoint.

    ``lora_config`` must match the one the expert trained with: a session doesn't inherit its
    LoRA rank and alpha from the checkpoint. ``load_optimizer=False`` skips the expert's optimizer
    state, which a teacher never uses. The generator loads the checkpoint's weights when the
    session starts, so the teacher scores with the expert's weights before any other call.
    """
    return resources.create_session(
        display_name=f"teacher-{name}",
        resume_from_checkpoint_id=training_checkpoint_id,
        load_optimizer=False,
        lora_config=lora_config,
    )


def _check_lengths(prompts: Sequence[Sequence[int]], completions: Sequence[Sequence[int]]) -> None:
    if len(prompts) != len(completions):
        raise ValueError(f"{len(prompts)} prompts but {len(completions)} completions")
    for index, (prompt, completion) in enumerate(zip(prompts, completions, strict=True)):
        if not prompt or not completion:
            raise ValueError(f"pair {index}: prompt and completion must both be non-empty")


def score_with_generator(
    teacher: SessionClient,
    prompts: Sequence[Sequence[int]],
    completions: Sequence[Sequence[int]],
    *,
    timeout: float | None = None,
) -> list[list[float]]:
    """The teacher's log-probability of every completion token, from its generator.

    Sends each ``prompt + completion`` as one prompt with ``prompt_logprobs=True`` and
    ``max_tokens=1``. Entry ``i`` of ``prompt_logprobs`` is the log-probability of input token ``i``
    given the tokens before it, so the completion's scores are the entries from ``len(prompt)`` on.
    """
    _check_lengths(prompts, completions)
    inputs = [make_prompt(list(prompt) + list(completion)) for prompt, completion in zip(prompts, completions)]
    kwargs = {"timeout": timeout} if timeout is not None else {}
    results = teacher.generator.sample_batch(
        inputs,
        num_samples=1,
        sampling_params=SamplingParams(max_tokens=1),
        prompt_logprobs=True,
        **kwargs,
    )
    scores: list[list[float]] = []
    for index, (prompt, completion, result) in enumerate(zip(prompts, completions, results, strict=True)):
        logprobs = result.prompt_logprobs
        expected = len(prompt) + len(completion)
        if logprobs is None or len(logprobs) != expected:
            got = None if logprobs is None else len(logprobs)
            raise RuntimeError(f"teacher returned {got} prompt logprobs for pair {index}, expected {expected}")
        scores.append([float(value) for value in logprobs[len(prompt) :]])
    return scores


def score_with_trainer(
    teacher: SessionClient,
    prompts: Sequence[Sequence[int]],
    completions: Sequence[Sequence[int]],
    *,
    timeout: float | None = None,
) -> list[list[float]]:
    """The teacher's log-probability of every completion token, from its trainer (forward only, no gradients)."""
    _check_lengths(prompts, completions)
    samples = [
        build_cross_entropy_sample(list(prompt) + list(completion), [0.0] * len(prompt) + [1.0] * len(completion))
        for prompt, completion in zip(prompts, completions)
    ]
    slot_logprobs = compute_logprobs(teacher, samples, timeout=timeout)
    # Slot i predicts token i + 1, so the completion's tokens sit in the last len(completion) slots.
    return [
        [float(value) for value in slots[len(prompt) - 1 :]]
        for prompt, slots in zip(prompts, slot_logprobs, strict=True)
    ]
