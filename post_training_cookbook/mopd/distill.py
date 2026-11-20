r"""The per-token MOPD signal and the training samples that carry it.

On-policy distillation trains the student on its own completions. For every completion token
:math:`x_t` the student sampled, the routed teacher scores the same token, and the per-token
advantage is the negative reverse-KL estimate

.. math::  A_t = \beta \, (\log \pi_{teacher}(x_t) - \log \pi_{student}(x_t)),

which is positive where the teacher likes the token more than the student does and negative where
it likes it less. Clipping :math:`A_t` to :math:`[-c, c]` keeps a few tokens the teacher finds
very unlikely from dominating a step. One ``forward_backward`` with the built-in importance-sampling loss
(``-ratio * A`` per token, ``ratio = pi_trainer / pi_sampler``) then moves the student toward the
teacher on exactly the tokens it produced. Only the sampled token's log-probability is used, which
is what the service exposes: losses see one log-probability per position, not the full vocabulary.

Everything here is pure, so it is unit-tested without a session.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from together.post_training import Sample

from post_training_cookbook.samples import build_sample

__all__ = ["Completion", "build_distillation_sample", "token_advantages"]


@dataclass(frozen=True)
class Completion:
    """One student completion, scored by its domain's teacher.

    ``student_logprobs`` come from the student's generator at sampling time and
    ``teacher_logprobs`` from the teacher's teacher-forced pass over the same tokens; both hold one
    value per entry of ``completion_tokens``.
    """

    domain: str
    prompt_tokens: list[int]
    completion_tokens: list[int]
    student_logprobs: list[float]
    teacher_logprobs: list[float]

    def __post_init__(self) -> None:
        """Check that every token has a student and a teacher log-probability."""
        n = len(self.completion_tokens)
        if n == 0:
            raise ValueError("a completion needs at least one token")
        if len(self.student_logprobs) != n or len(self.teacher_logprobs) != n:
            raise ValueError(
                f"{n} completion tokens but {len(self.student_logprobs)} student and "
                f"{len(self.teacher_logprobs)} teacher logprobs"
            )
        if not self.prompt_tokens:
            raise ValueError("a completion needs a non-empty prompt")

    @property
    def reverse_kl(self) -> np.ndarray:
        """Per-token reverse-KL estimate ``log pi_student - log pi_teacher`` (mean is the KL estimate)."""
        return np.asarray(self.student_logprobs, dtype=np.float64) - np.asarray(
            self.teacher_logprobs, dtype=np.float64
        )


def token_advantages(completion: Completion, *, kl_coef: float = 1.0, clip: float | None = None) -> list[float]:
    """``kl_coef * (teacher - student)`` log-probability for every completion token.

    With ``clip``, each advantage is clipped to ``[-clip, clip]``.
    """
    advantages = -kl_coef * completion.reverse_kl
    if clip is not None:
        advantages = np.clip(advantages, -clip, clip)
    return [float(value) for value in advantages]


def build_distillation_sample(
    completion: Completion, *, kl_coef: float = 1.0, advantage_clip: float | None = None
) -> Sample:
    """An importance-sampling training sample whose advantages are the per-token distillation signal.

    The prompt is masked out (weight 0, advantage 0); every completion token trains with advantage
    ``kl_coef * (teacher - student)``, clipped to ``[-advantage_clip, advantage_clip]``. The
    student's sampling log-probabilities go in the ``logprobs`` row, so the loss's importance ratio
    corrects for any drift between the generator and the trainer.
    """
    prompt_len = len(completion.prompt_tokens)
    advantages = token_advantages(completion, kl_coef=kl_coef, clip=advantage_clip)
    return build_sample(
        tokens=completion.prompt_tokens + completion.completion_tokens,
        loss_mask=[0.0] * prompt_len + [1.0] * len(completion.completion_tokens),
        advantages=[0.0] * prompt_len + advantages,
        logprobs=[0.0] * prompt_len + [float(value) for value in completion.student_logprobs],
    )
