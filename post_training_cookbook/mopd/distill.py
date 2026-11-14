r"""The per-token MOPD signal and the training samples that carry it.

On-policy distillation trains the student on its own completions. For every completion token
:math:`x_t` the student sampled, the routed teacher scores the same token, and the per-token
advantage is the negative reverse-KL estimate

.. math::  A_t = \beta \, (\log \pi_{teacher}(x_t) - \log \pi_{student}(x_t)),

which is positive where the teacher likes the token more than the student does and negative where
it likes it less. One ``forward_backward`` with the built-in importance-sampling loss
(``-ratio * A`` per token, ``ratio = pi_trainer / pi_sampler``) then moves the student toward the
teacher on exactly the tokens it produced. Only the sampled token's log-probability is used, which
is what the service exposes: losses see one log-probability per position, not the full vocabulary.

Everything here is pure, so it is unit-tested without a session.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from together.post_training import Sample

from post_training_cookbook.samples import build_sample

__all__ = [
    "Balance",
    "Completion",
    "balance_weights",
    "build_distillation_sample",
    "token_advantages",
]

Balance = Literal["token", "domain"]
"""How a step's loss is shared between domains.

- ``"token"``: every completion token counts the same, so a domain's share of the update is its
  share of the batch's tokens. Long-answer domains (math) dominate short-answer ones.
- ``"domain"``: each domain's tokens are reweighted so every domain contributes the same total
  weight, whatever its answer length.
"""


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


def token_advantages(completion: Completion, *, kl_coef: float = 1.0) -> list[float]:
    """``kl_coef * (teacher - student)`` log-probability for every completion token."""
    return [float(-kl_coef * value) for value in completion.reverse_kl]


def balance_weights(completions: Sequence[Completion], balance: Balance) -> list[float]:
    """One multiplier per completion that applies the ``balance`` policy to its advantages.

    Under ``"domain"``, a completion from domain ``d`` gets ``T / (D * T_d)``, where ``T`` is the
    batch's completion-token count, ``T_d`` domain ``d``'s, and ``D`` the number of domains present.
    Each domain's tokens then carry ``T / D`` total weight, and the batch total stays ``T``, so the
    loss scale (and so the learning rate) is comparable with ``"token"``.
    """
    if balance == "token":
        return [1.0] * len(completions)
    if balance != "domain":
        raise ValueError(f"unknown balance policy: {balance!r}")
    tokens_by_domain: dict[str, int] = {}
    for completion in completions:
        tokens_by_domain[completion.domain] = tokens_by_domain.get(completion.domain, 0) + len(
            completion.completion_tokens
        )
    total = sum(tokens_by_domain.values())
    num_domains = len(tokens_by_domain)
    return [total / (num_domains * tokens_by_domain[completion.domain]) for completion in completions]


def build_distillation_sample(completion: Completion, *, kl_coef: float = 1.0, weight: float = 1.0) -> Sample:
    """An importance-sampling training sample whose advantages are the per-token distillation signal.

    The prompt is masked out (weight 0, advantage 0); every completion token trains with advantage
    ``weight * kl_coef * (teacher - student)``. The student's sampling log-probabilities go in the
    ``logprobs`` row, so the loss's importance ratio corrects for any drift between the generator
    and the trainer.
    """
    prompt_len = len(completion.prompt_tokens)
    advantages = [weight * value for value in token_advantages(completion, kl_coef=kl_coef)]
    return build_sample(
        tokens=completion.prompt_tokens + completion.completion_tokens,
        loss_mask=[0.0] * prompt_len + [1.0] * len(completion.completion_tokens),
        advantages=[0.0] * prompt_len + advantages,
        logprobs=[0.0] * prompt_len + [float(value) for value in completion.student_logprobs],
    )
