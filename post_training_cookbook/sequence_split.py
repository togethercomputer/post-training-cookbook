"""Render trained turns into prefix-stable trainable sequences.

A ``Turn`` declares intent (train ``completion`` given ``prompt``); a
``TrainableSequence`` is what the loss consumes (tokens + per-token mask, plus
logprobs when sampled on-policy). ``split_turns`` maps between them.

Chat templates don't guarantee that the render up to turn N is a prefix of the
render up to turn N+1 — thinking stripping is the usual cause. When a turn's
prompt stops extending the emitted tokens, the current sequence is closed and a
new one starts, keeping tokens, masks, and logprobs aligned.
"""

from __future__ import annotations

from typing import NamedTuple


class Context(NamedTuple):
    """An untrained run: prompt, history, or template scaffolding."""

    tokens: list[int]


class Completion(NamedTuple):
    """A trained run, carrying generator logprobs when the tokens were sampled on-policy."""

    tokens: list[int]
    logprobs: list[float] | None = None
    routed_experts_key: str | None = None


class Turn(NamedTuple):
    """Train ``completion`` conditioned on ``prompt``."""

    prompt: Context
    completion: Completion


class TrainableSequence(NamedTuple):
    """A token stream for the loss.

    ``loss_mask`` is 1 over trained tokens. ``logprobs`` is aligned per-token
    (0.0 over untrained runs) when sampled on-policy, else ``None``.
    ``routed_experts_key`` is the last completion's: a sequence is exactly its
    last turn's prompt + completion, so that one capture covers every token.
    """

    tokens: list[int]
    loss_mask: list[int]
    logprobs: list[float] | None
    routed_experts_key: str | None = None


def split_turns(turns: list[Turn]) -> list[TrainableSequence]:
    """Partition ``turns`` into prefix-stable trainable sequences.

    While a turn's prompt extends the tokens emitted so far, turns share one
    sequence and only the newly exposed prompt is kept. When a prompt no longer
    extends them, the sequence is closed and a new one starts at the full prompt.
    """
    sequences: list[TrainableSequence] = []
    runs: list[Context | Completion] = []
    emitted: list[int] = []

    for prompt, completion in turns:
        if prompt.tokens[: len(emitted)] == emitted:
            new_prompt = prompt.tokens[len(emitted) :]
            if new_prompt:
                runs.append(Context(new_prompt))
        else:
            sequences.append(_flatten(runs))
            runs = [prompt]
        runs.append(completion)
        emitted = prompt.tokens + completion.tokens

    if runs:
        sequences.append(_flatten(runs))
    return sequences


def _flatten(runs: list[Context | Completion]) -> TrainableSequence:
    """Concatenate runs into token-aligned arrays, masking completions in."""
    on_policy = any(isinstance(run, Completion) and run.logprobs is not None for run in runs)
    tokens: list[int] = []
    loss_mask: list[int] = []
    logprobs: list[float] = []
    for run in runs:
        trained = isinstance(run, Completion)
        tokens.extend(run.tokens)
        loss_mask.extend([1 if trained else 0] * len(run.tokens))
        if on_policy:
            logprobs.extend(run.logprobs if trained and run.logprobs is not None else [0.0] * len(run.tokens))
    return TrainableSequence(tokens, loss_mask, logprobs if on_policy else None, runs[-1].routed_experts_key)
