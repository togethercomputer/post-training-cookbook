"""Fake sessions, resources and codec for the MOPD tests."""

from __future__ import annotations

import asyncio
import itertools
from types import SimpleNamespace
from typing import Any

from post_training_cookbook.datasets.registry import DatasetConfig
from post_training_cookbook.mopd.domains import Domain

_ids = itertools.count()


class CharCodec:
    """One token per character; prompts are ``[1, *chars, 2]``."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens if t >= 32)

    def encode_messages(self, messages, *, tools=None, add_generation_prompt=False) -> list[int]:
        out = [1]
        for message in messages:
            out.extend(self.encode(message["content"]))
        return [*out, 2]


def logprob_of(token: int, *, offset: float) -> float:
    """A deterministic per-token logprob, so tests can predict both policies' scores."""
    return -((token % 7) / 10.0) - offset


class FakeGenerator:
    """Answers every prompt with the token sequence ``answer`` and scores teacher-forced prompts.

    ``offset`` shifts every logprob, so a student (offset 0.0) and a teacher (offset 0.2) disagree by
    a known amount on every token.
    """

    def __init__(self, answer: list[int], offset: float) -> None:
        self.answer = answer
        self.offset = offset
        self.sample_calls: list[dict[str, Any]] = []

    def sample_batch(self, prompts, num_samples=None, sampling_params=None, *, prompt_logprobs=None, **kwargs):
        self.sample_calls.append(
            {
                "prompts": prompts,
                "num_samples": num_samples,
                "params": sampling_params,
                "prompt_logprobs": prompt_logprobs,
            }
        )
        results = []
        for prompt in prompts:
            tokens = list(prompt["chunks"][0]["encoded_text"]["tokens"])
            if prompt_logprobs:
                scores = [0.0] + [logprob_of(t, offset=self.offset) for t in tokens[1:]]
                results.append(SimpleNamespace(sequences=[_seq([0], [0.0])], prompt_logprobs=scores))
            else:
                seqs = [
                    _seq(self.answer, [logprob_of(t, offset=self.offset) for t in self.answer])
                    for _ in range(num_samples or 1)
                ]
                results.append(SimpleNamespace(sequences=seqs, prompt_logprobs=None))
        return results


def _seq(tokens, logprobs, stop_reason="STOP_REASON_STOP"):
    return SimpleNamespace(tokens=list(tokens), logprobs=list(logprobs), stop_reason=stop_reason)


class FakeTrainer:
    def __init__(self) -> None:
        self.forward_backward_calls: list[dict[str, Any]] = []
        self.optim_steps: list[dict[str, Any]] = []
        self.syncs = 0

    def forward_backward(self, *, samples, loss, timeout=None):
        self.forward_backward_calls.append({"samples": samples, "loss": loss})
        return SimpleNamespace(loss=0.123, metrics={"loss/ratio/mean": 1.0})

    def optim_step(self, *, adam_params, timeout=None):
        self.optim_steps.append(adam_params)

    def weights_sync(self, timeout=None):
        self.syncs += 1

    def forward(self, *, samples, loss, timeout=None):
        outputs = []
        for sample in samples:
            targets = sample["loss_fn_inputs"]["target_tokens"]["data"]
            weights = sample["loss_fn_inputs"]["weights"]["data"]
            values = [logprob_of(t, offset=0.2) if w else 0.0 for t, w in zip(targets, weights)]
            outputs.append(SimpleNamespace(tensors={"logprobs": SimpleNamespace(data=values)}))
        return SimpleNamespace(loss_fn_outputs=outputs)


class FakeSession:
    def __init__(self, name: str, answer: list[int], offset: float, **create_kwargs: Any) -> None:
        self.session_id = f"sess-{name}-{next(_ids)}"
        self.generator = FakeGenerator(answer, offset)
        self.trainer = FakeTrainer()
        self.create_kwargs = create_kwargs
        self.stopped = False
        self.checkpoints: list[str] = []

    def run(self, coro):
        return asyncio.run(coro)

    async def create_training_checkpoint_async(self):
        self.checkpoints.append("training")
        return SimpleNamespace(id=f"ckpt-train-{self.session_id}")

    async def create_inference_checkpoint_async(self):
        self.checkpoints.append("inference")
        return SimpleNamespace(id=f"ckpt-inf-{self.session_id}")

    def stop(self):
        self.stopped = True


class FakeResource:
    """Creates fake sessions: teachers (resumed from a checkpoint) score with offset 0.2, the student samples."""

    model_resources_id = "mr-fake"

    def __init__(self, answer: list[int]) -> None:
        self.answer = answer
        self.sessions: list[FakeSession] = []

    def create_session(
        self, *, display_name=None, resume_from_checkpoint_id=None, load_optimizer=None, lora_config=None
    ):
        is_teacher = bool(display_name and display_name.startswith("teacher-"))
        session = FakeSession(
            display_name or "student",
            self.answer,
            offset=0.2 if is_teacher else 0.0,
            display_name=display_name,
            resume_from_checkpoint_id=resume_from_checkpoint_id,
            load_optimizer=load_optimizer,
            lora_config=lora_config,
        )
        self.sessions.append(session)
        return session


def toy_domain(name: str, rows: list[dict], num_train: int, num_test: int) -> Domain:
    """A domain whose reward is 1.0 when the response contains the row's answer."""
    return Domain(
        name=name,
        dataset=DatasetConfig(
            hf_name=f"toy/{name}",
            hf_config=None,
            hf_split="train",
            format_question=lambda row: row["q"],
            extract_target=lambda row: row["a"],
            compute_reward=lambda prediction, target: float(target in prediction),
            loader=lambda: _dataset(rows),
        ),
        num_train=num_train,
        num_test=num_test,
    )


def _dataset(rows):
    from datasets import Dataset

    return Dataset.from_list(rows)
