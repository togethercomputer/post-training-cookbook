"""SFT ``train()`` batching, its step budget, and its pre-session schema check."""

from __future__ import annotations

import itertools
from contextlib import nullcontext
from unittest.mock import Mock

import datasets
import pytest

from post_training_cookbook.sft import train
from post_training_cookbook.sft.config import SFTConfig
from tests._codec_fakes import DummyCodec

CONVERSATION = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]


def _stub_remote(monkeypatch, rows):
    """Serve ``rows`` as the dataset and stub the codec and every remote call; return (session, factory)."""
    session = Mock()
    session.trainer.forward_backward.return_value = Mock(loss=1.0)
    factory = Mock(return_value=nullcontext(session))
    monkeypatch.setattr(train.datasets, "load_dataset", lambda *args, **kwargs: datasets.Dataset.from_list(rows))
    monkeypatch.setattr(train, "load_dotenv", lambda **kwargs: None)
    monkeypatch.setattr(train, "get_codec", lambda *args, **kwargs: DummyCodec())
    monkeypatch.setattr(train, "training_session", factory)
    monkeypatch.setattr(train, "track_run", lambda *args, **kwargs: nullcontext(Mock()))
    monkeypatch.setattr(train, "log_step", Mock())
    monkeypatch.setattr(train, "checkpoint_if_due", Mock())
    monkeypatch.setattr(train, "latest_checkpoint_ref", Mock(return_value=None))
    return session, factory


def test_each_epoch_reshuffles_and_drops_the_partial_batch() -> None:
    dataset = datasets.Dataset.from_list([{"q": i} for i in range(5)])

    epochs = [[], []]
    for i, batch in enumerate(itertools.islice(train._batches(dataset, batch_size=2), 4)):
        assert len(batch) == 2
        epochs[i // 2].extend(batch["q"])

    assert all(len(set(epoch)) == 4 for epoch in epochs)
    assert epochs[0] != epochs[1]


def test_batches_reject_a_dataset_smaller_than_one_batch() -> None:
    with pytest.raises(ValueError, match="cannot fill"):
        next(train._batches(datasets.Dataset.from_list([{"q": 0}]), batch_size=2))


@pytest.mark.parametrize(("max_steps", "expected_steps"), [(None, 6), (5, 5)])
def test_steps_span_every_epoch_unless_capped(monkeypatch, max_steps, expected_steps) -> None:
    session, _ = _stub_remote(monkeypatch, [{"messages": CONVERSATION}] * 5)  # 2 full batches per epoch

    train.train(SFTConfig(model="org/M", batch_size=2, num_epochs=3, max_steps=max_steps))

    assert session.trainer.optim_step.call_count == expected_steps


def test_schema_mismatch_fails_before_a_session_is_provisioned(monkeypatch) -> None:
    _, factory = _stub_remote(monkeypatch, [{"prompt": "p", "completion": "c"}] * 2)

    with pytest.raises(KeyError, match="messages"):
        train.train(SFTConfig(model="org/M", batch_size=2))
    factory.assert_not_called()


def test_config_rejects_unknown_train_on_what() -> None:
    with pytest.raises(ValueError, match="train_on_what"):
        SFTConfig(model="org/M", train_on_what="last_assistant_messages")
