"""Tests for the session lifecycle against fresh and shared model resources."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from together.post_training import LoraConfig, MuonConfig, OptimizerConfig, SessionMetadata

import post_training_cookbook.multi_lora as multi_lora
from post_training_cookbook.multi_lora import resolve_base_model, training_session

READY = "MODEL_RESOURCES_STATUS_READY"
CREATING = "MODEL_RESOURCES_STATUS_CREATING"

BASE_MODEL = "Qwen/Qwen3.5-0.8B"


@dataclass
class StubResourceModel:
    """What ``resource.retrieve()`` returns — the SDK's model object."""

    id: str = "mr-1"
    base_model: str = BASE_MODEL
    status: str = READY
    lora_enabled: bool = True
    num_generator_replicas: int = 1

    @property
    def compute_config(self):
        return SimpleNamespace(num_generator_replicas=self.num_generator_replicas)


@dataclass
class StubConfig:
    model: str | None = None
    model_resources_id: str | None = None


def install(monkeypatch, *, resource_model=None, create_session_error=None, stop_error=None):
    """Patch ``multi_lora.ModelResourcesClient`` with a fake; return (events, state).

    ``events`` records the ordered lifecycle calls; ``state`` captures the
    ``create``/``attach`` kwargs for assertions.
    """
    resource_model = resource_model or StubResourceModel()
    events: list = []
    state: dict = {"create_kwargs": None, "attach_kwargs": None, "session_kwargs": None}

    class FakeSession:
        session_id = "sess-1"

        def stop(self):
            events.append("session.stop")
            if stop_error is not None:
                raise stop_error

    class FakeHandle:
        model_resources_id = resource_model.id

        def retrieve(self):
            events.append("retrieve")
            return resource_model

        def create_session(
            self,
            *,
            lora_config=None,
            display_name=None,
            metadata=None,
            resume_from_checkpoint_id=None,
            load_optimizer=None,
        ):
            state["session_kwargs"] = {
                "lora_config": lora_config,
                "display_name": display_name,
                "metadata": metadata,
                "resume_from_checkpoint_id": resume_from_checkpoint_id,
                "load_optimizer": load_optimizer,
            }
            events.append(("create_session", lora_config))
            if create_session_error is not None:
                raise create_session_error
            return FakeSession()

        def stop(self):
            events.append("resource.stop")

        def detach(self):
            events.append("resource.detach")

    class FakeModelResources:
        @classmethod
        def create(cls, **kwargs):
            state["create_kwargs"] = kwargs
            events.append("create")
            return FakeHandle()

        @classmethod
        def attach(cls, **kwargs):
            state["attach_kwargs"] = kwargs
            events.append("attach")
            return FakeHandle()

    monkeypatch.setattr(multi_lora, "ModelResourcesClient", FakeModelResources)
    return events, state


def patch_together(monkeypatch, resource_model: StubResourceModel) -> list[str]:
    """Route ``multi_lora.Together`` (the resolve-base-model read); return retrieved IDs."""
    retrieve_calls: list[str] = []

    def retrieve(resource_id):
        retrieve_calls.append(resource_id)
        return resource_model

    class FakeClient:
        def __init__(self, **kwargs):
            self.post_training = SimpleNamespace(model_resources=SimpleNamespace(retrieve=retrieve))

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return None

    monkeypatch.setattr(multi_lora, "Together", FakeClient)
    return retrieve_calls


def test_fresh_provisions_then_stops_session_and_resource(monkeypatch):
    events, state = install(monkeypatch)

    with training_session(model=BASE_MODEL, num_generator_replicas=0, lora_enabled=False) as session:
        assert session.session_id == "sess-1"
        assert events == ["create", ("create_session", None)]

    assert events[-2:] == ["session.stop", "resource.stop"]
    assert state["create_kwargs"]["base_model"] == BASE_MODEL
    assert state["create_kwargs"]["compute_config"] == {"num_generator_replicas": 0}
    assert state["create_kwargs"]["lora_enabled"] is False


def test_forwards_session_identity(monkeypatch):
    _, state = install(monkeypatch)
    metadata = SessionMetadata()

    with training_session(model=BASE_MODEL, display_name="run-name", metadata=metadata):
        pass

    assert state["session_kwargs"]["display_name"] == "run-name"
    assert state["session_kwargs"]["metadata"] is metadata


def test_forwards_resume(monkeypatch):
    _, state = install(monkeypatch)

    with training_session(model=BASE_MODEL, resume_from_checkpoint_id="tckpt-1", load_optimizer=False):
        pass

    assert state["session_kwargs"]["resume_from_checkpoint_id"] == "tckpt-1"
    assert state["session_kwargs"]["load_optimizer"] is False


def test_attach_checks_then_stops_and_detaches(monkeypatch):
    events, state = install(monkeypatch)

    with training_session(model_resources_id="mr-1", lora_config=LoraConfig(rank=8)) as session:
        assert session.session_id == "sess-1"

    assert events == ["attach", "retrieve", ("create_session", LoraConfig(rank=8)), "session.stop", "resource.detach"]
    assert state["attach_kwargs"]["model_resources_id"] == "mr-1"


@pytest.mark.parametrize(
    ("lora_enabled", "lora_config"),
    [(False, None), (True, LoraConfig(rank=8)), (True, None)],
)
def test_fresh_lora_enabled_mapping(monkeypatch, lora_enabled, lora_config):
    events, state = install(monkeypatch)

    with training_session(model=BASE_MODEL, lora_enabled=lora_enabled, lora_config=lora_config):
        pass

    assert state["create_kwargs"]["lora_enabled"] is lora_enabled
    assert ("create_session", lora_config) in events


def test_full_weight_rejects_session_lora_config(monkeypatch):
    install(monkeypatch)

    with pytest.raises(ValueError, match="lora_config"):
        with training_session(model=BASE_MODEL, lora_enabled=False, lora_config=LoraConfig(rank=8)):
            pass


@pytest.mark.parametrize(
    "optimizer_config",
    [OptimizerConfig(muon=MuonConfig(scaling_strategy="MUON_SCALING_STRATEGY_MATCH_ADAM")), None],
)
def test_fresh_forwards_optimizer_config(monkeypatch, optimizer_config):
    events, state = install(monkeypatch)

    with training_session(model=BASE_MODEL, lora_enabled=False, optimizer_config=optimizer_config):
        pass

    assert state["create_kwargs"]["optimizer_config"] == optimizer_config


def test_create_session_failure_stops_owned_resource(monkeypatch):
    events, _ = install(monkeypatch, create_session_error=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        with training_session(model=BASE_MODEL):
            pass

    assert events == ["create", ("create_session", None), "retrieve", "resource.stop"]


def test_create_session_failure_detaches_borrowed_resource(monkeypatch):
    events, _ = install(monkeypatch, create_session_error=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        with training_session(model_resources_id="mr-1"):
            pass

    assert events == ["attach", "retrieve", ("create_session", None), "retrieve", "resource.detach"]


def test_attach_non_ready_detaches_without_opening_session(monkeypatch):
    events, _ = install(monkeypatch, resource_model=StubResourceModel(status=CREATING))

    with pytest.raises(RuntimeError, match=CREATING):
        with training_session(model_resources_id="mr-1"):
            pass

    assert events == ["attach", "retrieve", "retrieve", "resource.detach"]


def test_attach_rejects_non_lora_resource(monkeypatch):
    install(monkeypatch, resource_model=StubResourceModel(lora_enabled=False))

    with pytest.raises(ValueError, match="lora_enabled"):
        with training_session(model_resources_id="mr-1"):
            pass


def test_attach_full_weight_to_full_weight_resource(monkeypatch):
    """``lora_enabled=False`` attaches to a full-weight resource without a LoRA session."""
    events, _ = install(monkeypatch, resource_model=StubResourceModel(lora_enabled=False))

    with training_session(model_resources_id="mr-1", lora_enabled=False):
        pass

    assert ("create_session", None) in events


def test_attach_rejects_missing_generator(monkeypatch):
    install(monkeypatch, resource_model=StubResourceModel(num_generator_replicas=0))

    with pytest.raises(ValueError, match="generator"):
        with training_session(model_resources_id="mr-1", num_generator_replicas=1):
            pass


def test_attach_allows_trainer_only_without_generator(monkeypatch):
    events, _ = install(monkeypatch, resource_model=StubResourceModel(num_generator_replicas=0))

    with training_session(model_resources_id="mr-1", num_generator_replicas=0, lora_config=LoraConfig(rank=8)):
        pass

    assert ("create_session", LoraConfig(rank=8)) in events


def test_teardown_is_best_effort(monkeypatch, caplog):
    events, _ = install(monkeypatch, stop_error=RuntimeError("stop boom"))

    with caplog.at_level("WARNING"):
        with training_session(model_resources_id="mr-1", lora_config=LoraConfig(rank=8)):
            pass

    assert events[-2:] == ["session.stop", "resource.detach"]
    assert "stop boom" in caplog.text


def test_rejects_both_model_and_resource():
    with pytest.raises(ValueError, match="not both"):
        with training_session(model=BASE_MODEL, model_resources_id="mr-1"):
            pass


def test_rejects_neither_model_nor_resource():
    with pytest.raises(ValueError, match="Set model"):
        with training_session():
            pass


def test_resolve_base_model_uses_config_model_when_fresh():
    assert resolve_base_model(StubConfig(model=BASE_MODEL)) == BASE_MODEL


def test_resolve_base_model_reads_resource_when_attaching(monkeypatch):
    retrieve_calls = patch_together(monkeypatch, StubResourceModel(base_model="meta-llama/Llama-3-8B"))

    assert resolve_base_model(StubConfig(model_resources_id="mr-1")) == "meta-llama/Llama-3-8B"
    assert retrieve_calls == ["mr-1"]


def test_resolve_base_model_rejects_neither():
    with pytest.raises(ValueError, match="Set model"):
        resolve_base_model(StubConfig())
