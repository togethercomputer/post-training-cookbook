"""Session lifecycle for recipes, with shared model-resource support.

Together Post Training separates the GPU pods that host a base model (a *model
resource*) from the training sessions that run on them. When a resource is
created with ``lora_enabled=True``, every session attached to it is an
independent LoRA adapter training against the shared base weights, so N
adapters cost roughly one base model's worth of GPUs instead of N.

A run names exactly one model source: a ``model`` provisions a fresh resource
that the run owns, or a ``model_resources_id`` attaches to one that already
exists and is left running afterwards. When attaching, the resource is
authoritative — it defines the base model — so a run does not also name a model.

The resource lifecycle (provision, attach, stop) is owned by the SDK's
:class:`together.post_training.ModelResourcesClient`. This module covers the cookbook
side: :func:`resolve_base_model` reports which base model a config will train,
and :func:`training_session` is a context manager that provisions-or-attaches,
yields a ready session, and tears the resource/session down on exit.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager

import httpx
from together import Together
from together.post_training import ComputeConfig, ModelResourcesClient, SessionMetadata

__all__ = ["resolve_base_model", "training_session"]

_READY = "MODEL_RESOURCES_STATUS_READY"
_CLIENT_TIMEOUT = httpx.Timeout(timeout=300.0, connect=300.0)

logger = logging.getLogger(__name__)


def resolve_base_model(config) -> str:
    """The base model a run will train on.

    When attaching to a shared resource the resource is authoritative, so the
    model name is read from it; otherwise the config names the model to
    provision. Callers use this to pick the tokenizer/codec before the (more
    expensive) session is created.
    """
    _require_single_model_source(config.model, config.model_resources_id)
    if config.model_resources_id is None:
        return config.model
    api_key = os.environ.get("TOGETHER_API_KEY")
    base_url = os.environ.get("TOGETHER_BASE_URL")
    with Together(api_key=api_key, base_url=base_url, timeout=_CLIENT_TIMEOUT, max_retries=5) as client:
        return client.post_training.model_resources.retrieve(config.model_resources_id).base_model


@contextmanager
def training_session(
    *,
    model: str | None = None,
    model_resources_id: str | None = None,
    display_name: str | None = None,
    metadata: SessionMetadata | None = None,
    resume_from_checkpoint_id: str | None = None,
    load_optimizer: bool | None = None,
    lora_config=None,
    lora_enabled: bool = True,
    optimizer_config=None,
    num_generator_replicas: int = 1,
):
    """Provision-or-attach a model resource and yield a ready session.

    With ``model`` set, provision a fresh resource the run owns; on exit the
    session and the resource are both stopped, freeing the GPUs. With
    ``model_resources_id`` set, attach a session to that shared resource as a
    tenant — after checking it is READY and can host the session (LoRA enabled,
    generator present when sampling) — and on exit stop the session but leave
    the resource running for other tenants.

    The optimizer (``optimizer_config``) is a property of the resource, so it
    applies only when provisioning a fresh one; when attaching, the optimizer is
    inherited from the existing resource and this argument is unused. The caller's
    per-step optimizer params must still match the attached resource's optimizer —
    a mismatch (e.g. Muon params against an AdamW resource) is rejected server-side.

    ``lora_enabled`` selects the resource mode, so like ``optimizer_config`` it only
    applies when provisioning; when attaching it is checked against the existing
    resource rather than changing it. ``lora_config`` only configures a LoRA session
    and must be omitted for full-weight training.

    ``resume_from_checkpoint_id`` loads a training checkpoint onto the new session.
    ``load_optimizer=False`` loads weights only (fresh optimizer, step 0); omit or
    ``True`` restores optimizer state and the step counter.

    Teardown is best-effort: a cleanup failure is logged, not raised, so it can
    never mask an exception from the ``with`` body. ``api_key``/``base_url`` are
    read from the environment.
    """
    _require_single_model_source(model, model_resources_id)
    if not lora_enabled and lora_config is not None:
        raise ValueError("lora_config must be omitted when lora_enabled=False")
    api_key = os.environ.get("TOGETHER_API_KEY")
    base_url = os.environ.get("TOGETHER_BASE_URL")
    attached = model_resources_id is not None

    if attached:
        resource = ModelResourcesClient.attach(
            model_resources_id=model_resources_id,
            api_key=api_key,
            base_url=base_url,
        )
    else:
        resource = ModelResourcesClient.create(
            base_model=model,
            lora_enabled=lora_enabled,
            optimizer_config=optimizer_config,
            compute_config=ComputeConfig(num_generator_replicas=num_generator_replicas),
            api_key=api_key,
            base_url=base_url,
        )

    session = None
    try:
        if attached:
            _check_attachable(
                resource.retrieve(),
                want_lora=lora_enabled,
                num_generator_replicas=num_generator_replicas,
            )
        session = resource.create_session(
            display_name=display_name,
            metadata=metadata,
            resume_from_checkpoint_id=resume_from_checkpoint_id,
            load_optimizer=load_optimizer,
            lora_config=lora_config,
        )
        print(f"Session ready: {session.session_id}")
        yield session
    except Exception:
        _log_statuses(session, resource)
        raise
    finally:
        if session is not None:
            try:
                session.stop()
            except Exception as exc:
                logger.warning("stopping session %s failed (%s: %s)", session.session_id, type(exc).__name__, exc)
        try:
            resource.detach() if attached else resource.stop()
        except Exception as exc:
            logger.warning(
                "releasing resource %s failed (%s: %s)", resource.model_resources_id, type(exc).__name__, exc
            )


def _log_statuses(session, resource) -> None:
    """Report session/resource status next to a failure from the ``with`` body.

    A session with no server operation for the server's inactivity window (1 h)
    expires and its resource is reaped, so the failing operation itself says
    nothing about why — only the statuses do.
    """
    try:
        if session is not None:
            logger.warning("session %s status=%s", session.session_id, session.retrieve().status)
        logger.warning("resource %s status=%s", resource.model_resources_id, resource.retrieve().status)
    except Exception as exc:
        logger.warning("reading session/resource status failed (%s: %s)", type(exc).__name__, exc)


def _require_single_model_source(model, model_resources_id) -> None:
    has_model = model is not None
    has_resource = model_resources_id is not None
    if has_model and has_resource:
        raise ValueError(
            "Set model to provision a fresh resource OR model_resources_id to attach to an existing one, "
            "not both — a shared resource already defines its base model."
        )
    if not has_model and not has_resource:
        raise ValueError("Set model to provision a resource, or model_resources_id to attach to one.")


def _check_attachable(resource, *, want_lora: bool, num_generator_replicas: int) -> None:
    if resource.status != _READY:
        raise RuntimeError(f"Model resource {resource.id} is not ready (status {resource.status})")
    if want_lora and not resource.lora_enabled:
        raise ValueError(f"Resource {resource.id} was provisioned without lora_enabled; it cannot host LoRA sessions")
    if num_generator_replicas > 0 and resource.compute_config.num_generator_replicas == 0:
        raise ValueError(f"Resource {resource.id} has no generator replicas; sampling would fail")
