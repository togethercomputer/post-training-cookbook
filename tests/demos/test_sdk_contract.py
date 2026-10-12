"""Assert the pinned wrapper actually provides what the demos call.

Every other test in this directory stubs `together.post_training.tinker` with a mock, which
answers to any attribute — so a `together` pin predating a method the demos use still passes
them. This test imports the real wrapper instead.
"""

from __future__ import annotations

import pytest

tinker = pytest.importorskip("together.post_training.tinker", reason="together SDK is not installed")

REQUIRED_BY_DEMOS = {
    "ServiceClient": ("create_lora_training_client", "create_lora_training_client_async"),
    "SamplingClient": ("sample", "sample_async"),
    "TrainingClient": (
        "forward_backward",
        "forward_backward_async",
        "optim_step",
        "optim_step_async",
        "save_weights_and_get_sampling_client",
        "save_weights_and_get_sampling_client_async",
    ),
}

# The demos skip checkpointing, but the README's Checkpoints section says these are mapped.
CHECKPOINTING = {
    "ServiceClient": ("create_training_client_from_state", "create_training_client_from_state_with_optimizer"),
    "TrainingClient": ("save_state", "save_state_async", "save_weights_for_sampler", "save_weights_for_sampler_async"),
}

# The README lists these as not mapped. Assert their absence, otherwise the claim rots silently
# once the wrapper grows the methods.
UNSUPPORTED = {
    "ServiceClient": (
        "create_training_client_from_state_async",
        "create_training_client_from_state_with_optimizer_async",
        "create_sampling_client",
    ),
    "TrainingClient": ("load_state", "load_state_with_optimizer", "create_sampling_client"),
}


@pytest.mark.parametrize(
    ("class_name", "method"),
    [
        (class_name, method)
        for documented in (REQUIRED_BY_DEMOS, CHECKPOINTING)
        for class_name, methods in documented.items()
        for method in methods
    ],
)
def test_wrapper_exposes_method(class_name, method):
    assert hasattr(getattr(tinker, class_name), method)


@pytest.mark.parametrize(
    ("class_name", "method"),
    [(class_name, method) for class_name, methods in UNSUPPORTED.items() for method in methods],
)
def test_wrapper_lacks_unmapped_method(class_name, method):
    assert not hasattr(getattr(tinker, class_name), method)


def test_submitted_futures_resolve_both_ways():
    """The sync demos call `.result()`; the async demo awaits `.result_async()`."""
    assert hasattr(tinker.APIFuture, "result")
    assert hasattr(tinker.APIFuture, "result_async")
