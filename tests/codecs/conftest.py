from __future__ import annotations

import pytest

from post_training_cookbook.codecs import _CODECS, resolve_codec_class
from tests.codecs import REFERENCE_MODELS, THINKING_OFF_MODELS, load_reference_codec

_COVERED_CLASSES = {resolve_codec_class(model) for model in REFERENCE_MODELS.values()}
_SUPPORTED_CLASSES = {codec_cls for _, codec_cls in _CODECS}
assert _COVERED_CLASSES == _SUPPORTED_CLASSES, (
    "tests/codecs/__init__.py REFERENCE_MODELS leaves a codec class unexercised: "
    f"missing {_SUPPORTED_CLASSES - _COVERED_CLASSES}"
)

_CODEC_PARAMS = [pytest.param(model, id=key) for key, model in REFERENCE_MODELS.items()]
_NO_THINKING_PARAMS = [param for param in _CODEC_PARAMS if param.values[0] in THINKING_OFF_MODELS]


@pytest.fixture(params=_CODEC_PARAMS, scope="session")
def codec(request):
    return load_reference_codec(request.param)


@pytest.fixture(params=_NO_THINKING_PARAMS, scope="session")
def codec_no_thinking(request):
    return load_reference_codec(request.param, enable_thinking=False)


@pytest.fixture(params=_CODEC_PARAMS, scope="session")
def codec_preserve_thinking(request):
    return load_reference_codec(request.param, preserve_thinking=True)
