"""Load `demos/tinker-migration/` scripts without their heavy dependencies.

The demos are standalone scripts in a hyphenated directory, so they are not importable as a
package; and CI does not sync the `tinker` group. Stubbing the third-party imports lets the
demos' own control flow be exercised offline. `test_sdk_contract.py` covers what stubbing
cannot: that the wrapper really provides what the demos call.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

_DEMOS_DIR = Path(__file__).resolve().parents[2] / "demos" / "tinker-migration"

_STUB_MODULES = (
    "datasets",
    "dotenv",
    "wandb",
    "together",
    "together.post_training",
    "together.post_training.tinker",
    "tinker_cookbook",
    "tinker_cookbook.model_info",
    "tinker_cookbook.renderers",
    "tinker_cookbook.recipes",
    "tinker_cookbook.recipes.math_rl",
    "tinker_cookbook.recipes.math_rl.math_env",
    "tinker_cookbook.recipes.math_rl.math_grading",
    "tinker_cookbook.supervised",
    "tinker_cookbook.supervised.data",
    "tinker_cookbook.tokenizer_utils",
)


def _load_demo(filename: str) -> ModuleType:
    """Import a demo script with its third-party dependencies replaced by mocks."""
    module_name = f"{Path(filename).stem}_demo"
    saved = {name: sys.modules.get(name) for name in (*_STUB_MODULES, module_name)}
    try:
        for name in _STUB_MODULES:
            sys.modules[name] = MagicMock()

        spec = importlib.util.spec_from_file_location(module_name, _DEMOS_DIR / filename)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


@pytest.fixture(scope="module")
def grpo_demo() -> ModuleType:
    return _load_demo("grpo_gsm8k.py")


@pytest.fixture(scope="module")
def async_demo() -> ModuleType:
    return _load_demo("grpo_gsm8k_async.py")


@pytest.fixture(scope="module")
def sft_demo() -> ModuleType:
    return _load_demo("sft_no_robots.py")
