"""Top-level package exports for SMC engines.

Keep imports lazy so backend-specific dependencies stay isolated.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from smcsd.engine import SMCEngine
    from smcsd.vllm_backend.engine import SMCVLLMEngine

__all__ = ["SMCEngine", "SMCVLLMEngine"]


def __getattr__(name: str) -> Any:
    if name == "SMCEngine":
        return import_module("smcsd.engine").SMCEngine
    if name == "SMCVLLMEngine":
        return import_module("smcsd.vllm_backend.engine").SMCVLLMEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
