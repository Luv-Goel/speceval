"""Adapter registration — auto-registers available backends with the factory."""

from __future__ import annotations

import logging

from speceval.adapters.base import ModelAdapterFactory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HuggingFace adapter (optional dependency)
# ---------------------------------------------------------------------------

try:
    import transformers  # noqa: F401
    import torch  # noqa: F401

    from speceval.adapters.huggingface import HuggingFaceAdapter

    ModelAdapterFactory.register("huggingface", HuggingFaceAdapter)
    logger.debug("Registered HuggingFace adapter.")
except ImportError:
    logger.debug(
        "HuggingFace adapter not registered — install 'transformers' and 'torch'."
    )

# ---------------------------------------------------------------------------
# OpenAI adapter (optional dependency)
# ---------------------------------------------------------------------------

try:
    import httpx  # noqa: F401

    from speceval.adapters.openai import OpenAIAdapter

    ModelAdapterFactory.register("openai", OpenAIAdapter)
    logger.debug("Registered OpenAI adapter.")
except ImportError:
    logger.debug(
        "OpenAI adapter not registered — install 'httpx'."
    )


__all__ = [
    "ModelAdapterFactory",
]
