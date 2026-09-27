"""Adapter registration — auto-registers available backends with the factory."""

from __future__ import annotations

import logging

from speceval.adapters.base import ModelAdapterFactory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HuggingFace adapter (optional dependency)
# ---------------------------------------------------------------------------

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401

    from speceval.adapters.huggingface import HuggingFaceAdapter

    ModelAdapterFactory.register("huggingface", HuggingFaceAdapter)
    logger.debug("Registered HuggingFace adapter.")
except ImportError:
    logger.debug("HuggingFace adapter not registered — install 'transformers' and 'torch'.")

# ---------------------------------------------------------------------------
# OpenAI adapter (optional dependency)
# ---------------------------------------------------------------------------

try:
    import httpx  # noqa: F401

    from speceval.adapters.anthropic import AnthropicAdapter
    from speceval.adapters.openai import OpenAIAdapter

    ModelAdapterFactory.register("openai", OpenAIAdapter)
    ModelAdapterFactory.register("anthropic", AnthropicAdapter)
    logger.debug("Registered OpenAI and Anthropic adapters.")
except ImportError:
    logger.debug("OpenAI and Anthropic adapters not registered — install 'httpx'.")


__all__ = [
    "ModelAdapterFactory",
]
