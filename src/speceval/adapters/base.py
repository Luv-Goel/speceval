"""Model adapter — abstract interface for running inference on any backend."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ModelAdapter(ABC):
    """Abstract interface for model inference backends.

    Subclasses implement :meth:`predict` and optionally override :attr:`metadata`.
    """

    @abstractmethod
    async def predict(self, inputs: list[dict]) -> list[dict]:
        """Run inference on a batch of inputs and return predictions.

        Parameters
        ----------
        inputs : list[dict]
            A list of input dictionaries (e.g. ``[{"messages": [...]}]``).

        Returns
        -------
        list[dict]
            A list of prediction dictionaries (one per input), following
            the same ordering as *inputs*.

        Raises
        ------
        ModelAdapterError
            If inference fails for the entire batch.
        """
        ...

    @property
    def metadata(self) -> dict:
        """Return model/adapter metadata (model name, backend, dtype, …)."""
        return {}


class ModelAdapterFactory:
    """Registry and factory for creating :class:`ModelAdapter` instances.

    Usage::

        ModelAdapterFactory.register("huggingface", HuggingFaceAdapter)
        adapter = ModelAdapterFactory.create({"model": "bert-base-uncased", ...})
    """

    _registry: dict[str, type[ModelAdapter]] = {}

    @classmethod
    def register(cls, name: str, adapter_cls: type[ModelAdapter]) -> None:
        """Register an adapter class under *name*."""
        if not issubclass(adapter_cls, ModelAdapter):
            raise TypeError(f"{adapter_cls.__name__} must subclass ModelAdapter")
        cls._registry[name.lower()] = adapter_cls
        logger.debug("Registered model adapter '%s' -> %s", name, adapter_cls.__name__)

    @classmethod
    def create(cls, config: dict) -> ModelAdapter:
        """Instantiate an adapter from a configuration dictionary.

        The dict **must** contain a ``"model"`` key whose value is the
        model name.  The adapter backend is selected from the ``"backend"``
        key (default: ``"openai"``).
        """
        backend = config.get("backend", "openai").lower()
        adapter_cls = cls._registry.get(backend)
        if adapter_cls is None:
            available = ", ".join(sorted(cls._registry))
            raise KeyError(
                f"Unknown adapter backend '{backend}'. "
                f"Registered backends: [{available}]"
            )
        logger.info("Creating %s adapter for model '%s'", backend, config.get("model", ""))
        return adapter_cls(config)

    @classmethod
    def list_backends(cls) -> list[str]:
        """Return the list of registered backend names."""
        return list(cls._registry)

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a backend from the registry."""
        cls._registry.pop(name.lower(), None)
