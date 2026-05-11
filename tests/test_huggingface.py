"""Tests for the HuggingFace adapter."""

from __future__ import annotations

import pytest

from speceval.adapters.base import ModelAdapter, ModelAdapterFactory
from speceval.exceptions import ModelAdapterError


class TestHuggingFaceAdapterInit:
    """HuggingFace adapter construction (no actual models loaded)."""

    def test_registered_as_backend(self):
        """HuggingFaceAdapter is registered as 'huggingface' backend."""
        backends = ModelAdapterFactory.list_backends()
        assert "huggingface" in backends

    def test_factory_creates_hf_adapter(self):
        """Factory.create with backend='huggingface' returns a ModelAdapter."""
        adapter = ModelAdapterFactory.create(
            {"backend": "huggingface", "model": "bert-base-uncased"}
        )
        assert isinstance(adapter, ModelAdapter)

    def test_missing_model_key(self):
        """A missing or empty 'model' key should raise."""
        from speceval.adapters.huggingface import HuggingFaceAdapter

        with pytest.raises(ModelAdapterError, match="requires a 'model'"):
            HuggingFaceAdapter({})


class TestModelAdapterFactory:
    """Tests for the factory pattern."""

    def setup_method(self):
        ModelAdapterFactory._registry.clear()

    def test_register_backend(self):
        """Register a minimal adapter class."""

        class DummyAdapter(ModelAdapter):
            async def predict(self, inputs):
                return []

        ModelAdapterFactory.register("dummy", DummyAdapter)
        assert "dummy" in ModelAdapterFactory.list_backends()

    def test_register_non_adapter_raises(self):
        """Registering a class that doesn't subclass ModelAdapter raises TypeError."""

        class NotAdapter:
            pass

        with pytest.raises(TypeError, match="must subclass ModelAdapter"):
            ModelAdapterFactory.register("bad", NotAdapter)  # type: ignore

    def test_unknown_backend_raises(self):
        """Requesting an unregistered backend raises KeyError."""
        ModelAdapterFactory._registry.clear()
        with pytest.raises(KeyError, match="Unknown adapter backend"):
            ModelAdapterFactory.create({"backend": "nonexistent", "model": "x"})

    def test_list_backends(self):
        """list_backends() returns registered names."""

        class A(ModelAdapter):
            async def predict(self, inputs):
                return []

        class B(ModelAdapter):
            async def predict(self, inputs):
                return []

        ModelAdapterFactory.register("backend_a", A)
        ModelAdapterFactory.register("backend_b", B)
        backends = ModelAdapterFactory.list_backends()
        assert "backend_a" in backends
        assert "backend_b" in backends

    def test_unregister(self):
        """unregister() removes a backend."""

        class Dummy(ModelAdapter):
            async def predict(self, inputs):
                return []

        ModelAdapterFactory.register("dummy", Dummy)
        ModelAdapterFactory.unregister("dummy")
        assert "dummy" not in ModelAdapterFactory.list_backends()
