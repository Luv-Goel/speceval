"""Tests for the OpenAI adapter with mocked HTTP responses."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from speceval.adapters.openai import OpenAIAdapter
from speceval.exceptions import ModelAdapterError, ModelNotFoundError


class TestOpenAIAdapterInit:
    """Adapter construction / validation."""

    def test_requires_model_name(self):
        """Omitting 'model' raises ModelAdapterError."""
        with pytest.raises(ModelAdapterError, match="requires a 'model' name"):
            OpenAIAdapter({})

    def test_requires_api_key(self, monkeypatch):
        """Without an API key in config or env, raises ModelAdapterError."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ModelAdapterError, match="API key not found"):
            OpenAIAdapter({"model": "gpt-4o"})

    def test_api_key_from_env(self, monkeypatch):
        """API key is read from the OPENAI_API_KEY env var."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
        adapter = OpenAIAdapter({"model": "gpt-4o"})
        assert adapter._api_key == "sk-test-123"

    def test_api_key_from_config(self, monkeypatch):
        """API key in config takes precedence over env."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")
        adapter = OpenAIAdapter({"model": "gpt-4o", "api_key": "sk-cfg-key"})
        assert adapter._api_key == "sk-cfg-key"

    def test_default_values(self, monkeypatch):
        """Defaults are applied for optional config values."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        adapter = OpenAIAdapter({"model": "gpt-4o"})
        assert adapter._base_url == "https://api.openai.com/v1"
        assert adapter._temperature == 0.0
        assert adapter._max_tokens == 2048
        assert adapter._timeout == 120.0

    def test_custom_values(self, monkeypatch):
        """Custom config values override defaults."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        adapter = OpenAIAdapter(
            {
                "model": "gpt-4o",
                "temperature": 0.7,
                "max_tokens": 512,
                "timeout": 30.0,
                "base_url": "https://custom.example.com/v1",
            }
        )
        assert adapter._base_url == "https://custom.example.com/v1"
        assert adapter._temperature == 0.7
        assert adapter._max_tokens == 512
        assert adapter._timeout == 30.0

    def test_metadata_shape(self, monkeypatch):
        """metadata() returns expected keys."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        adapter = OpenAIAdapter({"model": "gpt-4o", "temperature": 0.5})
        meta = adapter.metadata
        assert meta["backend"] == "openai"
        assert meta["model"] == "gpt-4o"
        assert meta["temperature"] == 0.5


class TestOpenAIAdapterPredict:
    """Mocked HTTP calls to the Chat Completions API."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")

    async def _mk_adapter(self, **kwargs) -> OpenAIAdapter:
        cfg = {"model": "gpt-4o", **kwargs}
        return OpenAIAdapter(cfg)

    async def test_successful_response(self):
        """A 200 response returns the parsed text and metadata."""
        adapter = await self._mk_adapter()

        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "  Hello world  "}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "model": "gpt-4o-2024-05-13",
        }

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        # Patch _get_client to return our mock
        with patch.object(adapter, "_get_client", return_value=mock_client):
            results = await adapter.predict([{"prompt": "Hi"}])

        assert len(results) == 1
        assert results[0]["text"] == "Hello world"
        assert results[0]["finish_reason"] == "stop"
        assert results[0]["model"] == "gpt-4o-2024-05-13"

    async def test_messages_format_preserved(self):
        """Inputs with 'messages' are passed through directly."""
        adapter = await self._mk_adapter()

        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Sure!"}, "finish_reason": "stop"}]
        }
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What's 2+2?"},
        ]

        with patch.object(adapter, "_get_client", return_value=mock_client):
            results = await adapter.predict([{"messages": messages}])

        assert len(results) == 1
        assert results[0]["text"] == "Sure!"

        # Verify the API was called with the correct messages
        called_payload = mock_client.post.call_args[1]["json"]
        assert called_payload["messages"] == messages

    async def test_authentication_error(self):
        """A 401 status raises ModelNotFoundError."""
        adapter = await self._mk_adapter()

        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 401
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        with patch.object(adapter, "_get_client", return_value=mock_client):
            with pytest.raises(ModelNotFoundError, match="authentication failed"):
                await adapter.predict([{"prompt": "Hi"}])

    async def test_model_not_found(self):
        """A 404 status raises ModelNotFoundError."""
        adapter = await self._mk_adapter()

        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        with patch.object(adapter, "_get_client", return_value=mock_client):
            with pytest.raises(ModelNotFoundError, match="not found"):
                await adapter.predict([{"prompt": "Hi"}])

    async def test_rate_limit_retry_then_success(self):
        """A 429 followed by a 200 succeeds (retries once)."""
        adapter = await self._mk_adapter()

        error_response = AsyncMock(spec=httpx.Response)
        error_response.status_code = 429

        ok_response = AsyncMock(spec=httpx.Response)
        ok_response.status_code = 200
        ok_response.json.return_value = {
            "choices": [{"message": {"content": "Retried!"}, "finish_reason": "stop"}]
        }

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = [error_response, ok_response]

        with patch.object(adapter, "_get_client", return_value=mock_client):
            results = await adapter.predict([{"prompt": "Hi"}])

        assert results[0]["text"] == "Retried!"
        assert mock_client.post.call_count == 2

    async def test_unexpected_status_raises(self):
        """An unexpected 3xx status raises ModelAdapterError."""
        adapter = await self._mk_adapter()

        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 302
        mock_response.text = "Redirect"
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        with patch.object(adapter, "_get_client", return_value=mock_client):
            with pytest.raises(ModelAdapterError, match="302"):
                await adapter.predict([{"prompt": "Hi"}])

    async def test_connection_retry_then_success(self):
        """A httpx.ConnectError followed by a 200 succeeds."""
        adapter = await self._mk_adapter()

        ok_response = AsyncMock(spec=httpx.Response)
        ok_response.status_code = 200
        ok_response.json.return_value = {
            "choices": [{"message": {"content": "Connected!"}, "finish_reason": "stop"}]
        }

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = [
            httpx.ConnectError("connection refused"),
            ok_response,
        ]

        with patch.object(adapter, "_get_client", return_value=mock_client):
            results = await adapter.predict([{"prompt": "Hello"}])

        assert results[0]["text"] == "Connected!"
        assert mock_client.post.call_count == 2

    async def test_all_retries_exhausted(self):
        """After max retries, all failing, raises ModelAdapterError."""
        adapter = await self._mk_adapter()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = httpx.TimeoutException("timeout")

        with patch.object(adapter, "_get_client", return_value=mock_client):
            with pytest.raises(ModelAdapterError, match="failed after"):
                await adapter.predict([{"prompt": "Hi"}])

        # _MAX_RETRIES = 5
        assert mock_client.post.call_count == 5


class TestOpenAIAdapterBuildMessages:
    """Test the static _build_messages helper."""

    def test_messages_list_preserved(self):
        """A 'messages' key is returned as-is."""
        msgs = [{"role": "user", "content": "hello"}]
        result = OpenAIAdapter._build_messages({"messages": msgs})
        assert result == msgs

    def test_prompt_wrapped(self):
        """'prompt' key gets wrapped in a user message."""
        result = OpenAIAdapter._build_messages({"prompt": "Hello"})
        assert result == [{"role": "user", "content": "Hello"}]

    def test_text_wrapped(self):
        """'text' key gets wrapped in a user message."""
        result = OpenAIAdapter._build_messages({"text": "Hi there"})
        assert result == [{"role": "user", "content": "Hi there"}]

    def test_fallback_to_empty(self):
        """Missing keys result in an empty user message."""
        result = OpenAIAdapter._build_messages({"unexpected": "val"})
        assert result == [{"role": "user", "content": ""}]
