"""OpenAI adapter — calls the OpenAI Chat Completions API via ``httpx``."""

from __future__ import annotations

import asyncio
import logging
import os

import httpx

from speceval.adapters.base import ModelAdapter
from speceval.exceptions import ModelAdapterError, ModelNotFoundError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry / backoff constants
# ---------------------------------------------------------------------------
_MAX_RETRIES = 5
_BASE_DELAY_S = 1.0
_MAX_DELAY_S = 60.0


class OpenAIAdapter(ModelAdapter):
    """Model adapter that calls the OpenAI Chat Completions API.

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys:

        - **model** (*str*) – OpenAI model name (e.g. ``"gpt-4o"``).
        - **api_key** (*str*, optional) – API key.  Falls back to the
          ``OPENAI_API_KEY`` environment variable.
        - **base_url** (*str*, optional) – API base URL.  Defaults to
          ``"https://api.openai.com/v1"``.
        - **temperature** (*float*, optional) – Sampling temperature.
          Defaults to ``0.0``.
        - **max_tokens** (*int*, optional) – Maximum tokens in the response.
          Defaults to ``2048``.
        - **timeout** (*float*, optional) – HTTP request timeout in seconds.
          Defaults to ``120.0``.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._model_name: str = config.get("model", "")
        if not self._model_name:
            raise ModelAdapterError("OpenAIAdapter requires a 'model' name.")

        self._api_key: str = config.get("api_key", "") or os.environ.get(
            "OPENAI_API_KEY", ""
        )
        if not self._api_key:
            raise ModelAdapterError(
                "OpenAI API key not found. Set the OPENAI_API_KEY environment "
                "variable or pass 'api_key' in the config."
            )

        self._base_url: str = config.get(
            "base_url", "https://api.openai.com/v1"
        ).rstrip("/")
        self._temperature: float = float(config.get("temperature", 0.0))
        self._max_tokens: int = int(config.get("max_tokens", 2048))
        self._timeout: float = float(config.get("timeout", 120.0))

        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def predict(self, inputs: list[dict]) -> list[dict]:
        """Call the Chat Completions API for each input.

        Each input dict must contain a ``"messages"`` key (list of message
        dicts with ``role`` and ``content``).  If a dict has no ``"messages"``
        key, it is assumed to contain a single user message under ``"prompt"``
        or ``"text"``.
        """
        client = await self._get_client()
        results: list[dict] = []

        for inp in inputs:
            messages = self._build_messages(inp)
            response_data = await self._call_with_retry(client, messages)
            results.append(response_data)

        return results

    @property
    def metadata(self) -> dict:
        return {
            "backend": "openai",
            "model": self._model_name,
            "base_url": self._base_url,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    @staticmethod
    def _build_messages(inp: dict) -> list[dict]:
        """Convert an input dict to an OpenAI messages list."""
        if "messages" in inp:
            return inp["messages"]

        # Simple prompt format
        text = inp.get("prompt", inp.get("text", ""))
        return [{"role": "user", "content": text}]

    async def _call_with_retry(
        self,
        client: httpx.AsyncClient,
        messages: list[dict],
    ) -> dict:
        """Make the API call with exponential backoff retry."""
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = await client.post(
                    "/chat/completions",
                    json={
                        "model": self._model_name,
                        "messages": messages,
                        "temperature": self._temperature,
                        "max_tokens": self._max_tokens,
                        "stream": False,
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    choice = data["choices"][0]
                    return {
                        "text": choice["message"]["content"].strip(),
                        "finish_reason": choice.get("finish_reason"),
                        "usage": data.get("usage"),
                        "model": data.get("model"),
                    }

                # Handle specific status codes
                if response.status_code == 401:
                    raise ModelNotFoundError(
                        "OpenAI authentication failed — check your API key."
                    )
                if response.status_code == 404:
                    raise ModelNotFoundError(
                        f"Model '{self._model_name}' not found or not accessible."
                    )
                if response.status_code == 429:
                    # Rate limit — retry with backoff
                    delay = min(
                        _BASE_DELAY_S * (2 ** (attempt - 1)), _MAX_DELAY_S
                    )
                    logger.warning(
                        "Rate limited (429) on attempt %d/%d — retrying in %.1fs",
                        attempt,
                        _MAX_RETRIES,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                # Other server errors — retry
                if 500 <= response.status_code < 600:
                    delay = min(
                        _BASE_DELAY_S * (2 ** (attempt - 1)), _MAX_DELAY_S
                    )
                    logger.warning(
                        "Server error %d on attempt %d/%d — retrying in %.1fs",
                        response.status_code,
                        attempt,
                        _MAX_RETRIES,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                # Unexpected status — raise immediately
                raise ModelAdapterError(
                    f"OpenAI API returned status {response.status_code}: "
                    f"{response.text[:500]}"
                )

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                delay = min(_BASE_DELAY_S * (2 ** (attempt - 1)), _MAX_DELAY_S)
                logger.warning(
                    "Connection error on attempt %d/%d: %s — retrying in %.1fs",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        raise ModelAdapterError(
            f"OpenAI API call failed after {_MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
