"""HuggingFace adapter — loads models from the Hugging Face Hub via ``transformers``."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from speceval.adapters.base import ModelAdapter
from speceval.exceptions import ModelAdapterError, ModelNotFoundError

logger = logging.getLogger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf_adapter")


class HuggingFaceAdapter(ModelAdapter):
    """Model adapter backed by Hugging Face ``transformers``.

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys:

        - **model** (*str*) – Hugging Face model ID or local path.
        - **device** (*str*, optional) – Device to run on (``"auto"``, ``"cpu"``,
          ``"cuda"``, …).  Defaults to ``"auto"``.
        - **dtype** (*str*, optional) – Torch dtype (``"float16"``, ``"bfloat16"``,
          ``"float32"``).  Defaults to ``"auto"``.
        - **trust_remote_code** (*bool*, optional) – Whether to trust remote code.
          Defaults to ``False``.
        - **batch_size** (*int*, optional) – Inference batch size.  Defaults to 8.
        - **max_length** (*int*, optional) – Maximum generation length.  Defaults to 512.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._model_name: str = config.get("model", "")
        if not self._model_name:
            raise ModelAdapterError("HuggingFaceAdapter requires a 'model' name.")

        self._device = config.get("device", "auto")
        self._dtype_str = config.get("dtype", "auto")
        self._trust_remote_code = config.get("trust_remote_code", False)
        self._batch_size = int(config.get("batch_size", 8))
        self._max_length = int(config.get("max_length", 512))

        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def predict(self, inputs: list[dict]) -> list[dict]:
        """Run inference on a batch of inputs.

        The model is loaded lazily on the first call.
        """
        if not self._loaded:
            await self._load_model()

        # Run inference on the thread pool to avoid blocking the event loop
        return await asyncio.get_event_loop().run_in_executor(
            _EXECUTOR,
            self._predict_sync,
            inputs,
        )

    @property
    def metadata(self) -> dict:
        return {
            "backend": "huggingface",
            "model": self._model_name,
            "device": self._device,
            "dtype": self._dtype_str,
            "trust_remote_code": self._trust_remote_code,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _load_model(self) -> None:
        """Load the model and tokenizer in the thread pool."""
        logger.info("Loading HuggingFace model '%s' …", self._model_name)

        try:
            await asyncio.get_event_loop().run_in_executor(
                _EXECUTOR, self._load_model_sync
            )
        except Exception as exc:
            raise ModelNotFoundError(
                f"Failed to load model '{self._model_name}': {exc}"
            ) from exc

        self._loaded = True
        logger.info("Model '%s' loaded successfully.", self._model_name)

    def _load_model_sync(self) -> None:
        """Synchronous model loading (runs in thread pool)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Resolve torch dtype
        torch_dtype = self._resolve_dtype()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=self._trust_remote_code,
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype=torch_dtype,
            device_map=self._device,
            trust_remote_code=self._trust_remote_code,
        )
        self._model.eval()

    def _predict_sync(self, inputs: list[dict]) -> list[dict]:
        """Synchronous batched inference (runs in thread pool)."""
        import torch

        results: list[dict] = []

        for i in range(0, len(inputs), self._batch_size):
            batch = inputs[i : i + self._batch_size]
            batch_results = self._predict_batch(batch)
            results.extend(batch_results)

        return results

    def _predict_batch(self, batch: list[dict]) -> list[dict]:
        """Tokenize, generate, and decode a single batch."""
        # Extract prompts — input can be plain text or structured
        prompts = []
        for item in batch:
            if "messages" in item:
                # Chat template
                prompt = self._tokenizer.apply_chat_template(
                    item["messages"], tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = item.get("prompt", item.get("text", ""))
            prompts.append(prompt)

        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length * 2,  # allow room for generation
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._max_length,
                pad_token_id=self._tokenizer.pad_token_id,
                do_sample=False,
            )

        decoded = self._tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return [{"text": text.strip()} for text in decoded]

    def _resolve_dtype(self) -> torch.dtype:
        """Map a dtype string to a torch dtype."""
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float64": torch.float64,
            "auto": "auto",
        }
        dtype = mapping.get(self._dtype_str, "auto")
        if dtype == "auto":
            return "auto"
        return dtype
