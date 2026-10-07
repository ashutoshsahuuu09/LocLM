"""Ollama runtime implementation.

Communicates with the Ollama API on localhost for local model inference.
All HTTP requests are validated to be localhost-only before execution.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from loclm.models.base import ChatMessage, ModelInfo, ModelRuntime
from loclm.security.network import validate_ollama_url

logger = logging.getLogger(__name__)

# Default Ollama API endpoint
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"


class OllamaRuntime(ModelRuntime):
    """Ollama-based local model runtime.

    Wraps the Ollama REST API for:
    - Model listing and management
    - Chat completions (streaming and non-streaming)
    - Text generation
    - Model loading/unloading
    """

    def __init__(
        self,
        host: str = DEFAULT_OLLAMA_HOST,
        timeout: float = 120.0,
    ) -> None:
        # Validate the host is local before storing
        validate_ollama_url(host)
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._host,
                timeout=httpx.Timeout(self._timeout, connect=10.0),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @property
    def runtime_name(self) -> str:
        return "Ollama"

    async def is_available(self) -> bool:
        """Check if Ollama is running and responding."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags", timeout=5.0)
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError) as e:
            logger.debug("Ollama not available: %s", e)
            return False

    async def list_models(self) -> list[ModelInfo]:
        """List all locally installed Ollama models."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            response.raise_for_status()
            data = response.json()

            models: list[ModelInfo] = []
            for m in data.get("models", []):
                model_info = _parse_ollama_model(m)
                models.append(model_info)

            models.sort(key=lambda x: x.name)
            return models

        except (httpx.HTTPError, KeyError, ValueError) as e:
            logger.error("Failed to list Ollama models: %s", e)
            return []

    async def model_info(self, model_name: str) -> ModelInfo | None:
        """Get info about a specific model."""
        try:
            client = await self._get_client()
            response = await client.post(
                "/api/show",
                json={"name": model_name},
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()

            # Extract details from the show response
            details = data.get("details", {})
            size_bytes = data.get("size", 0)

            return ModelInfo(
                name=model_name,
                size_gb=round(size_bytes / (1024**3), 2) if size_bytes else 0.0,
                family=details.get("family", "unknown"),
                parameter_size=details.get("parameter_size", ""),
                quantization=details.get("quantization_level", ""),
                format=details.get("format", ""),
            )

        except (httpx.HTTPError, KeyError, ValueError) as e:
            logger.error("Failed to get model info for '%s': %s", model_name, e)
            return None

    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from the Ollama registry.

        Note: This requires internet for the initial download.
        After download, the model is fully local.
        """
        try:
            client = await self._get_client()
            response = await client.post(
                "/api/pull",
                json={"name": model_name, "stream": False},
                timeout=httpx.Timeout(1800.0, connect=30.0),  # 30 min for large models
            )
            response.raise_for_status()
            return True

        except (httpx.HTTPError, httpx.TimeoutException) as e:
            logger.error("Failed to pull model '%s': %s", model_name, e)
            return False

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat request and return the complete response."""
        payload = self._build_chat_payload(
            model, messages, stream=False, temperature=temperature, max_tokens=max_tokens
        )

        try:
            client = await self._get_client()
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")

        except httpx.TimeoutException:
            logger.error("Chat request timed out for model '%s'", model)
            raise
        except httpx.HTTPError as e:
            logger.error("Chat request failed: %s", e)
            raise

    async def chat_stream(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens."""
        payload = self._build_chat_payload(
            model, messages, stream=True, temperature=temperature, max_tokens=max_tokens
        )

        client = await self._get_client()
        async with client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    if chunk.get("done"):
                        break
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue

    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        stream: bool = False,
        temperature: float = 0.7,
    ) -> str:
        """Simple text generation."""
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }

        try:
            client = await self._get_client()
            response = await client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")

        except httpx.HTTPError as e:
            logger.error("Generate request failed: %s", e)
            raise

    async def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is currently loaded in Ollama's memory."""
        try:
            client = await self._get_client()
            response = await client.get("/api/ps")
            if response.status_code != 200:
                return False
            data = response.json()
            for m in data.get("models", []):
                if m.get("name", "").startswith(model_name.split(":")[0]):
                    return True
            return False
        except (httpx.HTTPError, KeyError):
            return False

    async def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory by sending a keep_alive=0 request."""
        try:
            client = await self._get_client()
            response = await client.post(
                "/api/generate",
                json={
                    "model": model_name,
                    "prompt": "",
                    "keep_alive": 0,
                },
            )
            return response.status_code == 200
        except httpx.HTTPError as e:
            logger.debug("Failed to unload model '%s': %s", model_name, e)
            return False

    def _build_chat_payload(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        stream: bool,
        temperature: float,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Build the JSON payload for Ollama /api/chat."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "options": {
                "temperature": temperature,
            },
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        return payload


def _parse_ollama_model(data: dict[str, Any]) -> ModelInfo:
    """Parse a model entry from the Ollama /api/tags response."""
    name = data.get("name", "unknown")
    size_bytes = data.get("size", 0)
    details = data.get("details", {})

    return ModelInfo(
        name=name,
        size_gb=round(size_bytes / (1024**3), 2) if size_bytes else 0.0,
        family=details.get("family", "unknown"),
        parameter_size=details.get("parameter_size", ""),
        quantization=details.get("quantization_level", ""),
        format=details.get("format", ""),
        digest=data.get("digest", ""),
        modified_at=data.get("modified_at", ""),
    )
